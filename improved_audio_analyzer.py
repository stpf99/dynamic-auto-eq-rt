#!/usr/bin/env python3
"""
Zaawansowana aplikacja do analizy audio z automatyczną korekcją EQ
Analizuje niedoskonałości w różnych zakresach częstotliwości i proponuje krzywe korekcyjne
"""
import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Gst', '1.0')
from gi.repository import Gtk, Gst, GLib, GObject
import cairo
import numpy as np
import threading
from scipy import signal
from scipy.fft import rfft, rfftfreq
from collections import deque
import math
import os
from pydub import AudioSegment

# Inicjalizacja GStreamer
Gst.init(None)

class AudioAnalyzer:
    """Klasa analizująca audio i wykrywająca niedoskonałości"""

    def __init__(self):
        self.sample_rate = 44100
        self.buffer_size = 4096
        self.freq_bands = {
            'low': (20, 250),          # Bass
            'mid-low': (250, 800),     # Low-mids
            'mid': (800, 3000),        # Mids
            'mid-high': (3000, 8000),  # High-mids
            'high': (8000, 20000)      # Highs
        }

        # Bufory dla analizy czasowej całego utworu
        self.time_analysis = {band: [] for band in self.freq_bands}
        self.correction_curves = {band: [] for band in self.freq_bands}
        self.time_stamps = []
        
        # Parametry detekcji niedoskonałości
        self.detection_params = {
            'resonance_threshold': 6.0,     # dB powyżej średniej
            'null_threshold': -12.0,        # dB poniżej średniej
            'harshness_factor': 1.5,        # Współczynnik ostrości
            'muddiness_threshold': 0.7      # Próg zamulenia
        }

        # Sztywne ustawienia korekcji
        self.fixed_corrections = {
            'low': 0.0,
            'mid-low': 0.0,
            'mid': 0.0,
            'mid-high': 0.0,
            'high': 0.0
        }

    def analyze_full_audio(self, audio_data, chunk_size=4096):
        """Analizuje cały utwór fragmentami"""
        self.time_analysis = {band: [] for band in self.freq_bands}
        self.correction_curves = {band: [] for band in self.freq_bands}
        self.time_stamps = []
        
        num_chunks = len(audio_data) // chunk_size
        
        for i in range(0, len(audio_data) - chunk_size, chunk_size):
            chunk = audio_data[i:i + chunk_size]
            timestamp = i / self.sample_rate
            self.time_stamps.append(timestamp)
            
            # Analiza spektralna fragmentu
            band_analysis, spectrum, freqs = self.analyze_spectrum(chunk)
            
            # Wykryj niedoskonałości dla fragmentu
            imperfections = self.detect_imperfections(band_analysis)
            
            # Zapisz analizę dla każdego pasma
            for band_name in self.freq_bands:
                if band_name in band_analysis:
                    self.time_analysis[band_name].append(band_analysis[band_name])
                    
                    # Oblicz korekcję dla tego fragmentu
                    correction = 0.0
                    if band_name in imperfections:
                        for issue in imperfections[band_name]:
                            correction += issue['correction']
                    
                    self.correction_curves[band_name].append(correction)
        
        return self.time_analysis, self.correction_curves

    def analyze_spectrum(self, audio_data):
        """Analizuje spektrum częstotliwości"""
        # Zastosuj okno Hamminga
        window = np.hamming(len(audio_data))
        windowed_data = audio_data * window

        # FFT
        spectrum = np.abs(rfft(windowed_data))
        freqs = rfftfreq(len(windowed_data), 1/self.sample_rate)

        # Analiza per band
        band_analysis = {}
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_spectrum = spectrum[mask]

            if len(band_spectrum) > 0:
                band_analysis[band_name] = {
                    'mean': np.mean(band_spectrum),
                    'peak': np.max(band_spectrum),
                    'std': np.std(band_spectrum),
                    'energy': np.sum(band_spectrum**2),
                    'centroid': np.sum(freqs[mask] * band_spectrum) / np.sum(band_spectrum)
                }
            else:
                band_analysis[band_name] = {
                    'mean': 0, 'peak': 0, 'std': 0, 'energy': 0, 'centroid': 0
                }

        return band_analysis, spectrum, freqs

    def detect_imperfections(self, band_analysis):
        """Wykrywa niedoskonałości w poszczególnych pasmach"""
        imperfections = {}

        for band_name, metrics in band_analysis.items():
            issues = []

            # Konwersja do dB
            mean_db = 20 * np.log10(metrics['mean'] + 1e-10)
            peak_db = 20 * np.log10(metrics['peak'] + 1e-10)

            # Detekcja rezonansów
            if peak_db - mean_db > self.detection_params['resonance_threshold']:
                issues.append({
                    'type': 'resonance',
                    'severity': (peak_db - mean_db) / self.detection_params['resonance_threshold'],
                    'frequency': metrics['centroid'],
                    'correction': -(peak_db - mean_db) * 0.7
                })

            # Detekcja dziur częstotliwościowych
            if mean_db < self.detection_params['null_threshold']:
                issues.append({
                    'type': 'null',
                    'severity': abs(mean_db / self.detection_params['null_threshold']),
                    'frequency': metrics['centroid'],
                    'correction': abs(mean_db) * 0.5
                })

            # Detekcja ostrości (high-mids/highs)
            if band_name in ['mid-high', 'high']:
                if metrics['std'] / metrics['mean'] > self.detection_params['harshness_factor']:
                    issues.append({
                        'type': 'harshness',
                        'severity': metrics['std'] / metrics['mean'],
                        'frequency': metrics['centroid'],
                        'correction': -3.0
                    })

            # Detekcja zamulenia (low/mid-low)
            if band_name in ['low', 'mid-low']:
                if metrics['energy'] / (metrics['mean'] + 1e-10) > self.detection_params['muddiness_threshold']:
                    issues.append({
                        'type': 'muddiness',
                        'severity': metrics['energy'] / (metrics['mean'] + 1e-10),
                        'frequency': metrics['centroid'],
                        'correction': -2.0
                    })

            imperfections[band_name] = issues

        return imperfections

    def generate_eq_curve(self, imperfections, manual_weights=None, num_bands=10):
        """Generuje krzywą korekcyjną EQ z uwzględnieniem ręcznych podbić"""
        eq_freqs = np.array([31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
        eq_gains = np.zeros(num_bands)

        # Domyślne wagi (brak dodatkowych podbić)
        if manual_weights is None:
            manual_weights = {band: 0.0 for band in self.freq_bands}

        # Agregacja korekcji z automatycznej analizy
        for band_name, issues in imperfections.items():
            for issue in issues:
                freq = issue['frequency']
                correction = issue['correction']

                # Znajdź najbliższe pasmo EQ
                closest_band = np.argmin(np.abs(eq_freqs - freq))
                weight = 1.0 / (1.0 + np.abs(eq_freqs[closest_band] - freq) / 1000)
                eq_gains[closest_band] += correction * weight * issue['severity']

                # Wpływ na sąsiednie pasma
                if closest_band > 0:
                    eq_gains[closest_band - 1] += correction * weight * 0.3
                if closest_band < num_bands - 1:
                    eq_gains[closest_band + 1] += correction * weight * 0.3

        # Dodaj ręczne podbicia/osłabienia
        band_to_eq_mapping = {
            'low': [0, 1, 2],          # 31, 62, 125 Hz
            'mid-low': [2, 3],         # 125, 250 Hz  
            'mid': [3, 4, 5],          # 250, 500, 1000 Hz
            'mid-high': [5, 6, 7],     # 1000, 2000, 4000 Hz
            'high': [7, 8, 9]          # 4000, 8000, 16000 Hz
        }

        for band_name, manual_boost in manual_weights.items():
            if band_name in band_to_eq_mapping:
                for eq_idx in band_to_eq_mapping[band_name]:
                    eq_gains[eq_idx] += manual_boost

        # Dodaj sztywne korekcje
        for band_name, fixed_correction in self.fixed_corrections.items():
            if band_name in band_to_eq_mapping:
                for eq_idx in band_to_eq_mapping[band_name]:
                    eq_gains[eq_idx] += fixed_correction

        # Normalizacja i ograniczenie
        eq_gains = np.clip(eq_gains, -12, 12)

        return eq_freqs, eq_gains

    def set_fixed_correction(self, band, value):
        """Ustawia sztywną korekcję dla pasma"""
        if band in self.fixed_corrections:
            self.fixed_corrections[band] = value

class SpectrumWidget(Gtk.DrawingArea):
    """Widget do rysowania spektrum i krzywych EQ"""

    def __init__(self):
        super().__init__()
        self.set_size_request(800, 400)
        self.spectrum_data = None
        self.eq_curve = None
        self.imperfections = {}
        self.set_draw_func(self.draw)

    def draw(self, area, cr, width, height):
        """Rysowanie przy użyciu Cairo"""
        # Tło
        cr.set_source_rgb(0.1, 0.1, 0.1)
        cr.rectangle(0, 0, width, height)
        cr.fill()

        # Siatka
        self.draw_grid(cr, width, height)

        # Spektrum
        if self.spectrum_data is not None:
            self.draw_spectrum(cr, width, height)

        # Krzywa EQ
        if self.eq_curve is not None:
            self.draw_eq_curve(cr, width, height)

        # Oznaczenia niedoskonałości
        self.draw_imperfections(cr, width, height)

    def draw_grid(self, cr, width, height):
        """Rysuje siatkę częstotliwości"""
        cr.set_source_rgba(0.3, 0.3, 0.3, 0.5)
        cr.set_line_width(1)

        # Linie poziome (dB)
        db_levels = [-40, -30, -20, -10, 0, 10]
        for db in db_levels:
            y = height * (1 - (db + 40) / 50)
            cr.move_to(0, y)
            cr.line_to(width, y)
            cr.stroke()

            # Etykiety
            cr.set_source_rgba(0.6, 0.6, 0.6, 1)
            cr.move_to(5, y - 3)
            cr.show_text(f"{db} dB")

        # Linie pionowe (częstotliwości)
        freq_marks = [100, 1000, 10000]
        for freq in freq_marks:
            x = width * np.log10(freq / 20) / np.log10(20000 / 20)
            cr.set_source_rgba(0.3, 0.3, 0.3, 0.5)
            cr.move_to(x, 0)
            cr.line_to(x, height)
            cr.stroke()

            # Etykiety
            cr.set_source_rgba(0.6, 0.6, 0.6, 1)
            cr.move_to(x + 3, height - 5)
            cr.show_text(f"{freq} Hz")

    def draw_spectrum(self, cr, width, height):
        """Rysuje spektrum częstotliwości"""
        if self.spectrum_data is None:
            return

        spectrum, freqs = self.spectrum_data
        spectrum = np.abs(spectrum)
        spectrum_db = 20 * np.log10(spectrum + 1e-10)
        spectrum_db = np.clip(spectrum_db, -40, 10)

        # Rysowanie
        cr.set_source_rgba(0.2, 0.8, 0.3, 0.7)
        cr.set_line_width(2)

        for i in range(1, len(freqs)):
            if freqs[i] < 20 or freqs[i] > 20000:
                continue

            if freqs[i-1] <= 0 or freqs[i] <= 0:
                continue

            x1 = width * np.log10(freqs[i-1] / 20) / np.log10(20000 / 20)
            x2 = width * np.log10(freqs[i] / 20) / np.log10(20000 / 20)
            y1 = height * (1 - (spectrum_db[i-1] + 40) / 50)
            y2 = height * (1 - (spectrum_db[i] + 40) / 50)

            if i == 1:
                cr.move_to(x1, y1)
            cr.line_to(x2, y2)

        cr.stroke()

    def draw_eq_curve(self, cr, width, height):
        """Rysuje krzywą korekcyjną EQ"""
        if self.eq_curve is None:
            return

        freqs, gains = self.eq_curve

        # Interpolacja dla gładkiej krzywej
        freq_interp = np.logspace(np.log10(20), np.log10(20000), 500)
        gains_interp = np.interp(np.log10(freq_interp), np.log10(freqs), gains)

        # Rysowanie
        cr.set_source_rgba(0.9, 0.5, 0.1, 0.8)
        cr.set_line_width(3)

        for i in range(len(freq_interp)):
            x = width * np.log10(freq_interp[i] / 20) / np.log10(20000 / 20)
            y = height * (0.5 - gains_interp[i] / 24)

            if i == 0:
                cr.move_to(x, y)
            else:
                cr.line_to(x, y)

        cr.stroke()

    def draw_imperfections(self, cr, width, height):
        """Oznacza wykryte niedoskonałości"""
        colors = {
            'resonance': (1.0, 0.2, 0.2, 0.6),
            'null': (0.2, 0.2, 1.0, 0.6),
            'harshness': (1.0, 1.0, 0.2, 0.6),
            'muddiness': (0.6, 0.3, 0.1, 0.6)
        }

        for band_name, issues in self.imperfections.items():
            for issue in issues:
                if issue['frequency'] < 20 or issue['frequency'] > 20000:
                    continue

                x = width * np.log10(issue['frequency'] / 20) / np.log10(20000 / 20)

                # Rysuj marker
                cr.set_source_rgba(*colors.get(issue['type'], (0.5, 0.5, 0.5, 0.5)))
                cr.arc(x, height * 0.1, 5 * issue['severity'], 0, 2 * math.pi)
                cr.fill()

    def update_spectrum(self, spectrum, freqs):
        """Aktualizuje dane spektrum"""
        self.spectrum_data = (spectrum, freqs)
        self.queue_draw()

    def update_eq_curve(self, freqs, gains):
        """Aktualizuje krzywą EQ"""
        self.eq_curve = (freqs, gains)
        self.queue_draw()

    def update_imperfections(self, imperfections):
        """Aktualizuje wykryte niedoskonałości"""
        self.imperfections = imperfections
        self.queue_draw()

class BandAnalysisWidget(Gtk.DrawingArea):
    """Widget do rysowania krzywych korekcyjnych dla poszczególnych pasm"""

    def __init__(self, band_name):
        super().__init__()
        self.set_size_request(600, 200)
        self.band_name = band_name
        self.time_stamps = []
        self.correction_curve = []
        self.set_draw_func(self.draw)

    def draw(self, area, cr, width, height):
        """Rysowanie krzywej korekcyjnej w czasie"""
        # Tło
        cr.set_source_rgb(0.05, 0.05, 0.05)
        cr.rectangle(0, 0, width, height)
        cr.fill()

        # Siatka
        self.draw_grid(cr, width, height)

        # Krzywa korekcyjna
        if len(self.correction_curve) > 1 and len(self.time_stamps) > 1:
            self.draw_correction_curve(cr, width, height)

    def draw_grid(self, cr, width, height):
        """Rysuje siatkę"""
        cr.set_source_rgba(0.2, 0.2, 0.2, 0.5)
        cr.set_line_width(1)

        # Linie poziome (dB)
        db_levels = [-12, -6, 0, 6, 12]
        for db in db_levels:
            y = height * (0.5 - db / 24)
            cr.move_to(0, y)
            cr.line_to(width, y)
            cr.stroke()

            # Etykiety dB
            cr.set_source_rgba(0.6, 0.6, 0.6, 1)
            cr.move_to(5, y - 3)
            cr.show_text(f"{db} dB")

        # Linie pionowe (czas)
        if self.time_stamps:
            max_time = max(self.time_stamps)
            for t in range(0, int(max_time) + 1, max(1, int(max_time / 10))):
                x = width * t / max_time if max_time > 0 else 0
                cr.move_to(x, 0)
                cr.line_to(x, height)
                cr.stroke()

                # Etykiety czasu
                cr.set_source_rgba(0.6, 0.6, 0.6, 1)
                cr.move_to(x + 3, height - 5)
                cr.show_text(f"{t}s")

    def draw_correction_curve(self, cr, width, height):
        """Rysuje krzywą korekcyjną w kolorze odpowiadającym typowi korekcji"""
        if not self.correction_curve or not self.time_stamps:
            return

        max_time = max(self.time_stamps) if self.time_stamps else 1
        
        # Wybierz kolor na podstawie pasma i średniej korekcji
        avg_correction = np.mean(self.correction_curve)
        
        if avg_correction > 2:      # Duże podbicie - zielony
            color = (0.2, 0.8, 0.2, 0.8)
        elif avg_correction > 0:    # Małe podbicie - żółto-zielony
            color = (0.6, 0.8, 0.2, 0.8)
        elif avg_correction > -2:   # Małe osłabienie - pomarańczowy
            color = (1.0, 0.6, 0.2, 0.8)
        else:                       # Duże osłabienie - czerwony
            color = (1.0, 0.2, 0.2, 0.8)

        cr.set_source_rgba(*color)
        cr.set_line_width(2)

        for i, (time, correction) in enumerate(zip(self.time_stamps, self.correction_curve)):
            x = width * time / max_time if max_time > 0 else 0
            y = height * (0.5 - correction / 24)  # Skala ±12dB
            
            if i == 0:
                cr.move_to(x, y)
            else:
                cr.line_to(x, y)

        cr.stroke()

        # Wypełnienie pod krzywą dla lepszego efektu wizualnego
        cr.set_source_rgba(*color[:3], 0.3)  # Przeźroczyste wypełnienie
        
        # Powrót do początku i rysowanie wypełnienia
        for i, (time, correction) in enumerate(zip(self.time_stamps, self.correction_curve)):
            x = width * time / max_time if max_time > 0 else 0
            y = height * (0.5 - correction / 24)
            
            if i == 0:
                cr.move_to(x, height * 0.5)  # Rozpocznij od linii 0dB
                cr.line_to(x, y)
            else:
                cr.line_to(x, y)
        
        # Zamknij wypełnienie
        if self.time_stamps:
            last_x = width * self.time_stamps[-1] / max_time if max_time > 0 else 0
            cr.line_to(last_x, height * 0.5)
            cr.line_to(width * self.time_stamps[0] / max_time if max_time > 0 else 0, height * 0.5)
        
        cr.fill()

    def update_correction_data(self, time_stamps, correction_curve):
        """Aktualizuje dane krzywej korekcyjnej"""
        self.time_stamps = time_stamps
        self.correction_curve = correction_curve
        self.queue_draw()

class ManualBoostWidget(Gtk.Box):
    """Widget do ręcznych podbić pasm (dodatkowych do automatycznych korekcji)"""

    def __init__(self, analyzer):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.analyzer = analyzer
        self.sliders = {}

        # Dodaj suwaki dla każdego pasma
        for band_name in self.analyzer.freq_bands.keys():
            frame = Gtk.Frame()
            frame.set_label(f"Ręczne podbicie: {band_name}")

            hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)

            # Etykieta
            label = Gtk.Label(label="Podbicie: ")
            hbox.append(label)

            # Suwak (-6 do +6 dB)
            slider = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
            slider.set_range(-6, 6)
            slider.set_value(0)
            slider.set_draw_value(True)
            slider.connect("value-changed", self.on_boost_changed, band_name)

            hbox.append(slider)
            frame.set_child(hbox)
            self.append(frame)
            self.sliders[band_name] = slider

    def on_boost_changed(self, slider, band_name):
        """Callback dla zmian ręcznych podbić"""
        value = slider.get_value()
        print(f"Ręczne podbicie dla pasma {band_name}: {value:.1f} dB")

    def get_manual_boosts(self):
        """Zwraca aktualne ręczne podbicia"""
        boosts = {}
        for band_name, slider in self.sliders.items():
            boosts[band_name] = slider.get_value()
        return boosts

class FixedCorrectionWidget(Gtk.Box):
    """Widget do ustawiania sztywnych korekcji"""

    def __init__(self, analyzer):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.analyzer = analyzer
        self.sliders = {}

        # Dodaj suwaki dla każdego pasma
        for band_name in self.analyzer.freq_bands.keys():
            frame = Gtk.Frame()
            frame.set_label(f"Sztywna korekcja: {band_name}")

            hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)

            # Etykieta
            label = Gtk.Label(label="Korekcja: ")
            hbox.append(label)

            # Suwak (-12 do +12 dB)
            slider = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
            slider.set_range(-12, 12)
            slider.set_value(0)
            slider.set_draw_value(True)
            slider.connect("value-changed", self.on_correction_changed, band_name)

            hbox.append(slider)
            frame.set_child(hbox)
            self.append(frame)
            self.sliders[band_name] = slider

    def on_correction_changed(self, slider, band_name):
        """Callback dla zmian sztywnych korekcji"""
        value = slider.get_value()
        self.analyzer.set_fixed_correction(band_name, value)
        print(f"Sztywna korekcja dla pasma {band_name}: {value:.1f} dB")

    def get_fixed_corrections(self):
        """Zwraca aktualne sztywne korekcje"""
        corrections = {}
        for band_name, slider in self.sliders.items():
            corrections[band_name] = slider.get_value()
        return corrections

class AudioAnalyzerApp(Gtk.ApplicationWindow):
    """Główne okno aplikacji"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_title("Analizator Audio z Auto-EQ v2.0")
        self.set_default_size(1400, 900)

        self.analyzer = AudioAnalyzer()
        self.pipeline = None
        self.current_file = None
        self.is_playing = False

        self.setup_ui()
        self.setup_gstreamer()

    def setup_ui(self):
        """Tworzy interfejs użytkownika"""
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        main_box.set_margin_top(10)
        main_box.set_margin_bottom(10)
        main_box.set_margin_start(10)
        main_box.set_margin_end(10)
        self.set_child(main_box)

        # Pasek narzędzi
        toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        main_box.append(toolbar)

        # Przycisk wyboru pliku
        self.file_button = Gtk.Button(label="Wybierz plik audio")
        self.file_button.connect("clicked", self.on_file_choose)
        toolbar.append(self.file_button)

        # Przycisk odtwarzania
        self.play_button = Gtk.Button(label="Odtwórz")
        self.play_button.connect("clicked", self.on_play_pause)
        self.play_button.set_sensitive(False)
        toolbar.append(self.play_button)

        # Przycisk analizy
        self.analyze_button = Gtk.Button(label="Analizuj cały utwór")
        self.analyze_button.connect("clicked", self.on_analyze)
        self.analyze_button.set_sensitive(False)
        toolbar.append(self.analyze_button)

        # Przycisk zastosowania EQ
        self.apply_eq_button = Gtk.Button(label="Zastosuj Auto-EQ")
        self.apply_eq_button.connect("clicked", self.on_apply_eq)
        self.apply_eq_button.set_sensitive(False)
        toolbar.append(self.apply_eq_button)

        # Przycisk eksportu
        self.export_button = Gtk.Button(label="Eksportuj")
        self.export_button.connect("clicked", self.on_export)
        self.export_button.set_sensitive(False)
        toolbar.append(self.export_button)

        # Separator
        toolbar.append(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL))

        # Etykieta statusu
        self.status_label = Gtk.Label(label="Gotowy")
        toolbar.append(self.status_label)

        # Notebook dla zakładek
        notebook = Gtk.Notebook()
        main_box.append(notebook)

        # Zakładka spektrum
        spectrum_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.spectrum_widget = SpectrumWidget()
        spectrum_box.append(self.spectrum_widget)
        notebook.append_page(spectrum_box, Gtk.Label(label="Spektrum i EQ"))

        # Zakładka analizy czasowej pasm
        bands_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)

        # Wykresy krzywych korekcyjnych dla każdego pasma
        self.band_widgets = {}
        for band_name in self.analyzer.freq_bands.keys():
            frame = Gtk.Frame()
            frame.set_label(f"Pasmo: {band_name}")
            band_widget = BandAnalysisWidget(band_name)
            frame.set_child(band_widget)
            bands_box.append(frame)
            self.band_widgets[band_name] = band_widget

        scroll = Gtk.ScrolledWindow()
        scroll.set_child(bands_box)
        notebook.append_page(scroll, Gtk.Label(label="Analiza czasowa pasm"))

        # Zakładka ręcznych podbić
        self.manual_boost_widget = ManualBoostWidget(self.analyzer)
        scroll_boosts = Gtk.ScrolledWindow()
        scroll_boosts.set_child(self.manual_boost_widget)
        notebook.append_page(scroll_boosts, Gtk.Label(label="Ręczne podbicia"))

        # Zakładka sztywnych korekcji
        self.fixed_correction_widget = FixedCorrectionWidget(self.analyzer)
        scroll_fixed = Gtk.ScrolledWindow()
        scroll_fixed.set_child(self.fixed_correction_widget)
        notebook.append_page(scroll_fixed, Gtk.Label(label="Sztywne korekcje"))

        # Zakładka raportu
        self.report_view = Gtk.TextView()
        self.report_view.set_editable(False)
        self.report_view.set_wrap_mode(Gtk.WrapMode.WORD)
        report_scroll = Gtk.ScrolledWindow()
        report_scroll.set_child(self.report_view)
        notebook.append_page(report_scroll, Gtk.Label(label="Raport"))

        # Panel kontroli EQ (10 pasm)
        eq_frame = Gtk.Frame()
        eq_frame.set_label("Kontrola EQ (10 pasm)")
        main_box.append(eq_frame)

        eq_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        eq_frame.set_child(eq_box)

        self.eq_sliders = []
        eq_freqs = [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]

        for freq in eq_freqs:
            vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)

            # Slider
            slider = Gtk.Scale(orientation=Gtk.Orientation.VERTICAL)
            slider.set_range(-12, 12)
            slider.set_value(0)
            slider.set_inverted(True)
            slider.set_size_request(40, 150)
            slider.set_draw_value(True)
            slider.connect("value-changed", self.on_eq_changed)

            # Etykieta
            label = Gtk.Label(label=f"{freq}Hz")
            label.set_size_request(40, -1)

            vbox.append(slider)
            vbox.append(label)
            eq_box.append(vbox)

            self.eq_sliders.append(slider)

        # Przycisk resetu EQ
        reset_button = Gtk.Button(label="Reset EQ")
        reset_button.connect("clicked", self.on_reset_eq)
        eq_box.append(reset_button)

    def setup_gstreamer(self):
        """Konfiguruje pipeline GStreamer"""
        # Pipeline z equalizerem 10-pasmowym
        self.pipeline = Gst.Pipeline.new("audio-pipeline")

        # Elementy
        self.filesrc = Gst.ElementFactory.make("filesrc", "source")
        self.decode = Gst.ElementFactory.make("decodebin", "decode")
        self.convert = Gst.ElementFactory.make("audioconvert", "convert")
        self.equalizer = Gst.ElementFactory.make("equalizer-10bands", "eq")
        self.spectrum = Gst.ElementFactory.make("spectrum", "spectrum")
        self.sink = Gst.ElementFactory.make("autoaudiosink", "sink")

        # Konfiguracja spectrum
        self.spectrum.set_property("bands", 1024)
        self.spectrum.set_property("threshold", -80)
        self.spectrum.set_property("interval", 50000000)  # 50ms

        # Dodaj elementy do pipeline
        for element in [self.filesrc, self.decode, self.convert,
                        self.equalizer, self.spectrum, self.sink]:
            if element:
                self.pipeline.add(element)

        # Łączenie elementów
        self.filesrc.link(self.decode)
        self.convert.link(self.equalizer)
        self.equalizer.link(self.spectrum)
        self.spectrum.link(self.sink)

        # Callback dla decodebin
        self.decode.connect("pad-added", self.on_pad_added)

        # Bus dla wiadomości
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_bus_message)

    def on_pad_added(self, element, pad):
        """Callback gdy decodebin utworzy pad"""
        sink_pad = self.convert.get_static_pad("sink")
        if not sink_pad.is_linked():
            pad.link(sink_pad)

    def on_bus_message(self, bus, message):
        """Obsługa wiadomości z GStreamer"""
        if message.type == Gst.MessageType.ELEMENT:
            struct = message.get_structure()
            if struct.get_name() == "spectrum":
                # Pobierz dane spektrum
                magnitudes = struct.get_value("magnitude")

                # Konwertuj do numpy array
                spectrum = np.array(magnitudes, dtype=np.float32)

                # Częstotliwości
                freqs = np.linspace(0, self.analyzer.sample_rate / 2, len(spectrum))

                # Analiza spektrum
                band_analysis, _, _ = self.analyzer.analyze_spectrum(spectrum)

                # Aktualizuj widget spektrum
                GLib.idle_add(self.spectrum_widget.update_spectrum, spectrum, freqs)

                # Zapisz analizę pasm
                self.band_analysis = band_analysis

        elif message.type == Gst.MessageType.EOS:
            self.pipeline.set_state(Gst.State.READY)
            self.is_playing = False
            GLib.idle_add(self.play_button.set_label, "Odtwórz")

        elif message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, {debug}")
            self.pipeline.set_state(Gst.State.NULL)

    def on_file_choose(self, widget):
        """Wybór pliku audio"""
        dialog = Gtk.FileChooserDialog(
            title="Wybierz plik audio",
            transient_for=self,
            modal=True,
            action=Gtk.FileChooserAction.OPEN
        )
        dialog.add_buttons("Anuluj", Gtk.ResponseType.CANCEL, "Otwórz", Gtk.ResponseType.OK)

        # Filtr plików audio
        filter_audio = Gtk.FileFilter()
        filter_audio.set_name("Pliki audio")
        filter_audio.add_mime_type("audio/*")
        dialog.add_filter(filter_audio)

        dialog.connect("response", self.on_file_dialog_response)
        dialog.present()

    def on_file_dialog_response(self, dialog, response):
        """Obsługa wyboru pliku"""
        if response == Gtk.ResponseType.OK:
            file = dialog.get_file()
            self.current_file = file.get_path()
            self.filesrc.set_property("location", self.current_file)

            self.status_label.set_text(f"Załadowano: {os.path.basename(self.current_file)}")
            self.play_button.set_sensitive(True)
            self.analyze_button.set_sensitive(True)

        dialog.destroy()

    def on_play_pause(self, widget):
        """Odtwarzanie/pauza"""
        if not self.is_playing:
            self.pipeline.set_state(Gst.State.PLAYING)
            self.play_button.set_label("Pauza")
            self.is_playing = True
        else:
            self.pipeline.set_state(Gst.State.PAUSED)
            self.play_button.set_label("Odtwórz")
            self.is_playing = False

    def on_analyze(self, widget):
        """Przeprowadza pełną analizę pliku"""
        if not self.current_file:
            return

        self.status_label.set_text("Analizowanie całego utworu...")

        # Analiza w osobnym wątku
        thread = threading.Thread(target=self.analyze_full_file)
        thread.start()

    def analyze_full_file(self):
        """Analizuje cały plik audio fragmentami"""
        try:
            # Wczytaj plik audio za pomocą pydub
            audio = AudioSegment.from_file(self.current_file)
            
            # Konwertuj na mono jeśli stereo
            if audio.channels == 2:
                audio = audio.set_channels(1)
            
            # Pobierz dane audio
            samples = np.array(audio.get_array_of_samples())
            audio_data = samples.astype(np.float32) / (2**15)  # Normalizacja

            # Analizuj cały utwór fragmentami
            time_analysis, correction_curves = self.analyzer.analyze_full_audio(audio_data)

            # Aktualizuj UI
            GLib.idle_add(self.update_full_analysis_results, time_analysis, correction_curves)

        except Exception as e:
            GLib.idle_add(self.status_label.set_text, f"Błąd analizy: {str(e)}")

    def update_full_analysis_results(self, time_analysis, correction_curves):
        """Aktualizuje wyniki pełnej analizy w UI"""
        # Aktualizuj wykresy krzywych korekcyjnych dla każdego pasma
        for band_name, band_widget in self.band_widgets.items():
            if band_name in correction_curves:
                band_widget.update_correction_data(
                    self.analyzer.time_stamps, 
                    correction_curves[band_name]
                )

        # Generuj końcową krzywą EQ na podstawie średnich korekcji
        final_imperfections = {}
        for band_name, corrections in correction_curves.items():
            if corrections:
                avg_correction = np.mean(corrections)
                final_imperfections[band_name] = [{
                    'type': 'average_correction',
                    'severity': 1.0,
                    'frequency': sum(self.analyzer.freq_bands[band_name]) / 2,  # Środek pasma
                    'correction': avg_correction
                }]

        # Uwzględnij ręczne podbicia i sztywne korekcje
        manual_boosts = self.manual_boost_widget.get_manual_boosts()
        eq_freqs, eq_gains = self.analyzer.generate_eq_curve(final_imperfections, manual_boosts)

        # Aktualizuj krzywę EQ
        self.spectrum_widget.update_eq_curve(eq_freqs, eq_gains)
        self.spectrum_widget.update_imperfections(final_imperfections)

        # Generuj raport
        self.generate_analysis_report(time_analysis, correction_curves, eq_freqs, eq_gains)

        self.status_label.set_text("Analiza całego utworu zakończona")
        self.apply_eq_button.set_sensitive(True)
        self.export_button.set_sensitive(True)

        # Zapisz wyniki
        self.last_eq_curve = (eq_freqs, eq_gains)
        self.last_time_analysis = time_analysis
        self.last_correction_curves = correction_curves

    def generate_analysis_report(self, time_analysis, correction_curves, eq_freqs, eq_gains):
        """Generuje szczegółowy raport analizy"""
        report = "=== RAPORT ANALIZY CAŁEGO UTWORU ===\n\n"
        
        # Statystyki czasowe dla każdego pasma
        report += "STATYSTYKI KOREKCJI W CZASIE:\n"
        for band_name, corrections in correction_curves.items():
            if corrections:
                avg_correction = np.mean(corrections)
                max_correction = np.max(corrections)
                min_correction = np.min(corrections)
                std_correction = np.std(corrections)
                
                report += f"\n{band_name.upper()}:\n"
                report += f"  - Średnia korekcja: {avg_correction:.2f} dB\n"
                report += f"  - Maksymalna korekcja: {max_correction:.2f} dB\n"
                report += f"  - Minimalna korekcja: {min_correction:.2f} dB\n"
                report += f"  - Odchylenie standardowe: {std_correction:.2f} dB\n"
                
                # Analiza problemów
                if avg_correction > 2:
                    report += f"  ⚠️ Pasmo wymaga znacznego podbicia\n"
                elif avg_correction < -2:
                    report += f"  ⚠️ Pasmo wymaga znacznego stłumienia\n"
                
                if std_correction > 3:
                    report += f"  ⚠️ Duża zmienność korekcji w czasie\n"

        # Ręczne podbicia
        manual_boosts = self.manual_boost_widget.get_manual_boosts()
        if any(abs(boost) > 0.1 for boost in manual_boosts.values()):
            report += "\nRĘCZNE PODBICIA:\n"
            for band_name, boost in manual_boosts.items():
                if abs(boost) > 0.1:
                    report += f"  - {band_name}: {boost:+.1f} dB\n"

        # Sztywne korekcje
        fixed_corrections = self.fixed_correction_widget.get_fixed_corrections()
        if any(abs(correction) > 0.1 for correction in fixed_corrections.values()):
            report += "\nSZTYWNE KOREKCJE:\n"
            for band_name, correction in fixed_corrections.items():
                if abs(correction) > 0.1:
                    report += f"  - {band_name}: {correction:+.1f} dB\n"

        report += "\n=== FINALNE USTAWIENIA EQ ===\n"
        for i, (freq, gain) in enumerate(zip(eq_freqs, eq_gains)):
            report += f"{freq:>5} Hz: {gain:+6.1f} dB\n"

        # Rekomendacje
        report += "\n=== REKOMENDACJE ===\n"
        total_boost = sum(max(0, gain) for gain in eq_gains)
        total_cut = sum(min(0, gain) for gain in eq_gains)
        
        if total_boost > 15:
            report += "⚠️ Duże łączne podbicie - rozważ obniżenie głośności głównej\n"
        if total_cut < -15:
            report += "⚠️ Duże łączne stłumienie - może być potrzebne podbicie głośności\n"
        
        # Sprawdź balans tonalny
        low_gain = sum(eq_gains[:3])  # 31-125 Hz
        mid_gain = sum(eq_gains[3:7])  # 250-2000 Hz
        high_gain = sum(eq_gains[7:])  # 4000-16000 Hz
        
        if low_gain > mid_gain + 6:
            report += "📢 Nagranie może brzmieć zbyt baso - rozważ redukcję niskich tonów\n"
        elif high_gain > mid_gain + 6:
            report += "📢 Nagranie może brzmieć zbyt ostro - rozważ redukcję wysokich tonów\n"
        elif mid_gain > max(low_gain, high_gain) + 6:
            report += "📢 Nagranie może brzmieć zbyt 'boksowato' - rozważ redukcję średnich tonów\n"

        self.report_view.get_buffer().set_text(report)

    def on_apply_eq(self, widget):
        """Zastosuj automatyczną krzywą EQ z uwzględnieniem wszystkich korekcji"""
        if not hasattr(self, 'last_correction_curves'):
            self.status_label.set_text("Najpierw przeprowadź analizę całego utworu")
            return

        # Generuj końcową krzywą EQ
        final_imperfections = {}
        for band_name, corrections in self.last_correction_curves.items():
            if corrections:
                avg_correction = np.mean(corrections)
                final_imperfections[band_name] = [{
                    'type': 'final_correction',
                    'severity': 1.0,
                    'frequency': sum(self.analyzer.freq_bands[band_name]) / 2,
                    'correction': avg_correction
                }]

        # Uwzględnij ręczne podbicia
        manual_boosts = self.manual_boost_widget.get_manual_boosts()
        eq_freqs, eq_gains = self.analyzer.generate_eq_curve(final_imperfections, manual_boosts)

        # Ustaw slidery i equalizer
        for i, (slider, gain) in enumerate(zip(self.eq_sliders, eq_gains)):
            slider.set_value(gain)
            self.equalizer.set_property(f"band{i}", gain)

        # Aktualizuj krzywą EQ w widoku
        self.spectrum_widget.update_eq_curve(eq_freqs, eq_gains)

        self.status_label.set_text("Zastosowano kompletny Auto-EQ")

    def on_eq_changed(self, slider):
        """Callback dla zmian sliderów EQ"""
        # Znajdź indeks slidera
        index = self.eq_sliders.index(slider)
        value = slider.get_value()

        # Ustaw wartość w equalizerze
        self.equalizer.set_property(f"band{index}", value)

        # Aktualizuj krzywą EQ w widoku
        if hasattr(self, 'last_eq_curve'):
            eq_freqs, eq_gains = self.last_eq_curve
            eq_gains[index] = value
            self.spectrum_widget.update_eq_curve(eq_freqs, eq_gains)

    def on_reset_eq(self, widget):
        """Resetuje ustawienia EQ"""
        for i, slider in enumerate(self.eq_sliders):
            slider.set_value(0)
            self.equalizer.set_property(f"band{i}", 0)

        self.status_label.set_text("EQ zresetowany")

    def on_export(self, widget):
        """Eksportuje przetworzony plik z zastosowanym EQ"""
        if not hasattr(self, 'last_eq_curve') or not self.current_file:
            self.status_label.set_text("Brak danych do eksportu")
            return

        # Pokaż dialog zapisu
        dialog = Gtk.FileChooserDialog(
            title="Zapisz przetworzony plik",
            transient_for=self,
            modal=True,
            action=Gtk.FileChooserAction.SAVE
        )
        dialog.add_buttons("Anuluj", Gtk.ResponseType.CANCEL, "Zapisz", Gtk.ResponseType.OK)
        
        # Ustaw domyślną nazwę
        basename = os.path.splitext(os.path.basename(self.current_file))[0]
        dialog.set_current_name(f"{basename}_eq.wav")

        dialog.connect("response", self.on_export_dialog_response)
        dialog.present()

    def on_export_dialog_response(self, dialog, response):
        """Obsługa dialogu eksportu"""
        if response == Gtk.ResponseType.OK:
            output_path = dialog.get_file().get_path()
            
            # Eksport w osobnym wątku
            thread = threading.Thread(target=self.export_file, args=(output_path,))
            thread.start()
            
        dialog.destroy()

    def export_file(self, output_path):
        """Eksportuje plik z zastosowanymi korekcjami"""
        try:
            GLib.idle_add(self.status_label.set_text, "Eksportowanie...")
            
            # Wczytaj oryginalny plik
            audio = AudioSegment.from_file(self.current_file)
            
            # Konwertuj na mono jeśli stereo
            if audio.channels == 2:
                audio = audio.set_channels(1)
            
            samples = np.array(audio.get_array_of_samples())
            audio_data = samples.astype(np.float32) / (2**15)

            # Zastosuj filtry EQ na podstawie krzywej
            eq_freqs, eq_gains = self.last_eq_curve
            filtered_audio = audio_data.copy()
            
            for freq, gain in zip(eq_freqs, eq_gains):
                if abs(gain) > 0.1:  # Tylko znaczące korekcje
                    # Stwórz filtr peaking EQ
                    sos = signal.iirpeak(freq, Q=2.0, fs=self.analyzer.sample_rate)
                    # Zastosuj gain
                    gain_linear = 10**(gain/20)
                    filtered_audio = signal.sosfilt(sos, filtered_audio) * gain_linear

            # Normalizuj audio po zastosowaniu EQ
            max_val = np.max(np.abs(filtered_audio))
            if max_val > 1.0:
                filtered_audio = filtered_audio / max_val * 0.95

            # Konwertuj z powrotem do int16
            audio_int16 = (filtered_audio * (2**15)).astype(np.int16)

            # Zapisz jako WAV
            processed_audio = AudioSegment(
                audio_int16.tobytes(),
                frame_rate=audio.frame_rate,
                sample_width=2,
                channels=1
            )
            processed_audio.export(output_path, format="wav")

            GLib.idle_add(self.status_label.set_text, f"Eksportowano: {os.path.basename(output_path)}")

        except Exception as e:
            GLib.idle_add(self.status_label.set_text, f"Błąd eksportu: {str(e)}")

def main():
    app = Gtk.Application(application_id="org.example.improvedaudioanalyzer")
    app.connect("activate", lambda a: AudioAnalyzerApp(application=a).present())
    app.run(None)

if __name__ == "__main__":
    main()