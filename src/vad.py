#!/usr/bin/env python3
"""
Module de segmentation audio intelligent pour le projet STC.
Supporte actuellement : VAD (WebRTC).
Produit : Graphiques de visualisation et fichiers audio segmentés.
"""

import wave
import sys
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import webrtcvad
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple, List

# --- Configuration du Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# --- Constantes Audio ---
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # 16-bit
CHANNELS = 1
CHUNK_DURATION_MS = 20
CHUNK_BYTES = int((SAMPLE_RATE / 1000) * CHUNK_DURATION_MS * SAMPLE_WIDTH * CHANNELS)
BYTES_PER_SECOND = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS

@dataclass
class Segment:
    """Représente un segment audio finalisé, prêt pour la distribution ou la sauvegarde."""
    start_time: float
    end_time: float
    audio_data: bytearray

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def __repr__(self):
        return f"Segment(start={self.start_time:.2f}s, duration={self.duration:.2f}s)"

    def save_to_wav(self, output_dir: Path, index: int) -> Path:
        """Sauvegarde les données du segment dans un fichier .wav individuel."""
        filename = f"seg_{index:03d}_{self.start_time:.2f}s-{self.end_time:.2f}s.wav"
        output_path = output_dir / filename
        
        try:
            with wave.open(str(output_path), 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(SAMPLE_WIDTH)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(self.audio_data)
        except Exception as e:
            logger.error(f"Échec de sauvegarde du segment {index}: {e}")
            raise
        return output_path


class AudioStreamProcessor:
    """Gère la lecture simulée d'un flux audio en direct."""
    
    def __init__(self, wav_path: Path):
        self.wav_path = wav_path
        self._validate_file()

    def _validate_file(self):
        if not self.wav_path.exists():
            raise FileNotFoundError(f"Fichier audio introuvable: {self.wav_path}")
        with wave.open(str(self.wav_path), 'rb') as wf:
            if (wf.getframerate() != SAMPLE_RATE or
                wf.getsampwidth() != SAMPLE_WIDTH or
                wf.getnchannels() != CHANNELS):
                raise ValueError(f"Format WAV invalide. Requis: {SAMPLE_RATE}Hz, Mono, 16-bit.")

    def stream(self) -> Iterator[Tuple[Optional[bytes], float]]:
        """Générateur simulant l'arrivée de paquets audio en temps réel."""
        with wave.open(str(self.wav_path), 'rb') as wf:
            total_duration = wf.getnframes() / float(SAMPLE_RATE)
            current_time = 0.0
            
            while True:
                chunk = wf.readframes(CHUNK_BYTES // SAMPLE_WIDTH)
                if not chunk:
                    break
                if len(chunk) == CHUNK_BYTES:
                    yield chunk, current_time
                current_time += CHUNK_DURATION_MS / 1000.0
            
            yield None, total_duration

    def load_full_waveform(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Charge tout l'audio pour la visualisation (hors simulation temps réel)."""
        with wave.open(str(self.wav_path), 'rb') as wf:
            n_frames = wf.getnframes()
            signal = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)
            duration = n_frames / float(SAMPLE_RATE)
            time_axis = np.linspace(0., duration, n_frames)
            return signal, time_axis, duration

class VadSegmenter:
    """
    Implémente une stratégie de segmentation VAD "sans perte" (lossless).
    
    Logique :
    1. Attend le début de la parole (STATE_SILENCE).
    2. Commence à bufferiser (STATE_SPEECH).
    3. Continue de bufferiser (parole ET silence).
    4. Ne coupe QUE si DEUX conditions sont remplies :
       a) Un silence suffisant est détecté (>= min_silence_ms)
       b) Le buffer total est assez long (>= min_duration_ms)
    5. Si un silence est détecté mais que le segment est trop court,
       la coupure est "retenue" et le buffer continue de grossir.
    """
    STATE_SILENCE = 0
    STATE_SPEECH = 1

    def __init__(self, aggressiveness: int, min_silence_ms: int, min_duration_ms: int):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.min_silence_chunks = max(1, min_silence_ms // CHUNK_DURATION_MS)
        self.min_duration_bytes = int((min_duration_ms / 1000.0) * BYTES_PER_SECOND)
        
        self.state = self.STATE_SILENCE
        self.buffer = bytearray()
        self.buffer_start_time = 0.0
        self.silence_counter = 0
        
        logger.info(f"VAD Init | Agressivité: {aggressiveness} | "
                    f"Silence min pour couper: {min_silence_ms}ms | "
                    f"Durée segment min: {min_duration_ms}ms (ne coupera pas avant)")

    def process(self, chunk: bytes, timestamp: float) -> Iterator[Segment]:
        """Traite un chunk audio et retourne un Segment si une coupure est décidée."""
        try:
            is_speech = self.vad.is_speech(chunk, SAMPLE_RATE)
        except Exception:
            is_speech = False # Gère les chunks invalides en fin de fichier

        if self.state == self.STATE_SILENCE:
            if is_speech:
                # Début de la parole, on commence à bufferiser
                self.state = self.STATE_SPEECH
                self.buffer_start_time = timestamp
                self.buffer.extend(chunk)
                self.silence_counter = 0
            else:
                # Toujours en silence, on ne fait rien, on ne bufferise rien
                pass

        elif self.state == self.STATE_SPEECH:
            # On est en train de bufferiser parole + silences
            self.buffer.extend(chunk)

            if not is_speech:
                self.silence_counter += 1
            else:
                self.silence_counter = 0 # La parole a repris, reset du compteur

            # --- Logique de coupure ---
            # Condition 1: Le silence est-il assez long ?
            is_silence_trigger = self.silence_counter >= self.min_silence_chunks
            # Condition 2: Le segment total est-il assez long ?
            is_duration_met = len(self.buffer) >= self.min_duration_bytes

            if is_silence_trigger and is_duration_met:
                # Les deux conditions sont remplies : on coupe.
                # Le segment inclut le silence de fin pour être "lossless".
                segment_end_time = timestamp + (CHUNK_DURATION_MS / 1000.0)
                
                yield Segment(self.buffer_start_time, segment_end_time, self.buffer)
                
                # Reset
                self.buffer = bytearray()
                self.state = self.STATE_SILENCE
                self.silence_counter = 0
            
            # Si le silence est détecté (is_silence_trigger) MAIS que
            # la durée n'est pas atteinte (is_duration_met=False),
            # on ne fait RIEN. On continue de bufferiser.
            # C'est ce qui fusionne "OK" + [silence] + "prochaine phrase".

    def flush(self) -> Iterator[Segment]:
        """Force la coupure du segment restant en fin de flux (sans perte)."""
        if self.buffer:
            # Émet tout ce qui reste, peu importe la durée, pour éviter la perte de données.
            end_timestamp = self.buffer_start_time + (len(self.buffer) / BYTES_PER_SECOND)
            yield Segment(self.buffer_start_time, end_timestamp, self.buffer)
            self.buffer = bytearray()


class GraphGenerator:
    """Gère la création des visualisations."""
    
    @staticmethod
    def plot_analysis(signal: np.ndarray, time_axis: np.ndarray, duration: float, 
                     segments: List[Segment], output_path: Path, title: str):
        
        plot_width = max(15, min(100, duration * 0.5))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(plot_width, 8), sharex=True,
                                      gridspec_kw={'height_ratios': [3, 1]})

        # 1. Signal Waveform
        ax1.plot(time_axis, signal, color='#7f8c8d', alpha=0.8)
        ax1.set_title(title, fontsize=14, pad=20)
        ax1.set_ylabel('Amplitude', fontsize=12)
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)
        ax1.margins(x=0.01)

        # 2. Segments Timeline
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f', '#e67e22']
        for i, seg in enumerate(segments, 1): # Commence à 1
            color = colors[i % len(colors)]
            ax2.broken_barh([(seg.start_time, seg.duration)], (0.4, 0.6), 
                          facecolors=color, edgecolors='black', linewidth=0.5)
            # Étiquette (juste le numéro)
            ax2.text(seg.start_time + seg.duration/2, 1.1, f"#{i}",
                     ha='center', va='bottom', fontsize=9, weight='bold')

        ax2.set_ylim(0, 1.5)
        ax2.set_yticks([])
        ax2.set_xlabel('Temps (secondes)', fontsize=12)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path, dpi=100)
        plt.close(fig)
        logger.info(f"Graphique généré: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Outil de segmentation VAD 'Lossless' - Projet STC")
    parser.add_argument("wav_file", type=Path, help="Fichier audio d'entrée (.wav 16kHz Mono)")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("output"), help="Dossier de sortie")
    parser.add_argument("-a", "--aggressiveness", type=int, choices=[0,1,2,3], default=3, help="Agressivité VAD (3=Max)")
    parser.add_argument("-s", "--silence-ms", type=int, default=500, help="Silence min pour couper (ms)")
    parser.add_argument("-d", "--min-duration-ms", type=int, default=1000, help="Durée min pour déclencher une coupure (ms)")
    args = parser.parse_args()

    # 1. Préparation
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_output_dir = args.output_dir / "audio"
    audio_output_dir.mkdir(exist_ok=True)
    
    processor = AudioStreamProcessor(args.wav_file)
    segmenter = VadSegmenter(args.aggressiveness, args.silence_ms, args.min_duration_ms)
    
    # 2. Traitement du flux
    logger.info("Démarrage de la simulation du flux...")
    segments: List[Segment] = []
    stream = processor.stream()
    total_duration = 0.0
    
    for chunk, timestamp in stream:
        if chunk is None:
            total_duration = timestamp
            break
        for seg in segmenter.process(chunk, timestamp):
            segments.append(seg)
            print(f" -> [LIVE] Nouveau segment détecté: {seg}")

    for seg in segmenter.flush():
        segments.append(seg)
        print(f" -> [LIVE] Segment final (flush): {seg}")

    logger.info(f"Traitement terminé. {len(segments)} segments trouvés sur {total_duration:.1f}s.")

    # 3. Sauvegarde des résultats
    logger.info("Sauvegarde des segments audio...")
    for i, seg in enumerate(segments, 1): # Commence l'index à 1
        seg.save_to_wav(audio_output_dir, i)
    
    # 4. Génération du graphique
    logger.info("Génération du graphique d'analyse...")
    signal, time_axis, _ = processor.load_full_waveform()
    graph_title = (f"Analyse VAD Lossless [Agr:{args.aggressiveness} | "
                   f"Silence:{args.silence_ms}ms | MinDur:{args.min_duration_ms}ms]")
    
    GraphGenerator.plot_analysis(
        signal, time_axis, total_duration, segments,
        args.output_dir / "analyse_resultat.png",
        graph_title
    )

if __name__ == "__main__":
    main()