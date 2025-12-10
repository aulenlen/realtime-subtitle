import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import signal
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QObject, pyqtSignal, QTimer

from audio_capture import AudioCapture
from transcriber import Transcriber
from translator import Translator
from overlay_window import OverlayWindow
from config import config

class WorkerSignals(QObject):
    update_text = pyqtSignal(int, str, str)  # (chunk_id, original, translated)

class Pipeline(QObject):
    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()
        self.running = True
        
        # Print config for debugging
        config.print_config()
        
        # Initialize components
        self.audio = AudioCapture(
            device_index=config.device_index,
            sample_rate=config.sample_rate,
            silence_threshold=config.silence_threshold,
            silence_duration=config.silence_duration,
            chunk_duration=config.chunk_duration,
            max_phrase_duration=config.max_phrase_duration,
            streaming_mode=config.streaming_mode,
            streaming_interval=config.streaming_interval,
            streaming_overlap=config.streaming_overlap
        )
        
        # Initialize Transcriber
        print(f"[Pipeline] Initializing Transcriber with device={config.whisper_device}...")
        self.transcriber = Transcriber(
            model_size=config.whisper_model,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type,
            language=config.source_language
        )
        
        # Initialize Translator
        print(f"[Pipeline] Initializing Translator (target={config.target_lang})...")
        self.translator = Translator(
            target_lang=config.target_lang,
            base_url=config.api_base_url,
            api_key=config.api_key,
            model=config.model
        )

    def start(self):
        """Start the processing pipeline in a dedicated thread"""
        self.audio.start()
        self.thread = threading.Thread(target=self.processing_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        print("\n[Pipeline] Stopping...")
        self.running = False
        self.audio.stop()
        if self.thread.is_alive():
            self.thread.join(timeout=2)
        print("[Pipeline] Stopped.")

    def processing_loop(self):
        """Fully parallel pipeline: multiple concurrent transcription + translation"""
        print("Pipeline processing loop started (FULLY PARALLEL mode).")
        
        # Create multiple transcribers for concurrent processing
        # CHECK: If using MLX, force 1 worker (MLX is not thread-safe for parallel inference in this way)
        is_mlx = getattr(self.transcriber, 'use_mlx', False)
        
        if is_mlx:
            print("[Pipeline] MLX backend detected - forcing single worker (MLX uses GPU parallelism internaly)")
            num_transcription_workers = 1
        else:
            num_transcription_workers = config.transcription_workers
            
        print(f"[Pipeline] Using {num_transcription_workers} transcription workers...")
        
        transcribers = [self.transcriber]  # Reuse existing one
        for i in range(num_transcription_workers - 1):
            t = Transcriber(
                model_size=config.whisper_model,
                device=config.whisper_device,
                compute_type=config.whisper_compute_type,
                language=config.source_language
            )
            transcribers.append(t)
        
        # Thread pool for parallel transcription AND translation
        transcribe_executor = ThreadPoolExecutor(max_workers=num_transcription_workers)
        translate_executor = ThreadPoolExecutor(max_workers=2)
        
        pending_transcriptions = []  # List of (chunk_id, future) tuples
        pending_translations = []    # List of (chunk_id, future) tuples
        chunk_id = 0
        transcriber_index = 0
        
        for audio_chunk in self.audio.get_audio_stream():
            if not self.running:
                break
            
            chunk_id += 1
            
            # Round-robin assign to transcribers
            current_transcriber = transcribers[transcriber_index % len(transcribers)]
            transcriber_index += 1
            
            # Submit transcription asynchronously
            print(f"[Chunk {chunk_id}] Submitting to Whisper worker {transcriber_index % len(transcribers)}...")
            future = transcribe_executor.submit(
                self._transcribe_chunk, current_transcriber, audio_chunk, chunk_id
            )
            pending_transcriptions.append((chunk_id, future))
            
            # DRAIN QUEUE: If we have too many pending tasks, we are falling behind.
            # Skip old chunks to catch up.
            if len(pending_transcriptions) > 10:
                print(f"[Pipeline] WARNING: Falling behind! Dropping {len(pending_transcriptions) - 5} old chunks.")
                # Cancel futures if possible (though distinct from cancelling running items)
                # Just drop them from our list to stop tracking
                pending_transcriptions = pending_transcriptions[-5:]
            
            # Collect and process completed transcriptions
            still_pending = []
            for cid, fut in pending_transcriptions:
                if fut.done():
                    try:
                        text = fut.result()
                        if text:
                            # EMIT IMMEDIATE TRANSCRIPTION (Async Display)
                            self.signals.update_text.emit(cid, text, "")
                            print(f"[Chunk {cid}] Emitted partial: {text}")

                            # Submit for translation immediately
                            trans_future = translate_executor.submit(
                                self._translate_and_log, text, cid
                            )
                            pending_translations.append((cid, trans_future))
                    except Exception as e:
                        print(f"[Chunk {cid}] Transcription error: {e}")
                else:
                    still_pending.append((cid, fut))
            pending_transcriptions = still_pending
            
            # Collect and emit completed translations
            still_pending_trans = []
            for cid, fut in pending_translations:
                if fut.done():
                    try:
                        # Unpack tuple (original, translated)
                        result = fut.result()
                        if result:
                            original, translated = result
                            # EMIT FULL UPDATE
                            self.signals.update_text.emit(cid, original, translated)
                            print(f"[Chunk {cid}] Emitted final: {original} -> {translated}")
                    except Exception as e:
                        print(f"[Chunk {cid}] Translation error: {e}")
                else:
                    still_pending_trans.append((cid, fut))
            pending_translations = still_pending_trans
        
        # Wait for remaining work
        print("[Pipeline] Waiting for remaining transcriptions...")
        for cid, fut in pending_transcriptions:
            try:
                text = fut.result(timeout=10)
                if text and text.strip():
                    translated = self.translator.translate(text)
                    self.signals.update_text.emit(cid, text, translated)
            except:
                pass
        
        for cid, fut in pending_translations:
            try:
                result = fut.result(timeout=5)
                if result:
                    original, translated = result
                    self.signals.update_text.emit(cid, original, translated)
            except:
                pass
        
        transcribe_executor.shutdown(wait=False)
        translate_executor.shutdown(wait=False)
    
    def _transcribe_chunk(self, transcriber, audio_chunk, chunk_id):
        """Transcribe a single chunk and log timing"""
        t0 = time.time()
        text = transcriber.transcribe(audio_chunk)
        t1 = time.time()
        print(f"[Chunk {chunk_id}] Transcribed in {t1-t0:.2f}s: {text if text else '(empty)'}")
        return text
    
    def _translate_and_log(self, text, chunk_id=0):
        """Translate text and log result"""
        t0 = time.time()
        translated_text = self.translator.translate(text)
        t1 = time.time()
        print(f"[Chunk {chunk_id}] Translated in {t1-t0:.2f}s: {translated_text}")
        return (text, translated_text)

# Global reference for signal handler
_pipeline = None
_app = None

def signal_handler(sig, frame):
    """Handle Ctrl-C gracefully"""
    print("\n[Main] Ctrl-C received, shutting down...")
    if _pipeline:
        _pipeline.stop()
    if _app:
        _app.quit()
    sys.exit(0)

def main():
    global _pipeline, _app
    
    # Set up signal handler for Ctrl-C
    signal.signal(signal.SIGINT, signal_handler)
    
    _app = QApplication(sys.argv)
    
    # Initialize Overlay Window
    window = OverlayWindow(
        display_duration=config.display_duration,
        window_width=config.window_width
        # window_height is omitted to let it default to 100% screen height
    )
    window.show()
    
    # Logic
    _pipeline = Pipeline()
    
    # Connect signals
    _pipeline.signals.update_text.connect(window.update_text)
    
    # Timer to let Python interpreter handle signals (Ctrl-C)
    timer = QTimer()
    timer.start(200)
    timer.timeout.connect(lambda: None)
    
    # Start pipeline
    _pipeline.start()
    
    try:
        sys.exit(_app.exec())
    except SystemExit:
        pass
    finally:
        _pipeline.stop()

if __name__ == "__main__":
    main()
