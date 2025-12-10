import sys
import time
import subprocess
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class RestartHandler(FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback
        self.last_reload = time.time()

    def on_modified(self, event):
        if event.is_directory:
            return
        
        # Only react to .py and .ini changes
        if event.src_path.endswith('.py') or event.src_path.endswith('.ini'):
            # Debounce: avoid double restarts for single save
            if time.time() - self.last_reload > 1.0:
                print(f"\n[Reloader] detected change in {event.src_path}")
                self.callback()
                self.last_reload = time.time()

def run_app():
    # Run main.py as a subprocess
    return subprocess.Popen([sys.executable, "main.py"])

def main():
    print("[Reloader] Starting development monitor...")
    
    # Try to import watchdog, if not present, warn user
    try:
        import watchdog
    except ImportError:
        print("[Reloader] Error: 'watchdog' library not found.")
        print("Please install it: pip install watchdog")
        return

    process = run_app()
    
    def restart_process():
        nonlocal process
        print("[Reloader] Restarting app...")
        if process:
            # Graceful termination first
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
        
        process = run_app()

    event_handler = RestartHandler(restart_process)
    observer = Observer()
    # Watch current directory
    observer.schedule(event_handler, path='.', recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
            # Check if process died (e.g. syntax error)
            if process.poll() is not None:
                # If it crashed, we just wait for a file change to restart it
                pass
    except KeyboardInterrupt:
        observer.stop()
        if process:
            process.terminate()
    
    observer.join()

if __name__ == "__main__":
    main()
