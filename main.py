if __name__ == "__main__":
    # This must run before importing Qt, NumPy, PaddleOCR, or other heavy modules.
    # In a PyInstaller build, Windows multiprocessing starts the same EXE with
    # special arguments. freeze_support() diverts those child processes instead
    # of launching another copy of the GUI.
    import multiprocessing

    multiprocessing.freeze_support()

    import faulthandler
    import os
    import sys
    from pathlib import Path

    log_dir = Path(os.getenv("LOCALAPPDATA", Path.home())) / "BoreholeJournalOCR"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_stream = (log_dir / "borehole_ocr.log").open("a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = log_stream
    if sys.stderr is None:
        sys.stderr = log_stream
    faulthandler.enable(log_stream, all_threads=True)

    from app.ui.main_window import run_app

    raise SystemExit(run_app())
