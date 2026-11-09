import logging
import os
import warnings

from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> None:
    # Silence noisy third-party chatter.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    warnings.filterwarnings("ignore", category=UserWarning, module="fastembed")
    warnings.filterwarnings("ignore", message="Cannot enable progress bars")
    try:  # PyMuPDF prints parser notes (e.g. "RapidOCR is missing") to stdout; log them at DEBUG.
        import pymupdf

        pymupdf.set_messages(pylogging_name="pymupdf", pylogging_level=logging.DEBUG)
    except Exception:  # noqa: BLE001 - older PyMuPDF without set_messages
        pass
    logging.basicConfig(
        level=level.upper(),
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True,
    )
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "fastembed"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
