"""app.py — Hugging Face Space entry point (under 50 lines).

This file only imports and launches the Gradio app.
All logic lives in src/handlers.py, src/ui_builder.py, and src/theme.py.

To run locally:
    python app.py

The Space runs this file automatically on startup.
"""

import logging

from src.ui_builder import build_demo

# Set up basic logging so errors appear in the Space logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)

# Build the Gradio app (layout in ui_builder.py, handlers in handlers.py, CSS in theme.py)
demo = build_demo()

if __name__ == "__main__":
    demo.launch()
