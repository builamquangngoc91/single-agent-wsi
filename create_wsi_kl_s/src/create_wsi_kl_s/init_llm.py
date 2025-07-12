from __future__ import annotations

"""Central place to configure the default LLM for the package.

This module is executed on import so every Agent/Task that omits an explicit
``llm=...`` parameter will automatically use the configured Gemini model.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from crewai import LLM  # type: ignore

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parents[2]
load_dotenv(_project_root / ".env", override=True)  # silently skip if missing

_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not _GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found in environment. Add it to your .env file or "
        'export it in your shell (e.g. `export GEMINI_API_KEY="sk-..."`).'
    )

# You can tweak temperature, max_tokens, etc. here if desired.
_default_llm = LLM(
    model="gemini/gemini-2.5-flash-preview-04-17",
    api_key=_GEMINI_API_KEY,
    temperature=0.2,
)
