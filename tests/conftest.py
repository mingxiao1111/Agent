from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Generator

import pytest

from app import web
from tests.utils import DATASET_DIR


@pytest.fixture
def datasets_dir() -> Path:
    return DATASET_DIR


@pytest.fixture
def web_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator:
    web.GENERAL_SESSIONS.clear()
    web.TCM_SESSIONS.clear()

    long_memory_file = tmp_path / "general_long_memory_test.jsonl"
    feedback_dir = web.BASE_DIR / "data" / f"_chat_feedback_test_{uuid.uuid4().hex}"

    monkeypatch.setattr(web, "GENERAL_LONG_MEMORY_FILE", long_memory_file)
    monkeypatch.setattr(web, "CHAT_FEEDBACK_DIR", feedback_dir)
    monkeypatch.setattr(web, "GENERAL_M2_ENABLED", True)
    monkeypatch.setattr(web, "GENERAL_M2_LLM_CLEAN_ENABLED", False)

    yield web

    if feedback_dir.exists():
        shutil.rmtree(feedback_dir, ignore_errors=True)
    web.GENERAL_SESSIONS.clear()
    web.TCM_SESSIONS.clear()


@pytest.fixture
def client(web_module):
    return web_module.web_app.test_client()
