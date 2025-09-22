from __future__ import annotations

import json
import os

from src.data import DataPipeline


def test_pipeline_end_to_end(tmp_path):
    out_dir = tmp_path / "processed"
    pipe = DataPipeline(output_dir=str(out_dir), max_items=5)
    pipe.run(["https://github.com/codeGeekPro/neural-chat-engine"], ["python", "pydantic"])

    jsonl = out_dir / "conversations.jsonl"
    csvf = out_dir / "embeddings.csv"

    assert jsonl.exists() and csvf.exists()
    with open(jsonl, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f.read().splitlines() if l.strip()]
    assert len(lines) >= 2
    # Validate structure
    assert isinstance(lines[0]["conversation"], list)
