#!/usr/bin/env python3
"""
Create local embeddings (offline) from JSON / JSONL transcripts.
"""

import json
import argparse
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


def normalize_text(s: str) -> str:
    return " ".join(s.strip().split())


def iter_records(path: str) -> Iterable[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            for r in obj:
                yield r
        else:
            yield obj


def extract_text(rec: Dict[str, Any], parsed_key: str, text_field: str) -> Optional[str]:
    pr = rec.get(parsed_key)
    if not isinstance(pr, dict):
        return None

    val = pr.get(text_field)

    # for --text-field argument_snippet
    if isinstance(val, str):
        return val

    # for --text-field justification_bullets
    if isinstance(val, list) and all(isinstance(x, str) for x in val):
        return " ".join(x.strip() for x in val if x.strip())
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-folder", required=True)
    ap.add_argument("--parsed-key", default="parsed_response")
    ap.add_argument("--text-field", default="argument_snippet") # or "justification_bullets"
    ap.add_argument("--model", default="all-mpnet-base-v2")
    args = ap.parse_args()

    output_dir = os.path.join("outputs", args.input_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"embeddings_{args.text_field}.jsonl")

    model = SentenceTransformer(args.model)

    texts: List[str] = []
    metas: List[Dict[str, Any]] = []

    for i, rec in enumerate(iter_records(os.path.join("outputs", args.input_folder, "transcripts.jsonl"))):
        text = extract_text(rec, args.parsed_key, args.text_field)
        if not text:
            continue
        texts.append(normalize_text(text))
        metas.append({
            "record_id": rec.get("response_id", f"row_{i:06d}"),
            "persona_id": rec.get("persona_id"),
            "run_id": rec.get("run_id"),
        })

    if not texts:
        raise RuntimeError("No valid texts to embed.")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True  # cosine-ready
    )

    with open(output_file, "w", encoding="utf-8") as f:
        for meta, emb, text in zip(metas, embeddings, texts):
            row = {
                **meta,
                "embedding_model": args.model,
                "embedded_at_utc": datetime.now(timezone.utc).isoformat(),
                "normalized_text": text,
                "embedding": emb.tolist()
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(texts)} embeddings to {output_dir}")


if __name__ == "__main__":
    main()
