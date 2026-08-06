from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

required = [
    "bm25_tokenized_corpus.json",
    "chunk_embeddings.npy",
    "chunks_metadata.jsonl",
    "retrieval_config.json",
]

missing = [name for name in required if not (ROOT / name).exists()]

if missing:
    raise SystemExit(
        "Missing required files:\n"
        + "\n".join(f"- {name}" for name in missing)
    )

chunks = pd.read_json(ROOT / "chunks_metadata.jsonl", lines=True)
embeddings = np.load(ROOT / "chunk_embeddings.npy")

with open(ROOT / "bm25_tokenized_corpus.json", "r", encoding="utf-8") as file:
    bm25 = json.load(file)

print("Artifact check passed.")
print(f"Chunks: {len(chunks):,}")
print(f"Embeddings: {embeddings.shape}")
print(f"BM25 rows: {len(bm25):,}")
print(f"Documents: {chunks['source_id'].nunique():,}")

if len(chunks) != len(embeddings) or len(chunks) != len(bm25):
    raise SystemExit("The artifact row counts do not match.")

print("All row counts match.")
