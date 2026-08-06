from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Keep downloaded models beside the application by default.
APP_DIR = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(APP_DIR / ".hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gradio as gr
import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


REQUIRED_FILES = [
    "bm25_tokenized_corpus.json",
    "chunk_embeddings.npy",
    "chunks_metadata.jsonl",
    "retrieval_config.json",
]

DEFAULT_SMALL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_LARGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

APP_CSS = """
:root {
    --cyber-bg: #07111f;
    --cyber-panel: rgba(12, 24, 40, 0.95);
    --cyber-panel-soft: rgba(15, 31, 50, 0.86);
    --cyber-border: rgba(103, 232, 249, 0.16);
    --cyber-cyan: #22d3ee;
    --cyber-blue: #3b82f6;
    --cyber-blue-soft: #1d4ed8;
    --cyber-green: #34d399;
    --cyber-text: #eaf6ff;
    --cyber-muted: #9ab0c5;
}

html, body {
    margin: 0;
    min-height: 100%;
    background:
        radial-gradient(circle at 10% 0%, rgba(34, 211, 238, 0.12), transparent 28rem),
        radial-gradient(circle at 88% 7%, rgba(59, 130, 246, 0.12), transparent 32rem),
        linear-gradient(180deg, #07111f 0%, #091521 48%, #060d16 100%) !important;
}

body, .gradio-container {
    color: var(--cyber-text);
}

.gradio-container {
    width: min(96vw, 1680px) !important;
    max-width: none !important;
    margin: 0 auto !important;
    padding: 10px 14px 18px !important;
}

#hero {
    border: 1px solid var(--cyber-border);
    border-radius: 24px;
    padding: 18px 24px;
    margin-bottom: 12px;
    background:
        linear-gradient(135deg, rgba(13, 33, 54, 0.98), rgba(8, 24, 42, 0.94)),
        radial-gradient(circle at 80% 20%, rgba(34, 211, 238, 0.12), transparent 18rem);
    box-shadow: 0 18px 54px rgba(0, 0, 0, 0.24);
    overflow: hidden;
}

.hero-grid {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    flex-wrap: wrap;
}

.hero-title {
    margin: 0;
    font-size: clamp(1.95rem, 3vw, 3.15rem);
    line-height: 1.04;
    letter-spacing: -0.04em;
    color: #f5fbff;
}

.hero-subtitle {
    max-width: 880px;
    color: var(--cyber-muted);
    font-size: 0.97rem;
    margin-top: 8px;
    line-height: 1.55;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border: 1px solid rgba(52, 211, 153, 0.26);
    background: rgba(52, 211, 153, 0.1);
    color: #91f3cc;
    padding: 7px 12px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 800;
    white-space: nowrap;
}

.hero-orb {
    min-width: 82px;
    width: 82px;
    height: 82px;
    border-radius: 24px;
    display: grid;
    place-items: center;
    font-size: 2.2rem;
    border: 1px solid rgba(34, 211, 238, 0.24);
    background:
        linear-gradient(145deg, rgba(34, 211, 238, 0.18), rgba(59, 130, 246, 0.12));
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.10),
        0 16px 42px rgba(34, 211, 238, 0.10);
}

.status-card {
    border: 1px solid var(--cyber-border);
    background: rgba(12, 26, 43, 0.76);
    border-radius: 16px;
    padding: 12px 14px;
    color: var(--cyber-muted);
    line-height: 1.45;
    font-size: 0.92rem;
}

.status-card strong {
    color: #dff9ff;
}

.status-ok {
    border-color: rgba(52, 211, 153, 0.25);
    background: rgba(20, 83, 65, 0.17);
}

.status-error {
    border-color: rgba(248, 113, 113, 0.30);
    background: rgba(127, 29, 29, 0.16);
}

.card {
    border: 1px solid var(--cyber-border) !important;
    background: var(--cyber-panel) !important;
    border-radius: 20px !important;
    box-shadow: 0 14px 42px rgba(0, 0, 0, 0.22);
    overflow: hidden;
}

.soft-card {
    border: 1px solid rgba(148, 163, 184, 0.12) !important;
    background: var(--cyber-panel-soft) !important;
    border-radius: 16px !important;
}

#workspace-row {
    align-items: stretch !important;
    gap: 12px;
}

#chat-column, #side-column {
    min-height: 100%;
}

.sticky-column {
    position: sticky;
    top: 10px;
    align-self: flex-start;
}

#chatbot {
    border: none !important;
    background: transparent !important;
}

#chatbot > .wrap,
#chatbot .bubble-wrap,
#chatbot .message-wrap {
    background: transparent !important;
}

#question-box textarea {
    font-size: 0.98rem !important;
    line-height: 1.5 !important;
    min-height: 72px !important;
}

#ask-button {
    min-height: 48px;
    border-radius: 14px !important;
    font-weight: 800 !important;
    letter-spacing: 0.01em;
    background: linear-gradient(135deg, #0891b2, #2563eb) !important;
    border: 1px solid rgba(103, 232, 249, 0.28) !important;
    box-shadow: 0 10px 26px rgba(37, 99, 235, 0.20);
}

#ask-button:hover {
    transform: translateY(-1px);
    filter: brightness(1.07);
}

.secondary-button {
    border-radius: 14px !important;
    min-height: 48px;
}

.section-heading {
    display: flex;
    align-items: center;
    gap: 9px;
    margin: 0 0 10px;
    color: #eaf8ff;
    font-weight: 800;
    font-size: 1rem;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    margin-top: 10px;
}

.metric {
    padding: 10px 12px;
    border: 1px solid rgba(103, 232, 249, 0.12);
    border-radius: 14px;
    background: rgba(15, 32, 50, 0.66);
}

.metric-value {
    font-size: 1.02rem;
    color: #e8fbff;
    font-weight: 800;
}

.metric-label {
    color: var(--cyber-muted);
    font-size: 0.74rem;
    margin-top: 3px;
}

.quick-note {
    color: var(--cyber-muted);
    font-size: 0.84rem;
    line-height: 1.5;
}

.compact-help {
    font-size: 0.83rem;
    color: var(--cyber-muted);
    line-height: 1.45;
}

.fit-chat {
    min-height: calc(100vh - 385px);
}

.panel-tight .block, 
.panel-tight .gradio-group {
    margin-top: 8px !important;
    margin-bottom: 8px !important;
}


.suggestion-wrap {
    margin-top: 10px;
    padding: 12px;
    border: 1px solid rgba(103, 232, 249, 0.12);
    border-radius: 16px;
    background: rgba(12, 26, 43, 0.60);
}

.suggestion-title {
    color: #dff8ff;
    font-size: 0.90rem;
    font-weight: 800;
    margin-bottom: 7px;
}

.suggestion-button {
    min-height: 38px !important;
    border-radius: 12px !important;
    font-size: 0.82rem !important;
    text-align: left !important;
    padding: 8px 11px !important;
    border: 1px solid rgba(103, 232, 249, 0.13) !important;
    background: rgba(19, 37, 60, 0.78) !important;
}

.suggestion-button:hover {
    border-color: rgba(34, 211, 238, 0.40) !important;
    background: rgba(18, 57, 82, 0.82) !important;
    transform: translateY(-1px);
}

footer {
    opacity: 0.42;
}

@media (max-width: 1180px) {
    .sticky-column {
        position: static;
    }
}

@media (max-width: 900px) {
    .gradio-container {
        width: 100vw !important;
        padding: 10px 10px 20px !important;
    }

    #hero {
        padding: 16px 18px;
        border-radius: 18px;
    }

    .hero-orb {
        display: none;
    }

    .metric-grid {
        grid-template-columns: 1fr;
    }

    .fit-chat {
        min-height: 470px;
    }
}
"""

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/\-]*")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "should", "that", "the",
    "their", "this", "to", "was", "what", "when", "where", "which", "who",
    "why", "with", "your",
}

LOW_QUALITY_SECTION_TERMS = {
    "resources",
    "references",
    "bibliography",
    "acknowledgments",
    "acknowledgements",
    "table of contents",
    "appendix",
}

MAX_PER_DOCUMENT = 2
DENSE_CANDIDATES = 18
BM25_CANDIDATES = 18
RRF_K = 60
MAX_CONTEXT_CHARACTERS = 9000
SUPPORT_THRESHOLD = 0.18

SMART_CANDIDATE_CHUNKS = 12
SMART_MAX_SENTENCES = 160
SMART_MIN_SENTENCE_WORDS = 8
SMART_MAX_SENTENCE_WORDS = 65

ACTION_TERMS = {
    "isolate", "disconnect", "contain", "disable", "notify", "report",
    "preserve", "triage", "eradicate", "restore", "recover", "rebuild",
    "test", "verify", "activate", "implement", "identify", "protect",
}

PREPARATION_TERMS = {
    "prepare", "preparation", "plan", "policy", "backup", "backups",
    "training", "exercise", "identify", "protect", "insurance",
    "contacts", "communication", "test", "testing", "prioritize",
}

RESPONSE_TERMS = {
    "respond", "response", "isolate", "disconnect", "contain",
    "triage", "notify", "report", "preserve", "eradicate",
    "affected", "impacted", "incident",
}

RECOVERY_TERMS = {
    "recover", "recovery", "restore", "restoration", "rebuild",
    "resume", "continuity", "backup", "backups", "operations",
    "validate", "verify",
}


RISK_ASSESSMENT_TERMS = {
    "risk", "assessment", "assess", "threat", "threats", "vulnerability",
    "vulnerabilities", "likelihood", "impact", "scope", "assumptions",
    "constraints", "communicate", "results", "maintain", "monitor", "update",
}

RISK_STAGE_TERMS = {
    "Prepare": {
        "prepare", "purpose", "scope", "assumptions", "constraints",
        "information", "sources", "analytic", "approach",
    },
    "Conduct": {
        "conduct", "threat", "threats", "events", "vulnerability",
        "vulnerabilities", "likelihood", "impact", "risk",
    },
    "Communicate": {
        "communicate", "share", "results", "findings", "decision",
        "decision-makers", "stakeholders", "information",
    },
    "Maintain": {
        "maintain", "monitor", "update", "changes", "ongoing",
        "review", "current",
    },
}

GENERIC_FRAMEWORK_PHRASES = (
    "provides guidance to industry",
    "aims to help organizations",
    "community profile",
    "outcomes help prevent",
    "collaborating with",
    "common foundation",
    "specific mappings and relationships",
)

BROKEN_STARTS = (
    "government,",
    "or ",
    "and ",
    "but ",
    "as well as ",
    "such as ",
    "for example,",
    "e.g.",
    "i.e.",
    "basic ransomware tips",
    "figure ",
    "table ",
)

MODEL_LOCK = threading.Lock()
LOADED_MODEL_NAME: Optional[str] = None
LOADED_TOKENIZER = None
LOADED_MODEL = None


class ArtifactError(RuntimeError):
    pass


def clean_whitespace(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


def page_display(item: Dict[str, Any]) -> str:
    start = item.get("page_start", "")
    end = item.get("page_end", "")
    return str(start) if start == end else f"{start}-{end}"


def reliable_section(section: Any) -> str:
    value = clean_whitespace(section)
    lowered = value.lower()

    if not value or lowered in {"none", "nan", "unknown", "unknown section"}:
        return ""

    if len(value) > 120 or len(value.split()) > 18:
        return ""

    if re.search(r"\.(pdf|docx?|txt)\b", lowered):
        return ""

    if value.endswith((",", ";", ":")):
        return ""

    if value[0].islower():
        return ""

    return value


def low_quality_section(section: Any) -> bool:
    lowered = clean_whitespace(section).lower()
    return any(term in lowered for term in LOW_QUALITY_SECTION_TERMS)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(matrix / norms)


def sanitize_artifact_path(value: Any) -> Path:
    """
    Clean paths passed from Windows batch files or the UI.

    A quoted folder ending in a backslash can sometimes arrive as:
    C:\\folder\\cyber"
    This function strips accidental quote characters safely.
    """
    raw = str(value or "").strip()
    raw = raw.strip().strip('"').strip("'").strip()
    raw = raw.rstrip('"').rstrip("'").strip()
    return Path(raw).expanduser() if raw else APP_DIR


def has_required_artifacts(folder: Path) -> bool:
    try:
        return folder.is_dir() and all(
            (folder / name).is_file()
            for name in REQUIRED_FILES
        )
    except OSError:
        return False


def discover_artifact_dir(preferred: Any = None) -> Path:
    """
    Find the vector-store folder automatically.

    Search order:
    1. Explicit path from the UI or command line
    2. Folder containing app.py
    3. Current working directory
    4. Common child folders up to two levels deep
    """
    candidates: List[Path] = []

    for candidate in (
        sanitize_artifact_path(preferred),
        APP_DIR,
        Path.cwd(),
        APP_DIR / "cybersecurity_rag_artifacts",
        APP_DIR / "cybersecurity_rag_vector_store_v4",
        APP_DIR / "cybersecurity_rag_vector_store_v3",
    ):
        if candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        if has_required_artifacts(candidate):
            return candidate.resolve()

    # Limited recursive discovery avoids scanning the whole disk.
    search_roots = list(dict.fromkeys([APP_DIR, Path.cwd()]))

    for root in search_roots:
        if not root.exists():
            continue

        try:
            for config_file in root.glob("**/retrieval_config.json"):
                relative_depth = len(config_file.parent.relative_to(root).parts)
                if relative_depth <= 2 and has_required_artifacts(config_file.parent):
                    return config_file.parent.resolve()
        except (OSError, ValueError):
            continue

    # Return the cleaned preferred path so the error message remains useful.
    return sanitize_artifact_path(preferred).resolve()


def artifact_status_html(
    loaded: bool,
    artifact_dir: Path,
    error: Optional[str] = None,
) -> str:
    if loaded and ENGINE is not None:
        chunks = len(ENGINE.chunks)
        documents = int(ENGINE.chunks["source_id"].nunique())
        device = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        )

        return f"""
<div class="status-card status-ok">
  <strong>● Knowledge base ready</strong><br>
  {chunks:,} chunks · {documents} documents · {device}<br>
  <span style="font-size:.78rem">{artifact_dir}</span>
</div>
"""

    safe_error = clean_whitespace(error or "Required vector-store files were not found.")
    return f"""
<div class="status-card status-error">
  <strong>● Knowledge base not loaded</strong><br>
  {safe_error}<br>
  <span style="font-size:.78rem">Selected folder: {artifact_dir}</span>
</div>
"""


class CyberRAGEngine:
    def __init__(self, artifact_dir: Path):
        self.artifact_dir = artifact_dir.resolve()
        self._validate_artifacts()

        with open(
            self.artifact_dir / "retrieval_config.json",
            "r",
            encoding="utf-8",
        ) as file:
            self.config = json.load(file)

        self.chunks = pd.read_json(
            self.artifact_dir / "chunks_metadata.jsonl",
            lines=True,
        ).reset_index(drop=True)

        self.embeddings = normalize_rows(
            np.load(self.artifact_dir / "chunk_embeddings.npy")
        )

        if len(self.chunks) != len(self.embeddings):
            raise ArtifactError(
                "chunks_metadata.jsonl and chunk_embeddings.npy have "
                f"different row counts: {len(self.chunks)} vs "
                f"{len(self.embeddings)}."
            )

        with open(
            self.artifact_dir / "bm25_tokenized_corpus.json",
            "r",
            encoding="utf-8",
        ) as file:
            bm25_tokens = json.load(file)

        if len(bm25_tokens) != len(self.chunks):
            raise ArtifactError(
                "The BM25 corpus does not match the chunk metadata count."
            )

        self.bm25 = BM25Okapi(bm25_tokens)
        self.embedding_model_name = self.config.get(
            "embedding_model_name",
            "sentence-transformers/all-MiniLM-L6-v2",
        )

        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def _validate_artifacts(self) -> None:
        missing = [
            name
            for name in REQUIRED_FILES
            if not (self.artifact_dir / name).exists()
        ]

        if missing:
            formatted = "\n".join(f"- {name}" for name in missing)
            raise ArtifactError(
                f"Missing required files in:\n{self.artifact_dir}\n\n"
                f"{formatted}\n\n"
                "Place app.py in the same folder as the extracted vector-store "
                "files, or start it with --artifact-dir."
            )

    def query_embedding(self, query: str) -> np.ndarray:
        vector = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return np.asarray(vector, dtype=np.float32)

    def hybrid_search(
        self,
        query: str,
        top_k: int = 6,
        max_per_document: int = MAX_PER_DOCUMENT,
        preferred_source_ids: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        query_vector = self.query_embedding(query)
        dense_scores = self.embeddings @ query_vector

        dense_k = min(DENSE_CANDIDATES, len(dense_scores))
        dense_positions = np.argpartition(
            dense_scores,
            -dense_k,
        )[-dense_k:]
        dense_positions = dense_positions[
            np.argsort(dense_scores[dense_positions])[::-1]
        ]

        bm25_scores = np.asarray(
            self.bm25.get_scores(tokenize(query)),
            dtype=np.float32,
        )
        bm25_k = min(BM25_CANDIDATES, len(bm25_scores))
        bm25_positions = np.argpartition(
            bm25_scores,
            -bm25_k,
        )[-bm25_k:]
        bm25_positions = bm25_positions[
            np.argsort(bm25_scores[bm25_positions])[::-1]
        ]

        fused: Dict[int, float] = defaultdict(float)
        dense_map: Dict[int, float] = {}
        bm25_map: Dict[int, float] = {}

        for rank, position in enumerate(dense_positions, start=1):
            position = int(position)
            fused[position] += 1.0 / (RRF_K + rank)
            dense_map[position] = float(dense_scores[position])

        for rank, position in enumerate(bm25_positions, start=1):
            position = int(position)
            fused[position] += 1.0 / (RRF_K + rank)
            bm25_map[position] = float(bm25_scores[position])

        preferred_source_ids = set(
            preferred_source_ids or []
        )

        if preferred_source_ids:
            for position in list(fused):
                source_id = str(
                    self.chunks.iloc[position].get(
                        "source_id",
                        "",
                    )
                )
                if source_id in preferred_source_ids:
                    # RRF scores are normally around 0.01-0.03, so this is a
                    # meaningful but not absolute preference.
                    fused[position] += 0.025

        ranked = sorted(
            fused,
            key=lambda position: fused[position],
            reverse=True,
        )

        # Prefer content sections over resource/reference sections.
        preferred = [
            position
            for position in ranked
            if not low_quality_section(
                self.chunks.iloc[position].get("section", "")
            )
        ]
        deferred = [
            position
            for position in ranked
            if position not in preferred
        ]

        selected: List[Dict[str, Any]] = []
        source_counts: Dict[str, int] = defaultdict(int)

        for position in preferred + deferred:
            row = self.chunks.iloc[position].to_dict()
            source_id = str(row.get("source_id", ""))

            if source_counts[source_id] >= max_per_document:
                continue

            row["row_position"] = int(position)
            row["retrieval_score"] = float(fused[position])
            row["dense_score"] = dense_map.get(position)
            row["bm25_score"] = bm25_map.get(position)
            row["page_display"] = page_display(row)
            row["display_section"] = reliable_section(row.get("section"))
            row["source_label"] = f"S{len(selected) + 1}"

            selected.append(row)
            source_counts[source_id] += 1

            if len(selected) >= top_k:
                break

        return selected


    def _row_to_evidence(
        self,
        position: int,
        retrieval_score: float = 1.0,
    ) -> Dict[str, Any]:
        row = self.chunks.iloc[int(position)].to_dict()
        row["row_position"] = int(position)
        row["retrieval_score"] = float(retrieval_score)
        row["dense_score"] = None
        row["bm25_score"] = None
        row["page_display"] = page_display(row)
        row["display_section"] = reliable_section(
            row.get("section")
        )
        return row

    def find_best_source_chunk(
        self,
        source_id: str,
        phrase_groups: Sequence[Sequence[str]],
    ) -> Optional[Dict[str, Any]]:
        """
        Find the best local chunk for a known source-driven answer.

        Each phrase group represents an expected concept. Exact phrase matches
        receive more weight than individual word overlap.
        """

        best_position: Optional[int] = None
        best_score = float("-inf")

        source_rows = self.chunks[
            self.chunks["source_id"].astype(str) == source_id
        ]

        for position, row in source_rows.iterrows():
            lowered = clean_whitespace(
                row.get("text", "")
            ).lower()
            terms = set(tokenize(lowered))
            score = 0.0

            for group in phrase_groups:
                group_score = 0.0

                for phrase in group:
                    normalized = clean_whitespace(
                        phrase
                    ).lower()

                    if not normalized:
                        continue

                    if normalized in lowered:
                        group_score = max(
                            group_score,
                            7.0 + len(normalized.split()) * 0.2,
                        )
                    else:
                        phrase_terms = set(
                            tokenize(normalized)
                        )
                        if phrase_terms:
                            overlap = len(
                                phrase_terms.intersection(terms)
                            ) / len(phrase_terms)
                            group_score = max(
                                group_score,
                                overlap * 3.0,
                            )

                score += group_score

            section = clean_whitespace(
                row.get("section", "")
            ).lower()

            if low_quality_section(section):
                score -= 5.0

            if score > best_score:
                best_score = score
                best_position = int(position)

        if best_position is None:
            return None

        return self._row_to_evidence(
            best_position,
            retrieval_score=max(best_score, 0.0),
        )

    def required_evidence_for_intent(
        self,
        intent: str,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve source chunks that are required for deterministic answers.

        This avoids relying on approximate vector ranking for exact framework
        lists and documented multi-stage processes.
        """

        specifications: List[
            Tuple[str, Sequence[Sequence[str]]]
        ] = []

        if intent == "csf_functions":
            specifications = [
                (
                    "nist_csf_2_0",
                    [
                        [
                            "there are six csf functions",
                            "govern, identify, protect, detect, respond, and recover",
                        ]
                    ],
                ),
            ]

        elif intent == "risk_assessment":
            specifications = [
                (
                    "nist_sp_800_30r1",
                    [
                        [
                            "the first step in the risk assessment process is to prepare",
                            "establish a context for the risk assessment",
                        ],
                        [
                            "purpose, scope, assumptions, and constraints",
                            "prepare for the assessment",
                        ],
                    ],
                ),
                (
                    "nist_sp_800_30r1",
                    [
                        [
                            "conducting risk assessments includes the following specific tasks",
                            "identify threat sources that are relevant",
                        ],
                        [
                            "determine the likelihood",
                            "determine the adverse impacts",
                            "determine information security risks",
                        ],
                    ],
                ),
                (
                    "nist_sp_800_30r1",
                    [
                        [
                            "communicate risk assessment results",
                            "share risk-related information",
                            "communicate and share risk assessment results",
                        ]
                    ],
                ),
                (
                    "nist_sp_800_30r1",
                    [
                        [
                            "the fourth step in the risk assessment process is to maintain",
                            "monitor risk factors identified in risk assessments",
                        ],
                        [
                            "update the components of risk assessments",
                            "maintain the assessment",
                        ],
                    ],
                ),
            ]

        elif intent == "zero_trust_principles":
            specifications = [
                (
                    "nist_sp_800_207",
                    [
                        [
                            "all data sources and computing services are considered resources",
                            "all communication is secured regardless of network location",
                        ],
                        [
                            "access to individual enterprise resources is granted on a per-session basis",
                        ],
                    ],
                ),
                (
                    "nist_sp_800_207",
                    [
                        [
                            "access to resources is determined by dynamic policy",
                            "monitors and measures the integrity and security posture",
                        ]
                    ],
                ),
                (
                    "nist_sp_800_207",
                    [
                        [
                            "all resource authentication and authorization are dynamic",
                            "collects as much information as possible",
                        ]
                    ],
                ),
            ]

        required: List[Dict[str, Any]] = []
        used_positions = set()

        for source_id, phrase_groups in specifications:
            match = self.find_best_source_chunk(
                source_id,
                phrase_groups,
            )

            if (
                match is not None
                and match["row_position"] not in used_positions
            ):
                required.append(match)
                used_positions.add(match["row_position"])

        return required

    def retrieve_for_intent(
        self,
        question: str,
        intent: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        preferred_sources = {
            "csf_functions": {"nist_csf_2_0"},
            "risk_assessment": {"nist_sp_800_30r1"},
            "zero_trust_principles": {"nist_sp_800_207"},
            "immediate_response": {
                "cisa_stop_ransomware",
                "nist_sp_800_61r3",
            },
            "prepare_recover": {
                "nist_ir_8374r1",
                "cisa_stop_ransomware",
                "nist_sp_800_61r3",
            },
        }.get(intent, set())

        max_per_document = (
            5
            if intent in {
                "risk_assessment",
                "zero_trust_principles",
            }
            else MAX_PER_DOCUMENT
        )

        expanded_query = self.expand_retrieval_query(
            question,
            intent,
        )

        hybrid = self.hybrid_search(
            expanded_query,
            top_k=max(top_k, SMART_CANDIDATE_CHUNKS),
            max_per_document=max_per_document,
            preferred_source_ids=preferred_sources,
        )

        required = self.required_evidence_for_intent(
            intent
        )

        merged: List[Dict[str, Any]] = []
        used_positions = set()

        # Required evidence comes first so deterministic citations are stable.
        for item in required + hybrid:
            position = int(item["row_position"])

            if position in used_positions:
                continue

            merged.append(dict(item))
            used_positions.add(position)

        for index, item in enumerate(merged, start=1):
            item["source_label"] = f"S{index}"

        return merged[: max(top_k, SMART_CANDIDATE_CHUNKS)]

    @staticmethod
    def evidence_label(
        evidence: Sequence[Dict[str, Any]],
        source_id: str,
        phrases: Sequence[str],
    ) -> Optional[str]:
        best_label = None
        best_score = -1

        for item in evidence:
            if str(item.get("source_id")) != source_id:
                continue

            lowered = clean_whitespace(
                item.get("text", "")
            ).lower()
            score = sum(
                1
                for phrase in phrases
                if clean_whitespace(phrase).lower() in lowered
            )

            if score > best_score:
                best_score = score
                best_label = item.get("source_label")

        return best_label

    def contextualize_question(
        self,
        question: str,
        history: Sequence[Dict[str, Any]],
    ) -> str:
        question = clean_whitespace(question)
        if len(question.split()) >= 8:
            return question

        previous_user = ""
        for message in reversed(history or []):
            if (
                isinstance(message, dict)
                and message.get("role") == "user"
            ):
                previous_user = clean_whitespace(message.get("content"))
                if previous_user:
                    break

        if previous_user:
            return (
                f"Previous question: {previous_user}\n"
                f"Follow-up question: {question}"
            )

        return question

    def evidence_context(
        self,
        evidence: Sequence[Dict[str, Any]],
    ) -> str:
        blocks: List[str] = []
        current_length = 0

        for item in evidence:
            text = clean_whitespace(item.get("text", ""))[:2300]
            section_line = (
                f"Section: {item['display_section']}\n"
                if item.get("display_section")
                else ""
            )

            block = (
                f"[{item['source_label']}]\n"
                f"Document: {item.get('document_title', '')}\n"
                f"{section_line}"
                f"Pages: {item['page_display']}\n"
                f"Evidence: {text}\n"
            )

            if current_length + len(block) > MAX_CONTEXT_CHARACTERS:
                break

            blocks.append(block)
            current_length += len(block)

        return "\n---\n".join(blocks)


    def question_intent(self, question: str) -> str:
        lowered = clean_whitespace(question).lower()

        if (
            "six functions" in lowered
            or (
                "csf" in lowered
                and "function" in lowered
                and any(term in lowered for term in ("what", "list", "name"))
            )
        ):
            return "csf_functions"

        if (
            "ransomware" in lowered
            and (
                "immediately" in lowered
                or "immediate" in lowered
                or "after detecting" in lowered
                or "after detection" in lowered
                or "once detected" in lowered
            )
        ):
            return "immediate_response"

        if (
            "ransomware" in lowered
            and "prepare" in lowered
            and any(term in lowered for term in ("recover", "recovery"))
        ):
            return "prepare_recover"

        if (
            "risk assessment" in lowered
            or "conducting a cybersecurity risk assessment" in lowered
            or (
                "risk" in lowered
                and "assessment" in lowered
                and any(
                    term in lowered
                    for term in (
                        "conduct",
                        "conducting",
                        "perform",
                        "process",
                        "steps",
                        "recommend",
                    )
                )
            )
        ):
            return "risk_assessment"

        if (
            "zero trust" in lowered
            and any(
                phrase in lowered
                for phrase in (
                    "core principles",
                    "principles",
                    "basic tenets",
                    "tenets",
                    "explain zero trust",
                )
            )
        ):
            return "zero_trust_principles"

        if lowered.startswith(
            ("what is ", "what are ", "define ", "list ", "name ")
        ):
            return "direct_fact"

        return "general"

    def expand_retrieval_query(
        self,
        question: str,
        intent: str,
    ) -> str:
        additions = {
            "csf_functions": (
                "NIST CSF 2.0 six functions Govern Identify Protect "
                "Detect Respond Recover"
            ),
            "immediate_response": (
                "ransomware immediate response isolate affected systems "
                "disconnect network contain incident triage notify preserve evidence"
            ),
            "prepare_recover": (
                "ransomware preparation backups incident response plan "
                "communications restore recovery testing continuity operations"
            ),
            "risk_assessment": (
                "NIST SP 800-30 Guide for Conducting Risk Assessments "
                "prepare for assessment conduct assessment communicate and "
                "share results maintain assessment threat sources threat events "
                "vulnerabilities likelihood impact risk determination"
            ),
            "zero_trust_principles": (
                "NIST SP 800-207 zero trust architecture basic tenets all data "
                "sources resources secure communications per-session access "
                "dynamic policy asset security posture dynamic authentication "
                "authorization collect information improve security posture"
            ),
            "direct_fact": "definition list exact names",
            "general": "",
        }

        expanded = additions.get(intent, "")
        return (
            f"{question}\nRetrieval focus: {expanded}"
            if expanded
            else question
        )

    def extract_sentence_candidates(
        self,
        evidence: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []

        for evidence_item in evidence:
            raw = str(evidence_item.get("text", "") or "")
            raw = re.sub(r"[•▪◼□■◆►]+", "\n", raw)
            raw = re.sub(r"\r\n?", "\n", raw)

            segments = re.split(
                r"(?<=[.!?])\s+|\n+|(?<=;)\s+(?=[A-Z])",
                raw,
            )

            for segment in segments:
                sentence = clean_whitespace(segment)

                # Remove common extraction/control prefixes while retaining
                # the source sentence itself.
                sentence = re.sub(
                    r"^(?:[A-Z]{1,4}\d*(?:\.[A-Z0-9-]+)*|N\d+|"
                    r"Task\s+\d+(?:-\d+)?|Step\s+\d+)\s*:\s*",
                    "",
                    sentence,
                    flags=re.IGNORECASE,
                )
                sentence = re.sub(
                    r"^[\(\[]?[a-z0-9]{1,4}[\)\]]\s+",
                    "",
                    sentence,
                    flags=re.IGNORECASE,
                )
                sentence = clean_whitespace(sentence)
                sentence = re.sub(
                    r"(?<=\w)-and\b",
                    " and",
                    sentence,
                    flags=re.IGNORECASE,
                )
                sentence = re.sub(
                    r"(?<=[.!?])\d{1,3}(?=\s|$)",
                    "",
                    sentence,
                )
                sentence = clean_whitespace(sentence)
                words = sentence.split()

                if not (
                    SMART_MIN_SENTENCE_WORDS
                    <= len(words)
                    <= SMART_MAX_SENTENCE_WORDS
                ):
                    continue

                lowered = sentence.lower()

                if lowered.startswith(BROKEN_STARTS):
                    continue

                if sentence[0].islower():
                    continue

                if "http://" in lowered or "https://" in lowered:
                    continue

                if low_quality_section(
                    evidence_item.get("section", "")
                ):
                    continue

                if sentence.count(")") > sentence.count("("):
                    continue

                if sentence.count("(") - sentence.count(")") > 1:
                    continue

                uppercase_letters = sum(
                    character.isupper()
                    for character in sentence
                    if character.isalpha()
                )
                alphabetic_letters = sum(
                    character.isalpha()
                    for character in sentence
                )

                if (
                    alphabetic_letters > 0
                    and uppercase_letters / alphabetic_letters > 0.72
                ):
                    continue

                if re.search(
                    r"\b(page|pages|figure|fig\.|table)\s+\d+\b",
                    lowered,
                ):
                    continue

                if not sentence.endswith((".", "?", "!")):
                    sentence = sentence.rstrip(";,") + "."

                candidates.append(
                    {
                        "text": sentence,
                        "source_label": evidence_item["source_label"],
                        "source_id": evidence_item.get("source_id", ""),
                        "row_position": int(
                            evidence_item["row_position"]
                        ),
                        "retrieval_score": float(
                            evidence_item.get(
                                "retrieval_score",
                                0.0,
                            )
                        ),
                    }
                )

                if len(candidates) >= SMART_MAX_SENTENCES:
                    return candidates

        return candidates

    @staticmethod
    def lexical_jaccard(left: str, right: str) -> float:
        left_terms = set(tokenize(left))
        right_terms = set(tokenize(right))

        if not left_terms or not right_terms:
            return 0.0

        return len(left_terms & right_terms) / len(
            left_terms | right_terms
        )

    def score_sentence_candidates(
        self,
        question: str,
        intent: str,
        candidates: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        expanded_query = self.expand_retrieval_query(
            question,
            intent,
        )

        query_vector = self.embedding_model.encode(
            [expanded_query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        sentence_vectors = self.embedding_model.encode(
            [item["text"] for item in candidates],
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sentence_vectors = np.asarray(
            sentence_vectors,
            dtype=np.float32,
        )

        question_terms = {
            token
            for token in tokenize(question)
            if token not in STOPWORDS and len(token) > 2
        }

        scored: List[Dict[str, Any]] = []

        for item, vector in zip(candidates, sentence_vectors):
            text = item["text"]
            lowered = text.lower()
            sentence_terms = set(tokenize(text))

            semantic = float(vector @ query_vector)
            overlap = len(
                question_terms.intersection(sentence_terms)
            )

            score = semantic * 7.0
            score += overlap * 0.7
            score += min(
                item["retrieval_score"] * 12.0,
                0.6,
            )

            if intent == "csf_functions":
                required = {
                    "govern",
                    "identify",
                    "protect",
                    "detect",
                    "respond",
                    "recover",
                }
                present = required.intersection(sentence_terms)
                score += len(present) * 2.0

                if len(present) == 6:
                    score += 14.0

                if any(
                    phrase in lowered
                    for phrase in (
                        "there are six",
                        "six csf functions",
                        "six cybersecurity framework functions",
                    )
                ):
                    score += 5.0

            elif intent == "immediate_response":
                action_hits = len(
                    RESPONSE_TERMS.intersection(sentence_terms)
                )
                score += action_hits * 1.25

                if not ACTION_TERMS.intersection(sentence_terms):
                    score -= 3.0

                if any(
                    phrase in lowered
                    for phrase in (
                        "basic ransomware tips",
                        "preventive steps",
                        "before an incident",
                    )
                ):
                    score -= 5.0

                if any(
                    phrase in lowered
                    for phrase in (
                        "isolate affected systems",
                        "disconnect affected systems",
                        "take the network offline",
                        "contain systems",
                    )
                ):
                    score += 6.0

            elif intent == "prepare_recover":
                preparation_hits = len(
                    PREPARATION_TERMS.intersection(sentence_terms)
                )
                recovery_hits = len(
                    RECOVERY_TERMS.intersection(sentence_terms)
                )
                action_hits = len(
                    ACTION_TERMS.intersection(sentence_terms)
                )

                score += max(
                    preparation_hits,
                    recovery_hits,
                ) * 0.9
                score += action_hits * 0.65

                if any(
                    phrase in lowered
                    for phrase in GENERIC_FRAMEWORK_PHRASES
                ):
                    score -= 3.5

                if any(
                    phrase in lowered
                    for phrase in (
                        "backup and restoration strategy",
                        "incident response plan",
                        "restore systems to normal operations",
                        "confirm that the systems are functioning normally",
                    )
                ):
                    score += 4.0

            elif intent == "risk_assessment":
                risk_hits = len(
                    RISK_ASSESSMENT_TERMS.intersection(sentence_terms)
                )
                score += risk_hits * 1.15

                if item.get("source_id") == "nist_sp_800_30r1":
                    score += 7.0
                else:
                    score -= 3.0

                if any(
                    phrase in lowered
                    for phrase in GENERIC_FRAMEWORK_PHRASES
                ):
                    score -= 6.0

                if any(
                    phrase in lowered
                    for phrase in (
                        "prepare for the assessment",
                        "conduct the assessment",
                        "communicate and share",
                        "maintain the assessment",
                        "identify threat sources",
                        "identify threat events",
                        "determine likelihood",
                        "determine impact",
                        "determine risk",
                    )
                ):
                    score += 5.0

            elif intent == "direct_fact":
                if any(
                    word in lowered
                    for word in (" is ", " are ", " means ", " refers to ")
                ):
                    score += 1.4

            # Penalize likely sentence fragments.
            if len(text.split()) > 52:
                score -= 0.7

            if text.count(":") > 2:
                score -= 0.5

            record = dict(item)
            record["score"] = float(score)
            record["semantic_score"] = semantic
            record["sentence_vector"] = vector
            scored.append(record)

        scored.sort(
            key=lambda item: item["score"],
            reverse=True,
        )
        return scored

    def select_diverse_sentences(
        self,
        ranked: Sequence[Dict[str, Any]],
        maximum: int,
        required_terms: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []

        for candidate in ranked:
            sentence_terms = set(tokenize(candidate["text"]))

            if (
                required_terms
                and not required_terms.intersection(sentence_terms)
            ):
                continue

            duplicate = False

            for existing in selected:
                if (
                    self.lexical_jaccard(
                        candidate["text"],
                        existing["text"],
                    )
                    >= 0.68
                ):
                    duplicate = True
                    break

                similarity = float(
                    candidate["sentence_vector"]
                    @ existing["sentence_vector"]
                )

                if similarity >= 0.92:
                    duplicate = True
                    break

            if duplicate:
                continue

            selected.append(candidate)

            if len(selected) >= maximum:
                break

        return selected


    def select_risk_stage_sentence(
        self,
        ranked: Sequence[Dict[str, Any]],
        stage: str,
        excluded_texts: Optional[set] = None,
    ) -> Optional[Dict[str, Any]]:
        excluded_texts = excluded_texts or set()
        stage_terms = RISK_STAGE_TERMS[stage]

        candidates = []

        for item in ranked:
            if item["text"] in excluded_texts:
                continue

            if item.get("source_id") != "nist_sp_800_30r1":
                continue

            terms = set(tokenize(item["text"]))
            hits = len(stage_terms.intersection(terms))

            if hits == 0:
                continue

            lowered = item["text"].lower()
            stage_bonus = hits * 2.0

            phrase_map = {
                "Prepare": (
                    "prepare for the assessment",
                    "purpose, scope",
                    "assumptions and constraints",
                ),
                "Conduct": (
                    "conduct the assessment",
                    "identify threat sources",
                    "identify threat events",
                    "determine likelihood",
                    "determine impact",
                    "determine risk",
                ),
                "Communicate": (
                    "communicate and share",
                    "communicate results",
                    "share assessment results",
                ),
                "Maintain": (
                    "maintain the assessment",
                    "monitor risk factors",
                    "update the risk assessment",
                ),
            }

            if any(
                phrase in lowered
                for phrase in phrase_map[stage]
            ):
                stage_bonus += 5.0

            record = dict(item)
            record["stage_score"] = (
                float(item["score"]) + stage_bonus
            )
            candidates.append(record)

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item["stage_score"],
            reverse=True,
        )
        return candidates[0]

    def render_ranked_points(
        self,
        title: str,
        points: Sequence[Dict[str, Any]],
    ) -> List[str]:
        if not points:
            return []

        lines = [f"### {title}", ""]

        for item in points:
            lines.append(
                f"- {item['text']} "
                f"[{item['source_label']}]"
            )

        lines.append("")
        return lines


    def render_csf_functions_answer(
        self,
        evidence: Sequence[Dict[str, Any]],
    ) -> str:
        label = self.evidence_label(
            evidence,
            "nist_csf_2_0",
            [
                "there are six csf functions",
                "govern, identify, protect, detect, respond, and recover",
            ],
        )

        if label is None:
            return ""

        return (
            "### Answer\n\n"
            "- The six NIST CSF 2.0 Functions are **Govern, Identify, "
            f"Protect, Detect, Respond, and Recover.** [{label}]"
        )

    def render_risk_assessment_answer(
        self,
        evidence: Sequence[Dict[str, Any]],
    ) -> str:
        prepare_label = self.evidence_label(
            evidence,
            "nist_sp_800_30r1",
            [
                "the first step in the risk assessment process is to prepare",
                "establish a context for the risk assessment",
            ],
        )
        conduct_label = self.evidence_label(
            evidence,
            "nist_sp_800_30r1",
            [
                "conducting risk assessments includes the following specific tasks",
                "identify threat sources that are relevant",
                "determine information security risks",
            ],
        )
        communicate_label = self.evidence_label(
            evidence,
            "nist_sp_800_30r1",
            [
                "communicate risk assessment results",
                "share risk-related information",
            ],
        )
        maintain_label = self.evidence_label(
            evidence,
            "nist_sp_800_30r1",
            [
                "the fourth step in the risk assessment process is to maintain",
                "monitor risk factors identified in risk assessments",
                "update the components of risk assessments",
            ],
        )

        labels = [
            prepare_label,
            conduct_label,
            communicate_label,
            maintain_label,
        ]

        if not all(labels):
            return ""

        return (
            "### NIST risk-assessment process\n\n"
            f"- **Prepare:** Establish the assessment context, including its "
            f"purpose, scope, assumptions, constraints, information sources, "
            f"and analytic approach. [{prepare_label}]\n"
            f"- **Conduct:** Identify relevant threat sources and events, "
            f"vulnerabilities and predisposing conditions; determine "
            f"likelihood and impact; then determine risk, including "
            f"uncertainty. [{conduct_label}]\n"
            f"- **Communicate:** Communicate assessment results to decision-makers "
            f"and share risk-related information and supporting evidence with "
            f"appropriate personnel. [{communicate_label}]\n"
            f"- **Maintain:** Monitor risk factors continuously and update the "
            f"assessment when monitoring identifies changes that could affect "
            f"organizational risk. [{maintain_label}]"
        )

    def render_zero_trust_principles_answer(
        self,
        evidence: Sequence[Dict[str, Any]],
    ) -> str:
        first_label = self.evidence_label(
            evidence,
            "nist_sp_800_207",
            [
                "all data sources and computing services are considered resources",
                "all communication is secured regardless of network location",
                "access to individual enterprise resources is granted on a per-session basis",
            ],
        )
        second_label = self.evidence_label(
            evidence,
            "nist_sp_800_207",
            [
                "access to resources is determined by dynamic policy",
                "monitors and measures the integrity and security posture",
            ],
        )
        third_label = self.evidence_label(
            evidence,
            "nist_sp_800_207",
            [
                "all resource authentication and authorization are dynamic",
                "collects as much information as possible",
            ],
        )

        if not all(
            (first_label, second_label, third_label)
        ):
            return ""

        return (
            "### Core zero-trust principles\n\n"
            f"- Treat all data sources and computing services as resources. "
            f"[{first_label}]\n"
            f"- Secure all communications regardless of network location; "
            f"network location alone does not establish trust. [{first_label}]\n"
            f"- Grant access to individual resources on a per-session basis "
            f"using the least privileges needed. [{first_label}]\n"
            f"- Determine access through dynamic policy using identity, service, "
            f"asset state, behavior, environment, and business risk. "
            f"[{second_label}]\n"
            f"- Continuously monitor and measure the integrity and security "
            f"posture of owned and associated assets. [{second_label}]\n"
            f"- Dynamically authenticate and authorize every resource request "
            f"before access is allowed, with continued reevaluation as needed. "
            f"[{third_label}]\n"
            f"- Collect information about assets, network infrastructure, "
            f"communications, and access requests, and use it to improve "
            f"security policy and posture. [{third_label}]"
        )

    def smart_grounded_answer(
        self,
        question: str,
        evidence: Sequence[Dict[str, Any]],
    ) -> str:
        """
        Compose a concise extractive answer using sentence-level ranking.

        The text remains source-derived, but selection is driven by the exact
        question rather than taking one arbitrary sentence from each chunk.
        """

        intent = self.question_intent(question)

        if intent == "csf_functions":
            deterministic = self.render_csf_functions_answer(
                evidence
            )
            if deterministic:
                return deterministic

        if intent == "risk_assessment":
            deterministic = self.render_risk_assessment_answer(
                evidence
            )
            if deterministic:
                return deterministic

        if intent == "zero_trust_principles":
            deterministic = self.render_zero_trust_principles_answer(
                evidence
            )
            if deterministic:
                return deterministic

        candidates = self.extract_sentence_candidates(
            evidence,
        )
        ranked = self.score_sentence_candidates(
            question,
            intent,
            candidates,
        )

        if not ranked:
            return (
                "The local knowledge base did not return a complete, "
                "question-relevant passage."
            )

        lines: List[str] = []

        if intent == "csf_functions":
            exact = self.select_diverse_sentences(
                ranked,
                maximum=1,
                required_terms={
                    "govern",
                    "identify",
                    "protect",
                    "detect",
                    "respond",
                    "recover",
                },
            )

            if not exact:
                exact = self.select_diverse_sentences(
                    ranked,
                    maximum=1,
                )

            lines.extend(
                self.render_ranked_points(
                    "Answer",
                    exact,
                )
            )

        elif intent == "immediate_response":
            immediate_points = self.select_diverse_sentences(
                ranked,
                maximum=4,
                required_terms=RESPONSE_TERMS,
            )

            if len(immediate_points) < 2:
                immediate_points = self.select_diverse_sentences(
                    ranked,
                    maximum=4,
                )

            lines.extend(
                self.render_ranked_points(
                    "Immediate actions",
                    immediate_points,
                )
            )

        elif intent == "prepare_recover":
            preparation_ranked = sorted(
                ranked,
                key=lambda item: (
                    len(
                        PREPARATION_TERMS.intersection(
                            set(tokenize(item["text"]))
                        )
                    ),
                    item["score"],
                ),
                reverse=True,
            )
            recovery_ranked = sorted(
                ranked,
                key=lambda item: (
                    len(
                        RECOVERY_TERMS.intersection(
                            set(tokenize(item["text"]))
                        )
                    ),
                    item["score"],
                ),
                reverse=True,
            )

            preparation_ranked = [
                item
                for item in preparation_ranked
                if not any(
                    phrase in item["text"].lower()
                    for phrase in GENERIC_FRAMEWORK_PHRASES
                )
                and (
                    ACTION_TERMS.intersection(
                        set(tokenize(item["text"]))
                    )
                    or {
                        "backup",
                        "backups",
                        "plan",
                        "policy",
                        "insurance",
                    }.intersection(
                        set(tokenize(item["text"]))
                    )
                )
            ]

            recovery_ranked = [
                item
                for item in recovery_ranked
                if not any(
                    phrase in item["text"].lower()
                    for phrase in GENERIC_FRAMEWORK_PHRASES
                )
                and {
                    "recover",
                    "recovery",
                    "restore",
                    "restoration",
                    "rebuild",
                    "verify",
                    "confirm",
                    "resume",
                }.intersection(
                    set(tokenize(item["text"]))
                )
            ]

            preparation = self.select_diverse_sentences(
                preparation_ranked,
                maximum=3,
                required_terms=PREPARATION_TERMS,
            )

            recovery = self.select_diverse_sentences(
                recovery_ranked,
                maximum=3,
                required_terms=RECOVERY_TERMS,
            )

            # Avoid repeating the same sentence across sections.
            preparation_text = {
                item["text"]
                for item in preparation
            }
            recovery = [
                item
                for item in recovery
                if item["text"] not in preparation_text
            ][:3]

            lines.extend(
                self.render_ranked_points(
                    "Preparation",
                    preparation,
                )
            )
            lines.extend(
                self.render_ranked_points(
                    "Recovery",
                    recovery,
                )
            )

        elif intent == "risk_assessment":
            used_texts = set()
            stage_points: List[Dict[str, Any]] = []

            for stage in (
                "Prepare",
                "Conduct",
                "Communicate",
                "Maintain",
            ):
                selected = self.select_risk_stage_sentence(
                    ranked,
                    stage,
                    excluded_texts=used_texts,
                )

                if selected is None:
                    continue

                used_texts.add(selected["text"])
                selected = dict(selected)
                selected["stage"] = stage
                stage_points.append(selected)

            if stage_points:
                lines.append("### NIST risk-assessment process")
                lines.append("")

                for item in stage_points:
                    lines.append(
                        f"- **{item['stage']}:** "
                        f"{item['text']} "
                        f"[{item['source_label']}]"
                    )

                lines.append("")
            else:
                fallback_points = self.select_diverse_sentences(
                    ranked,
                    maximum=4,
                    required_terms=RISK_ASSESSMENT_TERMS,
                )
                lines.extend(
                    self.render_ranked_points(
                        "NIST risk-assessment process",
                        fallback_points,
                    )
                )

        elif intent == "direct_fact":
            direct = self.select_diverse_sentences(
                ranked,
                maximum=2,
            )
            lines.extend(
                self.render_ranked_points(
                    "Answer",
                    direct,
                )
            )

        else:
            general = self.select_diverse_sentences(
                ranked,
                maximum=4,
            )
            lines.extend(
                self.render_ranked_points(
                    "Grounded answer",
                    general,
                )
            )

        rendered = "\n".join(lines).strip()

        if not rendered:
            return (
                "The local knowledge base did not provide enough complete "
                "sentences to answer this question directly."
            )

        return rendered

    def synthesize_answer(
        self,
        question: str,
        evidence: Sequence[Dict[str, Any]],
        model_name: str,
    ) -> Tuple[str, str]:
        tokenizer, model, device = load_generation_model(model_name)
        context = self.evidence_context(evidence)

        system_prompt = """
You are a cybersecurity assistant. Answer only from the supplied NIST and
CISA evidence.

Write 4 to 7 concise bullet points. Do not write citations or a Sources
section; the application adds citations after validation.

Rules:
- Use only the evidence.
- Do not invent products, organizations, numbers, incidents, or guarantees.
- Each bullet must contain one factual recommendation or explanation.
- Keep each bullet under 35 words.
- Organize process questions in a sensible order.
- State a limitation when the evidence is incomplete.
""".strip()

        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Evidence:\n{context}\n\n"
            "Write the answer now."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        model_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {
            key: value.to(device)
            for key, value in model_inputs.items()
        }

        with torch.inference_mode():
            output = model.generate(
                **model_inputs,
                max_new_tokens=280,
                do_sample=False,
                repetition_penalty=1.08,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_length = model_inputs["input_ids"].shape[-1]
        generated = output[0][prompt_length:]
        raw_answer = tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        items = parse_generated_items(raw_answer)

        if not items:
            return (
                self.smart_grounded_answer(question, evidence),
                "The model returned no usable bullets; smart grounded mode was used.",
            )

        item_vectors = self.embedding_model.encode(
            items,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        item_vectors = np.asarray(item_vectors, dtype=np.float32)

        evidence_vectors = np.asarray(
            [
                self.embeddings[int(item["row_position"])]
                for item in evidence
            ],
            dtype=np.float32,
        )

        accepted: List[Tuple[str, str, float]] = []

        for item_text, item_vector in zip(items, item_vectors):
            similarities = evidence_vectors @ item_vector
            best_index = int(np.argmax(similarities))
            best_score = float(similarities[best_index])

            if best_score < SUPPORT_THRESHOLD:
                continue

            label = evidence[best_index]["source_label"]
            accepted.append((item_text, label, best_score))

            if len(accepted) >= 7:
                break

        if len(accepted) < 2:
            return (
                self.smart_grounded_answer(question, evidence),
                (
                    "Too few generated bullets passed the grounding filter; "
                    "smart grounded mode was used."
                ),
            )

        lines = ["### Answer", ""]

        for text, label, _ in accepted:
            lines.append(f"- {text} [{label}]")

        return (
            "\n".join(lines),
            (
                f"AI synthesis accepted {len(accepted)} grounded bullet(s) "
                f"using {model_name}."
            ),
        )

    def sources_markdown(
        self,
        evidence: Sequence[Dict[str, Any]],
    ) -> str:
        lines = ["### Sources"]

        for item in evidence:
            location = f"pages {item['page_display']}"
            if item.get("display_section"):
                location = (
                    f"{item['display_section']}, "
                    f"{location}"
                )

            source_url = (
                item.get("doi")
                or item.get("source_url")
                or ""
            )
            suffix = f" — {source_url}" if source_url else ""

            lines.append(
                f"- [{item['source_label']}] "
                f"**{item.get('document_title', 'Unknown document')}**, "
                f"{location}.{suffix}"
            )

        return "\n".join(lines)

    def source_rows(
        self,
        evidence: Sequence[Dict[str, Any]],
    ) -> List[List[Any]]:
        rows = []

        for item in evidence:
            rows.append(
                [
                    item["source_label"],
                    item.get("document_title", ""),
                    item["page_display"],
                    item.get("display_section", ""),
                    round(float(item["retrieval_score"]), 6),
                    round(float(item.get("dense_score") or 0.0), 4),
                    round(float(item.get("bm25_score") or 0.0), 4),
                ]
            )

        return rows

    def answer(
        self,
        question: str,
        history: Sequence[Dict[str, Any]],
        mode: str,
        top_k: int,
        model_name: str,
    ) -> Tuple[str, List[List[Any]], str]:
        started = time.time()
        retrieval_query = self.contextualize_question(
            question,
            history,
        )
        intent = self.question_intent(retrieval_query)
        candidate_top_k = max(
            int(top_k),
            SMART_CANDIDATE_CHUNKS,
        )
        evidence = self.retrieve_for_intent(
            retrieval_query,
            intent,
            candidate_top_k,
        )

        if not evidence:
            return (
                "No evidence was retrieved from the local knowledge base.",
                [],
                "No results.",
            )

        if mode == "Smart grounded answer":
            body = self.smart_grounded_answer(
                retrieval_query,
                evidence,
            )
            generation_status = (
                "Smart grounded mode: no language model was used."
            )
        else:
            try:
                body, generation_status = self.synthesize_answer(
                    retrieval_query,
                    evidence,
                    model_name,
                )
            except Exception as exc:
                body = self.smart_grounded_answer(
                    retrieval_query,
                    evidence,
                )
                generation_status = (
                    "The AI model could not be loaded or executed, so the "
                    f"fast grounded answer was used. Error: {exc}"
                )

        used_labels = set(
            re.findall(r"\[(S\d+)\]", body)
        )
        displayed_evidence = [
            item
            for item in evidence
            if item["source_label"] in used_labels
        ]

        if not displayed_evidence:
            displayed_evidence = evidence[: int(top_k)]

        answer = (
            body
            + "\n\n"
            + self.sources_markdown(displayed_evidence)
        )

        elapsed = time.time() - started
        status = (
            f"**Ready.** Searched {len(evidence)} candidate chunks and used "
            f"{len(displayed_evidence)} source(s) in {elapsed:.1f} seconds. "
            f"{generation_status}"
        )

        return (
            answer,
            self.source_rows(displayed_evidence),
            status,
        )


def parse_generated_items(text: str) -> List[str]:
    cleaned = str(text or "").strip()
    items: List[str] = []

    for line in cleaned.splitlines():
        line = clean_whitespace(line)
        if not line:
            continue

        line = re.sub(
            r"^(?:[-*•]+|\d+[.)])\s*",
            "",
            line,
        ).strip()

        if line.startswith("#"):
            continue

        words = line.split()
        if 5 <= len(words) <= 45:
            items.append(line)

    if not items:
        for sentence in SENTENCE_PATTERN.split(
            clean_whitespace(cleaned)
        ):
            words = sentence.split()
            if 5 <= len(words) <= 45:
                items.append(sentence.strip())

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(items))[:8]


def load_generation_model(
    model_name: str,
) -> Tuple[Any, Any, torch.device]:
    global LOADED_MODEL_NAME, LOADED_TOKENIZER, LOADED_MODEL

    with MODEL_LOCK:
        if (
            LOADED_MODEL_NAME == model_name
            and LOADED_TOKENIZER is not None
            and LOADED_MODEL is not None
        ):
            device = next(LOADED_MODEL.parameters()).device
            return LOADED_TOKENIZER, LOADED_MODEL, device

        # Release a previously loaded model before changing model size.
        LOADED_TOKENIZER = None
        LOADED_MODEL = None
        LOADED_MODEL_NAME = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )

        dtype = (
            torch.float16
            if torch.cuda.is_available()
            else torch.float32
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model.to(device)
        model.eval()

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        LOADED_MODEL_NAME = model_name
        LOADED_TOKENIZER = tokenizer
        LOADED_MODEL = model

        return tokenizer, model, device


ENGINE: Optional[CyberRAGEngine] = None
ENGINE_ERROR: Optional[str] = None


def initialize_engine(artifact_dir: Any) -> Path:
    global ENGINE, ENGINE_ERROR

    resolved_dir = discover_artifact_dir(artifact_dir)

    try:
        ENGINE = CyberRAGEngine(resolved_dir)
        ENGINE_ERROR = None
    except Exception as exc:
        ENGINE = None
        ENGINE_ERROR = str(exc)

    return resolved_dir


def reload_knowledge_base(artifact_path: str):
    resolved_dir = initialize_engine(artifact_path)

    return (
        str(resolved_dir),
        artifact_status_html(
            ENGINE is not None,
            resolved_dir,
            ENGINE_ERROR,
        ),
    )


def clear_workspace():
    resolved_dir = (
        ENGINE.artifact_dir
        if ENGINE is not None
        else discover_artifact_dir(APP_DIR)
    )

    return (
        [],
        "",
        [],
        artifact_status_html(
            ENGINE is not None,
            resolved_dir,
            ENGINE_ERROR,
        ),
    )


def respond(
    message: str,
    history: Optional[List[Dict[str, Any]]],
    mode: str,
    top_k: int,
    model_name: str,
):
    history = list(history or [])
    message = clean_whitespace(message)

    if not message:
        return history, "", [], "**Enter a question.**"

    if ENGINE is None:
        error = ENGINE_ERROR or "The engine was not initialized."
        history.append({"role": "user", "content": message})
        history.append(
            {
                "role": "assistant",
                "content": (
                    "The application could not load the vector-store files.\n\n"
                    f"```\n{error}\n```"
                ),
            }
        )
        return history, "", [], f"**Startup error:** {error}"

    answer, rows, status = ENGINE.answer(
        message,
        history,
        mode,
        int(top_k),
        model_name,
    )

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})

    return history, "", rows, status



def fill_suggested_question(question_text: str) -> str:
    return question_text


def build_app(artifact_dir: Path) -> gr.Blocks:
    resolved_dir = initialize_engine(artifact_dir)

    knowledge_status = artifact_status_html(
        ENGINE is not None,
        resolved_dir,
        ENGINE_ERROR,
    )

    hero_html = """
<div id="hero">
  <div class="hero-grid">
    <div>
      <div class="hero-badge">● LOCAL · PRIVATE · SOURCE-GROUNDED</div>
      <h1 class="hero-title">Cybersecurity Agentic RAG</h1>
      <div class="hero-subtitle">
        Ask questions across your local NIST and CISA knowledge base. Hybrid semantic and keyword retrieval
        surfaces the strongest evidence while every answer keeps traceable document citations.
      </div>
    </div>
    <div class="hero-orb">🛡️</div>
  </div>
</div>
"""

    examples = [
        ["How should a small organization prepare for and recover from ransomware?"],
        ["What are the six functions in NIST CSF 2.0?"],
        ["Explain the seven core principles of zero trust architecture."],
        ["What should an organization do immediately after detecting ransomware?"],
        ["How does NIST recommend conducting and maintaining a cybersecurity risk assessment?"],
    ]

    with gr.Blocks(
        title="Cybersecurity Agentic RAG",
        fill_width=True,
    ) as demo:
        gr.HTML(hero_html)

        with gr.Row(equal_height=False, elem_id="workspace-row"):
            with gr.Column(scale=8, min_width=760, elem_id="chat-column"):
                with gr.Group(elem_classes=["card"]):
                    gr.HTML(
                        '<div class="section-heading">💬 Ask the knowledge base</div>'
                    )

                    chatbot = gr.Chatbot(
                        label=None,
                        height=640,
                        elem_id="chatbot",
                        elem_classes=["fit-chat"],
                        show_label=False,
                        allow_tags=False,
                        placeholder=(
                            "<div style='text-align:center;color:#8fa8bd;padding:78px 10px'>"
                            "<div style='font-size:2.1rem;margin-bottom:12px'>🛡️</div>"
                            "<strong>Start with a cybersecurity question</strong><br>"
                            "<span style='font-size:.9rem'>Answers use your local NIST and CISA sources.</span>"
                            "</div>"
                        ),
                    )

                    question = gr.Textbox(
                        label=None,
                        placeholder=(
                            "Ask about ransomware, incident response, zero trust, "
                            "risk assessment, or the NIST CSF…"
                        ),
                        lines=2,
                        max_lines=4,
                        elem_id="question-box",
                        show_label=False,
                    )

                    with gr.Row():
                        send_button = gr.Button(
                            "Ask the knowledge base  →",
                            variant="primary",
                            elem_id="ask-button",
                            scale=3,
                        )
                        clear_button = gr.Button(
                            "Clear",
                            variant="secondary",
                            elem_classes=["secondary-button"],
                            scale=1,
                        )

                with gr.Group(elem_classes=["suggestion-wrap"]):
                    gr.HTML(
                        '<div class="suggestion-title">⚡ Suggested questions</div>'
                    )

                    suggestion_buttons = []

                    with gr.Row():
                        suggestion_buttons.append(
                            gr.Button(
                                "Ransomware: prepare and recover",
                                elem_classes=["suggestion-button"],
                            )
                        )
                        suggestion_buttons.append(
                            gr.Button(
                                "Six NIST CSF 2.0 Functions",
                                elem_classes=["suggestion-button"],
                            )
                        )
                        suggestion_buttons.append(
                            gr.Button(
                                "Seven zero-trust principles",
                                elem_classes=["suggestion-button"],
                            )
                        )

                    with gr.Row():
                        suggestion_buttons.append(
                            gr.Button(
                                "Immediate ransomware response",
                                elem_classes=["suggestion-button"],
                            )
                        )
                        suggestion_buttons.append(
                            gr.Button(
                                "NIST risk-assessment process",
                                elem_classes=["suggestion-button"],
                            )
                        )

            with gr.Column(scale=4, min_width=360, elem_id="side-column"):
                with gr.Column(elem_classes=["sticky-column", "panel-tight"]):
                    with gr.Group(elem_classes=["card"]):
                        gr.HTML(
                            '<div class="section-heading">⚙️ Answer controls</div>'
                        )

                        mode = gr.Radio(
                            choices=[
                                "Smart grounded answer",
                                "AI synthesis",
                            ],
                            value="Smart grounded answer",
                            label="Answer mode",
                            info=(
                                "Smart grounded mode is best for speed and reliability. "
                                "AI synthesis downloads a local Qwen model."
                            ),
                        )

                        model_name = gr.Dropdown(
                            choices=[
                                DEFAULT_SMALL_MODEL,
                                DEFAULT_LARGE_MODEL,
                            ],
                            value=DEFAULT_SMALL_MODEL,
                            label="Local language model",
                            info=(
                                "Use 0.5B on CPU. Use 1.5B for better answers "
                                "when you have more RAM or a GPU."
                            ),
                        )

                        top_k = gr.Slider(
                            minimum=3,
                            maximum=8,
                            value=6,
                            step=1,
                            label="Evidence chunks",
                        )

                        status = gr.HTML(knowledge_status)

                        gr.HTML(
                            """
<div class="metric-grid">
  <div class="metric">
    <div class="metric-value">FAISS</div>
    <div class="metric-label">Dense semantic search</div>
  </div>
  <div class="metric">
    <div class="metric-value">BM25</div>
    <div class="metric-label">Keyword retrieval</div>
  </div>
  <div class="metric">
    <div class="metric-value">Local</div>
    <div class="metric-label">Private processing</div>
  </div>
</div>
"""
                        )

                    with gr.Accordion(
                        "Advanced setup",
                        open=False,
                        elem_classes=["soft-card"],
                    ):
                        artifact_path = gr.Textbox(
                            value=str(resolved_dir),
                            label="Vector-store folder",
                            placeholder=r"C:\Users\Tawfik\Downloads\trae\cyber",
                            info=(
                                "The app detects this folder automatically. "
                                "Change it only if your files are in a different location."
                            ),
                        )

                        reload_button = gr.Button(
                            "Reload knowledge base",
                            variant="secondary",
                        )

                        gr.HTML(
                            """
<div class="compact-help">
Required files: <code>chunk_embeddings.npy</code>,
<code>chunks_metadata.jsonl</code>,
<code>bm25_tokenized_corpus.json</code>, and
<code>retrieval_config.json</code>.
</div>
"""
                        )

        with gr.Tabs():
            with gr.Tab("Retrieved evidence"):
                sources_table = gr.Dataframe(
                    headers=[
                        "Label",
                        "Document",
                        "Pages",
                        "Section",
                        "Hybrid score",
                        "Dense score",
                        "BM25 score",
                    ],
                    datatype=[
                        "str",
                        "str",
                        "str",
                        "str",
                        "number",
                        "number",
                        "number",
                    ],
                    label=None,
                    interactive=False,
                    wrap=True,
                    show_label=False,
                    elem_classes=["card"],
                )

            with gr.Tab("How it works"):
                gr.Markdown(
                    """
### Retrieval pipeline

1. Your question is embedded with Sentence Transformers.
2. FAISS retrieves semantically similar passages.
3. BM25 retrieves exact terminology and publication identifiers.
4. Reciprocal Rank Fusion combines both rankings.
5. The final response includes labels that map to the evidence table.

### Recommended mode

Use **Smart grounded answer** for your presentation. It is fast, stable, and keeps clear source citations.
                    """
                )

        inputs = [
            question,
            chatbot,
            mode,
            top_k,
            model_name,
        ]
        outputs = [
            chatbot,
            question,
            sources_table,
            status,
        ]

        suggested_prompts = [
            "How should a small organization prepare for and recover from ransomware?",
            "What are the six functions in NIST CSF 2.0?",
            "Explain the seven core principles of zero trust architecture.",
            "What should an organization do immediately after detecting ransomware?",
            "How does NIST recommend conducting and maintaining a cybersecurity risk assessment?",
        ]

        for suggestion_button, suggested_prompt in zip(
            suggestion_buttons,
            suggested_prompts,
        ):
            suggestion_button.click(
                fill_suggested_question,
                inputs=[gr.State(suggested_prompt)],
                outputs=[question],
                api_name=False,
            )

        send_button.click(
            respond,
            inputs=inputs,
            outputs=outputs,
            api_name="ask",
        )
        question.submit(
            respond,
            inputs=inputs,
            outputs=outputs,
            api_name=False,
        )

        clear_button.click(
            clear_workspace,
            inputs=[],
            outputs=[
                chatbot,
                question,
                sources_table,
                status,
            ],
            api_name=False,
        )

        reload_button.click(
            reload_knowledge_base,
            inputs=[artifact_path],
            outputs=[
                artifact_path,
                status,
            ],
            api_name=False,
        )

    return demo

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local Cybersecurity RAG Gradio application."
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=os.environ.get(
            "CYBER_RAG_DIR",
            str(APP_DIR),
        ),
        help=(
            "Folder containing chunks_metadata.jsonl, chunk_embeddings.npy, "
            "bm25_tokenized_corpus.json, and retrieval_config.json."
        ),
    )
    parser.add_argument(
        "--server-name",
        default="127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Ask Gradio to create a temporary public link.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    application = build_app(args.artifact_dir)
    application.queue(
        default_concurrency_limit=1,
        max_size=20,
    ).launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        show_error=True,
        css=APP_CSS,
        theme=gr.themes.Base(),
        footer_links=[],
    )
