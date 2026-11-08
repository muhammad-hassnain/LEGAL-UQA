"""Judge every row of the human-eval set with JEV in noul mode.

For each (question, ground_truth_answer, llm_answer) triple we ask JEV one
yes/no question: "does the LLM answer convey the same meaning as the
ground-truth answer?". The reply is a calibrated probability in [0, 1]
(0 = definitely no, 1 = definitely yes, 0.5 = max uncertainty). We
checkpoint to JSONL so a crashed run resumes cheaply, then materialise a
per-model CSV joining the noul verdict back onto the human Score / Strict
labels.

Everything JEV sees comes from the human-eval file itself (`question`,
`answers`, the model's `prediction_text_*`); no external context is passed.

CLI entry point lives in scripts/jev/run.py. Agreement metrics live in
scripts/jev/agreement.py.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
HUMAN_DIR = DATA / "human_evaluations_qa"
OUT_DIR = DATA / "automated_evaluations_qa" / "jev"
REVISED_OUT_DIR = OUT_DIR / "reference-v2"


@dataclass(frozen=True)
class EvalFile:
    filename: str
    prediction_col: str
    question_col: str
    score_col: str
    answer_col: str = "answers"


# The nine models with human-eval files. Column names differ by file; this map
# is the single source of truth for which column holds what.
HUMAN_EVAL_FILES: dict[str, EvalFile] = {
    "claude-sonnet-5":   EvalFile("claude-sonnet-5-responses.csv",   "prediction_text_claude_sonnet_5", "question",  "Score"),
    "gemma-4-e4b":       EvalFile("gemma-4-e4b-responses.csv",       "prediction_text_gemma_4_e4b",     "question",  "Score"),
    "deepseek-v4-pro":   EvalFile("deepseek-v4-pro-responses.csv",   "prediction_text_deepseek_v4_pro", "question",  "Score"),
    "gpt-5.4":           EvalFile("gpt-5.4-responses.csv",           "prediction_text_gpt_5_4",         "question",  "Score"),
    "claude-3.5-sonnet": EvalFile("claude-3.5-sonnet_human_eval.xlsx", "prediction_text_claude",        "question",  "score"),
    "gemini-flash":      EvalFile("gemini-flash_human_eval.xlsx",    "prediction_text_gemini",          "question",  "score"),
    "gemini-pro":        EvalFile("gemini-pro_human_eval.xlsx",      "prediction_text_gemini",          "question",  "Score"),
    "mistral-large":     EvalFile("mistral-large_human_eval.xlsx",   "prediction_text_mistral",         "question",  "score"),
    "mt5-large":         EvalFile("mt5-large_human_eval.xlsx",       "prediction_text",                 "questions", "Score"),
}

JEV_MODEL = os.environ.get("JEV_MODEL", "jev-latest")


def _load_api_key() -> None:
    """Read .env and ensure TYPESAFE_API_KEY is set.

    Accepts either ``TYPESAFE_API_KEY`` (what the SDK reads natively) or
    ``JEV_API_KEY`` (the alias used in this project's .env), copying the
    latter into the former so the SDK finds it.
    """
    load_dotenv(ROOT / ".env")
    if "TYPESAFE_API_KEY" in os.environ:
        return
    if "JEV_API_KEY" in os.environ:
        os.environ["TYPESAFE_API_KEY"] = os.environ["JEV_API_KEY"]
        return
    raise RuntimeError(
        "No JEV API key found. Add JEV_API_KEY (or TYPESAFE_API_KEY) to .env."
    )

INSTRUCTIONS = (
    "You are judging a Urdu-language legal question-answering system. The "
    "<state> gives you the question, a model-generated reference answer "
    "translated into Urdu with limited manual checking, and a candidate "
    "answer produced by an LLM. Decide whether "
    "the LLM answer conveys the same meaning as the ground-truth answer. "
    "Wording, phrasing, and length may differ freely; only semantic "
    "equivalence matters."
)
CRITERIA_TEXT = {
    "true": (
        "The LLM answer expresses the same facts as the ground-truth answer "
        "and would be considered a correct response to the question. "
        "Paraphrasing, synonyms, re-ordering, and extra harmless detail are "
        "all fine."
    ),
    "false": (
        "The LLM answer contradicts the ground-truth answer, omits something "
        "the ground truth treats as essential, or adds a substantive claim "
        "the ground truth does not support."
    ),
}
PROMPT_VERSION = "reference-v2"


def load_rows(model_key: str) -> pd.DataFrame:
    """Load one model's human-eval file as (id, question, gold, prediction, scores)."""
    spec = HUMAN_EVAL_FILES[model_key]
    reader = pd.read_excel if spec.filename.endswith(".xlsx") else pd.read_csv
    df = reader(HUMAN_DIR / spec.filename)

    required = {"id", spec.prediction_col, spec.question_col,
                spec.answer_col, spec.score_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{spec.filename} missing columns: {sorted(missing)}")

    df = df.rename(columns={
        spec.prediction_col: "prediction",
        spec.question_col: "question",
        spec.answer_col: "ground_truth",
        spec.score_col: "human_score",
    })
    # Gemma CSV has one blank-id row; drop anything that would break the merge.
    df = df.dropna(subset=["id", "question", "ground_truth",
                           "prediction", "human_score"])
    df["id"] = df["id"].astype(int)

    keep = ["id", "question", "ground_truth", "prediction", "human_score"]
    if "Strict" in df.columns:
        keep.append("Strict")
    return df[keep].reset_index(drop=True)


def input_fingerprint(row: pd.Series) -> str:
    payload = {"question": row["question"], "reference": row["ground_truth"],
               "prediction": row["prediction"], "instructions": INSTRUCTIONS,
               "criteria": CRITERIA_TEXT, "requested_model": JEV_MODEL}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


async def _judge_one(client, row: pd.Series) -> dict:
    from typesafe_sdk import Noul, NoulCriteria
    state = {
        "question":            row["question"],
        "ground_truth_answer": row["ground_truth"],
        "llm_answer":          row["prediction"],
    }
    t0 = time.perf_counter()
    resp = await client.system_one(
        model=JEV_MODEL,
        state=state,
        questions={"is_correct": Noul(instructions=INSTRUCTIONS, criteria=NoulCriteria(**CRITERIA_TEXT))},
    )
    return {
        "id": int(row["id"]),
        "noul": float(resp.answers["is_correct"].noul),
        "latency_ms": int((time.perf_counter() - t0) * 1000),
        "input_tokens": int(getattr(resp.usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(resp.usage, "output_tokens", 0) or 0),
        "model": resp.model,
        "input_sha256": input_fingerprint(row),
        "prompt_version": PROMPT_VERSION,
    }


def _load_checkpoint(path: Path, fingerprints: dict[int, str]) -> set[int]:
    if not path.exists():
        return set()
    done: set[int] = set()
    with path.open() as f:
        for line in f:
            record = json.loads(line)
            ident = int(record["id"])
            if ident not in fingerprints or record.get("input_sha256") != fingerprints[ident]:
                raise ValueError(f"Stale/unversioned checkpoint {path}, ID {ident}; use a fresh output directory")
            if ident in done:
                raise ValueError(f"Duplicate checkpoint ID {ident}: {path}")
            done.add(ident)
    return done


async def run_model(
    model_key: str,
    *,
    concurrency: int = 8,
    limit: int | None = None,
    max_retries: int = 5,
    verbose: bool = True,
    output_dir: Path = REVISED_OUT_DIR,
) -> Path:
    """Judge every eligible row for one model; return the output CSV path."""
    global JEV_MODEL
    _load_api_key()
    JEV_MODEL = os.environ.get("JEV_MODEL", "jev-latest")
    from typesafe_sdk import AsyncTypeSafeClient
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = load_rows(model_key)
    rows = all_rows
    if limit is not None:
        rows = rows.head(limit)

    ckpt = output_dir / f"{model_key}_jev_verdicts.jsonl"
    fingerprints = {int(r['id']): input_fingerprint(r) for _, r in all_rows.iterrows()}
    done = _load_checkpoint(ckpt, fingerprints)
    todo = rows[~rows["id"].isin(done)]
    if verbose:
        print(f"[{model_key}] {len(rows)} total / {len(done)} cached / {len(todo)} to judge")

    if not todo.empty:
        sem = asyncio.Semaphore(concurrency)

        async with AsyncTypeSafeClient() as client:
            async def judge_with_retry(row: pd.Series) -> dict | None:
                async with sem:
                    for attempt in range(max_retries):
                        try:
                            return await _judge_one(client, row)
                        except Exception as exc:  # noqa: BLE001
                            wait = min(2 ** attempt, 30)
                            if verbose:
                                print(f"[{model_key} id={int(row['id'])}] attempt {attempt+1} failed: {exc!r}; retry in {wait}s")
                            await asyncio.sleep(wait)
                    if verbose:
                        print(f"[{model_key} id={int(row['id'])}] gave up after {max_retries} attempts")
                    return None

            tasks = [asyncio.create_task(judge_with_retry(r)) for _, r in todo.iterrows()]
            # Write results as they arrive so a kill mid-run keeps progress.
            with ckpt.open("a") as sink:
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    if result is None:
                        continue
                    sink.write(json.dumps(result) + "\n")
                    sink.flush()

    if not ckpt.exists() or ckpt.stat().st_size == 0:
        if verbose:
            print(f"[{model_key}] no verdicts yet; nothing to materialise")
        return ckpt
    verdicts = pd.read_json(ckpt, lines=True)
    merged = rows.merge(verdicts, on="id", how="inner")
    merged["jev_verdict"] = (merged["noul"] >= 0.5).astype(int)
    merged["human_verdict"] = (merged["human_score"] >= 0.5).astype(int)

    out_csv = output_dir / f"{model_key}_jev_verdicts.csv"
    merged.to_csv(out_csv, index=False)
    if verbose:
        print(f"[{model_key}] wrote {len(merged)} verdicts -> {out_csv.relative_to(ROOT)}")
    return out_csv


def run_model_sync(model_key: str, **kwargs) -> Path:
    """Convenience wrapper for notebooks."""
    return asyncio.run(run_model(model_key, **kwargs))
