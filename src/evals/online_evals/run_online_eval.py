import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric
from deepeval.test_case.llm_test_case import LLMTestCase
from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()

JUDGE_MODEL = "gpt-5.4-mini"  # keep in sync with params.yaml's evaluation.judge_llm
ENVIRONMENT_FILTER = "production"
ROOT_SPAN_NAME = "chat-response"
RETRIEVE_SPAN_NAME = "retrieve"
LOOKBACK_MINUTES = 15  # only used when no checkpoint exists yet

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = SCRIPT_DIR / "eval_checkpoint.json"

# Must stay in sync with src/app/rag_workflow_with_guardrails.py. Hardcoded
# rather than imported since this script has no access to the rest of the repo.
CRITICAL_FALLBACK_MESSAGE = (
    "Something went wrong while processing your request and it could not be "
    "completed safely. Please try again in a moment."
)
SOFT_FALLBACK_MESSAGE = (
    "I couldn't produce a fully reliable answer to that question. Could you "
    "rephrase it or ask something more specific?"
)
FALLBACK_MESSAGES = [CRITICAL_FALLBACK_MESSAGE, SOFT_FALLBACK_MESSAGE]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("online_eval")

# Same metric config as src/evals/application_evals/evaluate_rag_app.py,
# restricted to the reference-free "RAG Triad".
answer_relevancy_metric = AnswerRelevancyMetric(model=JUDGE_MODEL)
faithfulness_metric = FaithfulnessMetric(model=JUDGE_MODEL)
contextual_relevancy_metric = ContextualRelevancyMetric(model=JUDGE_MODEL)


def load_checkpoint() -> datetime:
    default = datetime.now(timezone.utc) - timedelta(minutes=LOOKBACK_MINUTES)
    if not CHECKPOINT_PATH.exists():
        return default
    try:
        data = json.loads(CHECKPOINT_PATH.read_text())
        return datetime.fromisoformat(data["last_processed_timestamp"])
    except (json.JSONDecodeError, KeyError, ValueError, OSError):
        logger.warning("Checkpoint missing/corrupt, defaulting to %s", default.isoformat())
        return default


def save_checkpoint(timestamp: datetime) -> None:
    payload = json.dumps({"last_processed_timestamp": timestamp.isoformat()})
    fd, tmp_path = tempfile.mkstemp(dir=SCRIPT_DIR, prefix=".eval_checkpoint_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(payload)
        os.replace(tmp_path, CHECKPOINT_PATH)
    except BaseException:
        os.remove(tmp_path)
        raise


def fetch_new_root_spans(client, since: datetime, until: datetime) -> list:
    spans = []
    cursor = None
    while True:
        response = client.api.observations.get_many(
            name=ROOT_SPAN_NAME,
            environment=ENVIRONMENT_FILTER,
            from_start_time=since,
            to_start_time=until,
            fields="core,basic,io",
            limit=100,
            cursor=cursor,
        )
        spans.extend(response.data)
        cursor = response.meta.cursor
        if not cursor:
            break
    return spans


def fetch_retrieval_chunks(client, trace_id: str) -> list[str]:
    """Best-effort: returns [] (never raises) if the retrieve span or its
    expected shape isn't found, so callers can fall back to skipping the
    context-dependent metrics for that trace."""
    try:
        response = client.api.observations.get_many(
            trace_id=trace_id,
            name=RETRIEVE_SPAN_NAME,
            fields="core,io",
            limit=5,
        )
    except Exception:
        logger.exception("Failed to fetch retrieve span for trace %s", trace_id)
        return []

    if not response.data:
        return []

    output = response.data[0].output
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            return []
    if not isinstance(output, dict):
        return []

    chunks = []
    for doc in output.get("retrieved_docs") or []:
        if isinstance(doc, str):
            chunks.append(doc)
        elif isinstance(doc, dict):
            if "page_content" in doc:
                chunks.append(doc["page_content"])
            elif isinstance(doc.get("kwargs"), dict) and "page_content" in doc["kwargs"]:
                # generic LangChain Serializable JSON shape
                chunks.append(doc["kwargs"]["page_content"])
    return chunks


def is_fallback_or_unusable(output_text) -> bool:
    if not output_text or not isinstance(output_text, str):
        return True
    return output_text.strip() in FALLBACK_MESSAGES


def _measure_and_push(client, trace_id: str, name: str, metric, test_case: LLMTestCase) -> None:
    metric.measure(test_case)
    if metric.error or metric.score is None:
        raise RuntimeError(metric.error or f"{name} produced no score")
    client.create_score(
        trace_id=trace_id,
        name=name,
        value=metric.score,
        data_type="NUMERIC",
        comment=metric.reason,
        score_id=f"{trace_id}_{name}",
    )


def score_trace(client, trace_id: str, query: str, response_text: str, chunks: list[str]) -> None:
    test_case = LLMTestCase(input=query, actual_output=response_text, retrieval_context=chunks or None)

    metrics_to_run = [("answer_relevancy", answer_relevancy_metric)]
    if chunks:
        metrics_to_run += [
            ("faithfulness", faithfulness_metric),
            ("contextual_relevancy", contextual_relevancy_metric),
        ]
    else:
        logger.warning("No retrieval chunks for trace %s; scoring answer_relevancy only", trace_id)

    for name, metric in metrics_to_run:
        try:
            _measure_and_push(client, trace_id, name, metric, test_case)
        except Exception:
            logger.exception("Failed to compute/push %s for trace %s", name, trace_id)


def main() -> None:
    run_started_at = datetime.now(timezone.utc)
    checkpoint = load_checkpoint()
    client = get_client()

    fetched = skipped = scored = failed = 0
    try:
        spans = fetch_new_root_spans(client, since=checkpoint, until=run_started_at)
        fetched = len(spans)

        for span in spans:
            trace_id = span.trace_id
            if not trace_id:
                continue
            if is_fallback_or_unusable(span.output) or not isinstance(span.input, str) or not span.input.strip():
                skipped += 1
                continue

            chunks = fetch_retrieval_chunks(client, trace_id)
            try:
                score_trace(client, trace_id, span.input, span.output, chunks)
                scored += 1
            except Exception:
                failed += 1
                logger.exception("Unhandled failure scoring trace %s", trace_id)
    except Exception:
        logger.exception("Run aborted before completion; checkpoint left untouched so the next run retries this window")
        client.flush()
        return

    client.flush()
    save_checkpoint(run_started_at)
    logger.info(
        "Run complete: fetched=%d skipped_fallback=%d scored=%d failed=%d checkpoint->%s",
        fetched, skipped, scored, failed, run_started_at.isoformat(),
    )


if __name__ == "__main__":
    main()
