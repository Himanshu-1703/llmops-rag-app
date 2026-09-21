# LLMOps RAG App

A production-oriented Retrieval-Augmented Generation service that demonstrates a complete LLMOps lifecycle: layered guardrails, externalized prompt management, synthetic evaluation-set generation, LLM-as-judge offline evaluation, experiment tracking, and automated regression/promotion gates.

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.12-blue.svg">
  <img alt="Package manager: uv" src="https://img.shields.io/badge/deps-uv-DE5FE9.svg">
  <img alt="Orchestration: LangGraph" src="https://img.shields.io/badge/orchestration-LangGraph-1C3C3C.svg">
  <img alt="API: FastAPI" src="https://img.shields.io/badge/api-FastAPI-009688.svg">
  <img alt="Eval: DeepEval" src="https://img.shields.io/badge/eval-DeepEval-6E56CF.svg">
  <img alt="Tracking: MLflow" src="https://img.shields.io/badge/tracking-MLflow%20%2F%20DagsHub-0194E2.svg">
  <img alt="CI/CD: GitHub Actions" src="https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF.svg">
  <img alt="Deploy: AWS CodeDeploy" src="https://img.shields.io/badge/deploy-AWS%20CodeDeploy-FF9900.svg">
</p>

---

## Overview

The service answers user questions grounded in a corpus of long-form transcript documents. Retrieval and generation run as an explicit LangGraph state machine, with validation gates wrapped around the user query, the retrieved context, and the generated answer. Around the runtime sits an offline evaluation and experiment-tracking pipeline that scores every change against a versioned metric baseline and blocks regressions before a prompt or configuration is promoted.

**Highlights**

- **Graph-based RAG** — retrieval, augmentation and generation as discrete, individually testable LangGraph nodes.
- **Three-layer guardrails** — jailbreak / PII / topic checks on input, prompt-injection detection on retrieved context, and relevancy checks on output, with graceful `exception` / `refrain` fallbacks.
- **Externalized prompts** — the system prompt is versioned and label-promoted in Langfuse, never hard-coded.
- **End-to-end tracing** — every `/chat` call is a Langfuse trace spanning retrieval, prompt, generation, and each guardrail decision, with the prompt version and the triggering session attached.
- **Synthetic evaluation data** — golden Q&A pairs are generated from the corpus with DeepEval's `Synthesizer` (filtration + evolution), then curated.
- **LLM-as-judge evaluation** — a 7-metric RAG suite (contextual recall/precision/relevancy, answer relevancy, faithfulness) plus custom `GEval` criteria.
- **Experiment tracking** — every evaluation run logs params, metrics, datasets, prompt and code artifacts to MLflow (DagsHub-hosted).
- **Statistical promotion gates** — noise- and drift-aware thresholds gate regression and champion promotion decisions in CI-style pytest checks.
- **Idempotent ingestion** — content-hashed chunks mean re-syncing the corpus only re-embeds what changed.
- **Champion-driven CD** — every promoted champion is built into containers, pushed to ECR, and rolled out to an auto-scaled EC2 fleet via CodeDeploy.

## Architecture

```
Streamlit UI ──HTTP──▶ FastAPI ──▶ LangGraph (guardrailed RAG) ──▶ Chroma + OpenAI
                          │                    │
                          │                    └─ trace + guardrail spans ──▶ Langfuse
                          └─ /internal/vector-store  ── ingestion (admin-key protected)

params.yaml ──▶ Pydantic config ──▶ shared LLM / retriever / logger
Langfuse ──▶ system prompt (fetched by label at runtime, linked back onto its trace)
```

The served graph:

![RAG workflow](reports/graph.png)

Each guardrail node emits a status of `ok`, `exception`, or `refrain`. `exception` routes to a critical fallback message; `refrain` routes to a soft fallback asking the user to rephrase. The plain (non-guardrailed) graph is retained for evaluation, where it produces the `actual_output` scored against the golden set.

### Repository layout

| Path | Purpose |
| --- | --- |
| `src/app/` | RAG graphs, guardrails, vector store, Langfuse prompt lifecycle |
| `src/api/` | FastAPI app — `chat`, `health`, and admin-key-protected `vector-store` / `debug` routers |
| `src/frontend/` | Streamlit chat client |
| `src/config/` | `params.yaml` loading + strict Pydantic validation |
| `src/data/` | Golden-set synthesis and evaluation-set generation |
| `src/evals/` | DeepEval metric suite and standalone metric checks |
| `utils/` | Transcript cleaning, MLflow helpers |
| `src/experiment/` | Experiment pipeline, vector-store rebuild, and challenger promote/reject scripts |
| `utils/compute_*_thresholds.py` | Recompute noise / historical metric thresholds from run history |
| `tests/` | `test_regression.py`, `test_promotion.py`, and manual API demo scripts |
| `notebooks/` | Component prototypes (baseline RAG, per-layer guardrails) |
| `reports/` | Timestamped evaluation reports and results |
| `docker/` | `Dockerfile.api`, `Dockerfile.frontend`, and the image's ingestion entrypoint |
| `deploy/codedeploy/` | CodeDeploy revision — `appspec.yml`, EC2 `docker-compose.yml`, lifecycle hook scripts |
| `deploy/ec2/manual-steps.md` | One-time manual EC2 bring-up (superseded by the ASG + CodeDeploy rollout) |
| `.github/workflows/` | `ci.yaml` (experiment → gates → promote) and `cd.yaml` (build → push → deploy) |
| `codedeploy_steps.md` | One-time AWS console setup for the ASG + ALB + CodeDeploy rollout |

## Tech stack

| Concern | Tooling |
| --- | --- |
| Orchestration | LangGraph, LangChain |
| Vector store | Chroma (persistent) |
| Models | OpenAI chat + embeddings (configurable in `params.yaml`) |
| Guardrails | Guardrails AI + custom LLM validators (jailbreak, PII, topic) + hub validators (prompt-injection, relevancy, reading-time) |
| Prompt registry & observability | Langfuse (prompt versions, request traces, guardrail spans) |
| Evaluation | DeepEval (`Synthesizer`, RAG metrics, `GEval`) |
| Experiment tracking | MLflow, DagsHub |
| API / UI | FastAPI, Uvicorn, Streamlit |
| Tooling | uv, Pydantic, pytest |
| Containers | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Deployment | AWS ECR, CodeDeploy, EC2 Auto Scaling Group + ALB, Secrets Manager |

## Getting started

### Prerequisites

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- API access for OpenAI, Langfuse, and a DagsHub-hosted MLflow tracking server

### Installation

```bash
uv sync
```

This resolves dependencies and installs the first-party packages (`api`, `app`, `config`, `data`, `evals`, `frontend`, `utils`) in editable mode. The guardrail validators are all LLM-based — no model download step.

### Configuration

Create a `.env` file in the project root:

```dotenv
OPENAI_API_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
ADMIN_API_KEY=...
```

Runtime behaviour (models, chunking, retrieval `k`, contextual compression, prompt label) is controlled by [`params.yaml`](params.yaml) and validated on load.

Register the system prompt in Langfuse before first run:

```bash
uv run python src/app/update_system_prompt.py
```

### Running the service

```bash
# API
uv run uvicorn api.main:app --reload --app-dir src

# Frontend (expects the API on http://127.0.0.1:8000)
uv run streamlit run src/frontend/app.py
```

### Ingesting documents

Place transcript files in `data/raw/`, then trigger a sync (cleans, chunks, embeds, and upserts — only changed chunks are re-embedded):

```bash
uv run python tests/demo_transcript_sync.py   # uses ADMIN_API_KEY from .env
```

## API

| Method | Endpoint | Auth | Description |
| --- | --- | --- | --- |
| `POST` | `/chat` | — | Streams a grounded answer for `{"query": "...", "session_id": "..."}` (`session_id` optional — groups the request's Langfuse trace by chat session) |
| `GET` | `/chat/files` | — | Lists indexed source documents |
| `GET` | `/health` | — | Liveness |
| `GET` | `/health/dependencies` | — | Live LLM + retriever probes |
| `GET` | `/internal/vector-store/chunks/count` | `X-Admin-Key` | Chunk count for the collection |
| `POST` | `/internal/vector-store/transcripts/sync` | `X-Admin-Key` | Clean, chunk, embed and upsert `data/raw/` |
| `POST` | `/internal/debug/rag` | `X-Admin-Key` | Runs the plain (non-guardrailed) graph and returns its full state |
| `POST` | `/internal/debug/rag_with_guardrails` | `X-Admin-Key` | Runs the guardrailed graph and returns its full state, incl. guardrail status |

## Observability

Every `/chat` request is fully traced in Langfuse, not just logged:

- **Root span per request** — [`chat.py`](src/api/routers/chat.py) opens a `chat-response` span around the graph call with the raw query as input and the final answer as output, and tags it with the request's `session_id` via `propagate_attributes` so a multi-turn conversation groups into one Langfuse session.
- **Auto-traced graph internals** — `graph.invoke(...)` runs with a `langfuse.langchain.CallbackHandler`, so every LangGraph node and LangChain call (retriever, prompt, LLM) shows up as a nested span under the root, with token usage and latency per step.
- **Guardrail decisions as first-class spans** — [`rag_workflow_with_guardrails.py`](src/app/rag_workflow_with_guardrails.py) wraps the input, retrieval, and output guardrail nodes in their own `guardrail`-typed observations, recording the `ok` / `refrain` / `exception` outcome and message — so trip rates per guardrail are queryable in the Langfuse UI, not just `grep`-able in logs.
- **Prompt-version attribution** — the augmentation node attaches `metadata={"langfuse_prompt": system_prompt}` to the prompt template, linking each generation to the exact Langfuse prompt version that produced it, which powers per-prompt-version metrics.
- **Environment separation** — the production compose file ([`deploy/codedeploy/docker-compose.yml`](deploy/codedeploy/docker-compose.yml)) sets `LANGFUSE_TRACING_ENVIRONMENT=production`, so EC2 traces are visually and queryably distinct from local/dev traces in Langfuse.

## LLMOps workflow

### Evaluation

```bash
uv run python src/experiment/execute_experiment_pipeline.py
```

Within a single MLflow run this:

1. logs the flattened `params.yaml`;
2. builds the evaluation set by running every golden question through the RAG graph;
3. scores it with the DeepEval suite (LLM-as-judge) and writes Markdown + JSON reports to `reports/`;
4. logs metrics, datasets, the active system prompt, and source files as artifacts;
5. tags the run `stage=challenger` (clearing the tag off any earlier challenger first);
6. appends the run to `historical_runs.json` (local history; no longer used by the gates).

Regenerate the golden set from the corpus:

```bash
uv run python src/data/generate_goldens.py
```

### Promotion & regression gates

Metric thresholds in [`thresholds.json`](thresholds.json) capture two kinds of variance:

- **noise** — run-to-run standard deviation for an unchanged configuration;
- **historical** — standard deviation across accepted runs.

The gates compare the challenger (the MLflow run tagged `stage=challenger`, set by the
pipeline) against the current champion (the run tagged `stage=champion`):

```bash
uv run pytest tests/test_regression.py    # fails on a drop beyond champion − 2·(historical + noise)
uv run pytest tests/test_promotion.py     # fails unless ≥ 5/7 metrics ≥ champion and none regress beyond 2·noise
```

If both pass, promote the challenger; if either fails, reject it:

```bash
uv run python src/experiment/promote_challenger.py   # champion → archived, challenger → champion
uv run python src/experiment/reject_challenger.py    # clears the challenger tag, champion untouched
```

The `stage` tag scheme has one invariant: exactly one `champion`, at most one `challenger`.
Bootstrap it once by hand in the MLflow UI — tag the current best run `stage=champion`.

Recompute thresholds after accumulating new runs:

```bash
uv run python utils/compute_noise_thresholds.py
uv run python utils/compute_historical_thresholds.py
```

### Prompt management

The system prompt is stored in Langfuse and fetched at runtime by the label set in `params.yaml` (`prompt_label`). `src/app/system_prompt_versioning.py` moves labels between versions to promote a prompt without a code change.

## Deployment

### Local, with Docker Compose

```bash
export OPENAI_API_KEY=...   # build-time secret baked into the API image's vector store
docker compose up --build
```

This builds the API and frontend images ([`docker/Dockerfile.api`](docker/Dockerfile.api), [`docker/Dockerfile.frontend`](docker/Dockerfile.frontend)) and serves them on `:8000` and `:8501`. The API image embeds `data/raw/` into a Chroma store at build time via [`docker/ingest_transcripts.py`](docker/ingest_transcripts.py), so it is self-contained and stateless at runtime; the frontend persists chat/session history to a named volume.

### Production, on AWS

Every push to `params.yaml` on `main` flows through CI into an automated CD rollout:

```
CI Pipeline (promote_challenger.py) ──▶ CD Pipeline
                                          │
                                          ├─ build & push API + frontend images ──▶ ECR
                                          └─ zip appspec.yml + compose file + hooks
                                             ──▶ S3 ──▶ CodeDeploy ──▶ EC2 fleet
```

- **[`cd.yaml`](.github/workflows/cd.yaml)** triggers on a successful `CI Pipeline` run (i.e. only after the champion has actually been promoted), builds both images, pushes them to ECR tagged with the commit SHA and `latest`, then assembles and uploads a CodeDeploy revision bundle and calls `aws deploy create-deployment`.
- **[`deploy/codedeploy/`](deploy/codedeploy/)** is that bundle: `appspec.yml` drives an in-place deployment through `ApplicationStop → BeforeInstall → ApplicationStart → ValidateService` hooks ([`scripts/`](deploy/codedeploy/scripts)), which pull the new images, write a runtime `.env` from the `api-keys` Secrets Manager secret, and restart the stack via the deployment-only `docker-compose.yml`.
- Instances run behind an **Auto Scaling Group + Application Load Balancer** (`campusx-rag-asg` / `campusx-rag-alb`), so CodeDeploy rolls new scale-out instances forward automatically. One-time console setup for this — S3 bucket, IAM roles, ASG/ALB, CodeDeploy app/group — is documented in [`codedeploy_steps.md`](codedeploy_steps.md); [`deploy/ec2/manual-steps.md`](deploy/ec2/manual-steps.md) covers the earlier single-instance manual deploy that CodeDeploy now automates.

Required repo secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (deploy), plus `OPENAI_API_KEY` (build-time secret for the API image's vector store, shared with CI).

## Testing

```bash
uv run pytest tests/
```

`test_regression.py` and `test_promotion.py` require network access to the MLflow tracking server, a run tagged `stage=champion`, and a run tagged `stage=challenger`. The `tests/demo_*.py` scripts are manual API exercises (a live server is required) and are excluded from collection by name.

## License

Not yet specified.
