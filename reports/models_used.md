# Models Used

| file_name | model_name | usecase |
|---|---|---|
| `src/app/rag_workflow.py` | `gpt-5-mini` | RAG generation node — answers the user query using retrieved context |
| `src/app/rag_workflow.py` | `text-embedding-3-small` | Embeds transcript chunks (on upsert) and queries (on retrieval) for the Chroma vector store |
| `src/data/generate_goldens.py` | `gpt-5.4-mini` | `Synthesizer` core model — generates and evolves synthetic golden Q&A pairs from transcripts |
| `src/data/generate_goldens.py` | `gpt-5-mini` | `FiltrationConfig` critic — scores synthetic input quality, filters low-quality goldens |
| `src/data/generate_goldens.py` | `gpt-5.4-mini` | `ContextConstructionConfig` critic — scores/filters context quality when building golden contexts |
| `src/data/generate_eval_dataset.py` | `gpt-5-mini` (indirect, via `rag_workflow.graph`) | Runs each golden's query through the RAG graph to produce `actual_output` for the eval dataset |
| `src/evals/application_evals/evaluate_rag_app.py` | `gpt-5.4-mini` | Judge model for the full metric suite: contextual recall/precision/relevancy, answer relevancy, faithfulness, and custom GEval metrics "answer correctness" & "simple explanation" |
| `src/evals/individual_evals/answer_correctness_metric.py` | `gpt-5.4` (default) | Standalone sanity check of the custom "answer correctness" GEval metric |
| `src/evals/individual_evals/simple_explanation_metric.py` | `gpt-5.4` (default) | Standalone sanity check of the custom "simple explanation" GEval metric |
| `src/evals/individual_evals/test_answer_relevancy_metric.py` | `gpt-5.4` (default) | Standalone sanity check of `AnswerRelevancyMetric` |
| `src/evals/individual_evals/test_faithfulness_metric.py` | `gpt-5.4` (default) | Standalone sanity check of `FaithfulnessMetric` |
| `src/evals/individual_evals/test_precision_metric.py` | `gpt-5.4` (default) | Standalone sanity check of `ContextualPrecisionMetric` |
| `src/evals/individual_evals/test_recall_metric.py` | `gpt-5.4` (default) | Standalone sanity check of `ContextualRecallMetric` |
| `src/evals/individual_evals/test_relevancy_metric.py` | `gpt-5.4` (default) | Standalone sanity check of `ContextualRelevancyMetric` |

**Notes:**
- `gpt-5-mini` is the app's actual production/generation model.
- `gpt-5.4-mini` / `gpt-5.4` is used everywhere as the evaluation judge/critic model, never as the app-facing model.
- `text-embedding-3-small` is the only non-chat model, used purely for vector retrieval, not judging or generation.
