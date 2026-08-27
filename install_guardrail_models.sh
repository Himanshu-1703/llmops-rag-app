#!/usr/bin/env bash
set -e

# guardrails_pii's post_install instantiates Presidio's AnalyzerEngine, which
# expects the spaCy model already present on disk (spacy doesn't auto-download it).
uv run -m spacy download en_core_web_lg

uv run -m guardrails_ai.detect_jailbreak.post_install
uv run -m guardrails_ai.guardrails_pii.post_install
