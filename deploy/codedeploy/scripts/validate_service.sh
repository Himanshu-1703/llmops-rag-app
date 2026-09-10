#!/usr/bin/env bash
# ValidateService -- wait out the cold start, then poll /health/dependencies a
# few times. If it never reports healthy, exit non-zero so CodeDeploy rolls back.
# Runs as root.
set -euo pipefail

API=http://localhost:8000
COLD_START_WAIT=120
RETRIES=3
RETRY_GAP=60

sleep "$COLD_START_WAIT"   # give the API time to load the vector store and warm the LLM

for ((attempt = 1; attempt <= RETRIES; attempt++)); do
  response=$(curl -fsS "$API/health/dependencies") || response=""
  echo "Attempt $attempt/$RETRIES: $response"

  if [ "$(echo "$response" | jq -r '.overall_health' 2>/dev/null)" = "healthy" ]; then
    echo "Healthy"
    exit 0
  fi

  if [ "$attempt" -lt "$RETRIES" ]; then
    sleep "$RETRY_GAP"
  fi
done

echo "Health check failed after $RETRIES attempts"
exit 1
