#!/usr/bin/env bash
# ApplicationStop -- CodeDeploy runs this from the PREVIOUS revision, so it does
# nothing on the first deploy. Stop the old stack and delete its images so the
# new revision pulls fresh. Runs as root.
set -euo pipefail

APP_DIR=/opt/campusx-rag

[ -f "$APP_DIR/docker-compose.yml" ] || exit 0

cd "$APP_DIR"
docker compose down --rmi all
