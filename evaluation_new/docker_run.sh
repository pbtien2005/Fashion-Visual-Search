#!/usr/bin/env bash
# Run a command in an existing project image (no build). Files are written as your user.
#   evaluation_new/docker_run.sh detect python -m evaluation_new.detect_cache
#   evaluation_new/docker_run.sh api    python -m evaluation_new.fetch_catalog_images
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-}" in
  detect) IMAGE=fashion-visual-search-object-detection:latest ;;
  api)    IMAGE=fashion-visual-search-api:latest ;;
  *) sed -n '2,4p' "$0"; exit 1 ;;
esac
shift

# .env is not passed with --env-file (it keeps quotes literally); scripts read it with python-dotenv.
# BENCH_NETWORK=fvs-bench joins the network of the detection service started for the colour cache.
docker run --rm ${BENCH_NETWORK:+--network "$BENCH_NETWORK"} \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e USER=bench -e LOGNAME=bench -e YOLO_CONFIG_DIR=/tmp -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 \
  -v "$REPO:/workspace" -w /workspace \
  --entrypoint "$1" "$IMAGE" "${@:2}"
