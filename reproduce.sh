#!/usr/bin/env bash
# reproduce.sh — verify every manuscript claim against the committed result tables.
#
# Tier 1 (this script): claim verification. Reads only the imagery-free CSV/parquet
# tables under code/. No dataset, no GPU, no environment variables. This is the
# reproduction path a reader can run directly.
#
# Tier 2 (documented, not run here): regenerating those tables from raw imagery.
# Needs the two datasets (access-restricted; see below) and a GPU. Per-dataset
# run order is in code/aihub_subtropical/README.md and code/dacon_cropdisease/README.md.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python}"
echo "Verifying manuscript claims against committed result tables..."
"$PY" test_assertions.py

cat <<'NOTE'

--- Full regeneration (Tier 2) ---
Regenerating the result tables from raw imagery requires:
  * the AI Hub dataset (aihub.or.kr; access-restricted, retrieve by title)
  * the CDD dataset (dacon.io; redistribution-prohibited)
  * a CUDA GPU for the fusion experiment
Point $TROPICAL_SCRATCH at a working directory and follow the run-order tables in
  code/aihub_subtropical/README.md   and   code/dacon_cropdisease/README.md
NOTE
