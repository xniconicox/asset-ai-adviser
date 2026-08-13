#!/usr/bin/env bash
set -euo pipefail

# LLM is intentionally not used in the scheduled daily job.
# This job only runs the local non-LLM pipeline: data refresh, ranks, quality checks,
# snapshot publication, and PDF report generation.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
poc_dir="$(cd -- "$script_dir/.." && pwd)"
mkdir -p "$poc_dir/logs"
{
    "$poc_dir/.venv/bin/asset-poc" daily
    "$poc_dir/.venv/bin/asset-poc" daily-report
} >>"$poc_dir/logs/daily.log" 2>&1
