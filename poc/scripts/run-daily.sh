#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
poc_dir="$(cd -- "$script_dir/.." && pwd)"
mkdir -p "$poc_dir/logs"
exec "$poc_dir/.venv/bin/asset-poc" daily >>"$poc_dir/logs/daily.log" 2>&1
