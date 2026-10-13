#!/bin/bash
# Recorded demo: 3 questions against the sample NDA, output saved to demo-output.txt
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="demo-output.txt"
: > "$OUT"
{
  echo "=== LocalLens demo — $(date -u '+%Y-%m-%d %H:%M UTC') ==="
  echo "Model: Qwen2.5-0.5B-Instruct (Q4_K_M, open weights, Apache 2.0), 100% local"
  echo "Document: sample_docs/freelance-nda.txt"
  echo ""
  echo "--- Q1: extraction ---"
  python3 -m locallens ask -f sample_docs/freelance-nda.txt \
    -q "What is the notice period for terminating the agreement?"
  echo ""
  echo "--- Q2: extraction ---"
  python3 -m locallens ask -f sample_docs/freelance-nda.txt \
    -q "How long do the confidentiality obligations survive after the agreement ends?"
  echo ""
  echo "--- Q3: honest abstention (answer is not in the document) ---"
  python3 -m locallens ask -f sample_docs/freelance-nda.txt \
    -q "What is the penalty for late payment of invoices?"
} 2>/dev/null | tee "$OUT"
echo ""
echo "Demo saved to $OUT"
