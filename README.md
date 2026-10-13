# LocalLens — private, offline document Q&A

Ask questions about a document. Everything runs on your own machine:
**no accounts, no API keys, no subscriptions, no data ever leaves the device.**

Built for the DEV Hacktoberfest 2026 Week 2 "Side Hustle" challenge:
something built with open-weight models that someone would actually pay for.

## Why this exists

Freelancers, accountants, HR teams, and small clinics all handle documents
they legally or ethically **can't paste into ChatGPT** — client contracts,
payroll files, patient forms. Their options today are: read 40 pages
manually, or violate confidentiality. LocalLens is the third option: point
it at a `.txt` file, ask in plain English, get an answer in seconds, with
the model telling you which sentences it used so you can verify.

The side-hustle angle: a one-time-purchase desktop tool for professionals
who handle sensitive documents. Privacy isn't a feature here — it's the
entire product.

## How it works

1. **Retrieve** (`locallens/retriever.py`, stdlib only): the document is
   split into sentences, ranked by keyword overlap with the question
   (stopword-filtered, naive stemming, tight-span tie-break so the most
   *focused* sentence wins).
2. **Generate** (`locallens/llm.py`): the top sentences go into a prompt
   for a local open-weight model via the llama.cpp binary. The model is
   instructed to answer in one sentence using only the provided text —
   and to say `Not found in the document.` when the answer isn't there,
   instead of hallucinating.

## Quickstart

Requirements: Python 3.8+ (standard library only — no pip packages needed),
about 500 MB disk for the model + binary.

```bash
# 1. One-time downloads (free, no account)
./scripts/download-llamacpp.sh   # llama.cpp inference binary (MIT)
./scripts/download-model.sh      # Qwen2.5-0.5B-Instruct Q4_K_M GGUF (Apache 2.0)

# 2. Ask a question
python3 -m locallens ask -f sample_docs/freelance-nda.txt \
  -q "What is the notice period for terminating the agreement?"

# 3. See exactly which sentences the model used
python3 -m locallens ask -f sample_docs/freelance-nda.txt \
  -q "How long do confidentiality obligations survive?" --show-context
```

Example output:

```
Q: What is the notice period for terminating the agreement?

The notice period for terminating the agreement is thirty (30) days.

(answered from sentence(s) [14, 17] of 32 — run with --show-context to see them)
```

A recorded demo (real output, nothing staged) is in
[`demo-output.txt`](demo-output.txt); regenerate it with `./scripts/demo.sh`.

## Tests

```bash
python3 -m unittest discover -s tests   # 12 tests, stdlib only
```

## Honest limitations

- The bundled model is **0.5B parameters** — deliberately tiny so it runs
  on any laptop CPU. It's good at lookup questions ("what is the notice
  period?") and honest abstention, but convoluted phrasing can stump it,
  and anything needing synthesis across distant clauses wants a bigger
  model. Swap any GGUF in via `--model` / `$LOCALLENS_MODEL`.
- Retrieval is keyword-based (no embeddings) — fast and dependency-free,
  but it won't catch pure paraphrases.
- `.txt` input for now; PDF/DOCX parsing is on the roadmap, not in v0.1.
- This is a challenge entry and a working prototype, not a hardened
  product. Answers should be verified against the source sentences shown
  with `--show-context` before acting on them.

## Licenses

- This project's code: MIT (see `LICENSE`)
- Inference: llama.cpp (MIT)
- Default model: Qwen2.5-0.5B-Instruct (Apache 2.0)
