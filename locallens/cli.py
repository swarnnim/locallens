"""Command-line interface for LocalLens."""

import argparse
import os
import sys

from . import __version__
from .retriever import rank, split_sentences, build_excerpts
from .llm import ask_llm


def cmd_ask(args):
    with open(args.file, "r", encoding="utf-8") as fh:
        text = fh.read()
    sentences = split_sentences(text)
    if not sentences:
        print("error: the document has no readable sentences.", file=sys.stderr)
        return 1
    ranked = rank(sentences, args.question, top_k=args.top_k)
    excerpts = build_excerpts(ranked)

    if args.show_context:
        print("--- excerpts sent to the model ---")
        print(excerpts)
        print("---------------------------------\n")

    reply = ask_llm(
        excerpts,
        args.question,
        model_path=args.model,
        cli_path=args.llamacpp,
        max_tokens=args.max_tokens,
        threads=args.threads,
    )
    print(f"Q: {args.question}\n")
    print(reply)
    if not args.show_context:
        print(f"\n(answered from sentence(s) {[i for i, _ in ranked]} of "
              f"{len(sentences)} — run with --show-context to see them)")
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        prog="locallens",
        description="Private, offline document Q&A. Nothing leaves your machine.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Ask a question about a document.")
    ask.add_argument("-f", "--file", required=True, help="Path to a .txt document.")
    ask.add_argument("-q", "--question", required=True, help="Question to ask.")
    ask.add_argument("--top-k", type=int, default=2,
                     help="How many sentences to feed the model (default: 2; "
                          "small local models stay most accurate with a short, "
                          "focused context).")
    ask.add_argument("--model", default=None,
                     help="Path to GGUF model (default: ./models/<bundled> or $LOCALLENS_MODEL).")
    ask.add_argument("--llamacpp", default=None,
                     help="Path to llama-cli binary (default: ./vendor/llamacpp/llama-cli or $LOCALLENS_LLAMACPP).")
    ask.add_argument("--max-tokens", type=int, default=160)
    ask.add_argument("--threads", type=int, default=2)
    ask.add_argument("--show-context", action="store_true",
                     help="Print the excerpts sent to the model.")
    ask.set_defaults(func=cmd_ask)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
