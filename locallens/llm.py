"""Thin wrapper around the llama.cpp `llama-cli` binary.

The model (an open-weight GGUF) and the binary both live on the user's
machine. This module builds the prompt, runs one single-turn generation,
and extracts just the model's reply from the CLI's output.
"""

import os
import shutil
import subprocess

SYSTEM_PROMPT = (
    "Answer the question using only the DOCUMENT text, in one sentence. "
    "If the DOCUMENT does not contain the answer, reply with exactly: "
    "Not found in the document."
)


def default_cli_path():
    """Locate the llama-cli binary: explicit env var, then the repo's
    vendor dir, then PATH."""
    env = os.environ.get("LOCALLENS_LLAMACPP")
    if env and os.path.isfile(env):
        return env
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vendored = os.path.join(here, "vendor", "llamacpp", "llama-cli")
    if os.path.isfile(vendored):
        return vendored
    found = shutil.which("llama-cli")
    if found:
        return found
    raise FileNotFoundError(
        "llama-cli not found. Run scripts/download-llamacpp.sh, set "
        "LOCALLENS_LLAMACPP, or put llama-cli on PATH."
    )


def default_model_path():
    env = os.environ.get("LOCALLENS_MODEL")
    if env and os.path.isfile(env):
        return env
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local = os.path.join(here, "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf")
    if os.path.isfile(local):
        return local
    raise FileNotFoundError(
        "Model not found. Run scripts/download-model.sh or set LOCALLENS_MODEL."
    )


def build_user_message(excerpts, question):
    return (
        "DOCUMENT:\n"
        f"{excerpts}\n\n"
        f"QUESTION: {question}\n"
        "ANSWER:"
    )


def extract_reply(cli_output):
    """Pull the generated text out of llama-cli's stdout.

    The CLI echoes the prompt as '> ' + prompt text (only the first line
    gets the '> ' prefix), so the reply begins on the line after the
    prompt's final line, which is always exactly 'ANSWER:'. For long
    prompts llama.cpp truncates the displayed echo with '... (truncated)',
    in which case the reply starts on the line after that marker.
    Everything up to the '[ Prompt:' timings line is the reply.
    """
    lines = cli_output.splitlines()
    anchor = None
    for i, line in enumerate(lines):
        if line.strip() == "ANSWER:":
            anchor = i
    if anchor is None:
        for i, line in enumerate(lines):
            if "(truncated)" in line:
                anchor = i
    if anchor is None:
        # Fallback: after the last '> ' line.
        for i, line in enumerate(lines):
            if line.startswith("> "):
                anchor = i
    if anchor is None:
        anchor = -1
    collected = []
    for line in lines[anchor + 1:]:
        if line.startswith("[ Prompt:"):
            break
        collected.append(line)
    return "\n".join(collected).strip()


def ask_llm(excerpts, question, model_path=None, cli_path=None,
            max_tokens=160, threads=2, ctx_size=2048, temperature=0.0,
            timeout=300):
    """Run one Q&A turn against the local model. Returns the reply text."""
    model_path = model_path or default_model_path()
    cli_path = cli_path or default_cli_path()
    cmd = [
        cli_path,
        "-m", model_path,
        "-sys", SYSTEM_PROMPT,
        "--prompt", build_user_message(excerpts, question),
        "-n", str(max_tokens),
        "-t", str(threads),
        "-c", str(ctx_size),
        "--temp", str(temperature),
        "--no-display-prompt",
        "-st",  # single turn, then exit (no interactive mode)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-cli failed (exit {proc.returncode}): {proc.stderr[-2000:]}"
        )
    reply = extract_reply(proc.stdout)
    if not reply:
        raise RuntimeError("llama-cli produced no reply text.")
    return reply
