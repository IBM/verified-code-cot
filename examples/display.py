"""Terminal display helpers for the CodeCoT pipeline."""
import json
import textwrap
import sys
from pathlib import Path

_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_GREEN  = "\033[32m"
_CYAN   = "\033[36m"
_YELLOW = "\033[33m"
_BLUE   = "\033[34m"
_RESET  = "\033[0m"

W = 88  # content width
SEP  = "  " + "-" * 56   # simple dash separator — renders reliably in all terminals
SEP2 = "  " + "=" * 56


def c(text: str, *codes: str) -> str:
    return "".join(codes) + str(text) + _RESET


def trunc(text: str, n: int = W) -> str:
    text = text.strip().replace("\n", " ")
    return text[:n] + "…" if len(text) > n else text


def wrap(text: str, width: int = W - 4, indent: str = "    ") -> list[str]:
    text = text.strip()
    return textwrap.wrap(text, width=width, subsequent_indent=indent) or [""]


def header(stage: int, total: int, label: str) -> None:
    print(f"\n{c(f'  [{stage}/{total}]', _BOLD, _CYAN)}  {c(label, _BOLD, _CYAN)}")
    print(c("  " + "-" * 52, _DIM))


def ok(msg: str) -> None:
    print(f"  {c('✓', _GREEN)}  {msg}")


def warn(msg: str) -> None:
    print(f"  {c('!', _YELLOW)}  {msg}", file=sys.stderr)


def pipeline_header(backend: str, model_id: str, output_dir: Path, log_path: Path) -> None:
    print()
    print(c("  CodeCoT Pipeline", _BOLD, _CYAN))
    print(c(f"  Model:  {backend} / {model_id}", _DIM))
    print(c(f"  Output: {output_dir}", _DIM))
    print(c(f"  Log:    {log_path}", _DIM))


def pipeline_done(output_dir: Path) -> None:
    print()
    print(c(f"  Done  --  CoT data in {output_dir}/raw/  |  summary: {output_dir}/demo_report.md", _BOLD, _CYAN))
    print()


def _assert_from_test(test: str) -> str:
    """Extract the assert line from a test function, falling back to def line."""
    for line in test.splitlines():
        stripped = line.strip()
        if stripped.startswith("assert "):
            return stripped
    # fallback: first non-def, non-comment, non-empty line
    for line in test.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("def ") and not stripped.startswith("#"):
            return stripped
    return test.strip().splitlines()[0] if test.strip() else ""


def sample_snippet(raw_dir: Path) -> None:
    """Print a clean vertical sample — verified pair then CoT."""
    results_path = raw_dir / "dual_exec_results.jsonl"
    if not results_path.exists():
        return
    entries = [json.loads(l) for l in results_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not entries:
        return
    e           = entries[0]
    concept     = e.get("concept", "") or e.get("concept_description", "")
    instruction = e.get("instruction", "")
    code        = (e.get("code_snippet") or [""])[0]
    tests       = e.get("passing_test_cases", [])
    score       = e.get("score", "—")

    fwd_q = fwd_r = bwd_q = bwd_r = ""
    conv_path = raw_dir / "conversations.jsonl"
    if conv_path.exists():
        for line in conv_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            msgs = json.loads(line).get("messages", [])
            if len(msgs) >= 4:
                fwd_q = msgs[0].get("content", "").strip()
                fwd_r = msgs[1].get("content", "").strip()
                bwd_q = msgs[2].get("content", "").strip()
                bwd_r = msgs[3].get("content", "").strip()
            break

    print()
    print(c(SEP2, _BLUE))
    print(c("  Sample verified pair", _BOLD, _CYAN))
    print(c(SEP, _DIM))

    # Concept + instruction
    if concept:
        print(f"\n  {c('Concept:', _DIM)}")
        for line in wrap(concept):
            print(c(f"    {line}", _CYAN))
    print(f"\n  {c('Instruction:', _DIM)}")
    for line in wrap(instruction):
        print(f"    {line}")

    # Code
    print(f"\n  {c('Code:', _DIM)}")
    code_lines = code.strip().splitlines()
    for line in code_lines[:10]:
        print(c(f"    {line}", _YELLOW))
    if len(code_lines) > 10:
        print(c(f"    ... ({len(code_lines) - 10} more lines)", _DIM))

    # Passing tests — show the assert, not the def
    print(f"\n  {c('Passing tests:', _DIM)}")
    for t in tests[:4]:
        assert_line = _assert_from_test(t)
        print(c(f"    {trunc(assert_line, W - 4)}", _GREEN))
    if len(tests) > 4:
        print(c(f"    ... ({len(tests) - 4} more)", _DIM))
    score_str = f"{score:.2f}" if isinstance(score, float) else str(score)
    print(c(f"\n  Score: {score_str}", _DIM))

    if not fwd_q:
        print(c("\n  (CoT not yet generated -- run --stage cot to populate)", _DIM))
        print(c(SEP2, _BLUE))
        return

    # CoT section
    print(c(f"\n{SEP}", _DIM))
    print(c("  Verified CoT", _BOLD, _CYAN))
    print(c(SEP, _DIM))

    print(f"\n  {c('Forward question:', _DIM)}")
    # fwd_q contains "code summary\nHere's the code:...\nActual question" -- extract last sentence
    fwd_q_clean = fwd_q.split("```")[-1].strip() if "```" in fwd_q else fwd_q
    for line in wrap(fwd_q_clean.splitlines()[-1] if fwd_q_clean.splitlines() else fwd_q_clean):
        print(c(f"    {line}", _CYAN))

    print(f"\n  {c('Forward reasoning:', _DIM)}")
    fwd_r_clean = fwd_r.split("<response>")[0].strip()
    for line in fwd_r_clean.splitlines()[:6]:
        print(f"    {line.rstrip()}")
    total = len(fwd_r_clean.splitlines())
    if total > 6:
        print(c(f"    ... ({total - 6} more lines in report)", _DIM))

    print(f"\n  {c('Backward question:', _DIM)}")
    for line in wrap(bwd_q.splitlines()[-1] if bwd_q.splitlines() else bwd_q):
        print(c(f"    {line}", _CYAN))

    print(f"\n  {c('Backward reasoning:', _DIM)}")
    bwd_r_clean = bwd_r.split("<response>")[0].strip()
    for line in bwd_r_clean.splitlines()[:6]:
        print(f"    {line.rstrip()}")
    total = len(bwd_r_clean.splitlines())
    if total > 6:
        print(c(f"    ... ({total - 6} more lines in report)", _DIM))

    print(c(f"\n{SEP2}", _BLUE))
