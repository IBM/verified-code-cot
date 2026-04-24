"""End-to-end CodeCoT pipeline runner."""
import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

try:
    import yaml
    def _load_config(path: str) -> dict:
        with open(path) as f:
            return yaml.safe_load(f) or {}
except ImportError:
    def _load_config(path: str) -> dict:
        raise SystemExit("pyyaml is required: pip install pyyaml")

import subprocess
from dataops_code_cot.components.execution.dual_exec import dual_exec, save_solution_test_cases
from dataops_code_cot.components.generation.concepts_generation import generate_concepts_concurrent
from dataops_code_cot.components.generation.solutions_and_testcases_generation import (
    get_instruction_code_test_pairs,
)
from dataops_code_cot.components.generation.split_tests import split_test_cases
from dataops_code_cot.utils.model_client import ModelClientFactory
from dataops_code_cot.scripts.get_traces import generate_traces
from dataops_code_cot.scripts.clean_trace import process_folder as clean_traces
from dataops_code_cot.scripts.generate_cots_batched import main as _cot_main

import display as _d



# ── Stage A: concept extraction ───────────────────────────────────────────────
def _stage_concepts(raw_dir: Path, input_file: str, client, a_cfg: dict = None) -> None:
    a_cfg = a_cfg or {}
    text = Path(input_file).read_text(encoding="utf-8")
    concepts = generate_concepts_concurrent({"0": {"0": text}}, client)
    (raw_dir / "concepts.json").write_text(json.dumps(concepts, indent=2), encoding="utf-8")
    cap = a_cfg.get("max_concepts", "all")
    _d.ok(f"{len(concepts)} concepts extracted  {_d.c(f'(stage_a.max_concepts={cap})', _d._DIM)}")
    for c in concepts[:4]:
        print(f"     {_d.c('-', _d._DIM)} {_d.c(_d.trunc(c.get('concept',''), 60), _d._DIM)}")


# ── Stage A: code + test synthesis ────────────────────────────────────────────
def _stage_synthesis(raw_dir: Path, client, model_id: str, a_cfg: dict) -> None:
    concepts = json.loads((raw_dir / "concepts.json").read_text(encoding="utf-8"))
    solutions_nested, tests_nested = get_instruction_code_test_pairs(
        concepts, client, model_id,
        difficulty_levels=a_cfg.get("difficulty_levels", ["medium", "hard"]),
        max_concepts=a_cfg.get("max_concepts", None),
        num_samples=a_cfg.get("num_samples", 2),
        max_output_tokens=a_cfg.get("max_output_tokens", 2048),
    )
    (raw_dir / "out_solutions_raw.json").write_text(
        json.dumps(solutions_nested, indent=2), encoding="utf-8"
    )
    solutions = [item for sub in solutions_nested for item in sub]
    tests     = [item for sub in tests_nested     for item in sub]
    solutions, tests = split_test_cases(solutions, tests)
    (raw_dir / "out_solutions.json").write_text(json.dumps(solutions, indent=2), encoding="utf-8")
    (raw_dir / "out_test_cases.json").write_text(json.dumps(tests, indent=2),    encoding="utf-8")
    with open(raw_dir / "out_solutions.jsonl", "w", encoding="utf-8") as f:
        for s in solutions:
            f.write(json.dumps(s) + "\n")
    diff   = a_cfg.get("difficulty_levels", ["medium", "hard"])
    nsamp  = a_cfg.get("num_samples", 2)
    _d.ok(f"{len(solutions)} instructions generated, each with {nsamp} code solution(s) and test set(s)  "
          f"{_d.c(f'(difficulty={diff})', _d._DIM)}")


# ── Stage B: code quality filter ──────────────────────────────────────────────
def _suppress_fd_stdout():
    """Context manager that silences stdout at the OS fd level (covers forked processes)."""
    import os, contextlib
    @contextlib.contextmanager
    def _ctx():
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved   = os.dup(1)
        os.dup2(devnull, 1)
        try:
            yield
        finally:
            os.dup2(saved, 1)
            os.close(saved)
            os.close(devnull)
    return _ctx()


def _stage_filter(raw_dir: Path, b_cfg: dict) -> None:
    solutions = json.loads((raw_dir / "out_solutions.json").read_text(encoding="utf-8"))
    tests     = json.loads((raw_dir / "out_test_cases.json").read_text(encoding="utf-8"))
    with _suppress_fd_stdout():
        solution_test_cases, handled_solutions = dual_exec(
            solutions, tests,
            timeout=b_cfg.get("timeout", 0.1),
            test_case_limit=b_cfg.get("test_case_limit", 5),
        )
    save_solution_test_cases(
        solution_test_cases, handled_solutions,
        str(raw_dir / "out_solutions.jsonl"), str(raw_dir), "dual_exec_results.jsonl",
    )
    n       = sum(len(v) for v in solution_test_cases.values())
    limit   = b_cfg.get("test_case_limit", 5)
    timeout = b_cfg.get("timeout", 0.1)
    _d.ok(f"{n} verified (code, test) pairs after execution + consensus filtering  "
          f"{_d.c(f'(up to {limit} test cases per solution, {timeout}s timeout)', _d._DIM)}")


# ── Stage C: trace annotation + CoT generation + symbolic verification ────────
def _stage_cot(raw_dir: Path, backend: str, model_id: str, config_path: str, client,
               max_pairs: int = None, trace_tests_per_pair: int = None) -> None:
    import shutil as _shutil
    traces_dir  = raw_dir / "traces"
    cleaned_dir = raw_dir / "traces_cleaned"
    # Clear previous trace runs to avoid accumulation
    if traces_dir.exists():  _shutil.rmtree(traces_dir)
    if cleaned_dir.exists(): _shutil.rmtree(cleaned_dir)
    traces_dir.mkdir(); cleaned_dir.mkdir()
    conversations = raw_dir / "conversations.jsonl"
    filtered_dir  = raw_dir / "filtered";     filtered_dir.mkdir(exist_ok=True)

    # Route all internal library logs to file only
    for _mod in ("dataops_code_cot", "pandarallel", "filter_qa_based_on_score"):
        logging.getLogger(_mod).setLevel(logging.CRITICAL)

    # Guard: skip if no verified pairs exist
    dual_exec_path = raw_dir / "dual_exec_results.jsonl"
    if not dual_exec_path.exists() or not dual_exec_path.stat().st_size:
        _d.warn("No verified pairs found — skipping CoT stage")
        return

    # 1. Annotate with execution traces
    data = [json.loads(l) for l in dual_exec_path
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    if max_pairs is not None:
        data = data[:max_pairs]
    if trace_tests_per_pair is not None:
        for entry in data:
            entry["passing_test_cases"] = entry.get("passing_test_cases", [])[:trace_tests_per_pair]
    with _suppress_fd_stdout():
        generate_traces(data, str(traces_dir / "data/test_code_1"),
                        str(traces_dir / "data/pysnooper_trace_uncleaned_v1"))
    _d.ok(f"Execution traces generated  {_d.c(f'→ {traces_dir}', _d._DIM)}")

    # 2. Clean traces
    uncleaned_dir = traces_dir / "data/pysnooper_trace_uncleaned_v1"
    if not uncleaned_dir.exists():
        _d.warn("No raw traces generated — skipping CoT stage")
        return
    clean_traces(str(uncleaned_dir), str(cleaned_dir))
    n_traces = len(list(cleaned_dir.glob("*.log")))
    n_pairs  = len(data)
    cap_str  = str(max_pairs) if max_pairs else "all"
    test_cap = str(trace_tests_per_pair) if trace_tests_per_pair else "all"
    _d.ok(f"{n_traces} execution traces  {_d.c(f'({n_pairs} pairs × up to {test_cap} tests, stage_c.max_pairs={cap_str})', _d._DIM)}")

    # 3. Generate CoT (forward + backward) grounded in traces
    _argv_backup = sys.argv[:]
    sys.argv = [
        "generate_cots_batched",
        "--exec_results_file", str(raw_dir / "dual_exec_results.jsonl"),
        "--trace_dir",         str(cleaned_dir),
        "--prompts_file",      "src/dataops_code_cot/scripts/prompts.json",
        "--output_file",       str(conversations),
        "--backend",           backend,
        "--model_id",          model_id,
        "--config",            config_path,
    ]
    try:
        _cot_main()
    finally:
        sys.argv = _argv_backup
    n_cot = sum(1 for l in conversations.read_text(encoding="utf-8").splitlines()
                if l.strip()) if conversations.exists() else 0
    _d.ok(f"{n_cot} verified CoT conversations (each has forward + backward reasoning)  {_d.c(f'→ {conversations}', _d._DIM)}")

    # 3b. Sliding window symbolic verification against traces
    from dataops_code_cot.scripts.cot_verifier import verify_conversations
    verified_path = raw_dir / "conversations_verified.jsonl"
    accepted, rejected = verify_conversations(conversations, cleaned_dir, verified_path, client)
    _d.ok(f"Symbolic verification: {accepted} accepted, {rejected} rejected")
    # Use verified conversations for downstream steps if any were accepted
    if accepted > 0:
        conversations = verified_path

    # 4. Annotate best test case by coverage
    from dataops_code_cot.scripts.best_test_case_annotation import process_folder as annotate_best
    annotated_dir = raw_dir / "annotated"; annotated_dir.mkdir(exist_ok=True)
    with _suppress_fd_stdout():
        annotate_best(str(raw_dir), str(annotated_dir))
    _d.ok(f"Test cases annotated by coverage  {_d.c(f'→ {annotated_dir}', _d._DIM)}")

    # 5. Filter to best test case per conversation
    subprocess.run([
        sys.executable, "src/dataops_code_cot/scripts/best_test_case_filter.py",
        str(annotated_dir), str(filtered_dir),
    ], check=True, capture_output=True, cwd=str(Path(__file__).parent.parent))
    _d.ok(f"CoT filtered to best test cases  {_d.c(f'→ {filtered_dir}', _d._DIM)}")


# ── Stage: report ─────────────────────────────────────────────────────────────
def _stage_report(output_dir: Path, raw_dir: Path) -> None:
    # Prefer CoT-filtered output when available; fall back to all verified pairs
    # Prefer CoT conversations (filtered) when available
    conv_filtered = raw_dir / "filtered" / "conversations_annotated_filtered.jsonl"
    if conv_filtered.exists():
        results_path = conv_filtered
        source_label = "CoT-verified"
    else:
        results_path = raw_dir / "dual_exec_results.jsonl"
        source_label = "execution-verified"

    if not results_path.exists():
        _d.warn("No results found — skipping report")
        return
    entries = [json.loads(l) for l in results_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not entries:
        _d.warn("No entries in results file — skipping report")
        return

    e           = entries[0]
    concept     = e.get("concept", "") or e.get("concept_description", "")
    instruction = e.get("instruction", "")
    code        = (e.get("code_snippet") or [""])[0]
    tests       = e.get("passing_test_cases", [])
    score       = e.get("score", "—")

    # Pull CoT from conversations.jsonl if available
    fwd_q = fwd_r = bwd_q = bwd_r = ""
    conv_path = raw_dir / "conversations.jsonl"
    if conv_path.exists():
        for line in conv_path.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            msgs = json.loads(line).get("messages", [])
            if len(msgs) >= 4:
                fwd_q = msgs[0].get("content", "").strip()
                fwd_r = msgs[1].get("content", "").strip().split("<response>")[0].strip()
                bwd_q = msgs[2].get("content", "").strip()
                bwd_r = msgs[3].get("content", "").strip().split("<response>")[0].strip()
            break

    (output_dir / "demo_report.md").write_text(
        _render_markdown(concept, instruction, code, tests, score, fwd_q, fwd_r, bwd_q, bwd_r, len(entries), source_label),
        encoding="utf-8"
    )
    # Point to the final CoT JSONL as the primary output
    final_jsonl = raw_dir / "filtered" / "conversations_annotated_filtered.jsonl"
    if not final_jsonl.exists():
        final_jsonl = raw_dir / "conversations_verified.jsonl"
    if not final_jsonl.exists():
        final_jsonl = raw_dir / "conversations.jsonl"
    _d.ok(f"{len(entries)} {source_label} CoT samples  {_d.c(f'→ {final_jsonl}', _d._DIM)}")
    _d.ok(f"Summary  {_d.c(f'→ {output_dir}/demo_report.md', _d._DIM)}")
    _d.sample_snippet(raw_dir)


# ── HTML / Markdown rendering ─────────────────────────────────────────────────
def _render_markdown(concept, instruction, code, tests, score,
                     fwd_q="", fwd_r="", bwd_q="", bwd_r="",
                     total=0, source_label="") -> str:
    test_block = "\n\n".join(tests[:5])
    score_str  = f"{score:.2f}" if isinstance(score, float) else str(score)
    cot_section = ""
    if fwd_r:
        cot_section = f"""
## Verified CoT  ({source_label}, {total} samples total)

### Forward question
{fwd_q}

### Forward reasoning (trace-grounded)
{fwd_r}

### Backward question
{bwd_q}

### Backward reasoning (trace-grounded)
{bwd_r}
"""
    return f"""# CodeCoT Demo Report

## Concept
{concept}

## Instruction
{instruction}

## Verified Code
```python
{code}
```

## Passing Tests ({len(tests)} total)
```python
{test_block}
```

## Score
{score_str}
{cot_section}"""



# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="CodeCoT pipeline")
    parser.add_argument("--input-file", default=None,  help="Seed .txt file")
    parser.add_argument("--backend",    default=None,  help="ollama | openai-compatible (overrides config)")
    parser.add_argument("--model-id",   default=None,  help="Model name (overrides config)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--input-dir",  default=None,  help="Reuse existing raw artifacts")
    parser.add_argument("--config",     default="pipeline_config.yaml")
    parser.add_argument("--stage",
        choices=["all", "concepts", "synthesis", "filter", "cot", "report"],
        default="all")
    args = parser.parse_args()

    cfg   = _load_config(args.config) if Path(args.config).exists() else {}
    a_cfg = cfg.get("stage_a", {})
    c_cfg = cfg.get("stage_c", {})
    m_cfg = cfg.get("model",   {})

    backend  = args.backend  or m_cfg.get("backend",  "ollama")
    model_id = args.model_id or m_cfg.get("model_id", "qwen2.5-coder:7b")

    if args.stage in {"all", "concepts"} and not args.input_file and not args.input_dir:
        parser.error("--input-file is required")

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(args.input_dir) if args.input_dir and args.stage != "all"
        else Path("outputs") / f"demo_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    )
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.input_dir and args.stage != "all":
        src = Path(args.input_dir) / "raw"
        for name in ["concepts.json", "out_solutions.json", "out_solutions_raw.json",
                     "out_test_cases.json", "dual_exec_results.jsonl", "out_solutions.jsonl"]:
            src_f, dst_f = src / name, raw_dir / name
            if src_f.exists() and not dst_f.exists():
                dst_f.write_text(src_f.read_text(encoding="utf-8"), encoding="utf-8")

    log_path = output_dir / "run.log"
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        handlers=[logging.FileHandler(log_path, mode="a")],
    )

    client = ModelClientFactory.create_client(backend=backend, model_id=model_id)
    total  = 5
    _d.pipeline_header(backend, model_id, output_dir, log_path)

    if args.stage in {"all", "concepts"}:
        _d.header(1, total, "Concept extraction")
        _stage_concepts(raw_dir, args.input_file, client, a_cfg)

    if args.stage in {"all", "synthesis"}:
        _d.header(2, total, "Code + test synthesis")
        _stage_synthesis(raw_dir, client, model_id, a_cfg)

    if args.stage in {"all", "filter"}:
        _d.header(3, total, "Code quality filter  (execution + consensus)")
        _stage_filter(raw_dir, cfg.get("stage_b", {}))

    if args.stage in {"all", "cot"}:
        _d.header(4, total, "Trace annotation, CoT generation + symbolic verification")
        _stage_cot(raw_dir, backend, model_id, args.config, client,
                   max_pairs=c_cfg.get("max_pairs", None),
                   trace_tests_per_pair=c_cfg.get("trace_tests_per_pair", None))

    if args.stage in {"all", "report"}:
        _d.header(5, total, "Report")
        _stage_report(output_dir, raw_dir)

    _d.pipeline_done(output_dir)


if __name__ == "__main__":
    main()
