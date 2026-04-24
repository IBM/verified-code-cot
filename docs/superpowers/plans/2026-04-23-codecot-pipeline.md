# CodeCoT Pipeline — Community Demo Polish Plan

**Goal:** Make the repo easy for community users to try without paid APIs, while showing clearly why execution-grounded CoT is better than a one-shot model answer.

**Principle:** Keep pipeline logic unchanged. Polish the entry points, defaults, and outputs.

**Default path:** free/local Ollama.

**Optional paths:** OpenAI-compatible local/hosted endpoints, RITS for internal use.

---

## Target User Experience

```bash
. setenv.sh
ollama pull qwen2.5-coder:7b
python examples/run_demo.py --backend ollama --model-id qwen2.5-coder:7b
```

Terminal output stays clean:

```text
CodeCoT Demo: Execution-Grounded Reasoning

Input: examples/test_concepts.txt
Backend: ollama / qwen2.5-coder:7b
Output: outputs/demo_2026-04-23_1432/

[1/5] Extracting concepts... found 6 concepts
[2/5] Generating code + tests... created 4 candidate problems
[3/5] Running dual execution... selected verified solution
[4/5] Capturing trace + CoT preview... captured evidence
[5/5] Building report... wrote demo_report.html

Open: outputs/demo_2026-04-23_1432/demo_report.html
```

The main output is a self-contained visual report:

```text
outputs/demo_<timestamp>/
  demo_report.html
  demo_report.md
  raw/
    concepts.json
    out_solutions_raw.json
    out_solutions.json
    out_test_cases.json
    dual_exec_results.jsonl
```

---

## What The Report Must Show

The report should be consumable in 30 seconds:

1. **Hero:** "CodeCoT Demo: Why Trace-Grounded Reasoning Helps"
2. **Pipeline strip:** Concept -> Problem + Tests -> Execution -> Trace -> Verified CoT
3. **Problem card:** generated instruction, code, and test case
4. **Trace snapshot:** compact table with line, variable state, and what happened
5. **Comparison:** one-shot baseline vs CodeCoT verified reasoning
6. **Forward CoT:** input -> trace states -> output
7. **Backward CoT:** output -> contributing states -> input behavior
8. **Artifacts:** links to raw files, collapsed or placed at the bottom

Do not print long JSON, full traces, or verbose logs in the terminal.

---

## Focused Implementation Tasks

### Task 1: Free/local model clients

**Files:**
- Create `src/dataops_code_cot/utils/model_client.py`
- Modify `src/dataops_code_cot/utils/__init__.py`

Requirements:

- Add `ModelClientFactory.create_client(backend, model_id, **kwargs)`.
- Support `ollama` as the default community backend.
- Support `openai-compatible` for local servers such as vLLM, llama.cpp, LM Studio, or LiteLLM.
- Keep `rits` optional/internal.
- Match the existing pipeline interface:

```python
client.get_model_response(
    system_prompt="...",
    user_prompt="...",
    model_id="...",
    max_new_tokens=2000,
    temperature=0.5,
)
```

Do not require paid OpenAI credentials for the default quickstart.

### Task 2: Single demo script

**Files:**
- Create `examples/run_demo.py`

Requirements:

- Run a small demo into `outputs/demo_<timestamp>/`.
- Default backend should be `ollama`.
- Default model should be a free local model, e.g. `qwen2.5-coder:7b`.
- Accept `--backend`, `--model-id`, `--input-file`, and `--output-dir`.
- Print only short stage summaries.
- Save raw intermediate artifacts under `raw/`.
- Generate both `demo_report.html` and `demo_report.md`.
- Include a `--sample-only` mode that builds the report from bundled sample outputs without calling an LLM.

### Task 3: Bundled sample output

**Files:**
- Create `examples/sample_outputs/raw/`
- Create `examples/sample_outputs/demo_report.md`
- Create `examples/sample_outputs/demo_report.html`

Requirements:

- Include one small curated example.
- Show generated problem/code/tests.
- Show a compact trace snippet.
- Show forward and backward CoT.
- Show one-shot baseline answer beside verified answer.
- Keep it honest: label it as a bundled illustrative sample.

### Task 4: README quickstart

**Files:**
- Modify `README.md`

Requirements:

- Put the free/local Ollama quickstart first.
- Make paid APIs optional, not the main path.
- Add a "No model installed?" path:

```bash
python examples/run_demo.py --sample-only
```

- Link to `outputs/.../demo_report.html` as the thing users should inspect.
- Keep `COT_PIPELINE_GUIDE.md` for full details.

### Task 5: Lightweight diagram

**Files:**
- Create `docs/pipeline.svg` or `docs/pipeline.png`

Requirements:

- Use a colorful repo-specific diagram, not the paper figure.
- Show the same high-level idea:

```text
Concepts -> Code + Tests -> Execution -> Trace -> Forward/Backward CoT
```

- Reference this diagram from README and/or the demo report.

---

## Explicit Non-Goals For This Pass

- Do not rewrite the pipeline internals.
- Do not introduce a `CodeCoT` wrapper unless it becomes necessary.
- Do not remove `spacy` setup yet; concept generation imports `spacy` and loads `en_core_web_sm`.
- Do not make OpenAI credentials part of the main quickstart.
- Do not optimize large-scale batching in this pass.

---

## Success Criteria

- A new user can run a demo without paid APIs.
- A user without Ollama/model setup can still inspect a polished sample report.
- Terminal output is short and uncluttered.
- The report visually explains what happened under the hood.
- Existing pipeline functions remain the source of truth.
