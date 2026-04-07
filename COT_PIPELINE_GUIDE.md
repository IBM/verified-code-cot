# COT Data Synthesis Pipeline - Complete Guide

This guide walks you through the entire process of generating Chain-of-Thought (CoT) training data using a three-stage pipeline.

---

## Overview

The COT data synthesis pipeline consists of three stages:

1. **Stage A**: Concept Sourcing and Curriculum-Driven Synthesis
2. **Stage B**: Execution-Based Verification and Agreement Clustering
3. **Stage C**: Execution-Grounded CoT Generation

Each stage builds upon the outputs of the previous one.

---

## Stage A: Concept Sourcing and Curriculum-Driven Synthesis

This stage generates coding concepts and their corresponding solutions from source documents.

### Prerequisites

- Text documents (`.txt` format) as seeds for concept generation
- If you have PDFs or Word documents, use **docling** to convert them to text first

Since this pipeline needs LLM to create data, we need to openai like server. Make sure you have access to LLM server.
It is possible to use vllm, ollama or litellm to have access to such server.

```
export OPENAI_API_KEY=""
export OPENAI_BASE_URL=""
```

You need to add model name to scripts which use client. You can check `examples/generate_concepts.py` on how to add it.
You may need to update client in `generate_cots_batched.py` too, with revelant model name.

```python
from dataops_code_cot.utils import OpenAIClient

def get_client():
    import os
    api_key = os.environ["RITS_API_KEY"]
    model_id = "" # choose the model name, this is the model served via openai compliant api
    client = OpenAIClient(
         model_id = model_id
    )
    return client

```
Once we have the setup with access to LLM ready, we can follow the next steps to generate COT data step by step.


### Step 1: Generate Concepts from Text

Extract coding concepts from your source documents:

```bash
python examples/generate_concepts.py --input-file input.txt --output-file concepts.json
```

**Output**: `concepts.json` — A JSON file containing extracted concepts

### Step 2: Generate Solutions and Test Cases from Concepts

Generate solutions and test cases based on the extracted concepts:

```bash
python examples/generate_sol_tests.py --input-file concepts.json --output-prefix out
```

**Outputs**:
- `out_solutions.json` — Solutions for each concept
- `out_solutions_raw.json` — Raw/unprocessed solutions
- `out_test_cases.json` — Test cases for each solution

---

## Stage B: Execution-Based Verification and Agreement Clustering

This stage verifies solutions by executing them and clustering results based on agreement.

### Step 1: Run Dual Execution Agreement

Execute solutions and generate agreement samples:

```bash
python examples/generate_samples.py \
  --solutions-file out_solutions.json \
  --solutions-raw-file out_solutions_raw.json \
  --test-cases-file out_test_cases.json
```

**Output**: `dual_exec_json.json` — Execution results with agreement clustering

### Step 2: Generate Execution Traces

Extract detailed execution traces from the dual execution results:

```bash
python src/dataops_code_cot/scripts/get_traces.py dual_exec_json.json traces_folder
```

**Output**: `traces_folder/` — Individual trace files for each execution

### Step 3: Clean Traces

Post-process and clean the execution traces:

```bash
python src/dataops_code_cot/scripts/clean_trace.py traces_folder cleaned_traces_folder
```

**Output**: `cleaned_traces_folder/` — Cleaned, standardized trace files

---

## Stage C: Execution-Grounded CoT Generation

This stage generates forward and backward questions with chain-of-thought reasoning grounded in execution traces.

### Step 1: Generate CoT Samples from Tests

Create CoT samples with forward/backward questions and execution traces:

```bash
cd src/dataops_code_cot/scripts/
python generate_cots_batched.py \
  --prompts_file prompts.json \
  --output_file conversations_test.jsonl \
  --trace_dir cleaned_traces_folder \
  --exec_results_file exec_file.jsonl
```

**Output**: `conversations_test.jsonl` — Conversations with CoT reasoning

**Note**: For large datasets, split the output into multiple files for parallel processing.

### Step 2: Filter Best Test Cases

Select the best test cases based on coverage:

```bash
cd src/dataops_code_cot/scripts
python best_test_case_filter.py -i input_folder -o output_folder
```

**Requirements**:
- Create a folder (`input_folder`) containing either:
  - Single `conversations_test.jsonl` file, or
  - Multiple split files from Step 1

**Output**: `output_folder/` — Filtered conversations with optimal test coverage

**How it works**: The script executes test cases and selects those with the best coverage metrics.

### Step 3: Prepare Final CoT Data

Convert the filtered CoT data into your desired format (e.g., conversation messages for training):

```bash
cd src/dataops_code_cot/scripts
python cot_jsonl_to_jsonl.py cot-data-file.jsonl
```

**Output**: Formatted CoT data ready for pretraining or RL fine-tuning

---

## File Format Reference

| File | Description | Format |
|------|-------------|--------|
| `concepts.json` | Extracted coding concepts | JSON array |
| `out_solutions.json` | Processed solutions | JSON |
| `out_solutions_raw.json` | Unprocessed solutions | JSON |
| `out_test_cases.json` | Test cases for solutions | JSON |
| `dual_exec_json.json` | Execution results with agreement | JSON |
| `conversations_test.jsonl` | CoT conversations | JSONL (one conversation per line) |
| Trace files | Individual execution traces | JSON |

---

## Tips and Best Practices

- **Parallel Processing**: For large datasets in Stage C, split `conversations_test.jsonl` into multiple files and run `best_test_case_filter.py` on each partition
- **Document Preparation**: Ensure source documents are well-formatted text before feeding into Stage A
- **Trace Quality**: Clean traces improve final CoT data quality—review the cleaned traces if results are unsatisfactory
- **Output Formats**: The final output from Stage C can be adapted to various training formats using custom conversion scripts

---

## Example Workflow

```bash
# Stage A: Generate concepts and solutions
python examples/generate_concepts.py --input-file data.txt --output-file concepts.json
python examples/generate_sol_tests.py --input-file concepts.json --output-prefix out

# Stage B: Execute and trace
python examples/generate_samples.py \
  --solutions-file out_solutions.json \
  --solutions-raw-file out_solutions_raw.json \
  --test-cases-file out_test_cases.json

python src/dataops_code_cot/scripts/get_traces.py dual_exec_json.json traces
python src/dataops_code_cot/scripts/clean_trace.py traces cleaned_traces

# Stage C: Generate CoT data
python src/dataops_code_cot/scripts/generate_cots_batched.py \
  --prompts_file prompts.json \
  --output_file conversations_test.jsonl \
  --trace_dir cleaned_traces \
  --exec_results_file exec_results.jsonl

python src/dataops_code_cot/scripts/best_test_case_filter.py -i . -o filtered
python src/dataops_code_cot/scripts/cot_jsonl_to_jsonl.py filtered/best_conversations.jsonl
```

---

## Troubleshooting

- **Stage A fails**: Verify input text files are UTF-8 encoded and in plain text format
- **Stage B traces are incomplete**: Rerun `clean_trace.py` and verify execution succeeded
- **Stage C produces poor CoT quality**: Check that prompts.json is properly formatted and contains comprehensive question templates
- **Performance issues**: Use parallel processing by splitting large files before Stage C processing

