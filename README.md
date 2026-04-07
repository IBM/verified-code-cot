## Verified Code CoT Data Pipeline

### Overview

This pipeline generates verifiable Chain-of-Thought (CoT) datasets for code reasoning tasks by grounding rationales directly in program execution traces. Unlike traditional synthetic datasets that rely on teacher-model explanations (which may hallucinate), this pipeline ensures every reasoning step is correct by construction.

The pipeline supports:

  >Forward reasoning (input → output prediction) <br/>
  Backward reasoning (output → input reconstruction)<br/>
  Trace-grounded natural language rationales<br/>
  Scalable synthesis across diverse programming concepts<br/>

### Key Features

**Execution-Grounded Verification:** Rationales are derived from actual runtime traces, eliminating logical hallucinations.<br/>
**Concept-First Curriculum:** Problems are synthesized from abstract programming concepts (e.g., recursion, dynamic programming), ensuring diversity and controlled difficulty.<br/>
**Bi-Directional Reasoning Dataset:** Supports both forward and backward reasoning tasks for robust model training.<br/>
**Dual Agreement Filtering:** A scalable verification method that clusters solutions/tests to identify high-quality artifacts.<br/>
**Open-Source Infrastructure:** Modular design for reproducibility and extensibility.<br/>

### Pipeline Architecture

**Stage A: Concept Sourcing & Curriculum-Driven Synthesis**

>**Document Processing:** Extract programming concepts from technical literature using Docling.<br/>
**Hybrid Concept Identification:** Combine statistical keyword extraction (PyTextRank) with LLM-based filtering.<br/>
**Deduplication & Quality Scoring:** Normalize, cluster, and score concepts by difficulty and relevance.<br/>

**Problem Synthesis:**

  >Instruction generation<br/>
  Signature generation (function/class metadata)<br/>
  Candidate code solutions <br/>
  Test scenario identification<br/>
  Unit test generation (strict assert-only format)<br/>

**Stage B: Execution-Based Verification**

Mass Execution: Run all solution-test pairs in sandboxed containers.<br/>
Pass/Fail Matrix: Construct binary matrix of results.<br/>
Dual Agreement Clustering:
Group solutions by identical test pass patterns.<br/>
Score clusters by solution agreement × test coverage.<br/>
Select highest-scoring cluster for canonical solution + tests.<br/>


**Stage C: CoT Generation**

Trace Extraction: Capture variable states, control flow, and transitions during execution.<br/>
Trace-to-Rationale Translation: Convert execution traces into natural language reasoning steps.<br/>

Bi-Directional CoT:<br/>
  >Forward reasoning: input → output<br/>
  Backward reasoning: output → input<br/>

Dataset Assembly: Package verified rationales, code, and tests into training-ready format.


## Getting Started

Prerequisites
  Python 3.10+

- **Clone the repo**

  The repo uses [uv](https://astral.sh/uv/) for python version and package management.

The following command can be run everytime we use the pipeline. When executed for the first time, it installs the required dependancies.

``` shell
. setenv.sh
```
 


## Supported LLMS

The pipeline supports OPENAI like api, which means it can be used with hosted models which are served via openai api, 
or behind litellm proxy. Even local LLMs can be served via `vllm serve` to run a openai like api server which can be 
used by the pipeline.

We use an abstraction called client, for abstracting model calls. 

NOTE: Advanced users who want to use GPU, can use vll based client which supports batching. However they need to install vllm and change the code
to use vllm based client instead of openai api based client.



## How to Run

 Please check [pipeline guide.](./COT_PIPELINE_GUIDE.md)


## Reference

