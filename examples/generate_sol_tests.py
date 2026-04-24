import json

import typer

from dataops_code_cot.components.generation.solutions_and_testcases_generation import (
    get_instruction_code_test_pairs,
)
from dataops_code_cot.components.generation.split_tests import split_test_cases
from dataops_code_cot.utils.model_client import ModelClientFactory

app = typer.Typer()


def flatten_list(nested_list: list) -> list:
    return [item for sublist in nested_list for item in sublist]


def save_to_json(data, file_path: str) -> None:
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


@app.command()
def main(
    input_file: str = typer.Option(
        "concepts.json", "--input-file", help="Path to concepts JSON produced by Stage A"
    ),
    output_prefix: str = typer.Option(
        "out", "--output-prefix", help="Prefix for output JSON files"
    ),
    backend: str = typer.Option(
        "ollama", "--backend", help="Model backend: ollama | openai-compatible | rits"
    ),
    model_id: str = typer.Option(
        "qwen2.5-coder:7b", "--model-id", help="Model name for the chosen backend"
    ),
):
    """Stage A/B — generate code solutions and test cases from concepts."""
    with open(input_file, "r") as f:
        concepts = json.load(f)

    client = ModelClientFactory.create_client(backend=backend, model_id=model_id)

    solutions_nested, test_cases_nested = get_instruction_code_test_pairs(
        concepts, client, model_id=model_id
    )
    print("Code and test-case generation done. Splitting test pairs…")

    save_to_json(solutions_nested, f"{output_prefix}_solutions_raw.json")

    solutions = flatten_list(solutions_nested)
    test_cases = flatten_list(test_cases_nested)
    solutions, test_cases = split_test_cases(solutions, test_cases)

    print("Test-case splitting done.")
    save_to_json(solutions, f"{output_prefix}_solutions.json")
    save_to_json(test_cases, f"{output_prefix}_test_cases.json")


if __name__ == "__main__":
    app()
