import json

import typer

from dataops_code_cot.components.generation.concepts_generation import generate_concepts
from dataops_code_cot.components.generation.solutions_and_testcases_generation import (
    get_instruction_code_test_pairs,
)
from dataops_code_cot.components.generation.split_tests import split_test_cases
from dataops_code_cot.utils import OpenAIClient

app = typer.Typer()


def get_client():
    import os

    api_key = os.environ["RITS_API_KEY"]
    client = OpenAIClient(
        model_id="", # Please add model name here and set OPENAI_API_BASE, OPENAI_API_KEY vars
    )
    return client


def flatten_list(nested_list):
    lst = [item for sublist in nested_list for item in sublist]
    return lst


def save_to_json(data, file_path):
    # Standard
    import json

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def save_to_jsonl(data, file_path):
    """
    Save a list (or generator) of Python objects as JSON‑Lines (JSONL) to a file.

    Parameters
    ----------
    data : list | generator
        List of dictionaries (or any JSON‑serializable objects) to write.
    file_path : str
        Path to the output JSONL file.
    """
    import json  # Import inside the function for clarity

    # Open the file in write mode with UTF‑8 encoding
    with open(file_path, "w", encoding="utf-8") as f:
        # Iterate over each record and dump it as a single JSON line
        for record in data:
            json.dump(record, f)  # Write without extra whitespace
            f.write("\n")


@app.command()
def main(
    input_file: str = typer.Option(
        "concepts.json", "--input-file", help="Path to input file with concepts"
    ),
    output_prefix: str = typer.Option(
        "output", "--output-prefix", help="Prefix for output JSON files"
    ),
):
    """Generate concepts from input documents."""
    with open(input_file, "r") as f:
        concepts = json.load(f)

    client = get_client()
    solutions, test_cases = get_instruction_code_test_pairs(
        concepts, client, model_id="mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    )
    print("Code and Test cases generation done. Now splitting the test pairs.")
    save_to_json(solutions, f"{output_prefix}_solutions_raw.json")

    solutions = flatten_list(solutions)
    test_cases = flatten_list(test_cases)

    solutions, test_cases = split_test_cases(solutions, test_cases)

    print("Test case splitting done..")
    save_to_json(solutions, f"{output_prefix}_solutions.json")
    save_to_json(test_cases, f"{output_prefix}_test_cases.json")


if __name__ == "__main__":
    app()
