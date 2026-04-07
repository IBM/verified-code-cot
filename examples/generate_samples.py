import json
from pathlib import Path

import typer

from dataops_code_cot.components.execution.dual_exec import (
    dual_exec,
    save_solution_test_cases,
)

app = typer.Typer()


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
    solutions_file: str = typer.Option(
        "solutions.json", "--solutions-file", help="Path to solutions JSON file"
    ),
    solutions_raw_file: str = typer.Option(
        "sd_solutions_raw.json",
        "--solutions-raw-file",
        help="Path to raw solutions JSON file",
    ),
    test_cases_file: str = typer.Option(
        "test_cases.json", "--test-cases-file", help="Path to test cases JSON file"
    ),
    output_file: str = typer.Option(
        "dual_exec_results.json", "--output-file", help="Path to output results file"
    ),
):
    """Execute dual execution on solutions and test cases."""
    with open(solutions_file) as s:
        solutions = json.load(s)

    # TODO: try to pass same jsonl as above to the code
    save_to_jsonl(solutions, "sols.jsonl")

    with open(test_cases_file) as s:
        test_cases = json.load(s)

    solution_test_cases, handled_solutions = dual_exec(solutions, test_cases)
    save_solution_test_cases(
        solution_test_cases,
        handled_solutions,
        "sols.jsonl",
        ".",
        output_file,
    )
    print("Validation done")


if __name__ == "__main__":
    app()
