import json

import typer

from dataops_code_cot.components.generation.concepts_generation import (
    generate_concepts_concurrent,
)
from dataops_code_cot.utils.model_client import ModelClientFactory

app = typer.Typer()


def convert_to_format(document_contents: list[str]) -> dict:
    return {"0": {str(i): text for i, text in enumerate(document_contents)}}


def save_to_json(data, file_path: str) -> None:
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


@app.command()
def main(
    input_file: str = typer.Option(
        "test_concepts.txt", "--input-file", help="Path to input .txt file"
    ),
    output_file: str = typer.Option(
        "concepts.json", "--output-file", help="Path to output JSON file"
    ),
    backend: str = typer.Option(
        "ollama", "--backend", help="Model backend: ollama | openai-compatible | rits"
    ),
    model_id: str = typer.Option(
        "qwen2.5-coder:7b", "--model-id", help="Model name for the chosen backend"
    ),
):
    """Stage A — extract programming concepts from a seed text file."""
    with open(input_file, "r") as f:
        text = f.read()

    client = ModelClientFactory.create_client(backend=backend, model_id=model_id)
    documents_dict = convert_to_format([text])
    concepts = generate_concepts_concurrent(documents_dict, client)

    print(f"Found {len(concepts)} concepts across all documents")
    save_to_json(concepts, output_file)


if __name__ == "__main__":
    app()
