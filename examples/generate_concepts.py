import typer

from dataops_code_cot.components.generation.concepts_generation import (
    generate_concepts,
    generate_concepts_concurrent,
)
from dataops_code_cot.utils import OpenAIClient

app = typer.Typer()


def get_client():
    import os

    api_key = os.environ["RITS_API_KEY"]
    model_id = (
        ""  # choose the model name, this is the model served via openai compliant api
    )
    client = OpenAIClient(model_id=model_id)
    return client


def flatten_list(nested_list):
    lst = [item for sublist in nested_list for item in sublist]
    return lst


def convert_to_format(document_contents):
    """
    Converts a list of document contents into the specified dictionary format with random keys.

    :param document_contents: List of document contents.
    :return: Dictionary in the specified format with random keys.
    """
    main_key = "0"
    sub_keys = [str(idx) for idx, _ in enumerate(document_contents)]

    formatted_dict = {
        main_key: {
            sub_keys[i]: document_contents[i] for i in range(len(document_contents))
        }
    }
    return formatted_dict


def save_to_json(data, file_path):
    # Standard
    import json

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def process_concepts(doc_list, client):
    documents_dict = convert_to_format(doc_list)
    return generate_concepts_concurrent(documents_dict, client)
    return generate_concepts(documents_dict, client)


@app.command()
def main(
    input_file: str = typer.Option(
        "test_concepts.txt", "--input-file", help="Path to input file with concepts"
    ),
    output_file: str = typer.Option(
        "concepts.json", "--output-file", help="Path to output JSON file"
    ),
):
    """Generate concepts from input documents."""
    with open(input_file, "r") as file:
        concepts = file.read()
    doc_list = [concepts]

    client = get_client()

    concepts = process_concepts(doc_list, client)
    print(f"Found {len(concepts)} across all documents")
    save_to_json(concepts, output_file)


if __name__ == "__main__":
    app()
