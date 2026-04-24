import os
import re

import typer


def clean_trace_content(trace_content):
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    cleaned_trace = ansi_escape.sub("", trace_content)

    cleaned_lines = []
    for line in cleaned_trace.split("\n"):
        line = line.strip()
        if line:
            cleaned_lines.append(line)

    cleaned_trace = "\n".join(cleaned_lines)
    return cleaned_trace


def clean_trace_file(input_file, output_file=None):
    """
    Clean PySnooper trace file by removing ANSI color codes and formatting.

    Args:
    - input_file: Path to the input trace file.
    - output_file: Path to the output file (optional).
    """
    with open(input_file, "r") as f:
        trace_content = f.read()
    # print(trace_content)
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    cleaned_trace = ansi_escape.sub("", trace_content)

    cleaned_lines = []
    for line in cleaned_trace.split("\n"):
        line = line.strip()
        if line:
            cleaned_lines.append(line)

    cleaned_trace = "\n".join(cleaned_lines)

    if output_file:
        with open(output_file, "w") as f:
            f.write(cleaned_trace)
    else:
        print(cleaned_trace)

    # print(cleaned_trace)


def process_folder(input_folder, output_folder=None):
    """
    Process all log files in a folder.

    Args:
    - input_folder: Path to the folder containing input log files.
    - output_folder: Path to the folder to save cleaned files (optional).
    """
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        input_file_path = os.path.join(input_folder, filename)

        if os.path.isfile(input_file_path):
            if output_folder:
                output_file_path = os.path.join(output_folder, filename)
            else:
                output_file_path = None

            clean_trace_file(input_file_path, output_file_path)


def main(
    input_folder: str = typer.Option(
        "data/pysnooper_trace_uncleaned_v1", help="Input folder path"
    ),
    output_folder: str = typer.Option(
        "data/pysnooper_trace_cleaned_v1", help="Output folder path"
    ),
):
    """
    Clean PySnooper trace files by removing ANSI color codes and formatting.
    """
    process_folder(input_folder, output_folder)


if __name__ == "__main__":
    typer.run(main)
