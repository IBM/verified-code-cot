#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
from enum import Enum

import pandas as pd
import typer


def extract_think_content(text):
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_response_content(text):
    match = re.search(r"<response>(.*?)</response>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def remove_feedback_part(text):
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        if line.strip().startswith("Feedback:"):
            break  # Stop processing once we hit "Feedback:"
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def get_answer_without_feedback(answer_text):
    answer_think = extract_think_content(answer_text)
    answer_response = extract_response_content(answer_text)
    answer_response_nofeedback = remove_feedback_part(answer_response)
    solution = f"{answer_think}\n{answer_response_nofeedback}"
    return solution


def join_messages(messages):
    # print (messages)
    user_messages = [
        message["content"] for message in messages if message["role"] == "user"
    ]
    assistant_messages = [
        get_answer_without_feedback(message["content"])
        for message in messages
        if message["role"] == "assistant"
    ]
    short_format = False
    if len(user_messages) < 3 and len(assistant_messages) < 2:
        if len(user_messages) == 2 and len(assistant_messages) == 1:
            short_format = True
        else:
            return ""

    comprehension = user_messages[0]
    question_1 = user_messages[1]
    answer_1 = assistant_messages[0]
    if not short_format:
        question_2 = user_messages[2]
        answer_2 = assistant_messages[1]

        output_format = f"""{comprehension}


1. {question_1}
{answer_1}


2. {question_2}
{answer_2}"""
    else:
        output_format = f"""{comprehension}


1. {question_1}
{answer_1}"""

    return output_format


def prepare_code_plus_test(row):
    code = row["code"]
    best_test_case = str(row["best_test_case_coverage"])
    if best_test_case == "-1":
        return ""
    test = row["test_cases"][best_test_case]

    return "\n".join((code, test))


def keep(row):
    # drop row if coverage test case is -1
    if str(row["best_test_case_coverage"]) == "-1":
        return False
    n_messages = len([message["content"] for message in row["messages"]])
    if n_messages < 5:
        # print (row["messages"])
        # retain if we have messages with role : user, user, assistant
        if n_messages == 3:
            roles = [x["role"] for x in row["messages"]]
            if roles[0] == "user" and roles[1] == "user" and roles[2] == "assistant":
                return True
            print(roles)
        return False
    return True


def prepare_data(row):
    """
    Prepare a single row of data by transforming and adding new columns.

    Parameters:
        row (pd.Series): A row from the DataFrame containing the following keys:
            - 'id': Unique identifier.
            - 'instruction': Instruction text.
            - 'code': Code content.
            - 'test_cases': Test cases associated with the code.
            - 'messages': List of messages, each containing a 'content' key.

    Returns:
        pd.Series: A new row with transformed and added columns:
            - 'id': New unique identifier using UUID.
            - 'source_id': Original ID from the input row.
            - 'source_instruction': Original instruction text from the input row.
            - 'source_code': Original code content from the input row.
            - 'source_tests': Original test cases from the input row.
            - 'content': Concatenated content of all messages in the input row.
    """
    import uuid

    # Index(['id', 'instruction', 'code', 'test_cases', 'messages', 'components'], dtype='object')
    id = row["id"]
    instruction = row["instruction"]
    code = row["code"]
    test_cases = row["test_cases"]
    messages = row["messages"]
    # content_list = [message["content"] for message in messages]
    content = join_messages(messages)
    keep_entry = keep(row)
    return pd.Series(
        {
            "document_id": str(uuid.uuid4()),
            "source_id": id,
            "source_instruction": instruction,
            "source_code": code,
            "source_tests": test_cases,
            "contents": content,
            "code_plus_test": prepare_code_plus_test(row),
            "best_test_case": row["best_test_case_coverage"],
            "keep": keep_entry,
            "number_messages": len(row["messages"]),
        }
    )


def main(file_name: str, output_file: str, dataset: str):
    """
    Process a JSONL file containing data and save the processed data to a Parquet file.

    Parameters:
        file_name (str): Path to the input JSON file.
        output_file (str): Path to the output Parquet file.
        dataset (str, optional): Name of the dataset to be added as a new column. Defaults to None.
    """
    df = pd.read_json(file_name, lines=True)
    print(f"Processing {file_name}")
    df_processed = df.apply(prepare_data, axis=1)
    if dataset:
        df_processed["source_dataset"] = len(df_processed) * [dataset]
    # Filter out empty content:
    print(f"Total rows before dropping ", len(df_processed))

    dfp = df_processed[df_processed["keep"]].drop(columns=["keep"]).reset_index()

    # df_processed.to_parquet(output_file)
    print(f"Total rows after dropping ", len(dfp))
    dfp.to_parquet(output_file)
    print(f"Writing output: {output_file}")


if __name__ == "__main__":
    typer.run(main)
