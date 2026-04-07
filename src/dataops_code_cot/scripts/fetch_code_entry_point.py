import json
import re

from dataops_code_cot.utils import OpenAIClient as Client


def process_file(input_filename, output_filename):
    # Initialize the VLLM client
    model_client = Client()
    processed_count = 0
    # System prompt to guide the model
    system_prompt = """You are a helpful assistant that analyzes Python code to identify the main entry point function or class.
The entry point is the main function or class that is being called in the test cases.
Please identify the main entry point and return it within <entry_point></entry_point> tags."""

    with open(input_filename, "r", encoding="utf-8") as infile:
        with open(output_filename, "w", encoding="utf-8") as outfile:
            for line in infile:
                entry = json.loads(line.strip())

                # if entry["entry_point"] in entry:
                #    continue
                # For each code snippet in the entry
                if "code_snippet" in entry and "passing_test_cases" in entry:
                    for i, code in enumerate(entry["code_snippet"]):
                        # Get test cases (up to 2 if available)
                        test_cases = (
                            entry["passing_test_cases"][:2]
                            if len(entry["passing_test_cases"]) > 1
                            else entry["passing_test_cases"][:1]
                        )
                        test_cases_str = "\n\nTest Cases:\n" + "\n".join(test_cases)

                        # Include instruction if available
                        instruction = (
                            f"\nInstruction: {entry.get('instruction', '')}\n"
                            if "instruction" in entry
                            else "\n"
                        )

                        # Create user prompt for the model
                        '''
                        user_prompt_1 = f"""Please analyze this Python code, its instruction, and test cases to identify the main entry point function or class.
{instruction}
Code:
{code}
{test_cases_str}

Return only the entry point name (function or class) within entry point tags.
Important: Always use <entry_point> tags to wrap your response, NOT the function name as tags.

Correct format:   <entry_point>count_word_frequency</entry_point>
Incorrect format: <count_word_frequency></count_word_frequency>

Return only the entry point name wrapped in <entry_point> tags."""
                        '''

                        user_prompt = f"""Please analyze the provided Python code, instruction, and test cases to identify the main entry point function or class being tested.

{instruction}

Code:
{code}

{test_cases_str}

### Important Instructions:
- The entry point is the **main function or class that the test cases are designed to evaluate**.
- If the test cases directly call a function/class, return its name.
- If the test cases only check properties (e.g., input/output types) **without calling a specific function**, carefully infer the most likely function/class being tested.
- If the correct entry point **cannot be determined with reasonable certainty**, return empty entry point tags.

### Response Format:
- Always wrap the identified function/class name in `<entry_point>` tags.
- If unsure, return: `<entry_point></entry_point>` (DO NOT GUESS incorrectly).
- DO NOT modify or add anything outside these tags.

✅ Correct format:
    <entry_point>count_word_frequency</entry_point>

❌ Incorrect formats:
    <count_word_frequency></count_word_frequency>
    count_word_frequency
    <entry_point>Possibly main_function?</entry_point>

Now, determine the correct entry point and return your answer in `<entry_point>` tags."""
                        # Get model response
                        response = model_client.get_model_response(
                            system_prompt=system_prompt, user_prompt=user_prompt
                        )

                        print(response)
                        # Extract entry point from response using regex
                        entry_point_match = re.search(
                            r"<entry_point>(.*?)</entry_point>", response
                        )
                        print(entry_point_match)
                        # print(user_prompt)
                        if entry_point_match:
                            entry_point = entry_point_match.group(1).strip()
                            # print(entry_point)
                            # Add entry point to the entry
                            # if "entry_point" not in entry:
                            print(entry["task_id"], entry_point)
                            entry["entry_point"] = entry_point
                            # outfile.flush()  # Force write to disk
                            # processed_count += 1
                            print(entry_point)
                        else:
                            entry["entry_point"] = ""

                # print(entry)
                # Write the processed entry

                outfile.write(json.dumps(entry) + "\n")
                outfile.flush()  # Force write to disk

                processed_count += 1

        print(f"Total entries processed: {processed_count}")  # Debug print


# Example usage:
# process_file("input.json", "output.json")


def main():
    import sys

    if len(sys.argv) != 3:
        print("Usage: python script.py <input_jsonl> <output_jsonl>")
        sys.exit(1)

    input_jsonl = sys.argv[1]
    output_jsonl = sys.argv[2]
    process_file(input_jsonl, output_jsonl)


if __name__ == "__main__":
    main()
