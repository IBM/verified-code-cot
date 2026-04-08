# Standard
import json
import sys


def split_test_functions(test_script):
    """Extract only top-level test functions with 'assert' statements."""
    lines = test_script.splitlines()
    test_functions = []
    current_function = []
    in_function = False

    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("def ") and not line.startswith(
            " "
        ):  # Top-level def only
            if in_function and "assert " in "\n".join(current_function):
                test_functions.append("\n".join(current_function))
            current_function = [line]
            in_function = True
        elif in_function and line.startswith(
            " "
        ):  # Indented line belongs to current function
            current_function.append(line)
        elif in_function and stripped_line and not stripped_line.startswith("#"):
            # Non-indented, non-comment line ends the function
            if "assert " in "\n".join(current_function):
                test_functions.append("\n".join(current_function))
            in_function = False
            current_function = []

    # Handle the last function
    if in_function and "assert " in "\n".join(current_function):
        test_functions.append("\n".join(current_function))

    # Process each test function for standalone asserts
    split_tests = []
    for func in test_functions:
        split_tests.extend(process_single_function(func))
    return split_tests


def process_single_function(test_script):
    """Split a test function with multiple standalone asserts into separate test functions."""
    lines = test_script.split("\n")
    func_start = next(
        (i for i, line in enumerate(lines) if line.strip().startswith("def ")), -1
    )
    if func_start == -1:
        return [test_script]

    func_name = lines[func_start].strip().split("(")[0].replace("def ", "")
    assert_lines = [line for line in lines[func_start:] if "assert " in line]

    if len(assert_lines) <= 1:
        return [test_script]

    is_standalone = True
    prev_assert_idx = None
    for i, line in enumerate(lines[func_start:], start=func_start):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            if "assert " in line:
                if prev_assert_idx is not None:
                    between_lines = [
                        l.strip()
                        for l in lines[prev_assert_idx + 1 : i]
                        if l.strip() and not l.strip().startswith("#")
                    ]
                    if between_lines:
                        is_standalone = False
                        break
                elif i > func_start:
                    before_lines = [
                        l.strip()
                        for l in lines[func_start + 1 : i]
                        if l.strip() and not l.strip().startswith("#")
                    ]
                    if before_lines:
                        is_standalone = False
                        break
                prev_assert_idx = i

    if not is_standalone:
        return [test_script]

    split_tests = []
    indent = " " * (len(lines[func_start]) - len(lines[func_start].lstrip()))
    for idx, assert_line in enumerate(assert_lines, 1):
        new_test_lines = [
            f"{indent}def {func_name}_case_{idx}():",
            f"{indent}    {assert_line.strip()}",
        ]
        split_tests.append("\n".join(new_test_lines))
    return split_tests


def process_jsonl_file(results, testcases):
    """Reads a JSONL file, processes the test functions, and writes the updated data back to a new JSONL file."""
    # with open(input_filename, 'r', encoding='utf-8') as infile:
    #     with open(output_filename, 'w', encoding='utf-8') as outfile:

    results_gl = []
    test_cases_gl = []
    for line_num, line in enumerate(results, 1):
        entry = json.loads(line)

        entry["task_id"] = line_num

        if "instruction_based_tests" in entry and entry["instruction_based_tests"]:
            processed_tests = []
            for test_case in entry["instruction_based_tests"]:
                if isinstance(test_case, list):
                    for sub_test in test_case:
                        if sub_test:
                            print(
                                f"\n--- Test Before Processing (ID: {entry['id']}) ---"
                            )
                            print(sub_test)
                            print("---------------------------------------------")

                            split_results = split_test_functions(sub_test)
                            processed_tests.extend(split_results)

                            print(f"--- Test After Processing (ID: {entry['id']}) ---")
                            for idx, split_test in enumerate(split_results, 1):
                                print(f"Split Test {idx}:\n{split_test}")
                            print("---------------------------------------------")
                else:
                    if test_case:
                        print(f"\n--- Test Before Processing (ID: {entry['id']}) ---")
                        print(test_case)
                        print("---------------------------------------------")

                        split_results = split_test_functions(test_case)
                        processed_tests.extend(split_results)

                        print(f"--- Test After Processing (ID: {entry['id']}) ---")
                        for idx, split_test in enumerate(split_results, 1):
                            print(f"Split Test {idx}:\n{split_test}")
                        print("---------------------------------------------")

            entry["tests_split"] = processed_tests

        outfile.write(json.dumps(entry) + "\n")


# print(f"Processed JSONL file saved as {output_filename}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_jsonl> <output_jsonl>")
        sys.exit(1)

    input_jsonl = sys.argv[1]
    output_jsonl = sys.argv[2]
    process_jsonl_file(input_jsonl, output_jsonl)


if __name__ == "__main__":
    main()
