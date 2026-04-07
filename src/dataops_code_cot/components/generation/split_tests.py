# Standard

"""
The `split_test_functions` function processes test scripts and splits them into individual test functions based on specific scenarios:

- Multiple Functions at the Same Level:
  Already partially handled by `split_by_functions`, which identifies top-level functions. This function refines the process by handling each top-level function individually via `process_single_function`.

- Single Assert Statements:
  If a test function contains zero or one `assert` statement, it is kept as-is and returned unchanged.

- Multiple Asserts on Direct Functions:
  - Standalone asserts calling the same function (e.g., `translate_text`) are split into separate test functions. Everything before the first `assert` is treated as the header and included in each split test.
  - Detects data structure tests (e.g., queue, stack, circular buffer) where asserts depend on sequential calls (using keywords like `push`, `pop`, `next`). These are not split to preserve dependency.

- Multiple Asserts on Class Methods:
  Leaves dependent class method asserts (e.g., `Scheduler().add_task`) as-is. Uses a heuristic to detect dependency, such as repeated calls to the same method or function names suggesting sequential logic (e.g., `sort`). These tests are returned unchanged.
"""


def split_by_functions(test_script):
    """Split the script into individual top-level functions based on 'def' statements, excluding trailing comments."""
    lines = test_script.splitlines()
    functions = []
    current_function = []
    base_indent = None
    in_function = False

    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if stripped_line.startswith("def "):
            current_indent = len(line) - len(line.lstrip())
            if base_indent is None:
                base_indent = current_indent
            if current_indent == base_indent:
                if current_function:
                    # Trim trailing empty lines or comments from the previous function
                    while current_function and (
                        not current_function[-1].strip()
                        or current_function[-1].strip().startswith("#")
                    ):
                        current_function.pop()
                    functions.append("\n".join(current_function))
                current_function = [line]
                in_function = True
            else:
                current_function.append(line)
        elif in_function:
            # Stop collecting lines for this function if we hit a non-indented line that's not part of the function
            if (
                stripped_line
                and not line.startswith(" ")
                and not stripped_line.startswith("#")
            ):
                in_function = False
                while current_function and (
                    not current_function[-1].strip()
                    or current_function[-1].strip().startswith("#")
                ):
                    current_function.pop()
                functions.append("\n".join(current_function))
                current_function = []
            else:
                current_function.append(line)

    if current_function:
        # Trim trailing empty lines or comments from the last function
        while current_function and (
            not current_function[-1].strip()
            or current_function[-1].strip().startswith("#")
        ):
            current_function.pop()
        functions.append("\n".join(current_function))

    return functions


def split_by_functions_simple(test_script):
    """Split the script into individual top-level functions based on 'def' statements."""
    lines = test_script.splitlines()
    functions = []
    current_function = []
    base_indent = None

    for line in lines:
        if line.strip().startswith("def "):
            current_indent = len(line) - len(line.lstrip())
            if base_indent is None:
                base_indent = current_indent
            if current_indent == base_indent:
                if current_function:
                    functions.append("\n".join(current_function))
                current_function = [line]
            else:
                current_function.append(line)
        else:
            if current_function:
                current_function.append(line)

    if current_function:
        functions.append("\n".join(current_function))

    return functions


def split_test_functions(test_script):
    """Split test script into individual test functions based on new scenarios."""
    test_script = test_script.split("python")[
        0
    ]  # Remove any "python" prefix if present

    lines = test_script.split("\n")

    if any(line.strip().startswith("class ") for line in lines):
        return [test_script]

    functions = split_by_functions(test_script)
    if not functions:
        return [test_script]

    if len(functions) > 1:
        split_tests = []
        for func in functions:
            split_tests.extend(process_single_function(func))
        return split_tests
    else:
        return process_single_function(test_script)


def process_single_function(test_script):
    """Process a single test function based on assert statements and dependencies."""
    lines = test_script.split("\n")

    import_lines = [
        line.strip()
        for line in lines
        if line.strip().startswith("import ") or line.strip().startswith("from ")
    ]
    imports = "\n".join(import_lines) + "\n" if import_lines else ""
    if import_lines:
        lines = [
            line
            for line in lines
            if not (
                line.strip().startswith("import ") or line.strip().startswith("from ")
            )
        ]

    func_start = next(
        (i for i, line in enumerate(lines) if line.strip().startswith("def ")), -1
    )
    if func_start == -1:
        # If no 'def', split anyway to ensure individual functions
        split_functions = split_by_functions(test_script)
        return split_functions if split_functions else [test_script]

    func_name = lines[func_start].strip().split("(")[0].replace("def ", "")
    assert_lines = [line for line in lines[func_start:] if "assert " in line]

    if len(assert_lines) <= 1:
        # If single assert, split into individual functions
        split_functions = split_by_functions(test_script)
        return split_functions if split_functions else [test_script]

    is_class_based = False
    called_functions = []
    for assert_line in assert_lines:
        assert_content = assert_line.split("assert ")[1].split("==")[0].strip()
        if "." in assert_content:
            is_class_based = True
            called_functions.append(assert_content.split(".")[1].split("(")[0])
        else:
            called_functions.append(assert_content.split("(")[0])

    if is_class_based:
        unique_methods = set(called_functions)
        if len(unique_methods) < len(called_functions) or "sort" in func_name.lower():
            # If dependent class methods, split into individual functions
            split_functions = split_by_functions(test_script)
            return split_functions if split_functions else [test_script]
        # If independent class methods, split into individual functions
        split_functions = split_by_functions(test_script)
        return split_functions if split_functions else [test_script]

    first_assert_idx = (
        next(i for i, line in enumerate(lines[func_start:]) if "assert " in line)
        + func_start
    )
    header_lines = lines[func_start + 1 : first_assert_idx]
    if header_lines:
        min_header_indent = min(
            len(line) - len(line.lstrip()) for line in header_lines if line.strip()
        )
        header = (
            "\n".join(line[min_header_indent:] for line in header_lines if line.strip())
            + "\n"
        )
    else:
        header = ""

    base_function = called_functions[0]
    all_same_function = all(func == base_function for func in called_functions)
    data_structure_keywords = {
        "push",
        "pop",
        "next",
        "enqueue",
        "dequeue",
        "append",
        "remove",
    }
    is_data_structure_test = any(
        keyword in base_function.lower() for keyword in data_structure_keywords
    )

    if all_same_function and not is_data_structure_test:
        split_tests = []
        assert_blocks = []
        current_block = []

        for line in lines[first_assert_idx:]:
            if "assert " in line:
                if current_block:
                    assert_blocks.append("\n".join(current_block))
                current_block = [line]
            else:
                if current_block:
                    current_block.append(line)
        if current_block:
            assert_blocks.append("\n".join(current_block))

        for idx, block in enumerate(assert_blocks, 1):
            block_lines = block.split("\n")
            if block_lines:
                min_block_indent = min(
                    len(line) - len(line.lstrip())
                    for line in block_lines
                    if line.strip()
                )
                normalized_block = "\n".join(
                    line[min_block_indent:] for line in block_lines if line.strip()
                )
            else:
                normalized_block = block

            new_test_lines = []
            new_test_lines.append(imports.rstrip())
            new_test_lines.append(f"def {func_name}_case_{idx}():")
            if header:
                new_test_lines.extend(
                    "    " + line for line in header.splitlines() if line.strip()
                )
            new_test_lines.extend(
                "    " + line for line in normalized_block.splitlines() if line.strip()
            )

            split_tests.append("\n".join(new_test_lines))
        return split_tests

    # Default case: split into individual functions
    split_functions = split_by_functions(test_script)
    return split_functions if split_functions else [test_script]


def normalize_instruction(instruction):
    """Normalize instruction text to handle minor variations."""
    return " ".join(instruction.lower().split())


def split_test_cases(results, test_cases):
    """Reads a JSONL file, processes the test functions, and writes the updated data back to a new JSONL file."""
    instruction_dict = {}
    next_instruction_id = 1

    results_gl = []
    test_cases_gl = []
    for line_num, line in enumerate(results):  # results_gl
        entry = results[line_num]

        entry["task_id"] = line_num

        # Process instruction_based_tests
        if "instruction_based_tests" in entry and entry["instruction_based_tests"]:
            processed_tests = []
            for test_case in entry["instruction_based_tests"]:
                if isinstance(test_case, list):
                    # Iterate over each element in the test_case list
                    for sub_test in test_case:
                        if sub_test:  # Skip empty elements
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
                    # Handle single string test_case
                    if test_case:  # Skip empty strings
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
            test_Case = test_cases[line_num]
            test_Case["tests_split"] = processed_tests
            test_Case["task_id"] = line_num
            results_gl.append(entry)
            test_cases_gl.append(test_Case)

    return results_gl, test_cases_gl
