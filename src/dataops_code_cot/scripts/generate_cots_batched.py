import argparse
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

# from dataops_code_cot.utils.vllm import OpenAIClient
from dataops_code_cot.utils import OpenAIClient

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# global variables
# Maximum token limit for microsoft/phi-4
MAX_TOKENS = 16384
CHARS_PER_TOKEN = 4
MAX_CHARS = MAX_TOKENS * CHARS_PER_TOKEN  # 65536
TRACE_CACHE = {}  # Cache trace file contents
OUTPUT_BUFFER = []  # Buffer conversations for writing
OUTPUT_BUFFER_SIZE = 10  # Write every 10 conversations

# Parse command line arguments
parser = argparse.ArgumentParser(description="Generate COT data")

parser.add_argument("--prompts_file", default="", help="Path to prompts.json")
parser.add_argument(
    "--output_file", default="conversations_test.jsonl", help="Path to output file"
)

parser.add_argument("--trace_dir", default="", help="Path to trace dir")
parser.add_argument(
    "--exec_results_file", default="", help="Path to exec results json file"
)

args = parser.parse_args()
output_file = args.output_file
raw_trace_directory = args.trace_dir
exec_results_file = args.exec_results_file
prompts_file = args.prompts_file


def read_raw_trace(raw_trace_directory, filename):
    """Read trace file, using cache if available."""
    global TRACE_CACHE
    trace_filename = os.path.join(raw_trace_directory, filename)
    if trace_filename in TRACE_CACHE:
        return TRACE_CACHE[trace_filename]
    try:
        with open(trace_filename, "r") as file:
            content = file.read().strip()
            TRACE_CACHE[trace_filename] = content
            return content
    except FileNotFoundError:
        logger.error(f"Raw trace file not found: {trace_filename}")
        return None
    except Exception as e:
        logger.error(f"Error reading {trace_filename}: {str(e)}")
        return None


def sanitize_text(text):
    replacements = {
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "--",
    }
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)
    return text


def is_standalone_assert(test_content):
    lines = [line.strip() for line in test_content.split("\n")]
    code_lines = [
        line
        for line in lines
        if line and not line.startswith("#") and not line.startswith("def ")
    ]
    return len(code_lines) == 1 and code_lines[0].startswith("assert ")


def extract_from_assert(test_content, entrypoint, signature_info):
    if signature_info["type"] == "function":
        call_pattern = rf"{entrypoint}\((.*?)\)"
    else:
        class_name = signature_info["name"]
        call_pattern = rf"(?:{class_name}\(\)|\w+)\.{entrypoint}\((.*?)\)"

    # Combine regex patterns into one with named groups
    patterns = [
        (
            r"assert\s+{call_pattern}\s*==\s*(.*?)(?:\n|$)",
            "equality",
            lambda m: (m.group(1), m.group(2).split("#")[0].strip(), None),
        ),
        (
            r"assert\s+{call_pattern}\s+is\s+(not\s+)?None\b",
            "is_none",
            lambda m: (m.group(1), "not None" if m.group(2) else "None", None),
        ),
        (
            r"assert\s+isinstance\({call_pattern},\s*(\w+)\)",
            "isinstance",
            lambda m: (m.group(1), m.group(2).strip(), None),
        ),
        (
            r"assert\s+len\({call_pattern}\)\s*==\s*(\d+|\w+)(?:\n|$)",
            "len",
            lambda m: (m.group(1), m.group(2).strip(), None),
        ),
        (
            r'try:\s*{call_pattern}\s*except\s+(\w+)\s+as\s+\w+:\s*assert\s+str\(\w+\)\s*==\s*"(.*?)"',
            "exception",
            lambda m: (m.group(1), f"{m.group(2)}: {m.group(3)}", None),
        ),
        (
            r"assert\s+(.+?)\s+in\s+{call_pattern}",
            "in",
            lambda m: (m.group(2), m.group(1).strip(), None),
        ),
        (r"assert\s+{call_pattern}\b", "boolean", lambda m: (m.group(1), "True", None)),
        (
            r"assert\s+{call_pattern}\s*(>|<=?|>=)\s*(.*?)(?:\n|$)",
            "comparison",
            lambda m: (
                m.group(1),
                m.group(3).split("#")[0].strip(),
                m.group(2).strip(),
            ),
        ),
    ]
    for pattern, assert_type, extract in patterns:
        pattern = pattern.format(call_pattern=call_pattern)
        match = re.search(pattern, test_content, re.DOTALL)
        if match:
            input_val, output_val, operator = extract(match)
            return input_val.strip(), output_val, assert_type, operator
    return None, None, None, None


def count_tokens(text):
    try:
        return len(text)
    except Exception as e:
        logger.error(f"Error counting characters: {str(e)}")
        return float("inf")


def filter_snippets_by_token_limit(
    trace_files_by_snippet, raw_trace_directory, max_chars
):
    """Filter snippets by file size without reading content."""
    filtered_snippets = {}
    threshold_chars = max_chars * 0.6  # 39,321.6
    excluded_count = 0
    total_snippets = len(trace_files_by_snippet)
    global TRACE_CACHE
    for snippet_key, trace_files in trace_files_by_snippet.items():
        valid_snippet = True
        for testcase_num, filename in trace_files:
            trace_filename = os.path.join(raw_trace_directory, filename)
            try:
                file_size = os.path.getsize(trace_filename)
                if file_size > threshold_chars:
                    logger.info(
                        f"Excluding snippet {snippet_key} due to trace file {filename} exceeding {threshold_chars} bytes"
                    )
                    valid_snippet = False
                    excluded_count += 1
                    break
                # Cache content for valid files
                content = read_raw_trace(raw_trace_directory, filename)
                if content is None:
                    logger.warning(
                        f"Skipping trace file {filename} for {snippet_key} due to read failure"
                    )
                    valid_snippet = False
                    excluded_count += 1
                    break
            except FileNotFoundError:
                logger.warning(f"Trace file {trace_filename} not found")
                valid_snippet = False
                excluded_count += 1
                break
        if valid_snippet:
            filtered_snippets[snippet_key] = trace_files
    logger.info(
        f"Filtered to {len(filtered_snippets)} snippets with traces within {threshold_chars} characters"
    )
    logger.info(
        f"Filtered out {excluded_count} of {total_snippets} snippets ({(excluded_count / total_snippets * 100):.2f}% of total)"
    )
    return filtered_snippets


def process_code_snippet(
    vllm_client,
    key,
    code_id,
    raw_trace_directory,
    data_lookup,
    processed_ids,
    user_prompt_templates,
):
    try:
        conv_id = f"{key}_{code_id}"
        if conv_id in processed_ids:
            logger.debug(f"Skipping already processed {conv_id}")
            return None
        try:
            entry = data_lookup[(key, code_id)]
        except KeyError:
            logger.warning(f"No data found for task_id {key}, code_id {code_id}")
            return None
        instruction = entry["instruction"]
        signature_info = entry["signature_info"]
        entrypoint = entry["entrypoint"]
        if not all([instruction, signature_info, entrypoint is not None]):
            logger.warning(
                f"Missing instruction, signature_info, or entrypoint for task_id {key}, code_id {code_id}"
            )
            return None
        if not entrypoint or not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", entrypoint):
            logger.error(
                f"Invalid entrypoint '{entrypoint}' for task_id {key}, code_id {code_id}"
            )
            return None
        try:
            first_function = entry["code_snippet"][0]
            if not first_function:
                logger.warning(
                    f"No code snippet found for task_id {key}, code_id {code_id}"
                )
                return None
        except (KeyError, IndexError):
            logger.warning(f"Invalid code_snippet for task_id {key}, code_id {code_id}")
            return None
        try:
            test_cases = entry["passing_test_cases"]
            if not test_cases:
                logger.warning(
                    f"No valid test cases for task_id {key}, code_id {code_id}"
                )
                return None
        except KeyError:
            logger.warning(
                f"No passing_test_cases for task_id {key}, code_id {code_id}"
            )
            return None
        test_case_data = []
        io_prompts = []
        io_prompt_indices = []
        for testcase_num, test_case_content in enumerate(test_cases):
            trace_filename = f"trace_{key}_{code_id}_testcase_{testcase_num}.log"
            raw_trace = read_raw_trace(raw_trace_directory, trace_filename)
            if raw_trace == "Execution timed out after 10 seconds":
                logger.info(
                    f"Skipping test case {testcase_num} for {conv_id} due to execution timeout"
                )
                continue
            if not raw_trace:
                logger.warning(
                    f"Skipping test case {testcase_num} for {conv_id} due to missing trace"
                )
                continue
            # Removed: Trace truncation (unnecessary due to filter_snippets_by_token_limit)
            test_case_lines = test_case_content.split("\n")
            test_case_no_comments = "\n".join(
                line for line in test_case_lines if not line.strip().startswith("#")
            )
            if is_standalone_assert(test_case_no_comments):
                input_val, output_val, assert_type, operator = extract_from_assert(
                    test_case_no_comments, entrypoint, signature_info
                )
                if input_val is None or output_val is None:
                    logger.warning(
                        f"Skipping test case {testcase_num} for {conv_id} due to failed input/output extraction"
                    )
                    continue
                test_case_data.append(
                    {
                        "testcase_num": testcase_num,
                        "content": test_case_content,
                        "no_comments": test_case_no_comments,
                        "raw_trace": raw_trace,
                        "input_val": input_val,
                        "output_val": output_val,
                        "assert_type": assert_type,
                        "operator": operator,
                    }
                )
            else:
                escaped_test_case_no_comments = test_case_no_comments.replace(
                    "{", "{{"
                ).replace("}", "}}")
                try:
                    io_prompt = user_prompt_templates[5].format(
                        code_type=signature_info["type"],
                        entrypoint=entrypoint,
                        first_function=first_function,
                        test_case_no_comments=escaped_test_case_no_comments,
                    )
                except KeyError as e:
                    logger.error(
                        f"KeyError formatting io_prompt for test case {conv_id}_{testcase_num}: {str(e)}"
                    )
                    continue
                io_prompts.append(io_prompt)
                io_prompt_indices.append(testcase_num)
                test_case_data.append(
                    {
                        "testcase_num": testcase_num,
                        "content": test_case_content,
                        "no_comments": test_case_no_comments,
                        "raw_trace": raw_trace,
                        "input_val": None,
                        "output_val": None,
                        "assert_type": None,
                        "operator": None,
                    }
                )
        if not test_case_data:
            logger.warning(f"No valid test cases with traces for {conv_id}")
            return None
        summary_prompt = f"""
        You are an expert in Python code analysis. Given a Python function and its associated instruction, generate a one-line description of what the function does in a clear, descriptive format. The description should translate the question in the instruction into a statement about the function's purpose, avoiding phrases like 'explain how to' and focusing on the action the code performs.

        Instruction: {instruction}
        Function Code: ```python
        {first_function}
        ```

        Output Format:
        <Summary>[One-line description]</Summary>
        """
        return {
            "conv_id": conv_id,
            "instruction": instruction,
            "first_function": first_function,
            "entrypoint": entrypoint,
            "signature_info": signature_info,
            "test_case_data": test_case_data,
            "summary_prompt": summary_prompt,
            "io_prompts": io_prompts,
            "io_prompt_indices": io_prompt_indices,
        }
    except Exception as e:
        logger.error(
            f"Error processing snippet task_id {key}, code_id {code_id}: {str(e)}"
        )
        return None


logger.info(f"Starting script with  trace_dir={raw_trace_directory}")
vllm_client = OpenAIClient()

with open(exec_results_file, "r") as f:
    data = [json.loads(line) for line in f]

data_lookup = {}
for entry in data:
    if "task_id" not in entry or "code_id" not in entry:
        logger.warning(f"Skipping entry with missing task_id or code_id: {entry}")
        continue
    task_id = str(entry["task_id"])
    code_id = str(entry["code_id"])
    signature_info = entry.get("signature_info", {})
    try:
        if signature_info.get("type") == "function":
            entrypoint = signature_info.get("name")
        elif signature_info.get("type") == "class":
            primary_method = entry.get("primary_method", "")
            if (
                primary_method is None
                or primary_method == ""
                or primary_method == "None"
            ):
                logger.warning(
                    f"primary_method is null, empty, or 'None' for task_id {task_id}, code_id {code_id}"
                )
                entrypoint = ""
            else:
                entrypoint = primary_method
        else:
            logger.warning(
                f"Invalid signature_info type for task_id {task_id}, code_id {code_id}: {signature_info.get('type')}"
            )
            continue
        if not entrypoint or not isinstance(entrypoint, str):
            logger.warning(
                f"Invalid or missing entrypoint for task_id {task_id}, code_id {code_id}, entrypoint: '{entrypoint}'"
            )
            continue
    except (KeyError, TypeError) as e:
        logger.warning(
            f"Malformed signature_info for task_id {task_id}, code_id {code_id}, error: {str(e)}"
        )
        continue
    data_lookup[(task_id, code_id)] = {
        "instruction": entry["instruction"],
        "signature_info": signature_info,
        "entrypoint": entrypoint,
        "code_snippet": entry.get("code_snippet", []),
        "passing_test_cases": entry.get("passing_test_cases", []),
        "primary_method": entry.get("primary_method"),
    }
logger.info(f"Loaded {len(data_lookup)} valid entries into data_lookup")

try:
    with open(prompts_file, "r", encoding="utf-8") as f:
        content = f.read()
        f.seek(0)
        try:
            prompts_data = json.load(f)
        except json.JSONDecodeError as e:
            lines = content.splitlines()
            error_line = e.lineno
            start_line = max(1, error_line - 2)
            end_line = min(len(lines), error_line + 2)
            context = "\n".join(
                f"Line {i}: {lines[i - 1]}" for i in range(start_line, end_line + 1)
            )
            logger.error(
                f"JSON decoding error in {prompts_file} at line {error_line}: {e.msg}\nContext:\n{context}"
            )
            exit(1)
    system_prompts = prompts_data["system_prompts"]
    user_prompt_templates = prompts_data["user_prompts"]
    assert_guidelines = prompts_data["assert_guidelines"]
except FileNotFoundError:
    logger.error(f"Prompts file not found: {prompts_file}")
    exit(1)
except UnicodeDecodeError as e:
    logger.error(f"Encoding error reading {prompts_file}: {str(e)}")
    exit(1)
except Exception as e:
    logger.error(f"Error reading prompts file {prompts_file}: {str(e)}")
    exit(1)

processed_ids = set()

if os.path.exists(output_file):
    with open(output_file, "r") as f:
        for line in f:
            try:
                conv = json.loads(line.strip())
                processed_ids.add(conv["id"])
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed line in {output_file}")

trace_files_by_snippet = {}
try:
    abs_trace_dir = os.path.abspath(raw_trace_directory)
    logger.info(f"Reading trace files from: {abs_trace_dir}")
    trace_files = [
        f
        for f in os.listdir(abs_trace_dir)
        if os.path.isfile(os.path.join(abs_trace_dir, f))
    ]
    for filename in trace_files:
        match = re.match(
            r"trace_(\d+)_(\d+)(?:_)?testcase_?(\d+)\.(log|txt)",
            filename,
            re.IGNORECASE,
        )
        if match:
            key, code_id, testcase_num = match.groups()[:-1]
            if (key, code_id) not in data_lookup:
                logger.warning(
                    f"Skipping trace file with invalid task_id {key} or code_id {code_id}: {filename}"
                )
                continue
            if int(testcase_num) >= len(
                data_lookup[(key, code_id)]["passing_test_cases"]
            ):
                logger.warning(
                    f"Skipping trace file with invalid testcase_num {testcase_num}: {filename}"
                )
                continue
            snippet_key = (key, code_id)
            if snippet_key not in trace_files_by_snippet:
                trace_files_by_snippet[snippet_key] = []
            trace_files_by_snippet[snippet_key].append((testcase_num, filename))
except FileNotFoundError:
    logger.error(f"Trace directory not found: {abs_trace_dir}")
    exit(1)
except Exception as e:
    logger.error(f"Error reading trace directory {abs_trace_dir}: {str(e)}")
    exit(1)

logger.info(f"Found {len(trace_files_by_snippet)} snippets with trace files")
trace_files_by_snippet = filter_snippets_by_token_limit(
    trace_files_by_snippet, raw_trace_directory, MAX_CHARS
)
logger.info(
    f"Found {len(trace_files_by_snippet)} snippets with trace files after character filtering"
)

# Dynamic batch sizing by test cases
TARGET_TEST_CASES_PER_BATCH = 300
conversations = []
snippets = sorted(trace_files_by_snippet.keys())


def flush_output_buffer():
    global OUTPUT_BUFFER
    if OUTPUT_BUFFER:
        with open(output_file, "a") as file:
            for conv in OUTPUT_BUFFER:
                json.dump(conv, file)
                file.write("\n")
            file.flush()
        logger.debug(f"Flushed {len(OUTPUT_BUFFER)} conversations to {output_file}")
        OUTPUT_BUFFER = []


batch_test_cases = []
current_test_count = 0
batch_snippets = []
for key, code_id in snippets:
    test_count = len(trace_files_by_snippet[(key, code_id)])
    if current_test_count + test_count > TARGET_TEST_CASES_PER_BATCH and batch_snippets:
        batch_test_cases.append((batch_snippets, current_test_count))
        batch_snippets = []
        current_test_count = 0
    batch_snippets.append((key, code_id))
    current_test_count += test_count
if batch_snippets:
    batch_test_cases.append((batch_snippets, current_test_count))

for batch_idx, (batch_snippets, total_test_cases) in enumerate(batch_test_cases):
    logger.debug(
        f"Processing batch {batch_idx} with {len(batch_snippets)} snippets, {total_test_cases} test cases"
    )

    # Parallelize snippet processing
    batch_data = []

    def process_snippet(key_code_id):
        key, code_id = key_code_id
        return process_code_snippet(
            vllm_client,
            key,
            code_id,
            raw_trace_directory,
            data_lookup,
            processed_ids,
            user_prompt_templates,
        )

    with ThreadPoolExecutor(
        max_workers=min(os.cpu_count() or 4, len(batch_snippets))
    ) as executor:
        results = executor.map(process_snippet, batch_snippets)
        for result in results:
            if result:
                batch_data.append(result)
    TRACE_CACHE.clear()  # Clear cache after batch
    if not batch_data:
        logger.info(f"No valid snippets in batch {batch_idx}")
        continue

    # Batch summaries
    summary_systems = ["You are an expert in Python code analysis."] * len(batch_data)
    summary_prompts = [data["summary_prompt"] for data in batch_data]
    try:
        summary_responses = vllm_client.get_model_response(
            summary_systems, summary_prompts
        )
        if not summary_responses or any(r is None for r in summary_responses):
            logger.warning(f"Failed to get summaries for batch {batch_idx}")
            continue
    except Exception as e:
        logger.error(f"Error getting summary responses for batch {batch_idx}: {str(e)}")
        continue

    # Batch input/output for non-standalone test cases
    io_systems = []
    io_prompts = []
    io_mapping = []
    for snippet_idx, data in enumerate(batch_data):
        for prompt_idx, prompt in enumerate(data["io_prompts"]):
            testcase_num = data["io_prompt_indices"][prompt_idx]
            io_systems.append("You are an expert in Python code execution analysis.")
            io_prompts.append(prompt)
            io_mapping.append((snippet_idx, testcase_num))
    io_responses = []
    if io_prompts:
        try:
            io_responses = vllm_client.get_model_response(io_systems, io_prompts)
            if not io_responses or any(r is None for r in io_responses):
                logger.warning(f"Failed to get IO responses for batch {batch_idx}")
                continue
        except Exception as e:
            logger.error(f"Error getting IO responses for batch {batch_idx}: {str(e)}")
            continue

    # Parse IO responses with single regex
    for resp_idx, (snippet_idx, testcase_num) in enumerate(io_mapping):
        resp = io_responses[resp_idx]
        try:
            match = re.match(
                r".*?<Input>(.*?)</Input>.*?<Output>(.*?)</Output>.*?(?:<AssertType>(.*?)</AssertType>)?(?:.*?<Operator>(.*?)</Operator>)?(?:.*?<AdditionalDetails>(.*?)</AdditionalDetails>)?",
                resp,
                re.DOTALL,
            )
            if match:
                input_val, output_val, assert_type, operator, additional_details = (
                    match.groups()
                )
                input_val = input_val.strip() if input_val else "unknown input"
                output_val = output_val.strip() if output_val else "unknown output"
                assert_type = assert_type.strip() if assert_type else None
                operator = operator.strip() if operator and operator != "None" else None
                additional_details = (
                    additional_details.strip()
                    if additional_details and additional_details != "None"
                    else None
                )
            else:
                input_val, output_val, assert_type, operator, additional_details = (
                    "unknown input",
                    "unknown output",
                    None,
                    None,
                    None,
                )
        except Exception:
            logger.warning(
                f"Malformed IO response for {batch_data[snippet_idx]['conv_id']}_{testcase_num}: {resp}"
            )
            input_val, output_val, assert_type, operator, additional_details = (
                "unknown input",
                "unknown output",
                None,
                None,
                None,
            )
        for test_case in batch_data[snippet_idx]["test_case_data"]:
            if test_case["testcase_num"] == testcase_num:
                test_case["input_val"] = input_val
                test_case["output_val"] = output_val
                test_case["assert_type"] = assert_type
                test_case["operator"] = operator
                test_case["additional_details"] = additional_details
                break

    # Prepare question prompts and validate test cases
    question_systems = []
    question_prompts = []
    test_case_mappings = {}
    valid_test_cases_by_snippet = [[] for _ in batch_data]
    for snippet_idx, data in enumerate(batch_data):
        if "entrypoint" not in data:
            logger.error(f"Missing entrypoint in data for {data['conv_id']}")
            continue
        valid_test_cases = []
        for test_case in data["test_case_data"]:
            input_val = test_case["input_val"]
            output_val = test_case["output_val"]
            if input_val == "unknown input":
                input_matches = re.findall(
                    rf"{data['entrypoint']}\((.*?)\)", test_case["no_comments"]
                )
                input_val = input_matches[0] if input_matches else "None"
                test_case["input_val"] = input_val
            if output_val == "unknown output":
                try:
                    output_match = re.search(
                        r"Return value:.. (.*)", test_case["raw_trace"]
                    )
                    output_val = (
                        output_match.group(1).strip()
                        if output_match
                        else "unknown output"
                    )
                    test_case["output_val"] = output_val
                except AttributeError:
                    logger.warning(
                        f"Malformed trace for {data['conv_id']}_{test_case['testcase_num']}"
                    )
                    continue
            if input_val == "unknown input" or output_val == "unknown output":
                logger.warning(
                    f"Skipping test case {data['conv_id']}_{test_case['testcase_num']} due to unresolved IO"
                )
                continue
            valid_test_cases.append(test_case)
        if not valid_test_cases:
            logger.warning(f"No valid test cases for {data['conv_id']}")
            continue
        data["test_case_data"] = valid_test_cases
        valid_test_cases_by_snippet[snippet_idx] = valid_test_cases
        for test_case in valid_test_cases:
            input_val = test_case["input_val"]
            output_val = test_case["output_val"]
            assert_type = test_case["assert_type"]
            operator = test_case["operator"]
            additional_details = test_case.get("additional_details", "None")
            raw_trace = test_case["raw_trace"]
            try:
                guideline_template = assert_guidelines.get(
                    assert_type, "Unknown assertion type"
                )
                if assert_type == "comparison" and operator:
                    formatted_guidelines = guideline_template.format(
                        operator=operator or "None",
                        additional_details=additional_details or "None",
                    )
                else:
                    formatted_guidelines = guideline_template.format(
                        additional_details=additional_details or "None"
                    )
            except KeyError as e:
                logger.error(
                    f"Failed to format assert_guidelines for {data['conv_id']}_{test_case['testcase_num']}, missing key: {e}"
                )
                continue
            try:
                question_prompt = user_prompt_templates[0].format(
                    first_function=data["first_function"],
                    raw_trace=raw_trace,
                    input_val=input_val,
                    output_val=output_val,
                    entrypoint=data.get("entrypoint", "unknown_function"),
                    assert_type=assert_type or "unknown",
                    operator=operator or "None",
                    additional_details=additional_details or "None",
                    assert_guidelines=formatted_guidelines,
                    test_case_content=test_case["content"],
                )
            except KeyError as e:
                logger.error(
                    f"Failed to format question prompt for {data['conv_id']}_{test_case['testcase_num']}, missing key: {e}"
                )
                continue
            question_systems.append(system_prompts[0])
            question_prompts.append(question_prompt)
            resp_idx = len(test_case_mappings)
            test_case_mappings[(snippet_idx, test_case["testcase_num"])] = resp_idx

    # Batch question model call
    question_responses = []
    if question_prompts:
        try:
            question_responses = vllm_client.get_model_response(
                question_systems, question_prompts
            )
            if not question_responses or any(r is None for r in question_responses):
                logger.warning(
                    f"Failed to get question responses for batch {batch_idx}"
                )
                continue
        except Exception as e:
            logger.error(
                f"Error getting question responses for batch {batch_idx}: {str(e)}"
            )
            continue

    # Update test_case_data with question responses
    for (snippet_idx, testcase_num), resp_idx in test_case_mappings.items():
        try:
            test_case = next(
                tc
                for tc in valid_test_cases_by_snippet[snippet_idx]
                if tc["testcase_num"] == testcase_num
            )
            resp = question_responses[resp_idx]
            try:
                match = re.match(
                    r".*?<ForwardQuestion>(.*?)</ForwardQuestion>.*?<BackwardQuestion>(.*?)</BackwardQuestion>",
                    resp,
                    re.DOTALL,
                )
                forward_q = (
                    match.group(1).strip()
                    if match
                    else f"What does `{batch_data[snippet_idx].get('entrypoint', 'unknown_function')}` return for {test_case['input_val']}? Show the steps."
                )
                backward_q = (
                    match.group(2).strip()
                    if match
                    else f"What input to `{batch_data[snippet_idx].get('entrypoint', 'unknown_function')}` could give {test_case['output_val']}? Explain how."
                )
            except Exception:
                logger.warning(
                    f"Malformed question response for {batch_data[snippet_idx]['conv_id']}_{testcase_num}"
                )
                forward_q = f"What does `{batch_data[snippet_idx].get('entrypoint', 'unknown_function')}` return for {test_case['input_val']}? Show the steps."
                backward_q = f"What input to `{batch_data[snippet_idx].get('entrypoint', 'unknown_function')}` could give {test_case['output_val']}? Explain how."
            test_case["forward_question"] = forward_q
            test_case["backward_question"] = backward_q
            test_case["question_response"] = resp
        except Exception as e:
            logger.error(
                f"Failed to process test case {batch_data[snippet_idx]['conv_id']}_{testcase_num}: {str(e)}"
            )
            valid_test_cases_by_snippet[snippet_idx] = [
                tc
                for tc in valid_test_cases_by_snippet[snippet_idx]
                if tc["testcase_num"] != testcase_num
            ]
            continue

    # Prepare forward/backward reasoning prompts
    forward_systems = []
    forward_prompts = []
    backward_systems = []
    backward_prompts = []
    for snippet_idx, data in enumerate(batch_data):
        for test_case in valid_test_cases_by_snippet[snippet_idx]:
            if (
                "forward_question" not in test_case
                or "backward_question" not in test_case
            ):
                logger.warning(
                    f"Skipping test case {data['conv_id']}_{test_case['testcase_num']} due to missing forward_question or backward_question"
                )
                continue
            input_val = test_case["input_val"]
            output_val = test_case["output_val"]
            raw_trace = test_case["raw_trace"]
            forward_q = test_case["forward_question"]
            backward_q = test_case["backward_question"]
            try:
                forward_prompt = user_prompt_templates[1].format(
                    first_function=data["first_function"],
                    raw_trace=raw_trace,
                    input_val=input_val,
                    output_val=output_val,
                    forward_question=forward_q,
                )
            except KeyError as e:
                logger.error(
                    f"Failed to format forward prompt for {data['conv_id']}_{test_case['testcase_num']}, missing key: {e}"
                )
                continue
            forward_systems.append(system_prompts[1])
            forward_prompts.append(forward_prompt)
            try:
                backward_prompt = user_prompt_templates[3].format(
                    first_function=data["first_function"],
                    raw_trace=raw_trace,
                    input_val=input_val,
                    output_val=output_val,
                    backward_question=backward_q,
                )
            except KeyError as e:
                logger.error(
                    f"Failed to format backward prompt for {data['conv_id']}_{test_case['testcase_num']}, missing key: {e}"
                )
                continue
            backward_systems.append(system_prompts[3])
            backward_prompts.append(backward_prompt)

    # Sequential forward/backward reasoning calls
    forward_responses = []
    if forward_prompts:
        try:
            forward_responses = vllm_client.get_model_response(
                forward_systems, forward_prompts
            )
            if not forward_responses or any(r is None for r in forward_responses):
                logger.warning(f"Failed to get forward responses for batch {batch_idx}")
        except Exception as e:
            logger.error(
                f"Error getting forward responses for batch {batch_idx}: {str(e)}"
            )
    backward_responses = []
    if backward_prompts:
        try:
            backward_responses = vllm_client.get_model_response(
                backward_systems, backward_prompts
            )
            if not backward_responses or any(r is None for r in backward_responses):
                logger.warning(
                    f"Failed to get backward responses for batch {batch_idx}"
                )
        except Exception as e:
            logger.error(
                f"Error getting backward responses for batch {batch_idx}: {str(e)}"
            )
    if not forward_responses or not backward_responses:
        logger.warning(f"Skipping batch {batch_idx} due to failed reasoning responses")
        continue

    # Prepare feedback prompts
    forward_feedback_systems = []
    forward_feedback_prompts = []
    backward_feedback_systems = []
    backward_feedback_prompts = []
    for (snippet_idx, testcase_num), resp_idx in test_case_mappings.items():
        test_case = next(
            tc
            for tc in valid_test_cases_by_snippet[snippet_idx]
            if tc["testcase_num"] == testcase_num
        )
        forward_resp = forward_responses[resp_idx]
        backward_resp = backward_responses[resp_idx]
        try:
            match = re.match(r".*?<output>(.*?)</output>", forward_resp, re.DOTALL)
            predicted_output = match.group(1).strip() if match else "unknown"
        except Exception:
            logger.warning(
                f"Malformed forward response for {batch_data[snippet_idx]['conv_id']}_{testcase_num}"
            )
            predicted_output = "unknown"
        try:
            match = re.match(r".*?<input>(.*?)</input>", backward_resp, re.DOTALL)
            predicted_input = match.group(1).strip() if match else "unknown"
        except Exception:
            logger.warning(
                f"Malformed backward response for {batch_data[snippet_idx]['conv_id']}_{testcase_num}"
            )
            predicted_input = "unknown"
        try:
            forward_feedback_prompt = user_prompt_templates[2].format(
                first_function=batch_data[snippet_idx]["first_function"],
                raw_trace=test_case["raw_trace"],
                input_val=test_case["input_val"],
                output_val=test_case["output_val"],
                predicted_output_placeholder=predicted_output,
            )
            backward_feedback_prompt = user_prompt_templates[4].format(
                first_function=batch_data[snippet_idx]["first_function"],
                raw_trace=test_case["raw_trace"],
                input_val=test_case["input_val"],
                output_val=test_case["output_val"],
                predicted_input_placeholder=predicted_input,
            )
        except KeyError as e:
            logger.error(
                f"Failed to format feedback prompts for {batch_data[snippet_idx]['conv_id']}_{testcase_num}, missing key: {e}"
            )
            continue
        forward_feedback_systems.append(system_prompts[2])
        forward_feedback_prompts.append(forward_feedback_prompt)
        backward_feedback_systems.append(system_prompts[4])
        backward_feedback_prompts.append(backward_feedback_prompt)

    # Sequential forward/backward feedback calls
    forward_feedback_responses = []
    if forward_feedback_prompts:
        try:
            forward_feedback_responses = vllm_client.get_model_response(
                forward_feedback_systems, forward_feedback_prompts
            )
            if not forward_feedback_responses or any(
                r is None for r in forward_feedback_responses
            ):
                logger.warning(
                    f"Failed to get forward feedback responses for batch {batch_idx}"
                )
        except Exception as e:
            logger.error(
                f"Error getting forward feedback responses for batch {batch_idx}: {str(e)}"
            )
    backward_feedback_responses = []
    if backward_feedback_prompts:
        try:
            backward_feedback_responses = vllm_client.get_model_response(
                backward_feedback_systems, backward_feedback_prompts
            )
            if not backward_feedback_responses or any(
                r is None for r in backward_feedback_responses
            ):
                logger.warning(
                    f"Failed to get backward feedback responses for batch {batch_idx}"
                )
        except Exception as e:
            logger.error(
                f"Error getting backward feedback responses for batch {batch_idx}: {str(e)}"
            )
    if not forward_feedback_responses or not backward_feedback_responses:
        logger.warning(f"Skipping batch {batch_idx} due to failed feedback responses")
        continue

    # Assemble conversations
    for snippet_idx, data in enumerate(batch_data):
        conv_id = data["conv_id"]
        try:
            summary_resp = summary_responses[snippet_idx]
            match = re.match(r".*?<Summary>(.*?)</Summary>", summary_resp, re.DOTALL)
            code_summary = (
                match.group(1).strip()
                if match
                else f"This code performs {data.get('entrypoint', 'unknown_function')}."
            )
        except Exception:
            logger.warning(f"Using fallback summary for {conv_id}")
            code_summary = (
                f"This code performs {data.get('entrypoint', 'unknown_function')}."
            )
        messages = []
        test_cases_dict = {}
        components = {}
        for idx, test_case in enumerate(data["test_case_data"]):
            testcase_num = test_case["testcase_num"]
            test_case_content = test_case["content"]
            input_val = test_case["input_val"]
            output_val = test_case["output_val"]
            assert_type = test_case["assert_type"]
            operator = test_case["operator"]
            try:
                resp_idx = test_case_mappings[(snippet_idx, testcase_num)]
            except KeyError:
                logger.warning(
                    f"Skipping test case {conv_id}_{testcase_num} due to missing mapping"
                )
                continue
            question_resp = test_case["question_response"]
            forward_q = test_case["forward_question"]
            backward_q = test_case["backward_question"]
            forward_resp = forward_responses[resp_idx]
            backward_resp = backward_responses[resp_idx]
            forward_feedback_resp = forward_feedback_responses[resp_idx]
            backward_feedback_resp = backward_feedback_responses[resp_idx]
            try:
                match = re.match(r".*?<output>(.*?)</output>", forward_resp, re.DOTALL)
                predicted_output = match.group(1).strip() if match else "unknown"
                forward_resp_stripped = re.sub(
                    r"<output>.*?</output>", "", forward_resp, flags=re.DOTALL
                ).strip()
            except Exception:
                logger.warning(
                    f"Malformed forward response for {conv_id}_{testcase_num}"
                )
                predicted_output = "unknown"
                forward_resp_stripped = forward_resp.strip()
            try:
                match = re.match(r".*?<input>(.*?)</input>", backward_resp, re.DOTALL)
                predicted_input = match.group(1).strip() if match else "unknown"
                backward_resp_stripped = re.sub(
                    r"<input>.*?</input>", "", backward_resp, flags=re.DOTALL
                ).strip()
            except Exception:
                logger.warning(
                    f"Malformed backward response for {conv_id}_{testcase_num}"
                )
                predicted_input = "unknown"
                backward_resp_stripped = backward_resp.strip()
            test_cases_dict[str(testcase_num)] = test_case_content
            filtered_test_case_data = {
                k: v
                for k, v in test_case.items()
                if k
                not in [
                    "raw_trace",
                    "forward_question",
                    "backward_question",
                    "question_response",
                ]
            }
            components[str(testcase_num)] = {
                "test_case_data": filtered_test_case_data,
                "question_response": question_resp,
                "forward_question": forward_q,
                "backward_question": backward_q,
                "forward_response": forward_resp,
                "backward_response": backward_resp,
                "forward_feedback_response": forward_feedback_resp,
                "backward_feedback_response": backward_feedback_resp,
                "predicted_output": predicted_output,
                "predicted_input": predicted_input,
            }
            if idx == 0:
                messages.append(
                    {
                        "content": sanitize_text(
                            f"{code_summary}\nHere's the code:\n```python\n{data['first_function']}\n```\n{forward_q}"
                        ),
                        "role": "user",
                    }
                )
            else:
                messages.append({"content": sanitize_text(forward_q), "role": "user"})
            messages.append(
                {
                    "content": sanitize_text(
                        f"{forward_resp_stripped}\n<response>Predicted Output: {predicted_output}\nFeedback: {forward_feedback_resp}</response>"
                    ),
                    "role": "assistant",
                }
            )
            messages.append({"content": sanitize_text(backward_q), "role": "user"})
            messages.append(
                {
                    "content": sanitize_text(
                        f"{backward_resp_stripped}\n<response>Predicted Input: {predicted_input}\nFeedback: {backward_feedback_resp}</response>"
                    ),
                    "role": "assistant",
                }
            )
        if not messages:
            logger.warning(f"No valid test cases processed for {conv_id}")
            continue
        conversation = {
            "id": conv_id,
            "instruction": data["instruction"],
            "code": data["first_function"],
            "test_cases": test_cases_dict,
            "messages": messages,
            "components": {
                "main_json": data_lookup.get(
                    (conv_id.split("_")[0], conv_id.split("_")[1]), {}
                ),
                "code_summary": code_summary,
                "test_cases_components": components,
            },
        }
        OUTPUT_BUFFER.append(conversation)
        conversations.append(conversation)
        processed_ids.add(conv_id)
        logger.debug(f"Prepared conversation for {conv_id}")
        if len(OUTPUT_BUFFER) >= OUTPUT_BUFFER_SIZE:
            flush_output_buffer()

# Flush remaining conversations
flush_output_buffer()
logger.info(f"Generated {len(conversations)} conversations total")
