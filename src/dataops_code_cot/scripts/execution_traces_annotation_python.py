import ast
import contextlib
import faulthandler
import io
import logging
import multiprocessing
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from functools import partial
from typing import Optional

import pandas as pd

logging.basicConfig(
    format="SystemLog: [%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


# Utilities (unchanged)
class TimeoutException(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class WriteOnlyStringIO(io.StringIO):
    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        return False


class redirect_stdin(contextlib._RedirectStream):
    _stream = "stdin"


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def reliability_guard(maximum_memory_bytes: Optional[int] = 2 * 10**9):
    import os

    original_rmtree = shutil.rmtree
    original_rmdir = os.rmdir
    original_chdir = os.chdir
    if maximum_memory_bytes is not None and platform.uname().system != "Darwin":
        try:
            import resource

            for rlimit in [resource.RLIMIT_AS, resource.RLIMIT_DATA]:
                try:
                    current_soft, current_hard = resource.getrlimit(rlimit)
                    if (
                        current_hard == resource.RLIM_INFINITY
                        or maximum_memory_bytes <= current_hard
                    ):
                        resource.setrlimit(rlimit, (maximum_memory_bytes, current_hard))
                    else:
                        logger.debug(
                            f"Skipping {rlimit} limit: requested {maximum_memory_bytes} exceeds hard limit {current_hard}"
                        )
                except (ValueError, resource.error) as e:
                    logger.debug(f"Failed to set {rlimit} limit: {e}")
                    continue
        except ImportError:
            logger.warning("resource module unavailable; skipping resource limits")
    elif platform.uname().system == "Darwin":
        logger.info("Skipping resource limits on macOS due to system restrictions")
    faulthandler.disable()
    import builtins

    builtins.exit = None
    builtins.quit = None
    os.environ["OMP_NUM_THREADS"] = "1"
    dangerous_os_functions = [
        "system",
        "remove",
        "removedirs",
        "fchdir",
        "setuid",
        "fork",
        "forkpty",
        "killpg",
        "rename",
        "renames",
        "truncate",
        "replace",
        "unlink",
        "fchmod",
        "fchown",
        "chmod",
        "chown",
        "chroot",
        "lchflags",
        "lchmod",
        "lchown",
    ]
    for func in dangerous_os_functions:
        setattr(os, func, None)
    shutil.move = None
    shutil.chown = None
    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None
    try:
        yield
    finally:
        shutil.rmtree = original_rmtree
        os.rmdir = original_rmdir
        os.chdir = original_chdir


# Unchanged functions (included for completeness)
def extract_test_function_content(test_functions_code):
    lines = [
        line
        for line in test_functions_code.split("\n")
        if line.strip() and not line.strip().startswith("#")
    ]
    test_body = []
    in_function = False
    base_indent = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def ") and not in_function:
            in_function = True
            base_indent = len(line) - len(line.lstrip())
            continue
        if in_function:
            test_body.append(line)
    if not test_body:
        logger.warning("No test function body found")
        return "", ""
    min_indent = min(
        len(line) - len(line.lstrip()) for line in test_body if line.strip()
    )
    extracted_body = "\n".join(line[min_indent:] for line in test_body if line.strip())
    return "", extracted_body


def extract_definitions(code):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        logger.error(f"SyntaxError in function_code: {e}")
        return None, None, None, None
    lines = code.split("\n")
    imports = []
    class_ranges = []
    func_ranges = []
    defined_names = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append((node.lineno - 1, lines[node.lineno - 1].rstrip()))
        elif isinstance(node, ast.ClassDef):
            class_ranges.append((node.lineno - 1, node.end_lineno, node.name))
            defined_names.add(node.name)
            for subnode in node.body:
                if isinstance(subnode, ast.FunctionDef):
                    defined_names.add(subnode.name)
        elif isinstance(node, ast.FunctionDef):
            func_ranges.append((node.lineno - 1, node.end_lineno, node.name))
            defined_names.add(node.name)
    return imports, class_ranges, func_ranges, defined_names


def get_signature_targets(signature_info):
    if not signature_info:
        return []
    sig_type = signature_info.get("type")
    if sig_type == "function":
        name = signature_info.get("name")
        return [name] if name else []
    elif sig_type == "class":
        methods = signature_info.get("methods", [])
        return [method["name"] for method in methods if method["name"] != "__init__"]
    return []


def pack_test_cases(
    function_code, test_functions_code, signature_targets, timeout, signature_info
):
    updated_test_functions, extracted_test_body = extract_test_function_content(
        test_functions_code
    )
    packed_code = ""
    lines = function_code.split("\n")
    sig_type = signature_info.get("type", "")
    if sig_type == "function":
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("def ") and any(
                target in stripped for target in signature_targets
            ):
                packed_code += f"@pysnooper.snoop(output=trace_file, depth=1)\n{line}\n"
            else:
                packed_code += f"{line}\n"
    elif sig_type == "class":
        in_class = False
        class_indent = 0
        for line in lines:
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            if stripped.startswith("class "):
                in_class = True
                class_indent = indent
                packed_code += f"{line}\n"
                continue
            if stripped.startswith("def ") and any(
                target in stripped for target in signature_targets
            ):
                indent_str = " " * indent
                packed_code += f"{indent_str}@pysnooper.snoop(output=trace_file, depth=1)\n{line}\n"
            else:
                packed_code += f"{line}\n"
            if (
                in_class
                and indent <= class_indent
                and stripped
                and not stripped.startswith("def ")
            ):
                in_class = False
    else:
        packed_code += function_code + "\n"
    packed_code += f"\n{updated_test_functions}\n"
    blank_4 = " " * 4
    blank_8 = " " * 8
    if extracted_test_body.strip():
        packed_code += f"def check():\n{blank_4}try:\n"
        for line in extracted_test_body.split("\n"):
            if line.strip():
                packed_code += f"{blank_8}{line.rstrip()}\n"
        packed_code += f"{blank_8}return True\n{blank_4}except Exception as e:\n{blank_8}print('Error: {{e}}', flush=True)\n{blank_8}return False\n"
    else:
        packed_code += f"def check():\n{blank_4}try:\n{blank_8}print('No test body', flush=True)\n{blank_8}return True\n{blank_4}except Exception:\n{blank_8}return False\n"
    packed_code += "\nglobal final_result\nfinal_result = check()\nprint('FINAL_RESULT:', final_result)"
    return packed_code


def process_results_batch(
    results, execution_traces, missing_modules, failed_executions
):
    for key, response_num, num_test, trace_content, result in results:
        if key not in execution_traces:
            execution_traces[key] = {}
        if response_num not in execution_traces[key]:
            execution_traces[key][response_num] = {}
        execution_traces[key][response_num][num_test] = result
        if isinstance(trace_content, str) and (
            "Error while executing" in trace_content
            or "timed out" in trace_content
            or "SyntaxError" in trace_content
        ):
            filename = (
                f"temp_script_{key}_response_{response_num}_testcase_{num_test}.py"
            )
            failed_executions.append((filename, trace_content))
        if isinstance(trace_content, str) and "ModuleNotFoundError" in trace_content:
            module_match = re.search(
                r"ModuleNotFoundError: No module named '([^']+)'", trace_content
            )
            if module_match:
                missing_modules[key].add(module_match.group(1))


def execute_single_test(args):
    """
    Execute a single code-test pair in an isolated process with pysnooper tracing.
    """
    try:
        trace_dir, logs_dir, value, key, code_id, num_test, timeout, result_queue = args
    except Exception as e:
        error_msg = f"Failed to unpack args: {e}"
        result_queue.put((None, None, None, error_msg, False))
        return

    trace_file = os.path.join(
        logs_dir, f"trace_{key}_{code_id}_testcase_{num_test}.log"
    )
    temp_script = os.path.join(
        trace_dir, f"temp_script_{key}_{code_id}_testcase_{num_test}.py"
    )

    # Ensure directories exist
    try:
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(trace_dir, exist_ok=True)
    except Exception as e:
        error_msg = f"Failed to create directories: {e}"
        result_queue.put((key, code_id, num_test, error_msg, False))
        return

    trace_content = "No trace generated"
    result = False

    # Save .py file early
    try:
        with open(temp_script, "w") as script_file:
            script_file.write("# Placeholder\n")
    except Exception as e:
        trace_content = f"Failed to create placeholder .py file: {e}"
        result_queue.put((key, code_id, num_test, trace_content, result))
        return

    try:
        with create_tempdir():
            with reliability_guard():
                # Validate inputs
                response_code = value.get("code_snippet")
                test_code = value.get("passing_test_cases", [None])[num_test]
                signature_info = value.get("signature_info", {})

                # Handle code_snippet as a list with one string
                if (
                    not isinstance(response_code, list)
                    or len(response_code) != 1
                    or not isinstance(response_code[0], str)
                ):
                    trace_content = f"Invalid code_snippet: expected list with one string, got {type(response_code)}"
                    logger.error(f"Task {key}_{code_id}_{num_test}: {trace_content}")
                    result_queue.put((key, code_id, num_test, trace_content, result))
                    return
                response_code_str = response_code[0]

                # Validate test_code as a string
                if not isinstance(test_code, str):
                    trace_content = (
                        f"Invalid test_code: expected string, got {type(test_code)}"
                    )
                    logger.error(f"Task {key}_{code_id}_{num_test}: {trace_content}")
                    result_queue.put((key, code_id, num_test, trace_content, result))
                    return

                if not response_code_str or not test_code:
                    trace_content = "Missing code_snippet or test_code content"
                    result_queue.put((key, code_id, num_test, trace_content, result))
                    return

                # Generate signature targets
                signature_targets = get_signature_targets(signature_info)

                if not signature_targets:
                    trace_content = "No valid signature_info provided"
                    result_queue.put((key, code_id, num_test, trace_content, result))
                    return

                # Generate combined code
                try:
                    packed_code = pack_test_cases(
                        response_code_str,
                        test_code,
                        signature_targets,
                        timeout,
                        signature_info,
                    )
                except Exception as e:
                    trace_content = f"Failed in pack_test_cases: {e}"
                    logger.error(f"Task {key}_{code_id}_{num_test}: {trace_content}")
                    result_queue.put((key, code_id, num_test, trace_content, result))
                    return

                combined_code = (
                    f"import pysnooper\ntrace_file = {repr(trace_file)}\n{packed_code}"
                )

                # Save .py file with actual code
                try:
                    with open(temp_script, "w") as script_file:
                        script_file.write(combined_code)
                except Exception as e:
                    trace_content = f"Failed to save .py file: {e}"
                    logger.error(f"Task {key}_{code_id}_{num_test}: {trace_content}")
                    result_queue.put((key, code_id, num_test, trace_content, result))
                    return

                # Execute the .py file using subprocess
                try:
                    with swallow_io():
                        process = subprocess.run(
                            [sys.executable, temp_script],
                            capture_output=True,
                            text=True,
                            timeout=timeout,
                        )
                    stdout = process.stdout
                    stderr = process.stderr

                    # Parse final_result from stdout
                    for line in stdout.splitlines():
                        if line.startswith("FINAL_RESULT:"):
                            try:
                                result = ast.literal_eval(
                                    line.split("FINAL_RESULT:")[1].strip()
                                )
                            except (ValueError, SyntaxError):
                                result = False
                            break
                    else:
                        result = False

                    # Handle execution outcomes
                    if process.returncode != 0:
                        trace_content = f"Subprocess error (returncode {process.returncode}): {stderr}"
                    elif stderr:
                        trace_content = f"Execution completed with stderr: {stderr}"
                    else:
                        trace_content = f"Execution completed, result={result}"

                except subprocess.TimeoutExpired as e:
                    trace_content = (
                        f"Execution timed out: stdout={e.stdout}, stderr={e.stderr}"
                    )
                    result = False
                except TypeError as e:
                    trace_content = (
                        f"TypeError in subprocess: {e}\n{traceback.format_exc()}"
                    )
                    result = False
                except subprocess.SubprocessError as e:
                    trace_content = f"Subprocess error: {e}\n{traceback.format_exc()}"
                    result = False
                except Exception as e:
                    trace_content = f"Unexpected error: {e}\n{traceback.format_exc()}"
                    result = False

    except Exception as e:
        tb = traceback.format_exc()
        trace_content = f"Setup error: {e}\n{tb}"
        result = False

    # Read the trace file content
    try:
        if os.path.exists(trace_file):
            with open(trace_file, "r") as f:
                trace_content = f.read()
        else:
            trace_content_ = f"Trace file not generated: {trace_content}"
            trace_content = trace_content_
    except Exception as e:
        trace_content = f"Failed to read trace file: {e}"

    result_queue.put((key, code_id, num_test, trace_content, result))


def sd_load_input_data(file_name):
    # Load JSONL
    df = pd.read_json(file_name, lines=True)
    return df


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


def process_row(row, result_queue):
    """
    Generates traces for code and test. It assumes that there is
    only single test.

        code: row["components"]["main_json"]["code_snippet"]
        test:

    It adds two new columns
       traces: execution traces.
       select_trace: boolean

    """
    # execute test case in the row
    key = row["index_id"]
    # prepare a value dict which has fields
    # code_snippet, , passing_test_cases, signature_info, num_test
    num_test = 0
    components = row["components"]
    signature_info = components["main_json"]["signature_info"]
    code = components["main_json"]["code_snippet"]
    for k, v in components["test_cases_components"].items():
        test = v["test_case_data"]["content"]
        break
    value = {
        "code_snippet": code,
        "passing_test_cases": {num_test: test},
        "signature_info": signature_info,
    }
    trace_dir = os.path.abspath("data/test_code_1")
    logs_dir = os.path.abspath("data/pysnooper_trace_uncleaned_v1")

    code_id = 0  # random.randint(1, 1000_000)
    TIMEOUT_DURATION = 5
    # trace_dir, logs_dir, value, key, code_id, num_test, TIMEOUT_DURATION, result_queue
    args = (
        trace_dir,
        logs_dir,
        value,
        key,
        code_id,
        num_test,
        TIMEOUT_DURATION,
        result_queue,
    )
    # print(args)
    trace_file = os.path.join(
        logs_dir, f"trace_{key}_{code_id}_testcase_{num_test}.log"
    )
    execute_single_test(args)

    try:
        with open(trace_file) as f:
            traces = f.read()
            row["traces"] = clean_trace_content(traces)
            row["select_trace"] = True
    except:
        row["traces"] = ""
        row["select_trace"] = False

    return row


def main():
    logger.info("Starting execution")

    # Create multiprocessing manager queue
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()

    import sys

    input_file, output_file = sys.argv[1:3]
    print(f"Input File: {input_file}")
    print(f"Output File: {output_file}")

    file_name = input_file
    data_df = sd_load_input_data(file_name)  # data is a list of jsons

    trace_dir = os.path.abspath("data/test_code_1")
    logs_dir = os.path.abspath("data/pysnooper_trace_uncleaned_v1")

    os.makedirs(trace_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    execution_traces = {}
    missing_modules = defaultdict(set)
    failed_executions = []
    TIMEOUT_DURATION = 5
    MAX_WORKERS = min(multiprocessing.cpu_count() * 2, 32)
    BATCH_SIZE = 1000  # Process 1000 JSONL entries at a time

    from pandarallel import pandarallel

    pandarallel.initialize()
    # Create a partial function tomapplu
    process_row_ = partial(process_row, result_queue=result_queue)
    data_df["index_id"] = [i for i in range(len(data_df))]
    df_out = data_df.parallel_apply(process_row_, axis=1)
    # df_out.to_json(output_file, orient="records", lines=True)
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    # data = [f for f in result_queue.get()]
    trace_list = {"index_id": [], "traces": []}
    for f in results:
        trace_list["index_id"].append(f[0])
        trace_list["traces"].append(f[3])
    print("traces len", len(trace_list["traces"]))
    print("df len", df_out)
    df_sorted = df_out.sort_values(by="index_id")
    traces_df = pd.DataFrame(trace_list).sort_values(by="index_id")
    df_sorted["traces"] = traces_df["traces"].to_list()
    df_sorted.to_json(output_file, orient="records", lines=True)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")
