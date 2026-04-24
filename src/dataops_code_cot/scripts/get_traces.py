import ast
import contextlib
import faulthandler
import io
import json
import logging
import multiprocessing
import os
import pickle
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from posixpath import exists
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import typer

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
def extract_test_function_content(test_functions_code: str) -> Tuple[str, str]:
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
            _ = base_indent
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


def extract_definitions(code: str) -> Tuple[List, List, List, Set]:
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


def get_signature_targets(signature_info: Dict[str, Any]) -> List[str]:
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
    function_code: str,
    test_functions_code: str,
    signature_targets: List[str],
    timeout: int,
    signature_info: Dict[str, Any],
) -> str:
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

    # Rename test function to 'check' if it exists
    if test_functions_code.strip():
        test_lines = test_functions_code.split("\n")
        renamed_test_code = []
        for line in test_lines:
            stripped = line.strip()
            if stripped.startswith("def ") and not stripped.startswith("def check("):
                # Replace the function name with 'check'
                indent = len(line) - len(line.lstrip())
                indent_str = " " * indent
                new_line = f"{indent_str}def check():"
                renamed_test_code.append(new_line)
                # Preserve the rest of the function body
                continue
            renamed_test_code.append(line)
        packed_code += "\n" + "\n".join(renamed_test_code) + "\n"
    else:
        # Fallback if no test code is provided
        blank_4 = " " * 4
        blank_8 = " " * 8
        packed_code += f"\ndef check():\n{blank_4}try:\n{blank_8}print('No test body', flush=True)\n{blank_8}return True\n{blank_4}except Exception:\n{blank_8}return False\n"

    packed_code += "\nglobal final_result\nfinal_result = check()\nprint('FINAL_RESULT:', final_result)"
    return packed_code


def process_results_batch(
    results: List[Tuple],
    execution_traces: Dict,
    missing_modules: Dict,
    failed_executions: List,
) -> None:
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


def execute_single_test(args: Tuple) -> None:
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
            trace_content = "Trace file not generated"
    except Exception as e:
        trace_content = f"Failed to read trace file: {e}"

    result_queue.put((key, code_id, num_test, trace_content, result))


def generate_traces(
    data: List[Dict], trace_dir: str, logs_dir: str,
    timeout_duration: int = 5,
    max_workers: int = None,
    batch_size: int = 1000,
) -> Tuple[Dict, Dict]:
    execution_traces = {}
    missing_modules = defaultdict(set)
    failed_executions = []
    TIMEOUT_DURATION = timeout_duration
    MAX_WORKERS = max_workers or min(multiprocessing.cpu_count() * 2, 32)
    BATCH_SIZE = batch_size

    # Calculate total tasks for logging
    total_tasks = sum(
        len(v.get("passing_test_cases", [])) for v in data if isinstance(v, dict)
    )

    # Process data in batches
    for batch_start in range(0, len(data), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(data))
        batch_data = data[batch_start:batch_end]
        logger.info(
            f"Processing batch {batch_start // BATCH_SIZE + 1} (entries {batch_start} to {batch_end - 1})"
        )

        test_cases = []
        manager = multiprocessing.Manager()
        result_queue = manager.Queue()
        processed_count = 0  # Tracks processed tasks across all batches for logging

        # Count processed tasks in earlier batches for accurate global task numbering
        if batch_start > 0:
            for idx in range(batch_start):
                value = data[idx]
                if not isinstance(value, dict) or not all(
                    k in value
                    for k in [
                        "code_snippet",
                        "passing_test_cases",
                        "signature_info",
                        "code_id",
                    ]
                ):
                    continue
                key = value.get("task_id", f"unknown_{idx}")
                code_id = value.get("code_id", "0")
                try:
                    num_test_cases = len(value["passing_test_cases"])
                except TypeError:
                    continue
                for num_test in range(num_test_cases):
                    script_file = os.path.join(
                        trace_dir, f"temp_script_{key}_{code_id}_testcase_{num_test}.py"
                    )
                    trace_file = os.path.join(
                        logs_dir, f"trace_{key}_{code_id}_testcase_{num_test}.log"
                    )
                    if os.path.exists(script_file) or os.path.exists(trace_file):
                        processed_count += 1

        # Generate test cases for the current batch
        for idx, value in enumerate(batch_data, start=batch_start):
            if not isinstance(value, dict) or not all(
                k in value
                for k in [
                    "code_snippet",
                    "passing_test_cases",
                    "signature_info",
                    "code_id",
                ]
            ):
                logger.warning(f"Skipping invalid entry at index {idx}")
                continue
            key = value.get("task_id", f"unknown_{idx}")
            code_id = value.get("code_id", "0")  # Use code_id instead of response_num
            try:
                num_test_cases = len(value["passing_test_cases"])
            except TypeError:
                logger.warning(f"Invalid passing_test_cases for task {key}")
                continue
            for num_test in range(num_test_cases):
                # Check if the task was already processed
                script_file = os.path.join(
                    trace_dir, f"temp_script_{key}_{code_id}_testcase_{num_test}.py"
                )
                trace_file = os.path.join(
                    logs_dir, f"trace_{key}_{code_id}_testcase_{num_test}.log"
                )
                if os.path.exists(script_file) or os.path.exists(trace_file):
                    processed_count += 1
                    logger.debug(
                        f"Skipping already processed task {key}_{code_id}_{num_test}"
                    )
                    continue
                try:
                    pickle.dumps(value)
                    test_cases.append(
                        (
                            trace_dir,
                            logs_dir,
                            value,
                            key,
                            code_id,
                            num_test,
                            TIMEOUT_DURATION,
                            result_queue,
                        )
                    )
                except Exception as e:
                    logger.warning(
                        f"Task {key}_{code_id}_{num_test}: Unpicklable data, skipping: {e}"
                    )
                    failed_executions.append(
                        (f"temp_script_{key}_{code_id}_testcase_{num_test}.py", str(e))
                    )
                    continue

        logger.info(
            f"Batch {batch_start // BATCH_SIZE + 1}: Submitting {len(test_cases)} new execution tasks"
        )

        try:
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = []
                for idx, args in enumerate(test_cases):
                    try:
                        future = executor.submit(execute_single_test, args)
                        futures.append(future)
                    except Exception as e:
                        logger.error(f"Failed to submit task {idx}: {e}")
                        continue

                results = []
                for idx, future in enumerate(as_completed(futures)):
                    try:
                        future.result()
                        while not result_queue.empty():
                            results.append(result_queue.get())
                    except Exception as e:
                        logger.warning(f"Task {idx} failed: {e}")
                    # Log global task number
                    global_task_num = processed_count + idx + 1
                    logger.info(
                        f"[{global_task_num}/{total_tasks}] execution completed"
                    )
        except Exception as e:
            logger.error(f"Batch {batch_start // BATCH_SIZE + 1}: Executor failed: {e}")

        logger.info(f"Batch {batch_start // BATCH_SIZE + 1}: Processing results")
        process_results_batch(
            results, execution_traces, missing_modules, failed_executions
        )

    # Write failed executions after all batches
    with open("failed_executions.txt", "w") as f:
        if failed_executions:
            f.write("Files where execution failed:\n")
            for filename, error in failed_executions:
                f.write(f"{filename}: {error}\n")
        else:
            f.write("No execution failures detected.\n")

    if missing_modules:
        logger.info("\n=== Missing Modules Summary ===")
        all_missing_modules = set()
        for key, modules in missing_modules.items():
            if modules:
                logger.info(f"Key {key}:")
                for module in modules:
                    logger.info(f"  - {module}")
                    all_missing_modules.add(module)
        if all_missing_modules:
            logger.info("\nTo install manually, use:")
            logger.info("pip install " + " ".join(all_missing_modules))

    logger.info("Execution completed")
    return execution_traces, missing_modules


def process_file(
    input_file: str = typer.Option(..., "--input", "-i", help="Input file path"),
    output_dir: str = typer.Option(..., "--output_dir", "-o", help="Output file path"),
    config: str = typer.Option("pipeline_config.yaml", "--config", help="Path to pipeline_config.yaml"),
) -> None:
    # Create output dir if it doesn't exist
    import pathlib

    output_dir_path = pathlib.Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir_path / "data/test_code_1"
    logs_dir = output_dir_path / "data/pysnooper_trace_uncleaned_v1"
    [dir.mkdir(parents=True, exist_ok=True) for dir in [trace_dir, logs_dir]]
    # Load config
    _timeout, _max_workers, _batch_size = 5, None, 1000
    import pathlib as _pl
    if _pl.Path(config).exists():
        try:
            import yaml as _yaml
            _c = (_yaml.safe_load(_pl.Path(config).read_text()) or {}).get("stage_c", {})
            _timeout     = _c.get("trace_timeout", _timeout)
            _max_workers = _c.get("max_workers",   _max_workers)
            _batch_size  = _c.get("batch_size",    _batch_size)
        except Exception:
            pass

    logger.info("Starting execution with optimized ProcessPoolExecutor...")
    start_time = time.time()
    try:
        with open(input_file, "r") as f:
            data = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(data)} JSONL entries")
    except Exception as e:
        logger.error(f"Failed to load JSONL: {e}")
        return {}, {}

    execution_traces, missing_modules = generate_traces(
        data, trace_dir, logs_dir,
        timeout_duration=_timeout,
        max_workers=_max_workers,
        batch_size=_batch_size,
    )
    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")
    logger.info("Done")


def main_() -> None:
    app = typer.Typer()
    app.command()(process_file)
    app()


if __name__ == "__main__":
    main_()
