# Standard
import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import re
import signal
import tempfile
from typing import Dict, Optional

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


def extract_test_function_content(test_functions_code):
    """
    Extracts the main test function body while preserving indentation.

    - If multiple functions exist, keep only the main test function's body inside `check()`
    - Other functions remain outside `check()`

    Args:
        test_functions_code (str): The test functions as a string.

    Returns:
        (str, str): Tuple containing (modified test functions, extracted test function body).
    """
    # print("extract_test_function_content")
    # print(test_functions_code)
    function_pattern = re.finditer(
        r"^\s*def (\w+)\(.*\):", test_functions_code, re.MULTILINE
    )
    function_bodies = {}
    main_test_function = None
    extracted_body = ""

    lines = test_functions_code.split("\n")
    function_start_lines = {match.group(1): match.start() for match in function_pattern}

    # Extract function contents
    current_function = None
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("def "):  # New function detected
            func_name = stripped_line.split("(")[0].split()[1]
            current_function = func_name
            function_bodies[current_function] = []
        if current_function:
            function_bodies[current_function].append(line)

    # Detect the main test function (the one containing assertions)
    for func_name, body_lines in function_bodies.items():
        if any("assert " in line for line in body_lines):  # Contains assertion
            main_test_function = func_name
            extracted_body = "\n".join(body_lines[1:])  # Exclude function signature
            break

    # Remove indentation for extracted body
    if extracted_body:
        body_lines = extracted_body.split("\n")
        min_indent = min(
            len(line) - len(line.lstrip()) for line in body_lines if line.strip()
        )
        extracted_body = "\n".join(line[min_indent:] for line in body_lines)

    # Remove the extracted function from test_functions_code
    updated_test_functions = "\n".join(
        "\n".join(body) if name != main_test_function else ""
        for name, body in function_bodies.items()
    ).strip()

    return updated_test_functions, extracted_body


def _pack_test_cases(function_code, test_functions_code, timeout):
    """
    Packs function definition, test functions, and sets the entry point dynamically.

    - Extracts and includes only the main test function inside `check()`
    - Other test functions remain outside `check()`

    Args:
        function_code (str): The function definition to be tested.
        test_functions_code (str): The test functions containing assertions.
        timeout (int): Timeout for execution.

    Returns:
        str: The packed code ready for execution.
    """
    updated_test_functions, extracted_test_body = extract_test_function_content(
        test_functions_code
    )

    blank_4 = " " * 4
    blank_8 = " " * 8
    blank_12 = " " * 12
    packed_code = f"{function_code}\n\n{updated_test_functions}\n\n"

    # Wrap extracted test body inside `check()`
    if extracted_test_body:
        packed_code += (
            f"def check():\n{blank_4}try:\n{blank_8}with time_limit({timeout}):\n"
        )
        packed_code += "\n".join(
            f"{blank_12}{line}" for line in extracted_test_body.split("\n")
        )
        packed_code += f"\n{blank_8}return True\n{blank_4}except Exception:\n{blank_8}return False\n"

        # Execute `check()`
        packed_code += "\n\nglobal final_result\nfinal_result = check()"

    # print(packed_code)
    return packed_code


def check_correctness_with_test_cases(
    task_id, prompt, function_code, test_functions_code, timeout
):
    """
    Evaluates the functional correctness of a solution_content by running the test
    suite provided in the problem.
    """
    extend_timeout = timeout * 2

    def unsafe_execute(result_queue):
        try:
            with create_tempdir():
                # These system calls are needed when cleaning up tempdir.
                # Standard
                import os
                import shutil

                rmtree = shutil.rmtree
                rmdir = os.rmdir
                chdir = os.chdir

                # Disable functionalities that can make destructive changes to the test.
                reliability_guard()

                result = []
                # Run each test function separately
                for test_code in test_functions_code:
                    check_program = _pack_test_cases(function_code, test_code, timeout)

                    try:
                        exec_globals = {"time_limit": time_limit}
                        with swallow_io():
                            exec(check_program, exec_globals)
                        result.append(exec_globals["final_result"])

                    except AssertionError:
                        result.append(False)  # Test failed but executed
                    except TimeoutException:
                        result.append("timed out")
                    except Exception as e:
                        result.append(f"failed: {str(e)}")

                # Cleanup
                shutil.rmtree = rmtree
                os.rmdir = rmdir
                os.chdir = chdir

                result_queue.put(result)
        except Exception as e:
            result_queue.put([f"failed: execution error: {e}"] * len(test_functions_code))

    # Use fork context so local functions can be passed (macOS defaults to spawn)
    ctx = multiprocessing.get_context("fork")
    result_queue = ctx.Queue()

    try:
        p = ctx.Process(target=unsafe_execute, args=(result_queue,))
        p.start()
        p.join(timeout=extend_timeout)

        if p.is_alive():
            os.kill(p.pid, signal.SIGTERM)
            p.join(timeout=1.0)
            if p.is_alive():
                os.kill(p.pid, signal.SIGKILL)
            result = ["timed out"] * len(test_functions_code)
        else:
            try:
                result = result_queue.get(timeout=1.0)
            except Exception:
                result = ["failed: result retrieval error"] * len(test_functions_code)
    except Exception as _exc:
        result = [f"failed: process error: {_exc}"] * len(test_functions_code)
    finally:
        if p.is_alive():
            try:
                os.kill(p.pid, signal.SIGKILL)
            except:
                pass
        try:
            result_queue.close()
        except:
            pass

    if not result:
        result = ["timed out"] * len(test_functions_code)

    # Only include test cases where result is True
    test_cases_passed = [
        test_case for test_case, res in zip(test_functions_code, result) if res is True
    ]

    result_dict = dict(
        task_id=task_id,
        test_cases=test_functions_code,
        completion=function_code,
        passed=(type(result) == list) and len(result) > 0 and any(result),
        result=result,
        test_cases_passed=test_cases_passed,
    )

    return result_dict


def check_correctness(
    task_id: str,
    prompt: str,
    completion: str,
    test: str,
    entry_point: str,
    timeout: float,
) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem.
    """

    def unsafe_execute():
        with create_tempdir():
            # These system calls are needed when cleaning up tempdir.
            # Standard
            import os
            import shutil

            rmtree = shutil.rmtree
            rmdir = os.rmdir
            chdir = os.chdir

            # Disable functionalities that can make destructive changes to the test.
            reliability_guard()

            # Construct the check program and run it.
            check_program = (
                prompt + completion + "\n" + test + "\n" + f"check({entry_point})"
            )

            try:
                exec_globals = {}
                with swallow_io():
                    with time_limit(timeout):
                        exec(check_program, exec_globals)
                result.append("passed")
            except TimeoutException:
                result.append("timed out")
            except BaseException as e:
                result.append(f"failed: {e}")

            # Needed for cleaning up.
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir

    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(target=unsafe_execute)
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()

    if not result:
        result.append("timed out")

    return dict(
        task_id=task_id,
        passed=result[0] == "passed",
        result=result[0],
        completion=completion,
    )


@contextlib.contextmanager
def time_limit(seconds: float):
    # signal.ITIMER_REAL / SIGALRM are not available on Windows or in subprocesses on macOS
    try:
        def signal_handler(signum, frame):
            raise TimeoutException("Timed out!")
        signal.setitimer(signal.ITIMER_REAL, seconds)
        signal.signal(signal.SIGALRM, signal_handler)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    except (AttributeError, OSError):
        # Fallback: no timeout enforcement — just yield
        yield


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


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
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
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        # Standard
        import resource

        resource.setrlimit(
            resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
        )
        if not platform.uname().system == "Darwin":
            resource.setrlimit(
                resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
            )

    faulthandler.disable()

    # Standard
    import builtins

    builtins.exit = None
    builtins.quit = None

    # Standard
    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    # Standard
    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    # Standard
    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    # Standard
    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None
