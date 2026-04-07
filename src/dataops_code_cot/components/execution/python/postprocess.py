# Standard
from collections import defaultdict

# Local
from dataops_code_cot.components.execution.python.io_utils import Tools

STOP_TOKEN = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]


class PostProcessor:
    @staticmethod
    def map_task_id_for_solution(source_list):
        result = []
        tasks = Tools.load_tasks(source_list)
        for task in tasks:
            for sample in task.get("responses", []):
                result.append(
                    {
                        "task_id": task["task_id"],
                        "prompt": task["instruction"],
                        "completion": sample,
                    }
                )
        return result

    @staticmethod
    def map_task_id_for_test_case(source_path):
        test_cases_by_task = defaultdict(list)
        tasks = Tools.load_tasks(source_path)
        for task in tasks:
            for test in task.get("tests_split", []):
                test_cases_by_task[task["task_id"]].append(test)

        return test_cases_by_task

    def test_case_extract_concepts(content):
        def _truncate(content):
            # Remove only comments, preserving commas and other syntax within assert statements
            return content.split("#")[0].strip()

        # Extract assert statements, ignoring comments
        split_by_assert = [
            f"assert {part}".strip()
            for part in f"assert {content}".split("assert ")
            if len(part.strip()) > 0 and not part.strip().startswith("#")
        ]

        # Apply _truncate to each assert statement to remove inline comments only
        truncated_test_cases = [_truncate(i) for i in split_by_assert]
        checked_assertions = [
            i
            for i in truncated_test_cases
            if i
            and not i.startswith("#")
            and PostProcessor._check_test_case_validation(i)
        ]
        # print(checked_assertions)
        return checked_assertions

    @staticmethod
    def test_case_extract(content):
        def _truncate(content):
            for identifier in STOP_TOKEN:
                if identifier in content:
                    content = content.split(identifier)[0]
            return content.strip()

        split_by_assert = [
            f"assert {part}".strip()
            for part in f"assert {content}".split("assert ")
            if len(part.strip()) > 0
        ]
        truncated_test_cases = [_truncate(i) for i in split_by_assert]
        checked_assertions = [
            i
            for i in truncated_test_cases
            if PostProcessor._check_test_case_validation(i)
        ]

        return checked_assertions

    @staticmethod
    def _check_test_case_validation(test_case):
        if len(test_case.strip()) < 1:
            return False
        if "assert" not in test_case:
            return False
        try:
            multi_line_test_case = test_case.replace("\n", "\n    ")
            assert_in_a_block = f"try:\n    {multi_line_test_case}\nexcept:\n    pass\n"
            compile(assert_in_a_block, "", "exec")
            return True
        except Exception:
            return False
