# Standard
import json
import logging
import os

# Local
from dataops_code_cot.components.execution.python.agreement import (
    DataManager,
    DualAgreement,
)
from dataops_code_cot.components.execution.python.execution import (
    evaluate_with_test_cases,
)
from dataops_code_cot.components.execution.python.postprocess import PostProcessor


logger = logging.getLogger(__name__)


def save_detailed_results(
    dual_exec_result,
    ranked_result,
    solution_test_cases,
    data_manager,
    save_path,
    cache_dir,
    handled_solutions,
):
    # Create a mapping of task_id to instruction from handled_solutions
    task_instructions = {}
    task_instructions = {
        solution["task_id"]: solution["prompt"] for solution in handled_solutions
    }

    detailed_results_passed = []
    # Create a mapping of task_id and completion to their scores
    scores_map = {}
    for task_id, solutions in ranked_result.items():
        for solution_group, score in solutions:
            solutions = (
                solution_group if isinstance(solution_group, list) else [solution_group]
            )
            unique_solutions = list(set(solutions))
            for solution in unique_solutions:
                scores_map[(task_id, solution)] = score

                test_case_info = solution_test_cases[task_id][solution]
                entry = {
                    "task_id": task_id,
                    "instruction": task_instructions[task_id],  # Add instruction
                    "code_snippet": [solution],
                    "score": score,
                    "passing_test_cases": test_case_info["test_cases"],
                    "solution_group_size": len(unique_solutions),
                }
                detailed_results_passed.append(entry)

    detailed_results = []
    # Combine dual execution results with scores and passing test cases
    for result in dual_exec_result:
        task_id = result["task_id"]
        completion = result["completion"]
        score = scores_map.get(
            (task_id, completion), 0
        )  # default score 0 if not ranked

        # Get passing test cases for this solution
        passing_pairs = data_manager.passed_solution_test_case_pairs_by_task[task_id]
        passing_test_cases = [
            test_case for sol, test_case in passing_pairs if sol == completion
        ]

        # Get execution results only for passing test cases
        passing_results = []
        for idx, test_case in enumerate(result["test_cases"]):
            if test_case in passing_test_cases:
                passing_results.append(result["result"][idx])

        if passing_test_cases:
            entry = {
                "instruction": task_instructions[task_id],  # Add instruction
                "code": completion,
                "test_cases": passing_test_cases,
            }
            detailed_results.append(entry)

    # Save the detailed results
    results_path = os.path.join(cache_dir, save_path)
    with open(results_path, "w") as f:
        for entry in detailed_results:
            f.write(json.dumps(entry) + "\n")

    passed_results_path = os.path.join(
        cache_dir, save_path.split(".")[0] + "_passed.json"
    )
    with open(passed_results_path, "w") as f:
        for entry in detailed_results_passed:
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Detailed results saved to {results_path}")
    return detailed_results


def load_signature_info(jsonl_path):
    """
    Load signature_info and additional fields from a JSONL file, mapping task_id to a dictionary of fields.
    """
    task_data_map = {}
    try:
        with open(jsonl_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    task_id = data.get("task_id")
                    if task_id is not None:
                        task_data_map[task_id] = {
                            "signature_info": data.get("signature_info", None),
                            "difficulty": data.get("difficulty", None),
                            "signature_details": data.get("signature_details", None),
                            "primary_method": data.get("primary_method", None),
                            "signature_type": data.get("signature_type", None),
                            "required_tests_str": data.get("required_tests_str", None),
                            "concept_description": data.get(
                                "concept_description", None
                            ),
                            "concept": data.get("concept", None),
                            "concept_examples": data.get("concept_examples", None),
                            "concept_id": data.get("concept_id", None),
                        }
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to parse JSON line: {line.strip()[:50]}... Error: {e}"
                    )
                except AttributeError as e:
                    logger.warning(
                        f"Invalid data in line: {line.strip()[:50]}... Error: {e}"
                    )
    except FileNotFoundError:
        logger.error(f"Signature info file not found: {jsonl_path}")
        return {}
    except Exception as e:
        logger.error(f"Error reading signature info file {jsonl_path}: {e}")
        return {}
    return task_data_map


def save_solution_test_cases(
    solution_test_cases,
    handled_solutions,
    source_path,
    cache_dir,
    save_path_prefix="solution_test_cases",
):
    """
    Save one entry per solution from solution_test_cases with task_id, code_id, instruction, code snippet, test cases, score, and additional fields from JSON.
    Only create entries if the test_cases field is non-empty.
    """
    # Create a mapping of task_id to instruction from handled_solutions
    task_instructions = {
        solution["task_id"]: solution["prompt"] for solution in handled_solutions
    }

    # Load signature_info and additional fields from the JSONL file
    jsonl_path = source_path
    task_data_map = load_signature_info(jsonl_path)
    if not task_data_map:
        logger.warning(
            "No data loaded from JSON; proceeding with empty fields for all entries"
        )

    entries = []
    code_id_counter = 0  # Counter for generating unique code_id
    for task_id in solution_test_cases:
        for solution, info in solution_test_cases[task_id].items():
            # Check if test_cases is non-empty
            if not info["test_cases"]:
                logger.debug(
                    f"Skipping entry for task_id {task_id} due to empty test_cases for solution"
                )
                continue

            # Get JSON entry for task_id
            json_entry = task_data_map.get(task_id, {})
            entry = {
                "task_id": task_id,
                "code_id": str(code_id_counter),  # Unique code_id for each solution
                "instruction": task_instructions.get(task_id, ""),
                "code_snippet": [solution],  # Store as list for consistency
                "passing_test_cases": info["test_cases"],
                "score": info["score"],
                "signature_info": json_entry.get("signature_info", None),
                "difficulty": json_entry.get("difficulty", None),
                "signature_details": json_entry.get("signature_details", None),
                "primary_method": json_entry.get("primary_method", None),
                "signature_type": json_entry.get("signature_type", None),
                "required_tests_str": json_entry.get("required_tests_str", None),
                "concept_description": json_entry.get("concept_description", None),
                "concept": json_entry.get("concept", None),
                "concept_examples": json_entry.get("concept_examples", None),
                "concept_id": json_entry.get("concept_id", None),
            }
            entries.append(entry)
            code_id_counter += 1  # Increment for next solution

    # Save to a new JSONL file
    save_path = os.path.join(cache_dir, f"{save_path_prefix}")
    with open(save_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Solution test cases saved to {save_path}")
    return entries


def dual_exec(results, instruction_based_test_cases, timeout: float = 0.1, test_case_limit: int = 5):
    cache_dir = "."
    save_path = "dual_exec_results.json"

    handled_solutions = PostProcessor.map_task_id_for_solution(results)
    handled_test_cases = PostProcessor.map_task_id_for_test_case(
        instruction_based_test_cases
    )

    dual_exec_result = evaluate_with_test_cases(
        handled_solutions, handled_test_cases, timeout, limit=test_case_limit
    )

    data_manager = DataManager(
        dual_exec_result, handled_solutions, handled_test_cases, test_case_limit
    )
    set_consistency = DualAgreement(data_manager)
    ranked_result, solution_test_cases = (
        set_consistency.get_sorted_solutions_without_iter()
    )

    return (solution_test_cases, handled_solutions)
