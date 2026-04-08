# Standard
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# Local
from dataops_code_cot.components.execution.python._execution import (
    check_correctness_with_test_cases,
)

logging.basicConfig(
    format="SystemLog: [%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def evaluate_with_test_cases(solutions, test_cases_dict, timeout, limit):
    logger.info(f"Start evaluation with test cases, timeout={timeout}, limit={limit}")
    with ProcessPoolExecutor() as executor:
        futures = []
        results_list = []
        existed_completion = defaultdict(set)

        for solution in solutions:
            task_id = solution["task_id"]
            prompt = solution["prompt"]
            completion = solution["completion"]

            if completion in existed_completion[task_id]:
                continue
            existed_completion[task_id].add(completion)
            task_test_cases = test_cases_dict[task_id]  # Use prompt instead of task_id
            if not task_test_cases:
                continue

            args = (task_id, prompt, completion, task_test_cases, timeout)
            future = executor.submit(check_correctness_with_test_cases, *args)
            futures.append(future)

        logger.info(f"{len(futures)} execution requests are submitted")
        for idx, future in enumerate(as_completed(futures)):
            logger.info("[{}/{}] execution completed".format(idx + 1, len(futures)))

            result = future.result()
            results_list.append(result)

    logger.info("execution finished!")
    return results_list
