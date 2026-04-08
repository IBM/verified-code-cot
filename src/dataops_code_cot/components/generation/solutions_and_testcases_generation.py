# Standard
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Third Party
from tqdm import tqdm

# from utils import VLLMClient_batch

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logging.getLogger("utils").setLevel(logging.DEBUG)
logger.debug("Logging initialized at DEBUG level")


InstructMode = Literal["I->R", "S->C", "C->I"]


@dataclass(frozen=True)
class Args:
    instruct_mode: InstructMode
    seed_data_file: str
    model_type: str
    model: str = "microsoft/phi-4"
    seed_code_start_index: int = 0
    max_new_data: int = 10
    continue_from: str | None = None
    seed: int = 3407
    temperature: float = 0.8
    max_output_tokens: int = 2048
    prompting_mode: Literal["chat", "completion"] = "completion"
    num_batched_requests: int = 1
    num_seed_examples: int = 10
    num_sample_per_request: int = 32
    sleep: float | None = None
    delay: float | None = None
    tag: str = ""
    save_dir: str = "./"


def load_prompts(prompt_file="prompts_batch.json"):
    try:
        with open(prompt_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file '{prompt_file}' not found.")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in '{prompt_file}'.")


def extract_json_block(response):
    json_block_pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(json_block_pattern, response, re.DOTALL)
    return match.group(1) if match else None


def parse_response(example, chunk, response):
    parsed_concepts = []
    pattern = r"'Concept':\s*'([^']+)',\s*'Description':\s*'([^']+)',\s*'Examples':\s*\[(.*?)\]"
    matches = re.findall(pattern, response, re.DOTALL)

    for concept, description, examples_str in matches:
        examples = [
            ex.strip().strip("'\"") for ex in examples_str.split(",") if ex.strip()
        ]
        parsed_concepts.append(
            {
                "id": example["example_id"],
                "seed": chunk,
                "Concept": concept.strip(),
                "Description": description.strip(),
                "Examples": examples,
            }
        )
    return parsed_concepts


def generate_concepts(client, examples, args, prompts):
    concept_responses = []
    for example in examples:
        seed = example["seed"]
        chunks = [seed[i : i + 4000] for i in range(0, len(seed), 4000)]

        for chunk in chunks:
            prompt = prompts["generate_concepts_prompt"].format(chunk=chunk)
            response = client.get_model_response(
                system_prompt="You are a helpful assistant who is an expert in Python programming.",
                user_prompt=prompt,
                model_id=args.model,
                max_new_tokens=args.max_output_tokens,
                temperature=args.temperature,
                top_k=50,
                top_p=1,
            )
            json_block = extract_json_block(response)
            if json_block:
                response = json_block
            concept_responses.extend(parse_response(example, chunk, response))
    return concept_responses


def extract_python_code(response_text):
    patterns = [r"```python\n(.*?)\n```", r"```\n(.*?)\n```", r"```(.*?)```"]
    all_code_blocks = []

    for pattern in patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        for match in matches:
            code = match.strip()
            if (
                code
                and (re.search(r"^(def|class)\s+\w+", code, re.MULTILINE))
                and (re.search(r"\n\s+.*", code))
            ):
                if not code.rstrip().endswith(":"):
                    all_code_blocks.append(code)
        if all_code_blocks:
            break

    return (
        all_code_blocks
        if all_code_blocks
        else (
            [response_text.strip()]
            if response_text.strip()
            else ["# No valid Python code found"]
        )
    )


def parse_signature_details(signature_text):
    match = re.search(r"```text\n(.*?)\n```", signature_text, re.DOTALL)
    signature_text = match.group(1).strip() if match else signature_text.strip()
    signature_text = (
        signature_text.replace("<br>", "").replace("\n", " ").replace("  ", " ")
    )

    if signature_text.startswith("Function:"):
        try:
            func_sig = signature_text[len("Function:") :].strip()
            name = func_sig.split("(")[0].strip()
            if "(" not in func_sig or ")" not in func_sig or "->" not in func_sig:
                raise ValueError("Malformed function signature")
            params_part = func_sig.split("(")[1].split(")")[0].strip()
            return_type = func_sig.split("->")[1].strip()
            inputs = [p.strip() for p in params_part.split(",")] if params_part else []
            return {
                "type": "function",
                "name": name,
                "inputs": inputs,
                "return_type": return_type,
                "methods": None,
                "constructor": None,
            }
        except (IndexError, ValueError, AttributeError) as e:
            logger.warning(
                f"Failed to parse function signature '{signature_text}': {str(e)}"
            )
            return {
                "type": "unknown",
                "name": "",
                "inputs": [],
                "return_type": "unknown",
                "methods": None,
                "constructor": None,
            }

    elif signature_text.startswith("Class:"):
        try:
            parts = signature_text[len("Class:") :].split(";")
            class_name = parts[0].strip()
            constructor = None
            methods = []
            for m in parts[1:]:
                m = m.strip()
                if not m:
                    continue
                try:
                    method_name = m.split("(")[0].strip()
                    if "(" not in m or ")" not in m or "->" not in m:
                        raise ValueError(f"Malformed method '{m}'")
                    params_part = m.split("(")[1].split(")")[0].strip()
                    return_type = m.split("->")[1].strip()
                    inputs = (
                        [p.strip() for p in params_part.split(",")]
                        if params_part
                        else []
                    )
                    if method_name == "__init__":
                        constructor = {
                            "name": method_name,
                            "inputs": inputs,
                            "return_type": return_type,
                        }
                    else:
                        methods.append(
                            {
                                "name": method_name,
                                "inputs": inputs,
                                "return_type": return_type,
                            }
                        )
                except (IndexError, ValueError):
                    # logger.warning(f"Skipping malformed method '{m}' in class '{class_name}': {str(e)}")
                    continue
            return {
                "type": "class",
                "name": class_name,
                "inputs": None,
                "return_type": None,
                "methods": methods if methods else [],
                "constructor": constructor,
            }
        except (IndexError, AttributeError) as e:
            logger.warning(
                f"Failed to parse class signature '{signature_text}': {str(e)}"
            )
            return {
                "type": "unknown",
                "name": "",
                "inputs": [],
                "return_type": "unknown",
                "methods": None,
                "constructor": None,
            }

    logger.warning(f"Unrecognized signature format '{signature_text}'")
    return {
        "type": "unknown",
        "name": "",
        "inputs": [],
        "return_type": "unknown",
        "methods": None,
        "constructor": None,
    }


def extract_python_test_code(response_text):
    patterns = [r"```python\n(.*?)\n```", r"```\n(.*?)\n```", r"```(.*?)```"]
    for pattern in patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        if matches:
            return "\n\n".join(match.strip() for match in matches)
    return response_text.strip() if response_text.strip() else ""


def split_test_cases(test_code):
    test_functions = re.findall(
        r"(def test_[^(]+\([^)]*\):(?:\n\s+[^\n]+)*)", test_code, re.MULTILINE
    )
    return [func.strip() for func in test_functions if func.strip()]


def extract_test_scenarios(response):
    match = re.search(r"```text\n(.*?)\n```", response, re.DOTALL)
    if match:
        scenarios = match.group(1).strip().split("\n")
        return [s.strip() for s in scenarios if s.strip()]
    lines = response.strip().split("\n")
    test_lines = [
        line.strip()
        for line in lines
        if line.strip() and line.strip().startswith(("Test ", "Test scenario"))
    ]
    if test_lines:
        return test_lines
    return []


def generate_instruction_response(client, concepts, args, prompts):
    difficulty_levels = ["medium", "hard"]
    # instruction_regex = re.compile(r"Instruction\d+:\n(.*?)\n", re.DOTALL)
    instruction_regex = re.compile(r"Instruction\d+:\n(.*?)(?:\n|$)", re.DOTALL)
    current_id = args.seed_code_start_index
    results_gl, test_cases_gl = [], []

    difficulty_desc = {
        "medium": {
            "complexity_description": "moderately complex problems requiring algorithmic thinking (5-7 difficulty)",
            "expected_lines": "30-60 lines",
            "difficulty_score": "7-8 on a scale of 1-10",
        },
        "hard": {
            "complexity_description": "sophisticated long problems requiring complex algorithms, deep logical and complex thinking and data structures (8-10 difficulty)",
            "expected_lines": "50-100+ lines",
            "difficulty_score": "9-10 on a scale of 1-10",
        },
    }

    # with open(Path("synthesized_ICT_phi_batch.json"), 'a') as f_res:
    # Batch instruction generation
    instruction_prompts = [
        (
            concept,
            d,
            prompts["instruction_prompt"].format(
                difficulty=d,
                complexity_description=difficulty_desc[d]["complexity_description"],
                expected_lines=difficulty_desc[d]["expected_lines"],
                difficulty_score=difficulty_desc[d]["difficulty_score"],
                concept=concept["concept"],
                description=concept["description"],
                examples=", ".join(concept["examples"]),
            ),
        )
        for concept in concepts
        for d in difficulty_levels
    ]
    system_prompts = [
        "You are a helpful assistant with expertise in Python and instructional design."
    ] * len(instruction_prompts)
    user_prompts = [prompt for _, _, prompt in instruction_prompts]

    # print(f"Starting instruction generation for {len(user_prompts)} prompts")
    # for i, prompt in enumerate(user_prompts[:3]):
    #    logger.debug(f"Instruction prompt {i} (full): {prompt}")
    # if len(user_prompts) > 3:
    #    logger.debug(f"Truncated {len(user_prompts) - 3} additional prompts")

    instruction_responses = client.get_model_response(
        system_prompt=system_prompts,
        user_prompt=user_prompts,
        model_id=args.model,
        max_new_tokens=args.max_output_tokens,
        temperature=args.temperature,
        top_k=50,
        top_p=0.6,
    )

    all_instructions = []
    for (concept, d, _), response in zip(instruction_prompts, instruction_responses):
        # print(f"Raw instruction response for {concept['concept']} ({d}): {response}")
        instructions = instruction_regex.findall(response)
        # print(f"Instructions for {concept['concept']} ({d}): {instructions}")
        # print(f"Parsed instructions for {concept['Concept']} ({d}): {instructions}")
        all_instructions.extend([(concept, d, instr) for instr in instructions])

    # if not all_instructions:
    #    logger.error("No instructions parsed from any responses. Check prompt, model output, or regex.")
    #    return current_id

    # Batch signature generation
    signature_prompts = [
        (concept, d, instr, prompts["signature_prompt"].format(instruction=instr))
        for concept, d, instr in all_instructions
    ]
    signature_system_prompts = ["You are a Python programming expert."] * len(
        signature_prompts
    )
    signature_user_prompts = [prompt for _, _, _, prompt in signature_prompts]

    # print(f"Starting signature generation for {len(signature_user_prompts)} instructions")
    signature_responses = client.get_model_response(
        system_prompt=signature_system_prompts,
        user_prompt=signature_user_prompts,
        model_id=args.model,
        max_new_tokens=256,
        temperature=0.5,
        top_k=50,
        top_p=0.9,
    )

    signature_infos = []
    for (concept, d, instr, _), response in zip(signature_prompts, signature_responses):
        # print(f"Raw signature response for {instr[:50]}...: --> {response}...")
        # print(f"Raw signature response for {instr[:50]}...: --> {response}...")
        signature_skeleton = re.search(r"```text\n(.*?)\n```", response, re.DOTALL)
        signature_skeleton = (
            signature_skeleton.group(1).strip()
            if signature_skeleton
            else response.strip()
        )
        # print(f"signrature skeleton {signature_skeleton}\n")
        signature_info = parse_signature_details(signature_skeleton)
        if signature_info["type"] == "unknown":
            # print(f"Failed to determine valid signature for: {instr}")
            logger.warning(f"Failed to determine valid signature for: {instr}")
            continue
        signature_infos.append((concept, d, instr, signature_info))

    # Group signatures by type
    function_signatures = [
        info for info in signature_infos if info[3]["type"] == "function"
    ]
    class_signatures = [info for info in signature_infos if info[3]["type"] == "class"]

    # Batch code generation by type
    func_code_prompts = []
    func_code_system_prompts = []
    class_code_prompts = []
    class_code_system_prompts = []

    for concept, d, instruction, signature_info in function_signatures:
        code_prompt = prompts["code_prompt_function"].format(
            difficulty=d,
            complexity_description=difficulty_desc[d]["complexity_description"],
            expected_lines=difficulty_desc[d]["expected_lines"],
            difficulty_score=difficulty_desc[d]["difficulty_score"],
            instruction=instruction,
            function_name=signature_info["name"],
            input_params=", ".join(signature_info["inputs"]),
            return_type=signature_info["return_type"],
        )
        func_code_prompts.extend([code_prompt] * 2)  # 2 calls per instruction
        func_code_system_prompts.extend(["You are a Python programming expert."] * 2)

    for concept, d, instruction, signature_info in class_signatures:
        constructor_sig = (
            f"def {signature_info['constructor']['name']}({', '.join(signature_info['constructor']['inputs'])}) -> {signature_info['constructor']['return_type']}"
            if signature_info["constructor"]
            else "# No constructor specified"
        )
        method_sigs = [
            f"def {m['name']}({', '.join(m['inputs'])}) -> {m['return_type']}"
            for m in signature_info["methods"]
        ]
        code_prompt = prompts["code_prompt_class"].format(
            difficulty=d,
            complexity_description=difficulty_desc[d]["complexity_description"],
            expected_lines=difficulty_desc[d]["expected_lines"],
            difficulty_score=difficulty_desc[d]["difficulty_score"],
            instruction=instruction,
            class_name=signature_info["name"],
            constructor_signature=constructor_sig,
            method_signatures=(
                "\n".join(method_sigs) if method_sigs else "# No methods specified"
            ),
        )
        class_code_prompts.extend([code_prompt] * 2)  # 2 calls per instruction
        class_code_system_prompts.extend(["You are a Python programming expert."] * 2)

    func_code_responses = []
    class_code_responses = []
    if func_code_prompts:
        # print(f"Starting batched function code generation for {len(func_code_prompts)} prompts")
        func_code_responses = client.get_model_response(
            system_prompt=func_code_system_prompts,
            user_prompt=func_code_prompts,
            model_id=args.model,
            max_new_tokens=args.max_output_tokens,
            temperature=args.temperature,
            top_k=50,
            top_p=0.8,
        )
    if class_code_prompts:
        # print(f"Starting batched class code generation for {len(class_code_prompts)} prompts")
        class_code_responses = client.get_model_response(
            system_prompt=class_code_system_prompts,
            user_prompt=class_code_prompts,
            model_id=args.model,
            max_new_tokens=args.max_output_tokens,
            temperature=args.temperature,
            top_k=50,
            top_p=0.8,
        )

    # Batch test scenario identification
    test_scenario_prompts = []
    test_scenario_system_prompts = []
    signature_details_list = []

    for concept, d, instruction, signature_info in signature_infos:
        if signature_info["type"] == "function":
            signature_details = f"Function: {signature_info['name']}({', '.join(signature_info['inputs'])}) -> {signature_info['return_type']}"
        else:
            method_sigs = [
                f"def {m['name']}({', '.join(m['inputs'])}) -> {m['return_type']}"
                for m in signature_info["methods"]
            ]
            primary_method = next(
                (
                    m["name"]
                    for m in signature_info["methods"]
                    if m["name"] == "compute"
                ),
                (
                    signature_info["methods"][0]["name"]
                    if signature_info["methods"]
                    else "unknown"
                ),
            )
            # signature_details = f"Class: {signature_info['name']}\nClass Methods:\n{'\n'.join(method_sigs) if method_sigs else '# No methods specified'}\nPrimary Method: {primary_method}"
            methods_text = (
                "\n".join(method_sigs) if method_sigs else "# No methods specified"
            )
            signature_details = (
                f"Class: {signature_info['name']}\n"
                f"Class Methods:\n{methods_text}\n"
                f"Primary Method: {primary_method}"
            )

        prompt = prompts["identify_tests_prompt"].format(
            instruction=instruction, signature_details=signature_details
        )
        test_scenario_prompts.append(prompt)
        test_scenario_system_prompts.append(
            "You are an expert at Python testing and requirements analysis."
        )
        signature_details_list.append(signature_details)

    # print(f"Starting batched test scenario identification for {len(test_scenario_prompts)} prompts")
    test_scenario_responses = client.get_model_response(
        system_prompt=test_scenario_system_prompts,
        user_prompt=test_scenario_prompts,
        model_id=args.model,
        max_new_tokens=256,
        temperature=0.5,
        top_k=50,
        top_p=0.9,
    )

    # Batch test generation by type
    func_test_prompts = []
    func_test_system_prompts = []
    class_test_prompts = []
    class_test_system_prompts = []

    for idx, (concept, d, instruction, signature_info) in enumerate(signature_infos):
        required_tests = extract_test_scenarios(test_scenario_responses[idx])
        required_tests_str = (
            "\n".join(required_tests)
            if required_tests
            else "No test scenarios identified"
        )

        if signature_info["type"] == "function":
            test_prompt = prompts["test_prompt_function"].format(
                difficulty=d,
                complexity_description=difficulty_desc[d]["complexity_description"],
                expected_lines=difficulty_desc[d]["expected_lines"],
                difficulty_score=difficulty_desc[d]["difficulty_score"],
                instruction=instruction,
                function_name=signature_info["name"],
                function_signature=f"def {signature_info['name']}({', '.join(signature_info['inputs'])}) -> {signature_info['return_type']}",
                required_tests=required_tests_str,
            )
            func_test_prompts.extend([test_prompt] * 3)  # 3 calls per instruction
            func_test_system_prompts.extend(
                ["You are an expert at Python testing and requirements analysis."] * 3
            )
        else:
            method_names = [m["name"] for m in signature_info["methods"]]
            primary_method = (
                "compute"
                if "compute" in method_names
                else (method_names[0] if method_names else "unknown")
            )
            method_sigs = [
                f"def {m['name']}({', '.join(m['inputs'])}) -> {m['return_type']}"
                for m in signature_info["methods"]
            ]
            test_prompt = prompts["test_prompt_class"].format(
                difficulty=d,
                complexity_description=difficulty_desc[d]["complexity_description"],
                expected_lines=difficulty_desc[d]["expected_lines"],
                difficulty_score=difficulty_desc[d]["difficulty_score"],
                instruction=instruction,
                class_name=signature_info["name"],
                primary_method=primary_method,
                method_signatures=(
                    "\n".join(method_sigs) if method_sigs else "# No methods specified"
                ),
                required_tests=required_tests_str,
            )
            class_test_prompts.extend([test_prompt] * 3)  # 3 calls per instruction
            class_test_system_prompts.extend(
                ["You are an expert at Python testing and requirements analysis."] * 3
            )

    func_test_responses = []
    class_test_responses = []
    if func_test_prompts:
        # print(f"Starting batched function test generation for {len(func_test_prompts)} prompts")
        func_test_responses = client.get_model_response(
            system_prompt=func_test_system_prompts,
            user_prompt=func_test_prompts,
            model_id=args.model,
            max_new_tokens=args.max_output_tokens,
            temperature=args.temperature,
            top_k=50,
            top_p=1,
        )
    if class_test_prompts:
        # print(f"Starting batched class test generation for {len(class_test_prompts)} prompts")
        class_test_responses = client.get_model_response(
            system_prompt=class_test_system_prompts,
            user_prompt=class_test_prompts,
            model_id=args.model,
            max_new_tokens=args.max_output_tokens,
            temperature=args.temperature,
            top_k=50,
            top_p=1,
        )

    # Combine results
    func_code_idx = 0
    class_code_idx = 0
    func_test_idx = 0
    class_test_idx = 0

    for idx, (concept, d, instruction, signature_info) in enumerate(signature_infos):
        required_tests = extract_test_scenarios(test_scenario_responses[idx])
        required_tests_str = (
            "\n".join(required_tests)
            if required_tests
            else "No test scenarios identified"
        )
        signature_details = signature_details_list[idx]

        if signature_info["type"] == "function":
            code_start = func_code_idx * 2
            code_end = code_start + 2
            response_codes = [
                extract_python_code(resp)
                for resp in func_code_responses[code_start:code_end]
            ]
            response_codes = [
                code for sublist in response_codes for code in sublist
            ]  # Flatten
            func_code_idx += 1

            test_start = func_test_idx * 3
            test_end = test_start + 3
            instruction_based_test_cases = [
                extract_python_test_code(resp)
                for resp in func_test_responses[test_start:test_end]
                if extract_python_test_code(resp)
            ]
            func_test_idx += 1
        else:
            code_start = class_code_idx * 2
            code_end = code_start + 2
            response_codes = [
                extract_python_code(resp)
                for resp in class_code_responses[code_start:code_end]
            ]
            response_codes = [
                code for sublist in response_codes for code in sublist
            ]  # Flatten
            class_code_idx += 1

            test_start = class_test_idx * 3
            test_end = test_start + 3
            instruction_based_test_cases = [
                extract_python_test_code(resp)
                for resp in class_test_responses[test_start:test_end]
                if extract_python_test_code(resp)
            ]
            class_test_idx += 1

        if not response_codes or all(
            code == "# No valid Python code found" for code in response_codes
        ):
            logger.warning(f"No valid code generated for: {instruction}")
            response_codes = ["# No valid Python code generated"]
        if not instruction_based_test_cases:
            logger.warning(f"No valid tests generated for: {instruction}")
            instruction_based_test_cases = ["# No valid tests generated"]

        current_id += 1
        result = {
            "concept_id": concept["id"],
            "id": current_id,
            "seed": concept["seed"],
            "concept": concept["concept"],
            "concept_description": concept["description"],
            "concept_examples": concept["examples"],
            "instruction": instruction,
            "responses": response_codes,
            "instruction_based_tests": instruction_based_test_cases,
            "difficulty": d,
            "instruction_prompt_difficulty": d,
            "instruction_prompt_complexity_description": difficulty_desc[d][
                "complexity_description"
            ],
            "instruction_prompt_expected_lines": difficulty_desc[d]["expected_lines"],
            "signature_info": signature_info,
            "signature_details": signature_details,
            "signature_type": signature_info["type"],
            "function_signature": (
                f"def {signature_info['name']}({', '.join(signature_info['inputs'])}) -> {signature_info['return_type']}"
                if signature_info["type"] == "function"
                else None
            ),
            "primary_method": (
                "compute"
                if signature_info["type"] == "class"
                and "compute" in [m["name"] for m in signature_info["methods"]]
                else (
                    signature_info["methods"][0]["name"]
                    if signature_info["type"] == "class" and signature_info["methods"]
                    else None
                )
            ),
            "required_tests_str": required_tests_str,
        }
        test_cases_with_id = {
            "id": result["id"],
            "tests": instruction_based_test_cases,
        }
        results_gl.append(result)
        test_cases_gl.append(test_cases_with_id)

    return current_id, results_gl, test_cases_gl


def load_bug_data(file_path):
    bug_data = []
    with open(file_path, "r") as file:
        for line in file:
            try:
                bug_data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return bug_data


def get_instruction_code_test_pairs(concepts, client, model_id):
    # Third Party

    args = Args(
        instruct_mode="C->I",
        seed_data_file="concepts.json",
        model=model_id,
        model_type="rits",
    )

    # Load prompts with error handling
    script_dir = Path(__file__).parent
    file_path = script_dir / "prompts_batch.json"
    prompts = load_prompts(file_path)
    unprocessed_concepts = [c for c in concepts]
    if not unprocessed_concepts:
        print("All concepts have been synthesized.")
        return [], []

    print(f"Found {len(unprocessed_concepts)} unsynthesized concepts to process.")

    batch_size = 10

    results = []
    instruction_based_test_cases = []

    print(f"Found {len(unprocessed_concepts)} unsynthesized concepts to process.")
    with tqdm(total=len(unprocessed_concepts), desc="Synthesizing Concepts") as pbar:
        for i in range(0, len(unprocessed_concepts), batch_size):
            batch = unprocessed_concepts[i : i + batch_size]
            print(f"Processing batch {i // batch_size + 1} with {len(batch)} concepts")
            start_id, result, test_case = generate_instruction_response(
                client, batch, args, prompts
            )

            # print(f"result {result}\n")

            results.append(result)
            instruction_based_test_cases.append(test_case)
            pbar.update(len(batch))

    return results, instruction_based_test_cases


if __name__ == "__main__":
    get_instruction_code_test_pairs("", "")
