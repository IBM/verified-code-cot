import json
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def compute_test_case_score(fwd_score: float, bwd_score: float) -> float:
    """Compute combined score for a test case."""
    score = 0.6 * fwd_score + 0.4 * bwd_score
    logger.debug(
        f"Computed combined score: forward={fwd_score}, backward={bwd_score}, combined={score}"
    )
    return score


def format_signature_details(signature: Any) -> str:
    """Format input/output parameter details from the signature_info field, handling both functions and classes."""
    if not signature:
        return "Signature Details:\nInput Parameters: Not available\nOutput Parameters: Not available"

    try:
        if isinstance(signature, str):
            # Handle string signature like "solution(expressions: str) -> list[float]"
            return f"Function Signature Details: {signature}"
        elif isinstance(signature, dict):
            signature_type = signature.get("type", "unknown")
            if signature_type == "function":
                # Handle function signature
                func_name = signature.get("name", "unknown")
                inputs = signature.get("inputs", [])
                return_type = signature.get("return_type", "unknown")

                # Format input parameters
                input_details = []
                for input_str in inputs:
                    if ": " in input_str:
                        name, type_ = input_str.split(": ", 1)
                        if name == "expressions" and type_ == "str":
                            description = "A multi-line string containing arithmetic expressions to be evaluated."
                        else:
                            description = (
                                f"A {type_} value used as input to the function."
                            )
                        input_details.append(f"- {name} ({type_}): {description}")
                    else:
                        logger.warning(f"Invalid input format: {input_str}")
                        input_details.append(f"- {input_str}: Unknown input parameter.")

                input_section = "Input Parameters:\n" + (
                    "\n".join(input_details) if input_details else "Not available"
                )

                # Format output parameters
                if return_type == "list[float]":
                    output_description = "A list of floating-point numbers representing the results of the evaluated expressions."
                else:
                    output_description = (
                        f"A {return_type} value returned by the function."
                    )
                output_section = f"Output Parameters:\n- return ({return_type}): {output_description}"

                return f"Function Signature Details:\n{input_section}\n{output_section}"

            elif signature_type == "class":
                # Handle class signature
                class_name = signature.get("name", "unknown")
                constructor = signature.get("constructor", {})
                methods = signature.get("methods", [])

                # Format constructor
                constructor_section = "Constructor:\n- No constructor defined"
                if constructor and constructor.get("name") == "__init__":
                    inputs = constructor.get("inputs", [])
                    return_type = constructor.get("return_type", "unknown")

                    # Exclude 'self' from inputs
                    input_details = []
                    for input_str in inputs:
                        if input_str == "self":
                            continue
                        if ": " in input_str:
                            name, type_ = input_str.split(": ", 1)
                            description = f"A {type_} value used to initialize the {class_name} class."
                            input_details.append(f"- {name} ({type_}): {description}")
                        else:
                            logger.warning(
                                f"Invalid constructor input format: {input_str}"
                            )
                            input_details.append(
                                f"- {input_str}: Unknown constructor parameter."
                            )

                    input_section = "  Input Parameters:\n" + (
                        "  " + "\n  ".join(input_details)
                        if input_details
                        else "  - None: Initializes the class without additional parameters."
                    )
                    output_section = f"  Output Parameters:\n  - return ({return_type}): Creates an instance of the {class_name} class."
                    constructor_section = (
                        f"Constructor:\n- __init__:\n{input_section}\n{output_section}"
                    )

                # Format methods
                method_sections = []
                for method in methods:
                    method_name = method.get("name", "unknown")
                    inputs = method.get("inputs", [])
                    return_type = method.get("return_type", "unknown")

                    # Exclude 'self' from inputs
                    input_details = []
                    for input_str in inputs:
                        if input_str == "self":
                            continue
                        if ": " in input_str:
                            name, type_ = input_str.split(": ", 1)
                            if (
                                method_name == "count_words"
                                and name == "text"
                                and type_ == "str"
                            ):
                                description = "A string to be processed to count word occurrences."
                            else:
                                description = f"A {type_} value used as input to the {method_name} method."
                            input_details.append(f"- {name} ({type_}): {description}")
                        else:
                            logger.warning(f"Invalid method input format: {input_str}")
                            input_details.append(
                                f"- {input_str}: Unknown method parameter."
                            )

                    input_section = "  Input Parameters:\n" + (
                        "  " + "\n  ".join(input_details)
                        if input_details
                        else "  - None: No additional input parameters."
                    )
                    if method_name == "count_words" and return_type == "dict[str, int]":
                        output_description = (
                            "A dictionary mapping words to their frequency counts."
                        )
                    else:
                        output_description = f"A {return_type} value returned by the {method_name} method."
                    output_section = f"  Output Parameters:\n  - return ({return_type}): {output_description}"
                    method_sections.append(
                        f"- {method_name}:\n{input_section}\n{output_section}"
                    )

                methods_section = "Methods:\n" + (
                    "\n".join(method_sections)
                    if method_sections
                    else "- No methods defined"
                )

                return f"Class Signature Details:\n{constructor_section}\n{methods_section}"

            else:
                logger.warning(f"Unknown signature type: {signature_type}")
                return "Signature Details:\nInput Parameters: Unknown type\nOutput Parameters: Unknown type"
        else:
            logger.warning(f"Unknown signature format: {type(signature)}")
            return "Signature Details:\nInput Parameters: Unknown format\nOutput Parameters: Unknown format"
    except Exception as e:
        logger.error(f"Error formatting signature: {e}")
        return "Signature Details:\nInput Parameters: Error processing signature\nOutput Parameters: Error processing signature"


def augment_first_message(content: str, signature: Any) -> str:
    """Augment the first message by removing the question and adding signature details and contextual line."""
    # signature_details = format_signature_details(signature)
    # contextual_line = (
    #    "Given a task description and its implementation in Python."
    #    "This conversation involves analyzing the execution flow of the provided Python code "
    #    "to understand its behavior, predicting outputs for given inputs (forward direction) "
    #    "and inputs for given outputs (backward direction), with responses including "
    #    "detailed reasoning in `<think>` tags and answers in `<response>` tags."
    # )
    signature_details = ""  # shiva added
    contextual_line = ""  # shiva added
    # Find the end of the code block
    code_block_start = content.find("```python")
    if code_block_start != -1:
        code_block_end = content.find("```", code_block_start + 3)
        if code_block_end != -1:
            # Keep content up to the end of the code block, discard the question
            return (
                content[: code_block_end + 3]
                +
                # "\n\n" + signature_details +
                ""
                + signature_details  # shiva
                +
                # "\n\n" + contextual_line
                ""
                + contextual_line  # shiva
            )
        else:
            # Incomplete code block; keep content up to start and append details
            logger.warning("Incomplete code block in first message")
            return (
                content[:code_block_start]
                + ""
                + signature_details
                + ""
                + contextual_line
            )
    else:
        # No code block; remove the last paragraph (assumed to be the question)
        paragraphs = content.strip().split("\n\n")
        if len(paragraphs) > 1:
            # Keep all but the last paragraph
            return (
                "".join(paragraphs[:-1]) + "" + signature_details + "" + contextual_line
            )
        else:
            # No paragraphs to remove; append details
            return content + "" + signature_details + "" + contextual_line


def map_messages_to_test_cases(
    messages: List[Dict[str, str]], test_cases_components: Dict[str, Any]
) -> Dict[int, str]:
    """Map message indices to test case IDs based on question matching, excluding first message."""
    message_to_test_case = {}
    for i, message in enumerate(messages[1:], start=1):  # Skip first message
        if message.get("role") != "user":
            continue
        question = message.get("content", "").strip()
        if not question:
            continue
        # Match against forward and backward questions
        for test_case_id, data in test_cases_components.items():
            if not data:
                continue
            forward_question = data.get("forward_question", "").strip()
            backward_question = data.get("backward_question", "").strip()
            if question == forward_question or question == backward_question:
                # Map the user message and the next assistant message (if exists)
                message_to_test_case[i] = test_case_id
                if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                    message_to_test_case[i + 1] = test_case_id
                break
        else:
            logger.debug(
                f"Message {i} (user): '{question[:50]}...' does not match any test case question"
            )

    logger.debug(f"Message to test case mapping: {message_to_test_case}")
    return message_to_test_case


def filter_conversation_with_coverage(
    conversation: Dict[str, Any], top_n: int = 1
) -> Dict[str, Any]:
    """Filter messages to keep only Q/A pairs for top N test cases, retaining and augmenting first message."""
    conversation_id = conversation.get("id", "unknown")
    logger.info(f"Filtering conversation {conversation_id}")

    # Extract components and messages
    test_cases_components = conversation.get("components", {}).get(
        "test_cases_components", {}
    )
    messages = conversation.get("messages", [])
    # Extract signature from components -> main_json -> signature_info
    signature = ""

    if not test_cases_components:
        logger.warning(
            f"No test case components found in conversation {conversation_id}"
        )
        if messages:
            first_message = messages[0].copy()
            if first_message.get("role") == "user":
                first_message["content"] = augment_first_message(
                    first_message["content"], signature
                )
            conversation["messages"] = [first_message]
        else:
            conversation["messages"] = []
        conversation["best_test_case_index"] = -1
        return conversation

    new_best_index = str(conversation["best_test_case_coverage"])
    if new_best_index == "-1":
        logger.info("SD: ")
        new_best_index = list(test_cases_components.keys())[
            0
        ]  # in case of failure, pick first fine. shiva
    top_test_case_ids = [new_best_index]

    logger.info(f"Top {top_n} test cases: {top_test_case_ids}")

    # Filter components to keep only top test cases
    filtered_components = {
        tc_id: test_cases_components[tc_id] for tc_id in top_test_case_ids
    }

    # Create new index mapping for retained test cases
    test_case_index_map = {tc_id: idx for idx, tc_id in enumerate(top_test_case_ids)}

    # Map messages to test cases (excluding first message)
    message_to_test_case = map_messages_to_test_cases(messages, test_cases_components)

    # Filter messages, always including the first message
    filtered_messages = []
    if messages:
        first_message = messages[0].copy()
        if first_message.get("role") == "user":
            first_message["content"] = augment_first_message(
                first_message["content"], signature
            )
        filtered_messages.append(first_message)
        logger.debug(
            "Keeping first message (augmented with signature and contextual line, question removed)"
        )

    for i, message in enumerate(messages[1:], start=1):
        test_case_id = message_to_test_case.get(i)
        if test_case_id in top_test_case_ids:
            filtered_messages.append(message)
            logger.debug(
                f"Keeping message {i} (role: {message.get('role')}): tied to test case {test_case_id}"
            )
        else:
            logger.debug(
                f"Skipping message {i} (role: {message.get('role')}): not tied to top test cases"
            )

    # Update conversation
    conversation["components"]["test_cases_components"] = filtered_components
    conversation["best_test_case_index"] = new_best_index
    conversation["messages"] = filtered_messages
    logger.info(
        f"Filtered messages: kept {len(filtered_messages)} out of {len(messages)}"
    )

    return conversation


def filter_conversation(conversation: Dict[str, Any], top_n: int = 2) -> Dict[str, Any]:
    """Filter messages to keep only Q/A pairs for top N test cases, retaining and augmenting first message."""
    conversation_id = conversation.get("id", "unknown")
    logger.info(f"Filtering conversation {conversation_id}")

    # Extract components and messages
    test_cases_components = conversation.get("components", {}).get(
        "test_cases_components", {}
    )
    messages = conversation.get("messages", [])
    # Extract signature from components -> main_json -> signature_info
    signature = (
        conversation.get("components", {})
        .get("main_json", {})
        .get("signature_info", None)
    )

    if not test_cases_components:
        logger.warning(
            f"No test case components found in conversation {conversation_id}"
        )
        if messages:
            # Keep only the first message, augmented with signature and contextual line
            first_message = messages[0].copy()
            if first_message.get("role") == "user":
                first_message["content"] = augment_first_message(
                    first_message["content"], signature
                )
            conversation["messages"] = [first_message]
        else:
            conversation["messages"] = []
        conversation["best_test_case_index"] = -1
        return conversation

    # Compute combined scores and select top N test cases
    test_case_scores = {}
    for test_case_id, test_case_data in test_cases_components.items():
        forward_score = test_case_data.get("forward_score", {}).get("total", None)
        backward_score = test_case_data.get("backward_score", {}).get("total", None)
        if forward_score is None or backward_score is None:
            logger.warning(
                f"Test case {test_case_id} missing forward or backward score, skipping"
            )
            continue
        combined_score = compute_test_case_score(forward_score, backward_score)
        test_case_scores[test_case_id] = {
            "combined": combined_score,
            "forward_score": forward_score,
            "backward_score": backward_score,
        }

    if not test_case_scores:
        logger.warning(f"No valid test case scores for conversation {conversation_id}")
        if messages:
            # Keep only the first message, augmented with signature and contextual line
            first_message = messages[0].copy()
            if first_message.get("role") == "user":
                first_message["content"] = augment_first_message(
                    first_message["content"], signature
                )
            conversation["messages"] = [first_message]
        else:
            conversation["messages"] = []
        conversation["best_test_case_index"] = -1
        return conversation

    # Sort test cases by combined score and select top N
    sorted_test_cases: List[Tuple[str, float]] = sorted(
        [(tc_id, scores["combined"]) for tc_id, scores in test_case_scores.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    top_test_case_ids = [tc_id for tc_id, _ in sorted_test_cases[:top_n]]
    logger.info(f"Top {top_n} test cases: {top_test_case_ids}")

    # Filter components to keep only top test cases
    filtered_components = {
        tc_id: test_cases_components[tc_id] for tc_id in top_test_case_ids
    }

    # Create new index mapping for retained test cases
    test_case_index_map = {tc_id: idx for idx, tc_id in enumerate(top_test_case_ids)}

    # Update best_test_case_index with enhanced logging
    best_test_case_id = None
    best_score = -1
    for tc_id, scores in test_case_scores.items():
        if tc_id in top_test_case_ids and scores["combined"] > best_score:
            best_score = scores["combined"]
            best_test_case_id = tc_id
    new_best_index = test_case_index_map.get(best_test_case_id, -1)
    logger.info(
        f"Updated best_test_case_index: test_case_id={best_test_case_id}, "
        f"combined_score={best_score:.4f}, new_index={new_best_index}"
    )

    # Map messages to test cases (excluding first message)
    message_to_test_case = map_messages_to_test_cases(messages, test_cases_components)

    # Filter messages, always including the first message
    filtered_messages = []
    if messages:
        first_message = messages[0].copy()
        if first_message.get("role") == "user":
            first_message["content"] = augment_first_message(
                first_message["content"], signature
            )
        filtered_messages.append(first_message)
        logger.debug(
            "Keeping first message (augmented with signature and contextual line, question removed)"
        )

    for i, message in enumerate(messages[1:], start=1):
        test_case_id = message_to_test_case.get(i)
        if test_case_id in top_test_case_ids:
            filtered_messages.append(message)
            logger.debug(
                f"Keeping message {i} (role: {message.get('role')}): tied to test case {test_case_id}"
            )
        else:
            logger.debug(
                f"Skipping message {i} (role: {message.get('role')}): not tied to top test cases"
            )

    # Update conversation
    conversation["components"]["test_cases_components"] = filtered_components
    conversation["best_test_case_index"] = new_best_index
    conversation["messages"] = filtered_messages
    logger.info(
        f"Filtered messages: kept {len(filtered_messages)} out of {len(messages)}"
    )

    return conversation


def filter_top_test_cases(input_file: str, output_file: str, top_n: int = 2):
    """Process scored JSONL to filter Q/A pairs for top N test cases."""
    logger.info(f"Reading scored JSONL from {input_file}")
    with open(input_file, "r", encoding="utf-8") as f_in, open(
        output_file, "w", encoding="utf-8"
    ) as f_out:
        for i, line in enumerate(f_in):
            try:
                conversation = json.loads(line.strip())
                logger.debug(
                    f"Processing conversation {i + 1} (ID: {conversation.get('id', 'unknown')})"
                )
                filtered_conversation = filter_conversation(conversation, top_n=top_n)
                json.dump(filtered_conversation, f_out)
                f_out.write("\n")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON line {i + 1}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing conversation {i + 1}: {e}")
                continue

    logger.info(f"Filtered conversations written to {output_file}")


if __name__ == "__main__":
    input_jsonl = "conversations/combined_conversations_deepcoder_scored.jsonl"
    output_jsonl = "conversations/conversations_deepcoder_filteredQA_correct.jsonl"
    filter_top_test_cases(input_jsonl, output_jsonl, top_n=1)
