import json
import logging
import traceback
from typing import Any, Dict

import nltk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Configure logging with DEBUG level
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download("punkt", quiet=True)


# VLLMClient_batch class (unchanged)
class VLLMClient_batch:
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-14B-Instruct"):
        logger.info(f"Initializing VLLMClient with model_name={model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        llm_args = {
            "model": model_name,
            "gpu_memory_utilization": 0.85,
        }
        self.llm = LLM(**llm_args)
        self.default_sampling_params = SamplingParams(max_tokens=8192)

    def get_model_response(
        self,
        system_prompt,
        user_prompt,
        model_id="Qwen/Qwen2.5-Coder-14B-Instruct",
        max_new_tokens=8000,
        min_new_tokens=30,
        temperature=0.5,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.05,
    ):
        logger.debug(
            f"get_model_response called with max_new_tokens={max_new_tokens}, temperature={temperature}"
        )
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
        if isinstance(system_prompt, str) and isinstance(user_prompt, str):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            logger.debug(f"Single prompt (first 500 chars): {text[:500]}...")
            outputs = self.llm.generate([text], sampling_params=sampling_params)
            response = outputs[0].outputs[0].text
            logger.debug(f"Single response (first 500 chars): {response[:500]}...")
            return response
        elif isinstance(system_prompt, list) and isinstance(user_prompt, list):
            if len(system_prompt) != len(user_prompt):
                raise ValueError(
                    "system_prompt and user_prompt lists must have the same length"
                )
            combined_prompts = [
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": sys},
                        {"role": "user", "content": usr},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for sys, usr in zip(system_prompt, user_prompt)
            ]
            logger.debug(f"Batch processing {len(combined_prompts)} prompts")
            for i, prompt in enumerate(combined_prompts[:3]):
                logger.debug(f"Batch prompt {i} (first 500 chars): {prompt[:500]}...")
            if len(combined_prompts) > 3:
                logger.debug(
                    f"Truncated {len(combined_prompts) - 3} additional prompts"
                )
            outputs = self.llm.generate(
                combined_prompts, sampling_params=sampling_params
            )
            results = [output.outputs[0].text for output in outputs]
            for i, response in enumerate(results[:3]):
                logger.debug(
                    f"Batch response {i} (first 500 chars): {response[:500]}..."
                )
            if len(results) > 3:
                logger.debug(f"Truncated {len(results) - 3} additional responses")
            return results
        else:
            raise ValueError(
                "system_prompt and user_prompt must both be strings or both be lists"
            )


# SCORING_PROMPT (updated to include logical reverse rationale for backward reasoning)
SCORING_PROMPT = """
You are tasked with critically evaluating a question/answer/reasoning/feedback (QRAF) set from a conversation about a coding problem to determine how effectively it captures the code’s core intent. The conversation includes an instruction, code, a test case, and a QRAF set (question, answer, reasoning, feedback). Score the QRAF set based on Intent Alignment (IA), Reasoning and Feedback Quality (RQ), and Reasoning and Feedback Conciseness (RC), applying a stringent dynamic penalty for verbose, unclear, or low-quality content to filter out poor-quality QRAFs. For backward QRAF sets, the reasoning must demonstrate a logical reverse rationale, correctly deducing the input from the output.

### Input
- **Instruction**: {instruction}
- **Code**: {code}
- **Test Case**: {test_case}
- **Question**: {question}
- **Answer**: {answer}
- **Reasoning**: {reasoning}
- **Feedback**: {feedback}
- **Is Backward**: {is_backward}

### Scoring Criteria
Apply a highly critical lens to ensure only high-quality QRAFs receive high scores. Low-quality questions (vague, irrelevant), incorrect or incomplete answers, unfocused/verbose reasoning, or irrelevant/redundant feedback should receive low scores.

1. **Intent Alignment (IA)** (0-10):
   - Measures how well the QRAF set tests and explains the code’s core intent (primary functionality from the instruction).
   - Evaluation:
     - Extract intent keywords from the instruction (e.g., key verbs/nouns).
     - Assess whether the question targets the core intent, with the answer, reasoning, and feedback reinforcing this focus.
     - Feedback must provide meaningful critique or insight aligned with the intent.
     - Penalize heavily if any component is vague, focuses on trivial edge cases (unless central), or misaligns with the instruction.
   - 10: QRAF directly tests and explains the core intent with a precise question, correct answer, focused reasoning, and insightful feedback.
   - 5: QRAF tests secondary or generic behavior, answer is partially correct, or feedback is generic.
   - 0: QRAF is unrelated to intent, question is vague, answer is incorrect, or feedback is irrelevant.
2. **Reasoning and Feedback Quality (RQ)** (0-10):
   - Assesses the clarity, structure, and focus of the reasoning and feedback in supporting the question and answer.
   - Evaluation:
     - Verify that reasoning is structured (e.g., Understand, Plan, Execute, Reflect) and clearly ties the answer to the question and code’s intent.
     - For backward QRAF sets (Is Backward = True), ensure the reasoning follows a logical reverse rationale, correctly deducing the input from the output with clear, rational steps that align with the code’s logic.
     - Ensure feedback provides specific, actionable critique or validation of the answer, enhancing understanding of the intent.
     - Check that reasoning and feedback focus on intent keywords, avoiding irrelevant details or excessive edge case discussion.
     - Penalize heavily for jargon, lack of clarity, logical gaps, feedback that repeats reasoning without adding value, or, for backward QRAFs, failure to demonstrate logical reverse deduction.
   - 10: Reasoning and feedback are highly structured, clear, and tightly focused on intent, perfectly justifying the answer and adding value; backward reasoning correctly deduces input from output.
   - 5: Reasoning or feedback explains the answer but lacks clarity, structure, or focus; feedback may be generic; backward reasoning has partial or unclear reverse deduction.
   - 0: Reasoning or feedback is confusing, incomplete, or fails to support the answer; backward reasoning lacks logical reverse rationale.
3. **Reasoning and Feedback Conciseness (RC)** (0-10):
   - Evaluates the combined length of reasoning and feedback, with a stringent dynamic penalty based on IA and RQ.
   - Evaluation:
     - Count total words in reasoning and feedback.
     - Estimate expected length based on complexity: Simple (150-200 words total), Moderate (200-300 words), Complex (300-400 words).
     - Compute Information Relevance Ratio (intent-related terms / total words).
     - Dynamic Penalty:
       - Quality Score: Q = (IA + RQ) / 2.
       - Penalty Scale: 1 - (Q / 20) (range 0.5 to 1).
       - If length > 1.5x expected, reduce RC by Penalty_Scale * (Excess_Length / Expected_Length).
       - If length < 50% expected, reduce RC by (1 - Length / Expected_Length).
       - Penalize heavily for verbose or overly brief content that dilutes quality.
   - 10: Matches expected length, high relevance ratio, no unnecessary content.
   - 5: Moderately verbose or brief, adjusted by penalty.
   - 0: Excessively long or critically short, low relevance.

### Output
Return **only** a JSON object wrapped in ```json\n``` delimiters, with no additional text, explanations, or prose. The JSON object must contain the following keys with integer scores between 0 and 10, reflecting the quality of the QRAF set:
```json
{{
  "IA": score,
  "RQ": score,
  "RC": score,
  "total": score
}}
```
"""


def extract_reasoning(response: str) -> str:
    logger.debug(
        f"Extracting reasoning from response (first 200 chars): {response[:200]}..."
    )
    try:
        start_tag = "<think>"
        end_tag = "</think>"
        start_idx = response.find(start_tag) + len(start_tag)
        end_idx = response.find(end_tag)
        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            logger.debug("No valid <think> tags found in response")
            return ""
        reasoning = response[start_idx:end_idx].strip()
        logger.debug(f"Extracted reasoning (first 200 chars): {reasoning[:200]}...")
        return reasoning
    except Exception as e:
        logger.error(
            f"Error extracting reasoning: {e}, response (first 200 chars): {response[:200]}..."
        )
        return ""


def extract_answer(response: str, is_forward: bool = True) -> str:
    logger.debug(
        f"Extracting answer from response (first 1000 chars): {response[:1000]}..., is_forward={is_forward}"
    )
    try:
        if not isinstance(response, str):
            logger.error(
                f"Response is not a string: {type(response)}, value: {response}"
            )
            return ""
        # Use <output> for forward_response, <input> for backward_response
        tag = "<output>" if is_forward else "<input>"
        end_tag = "</output>" if is_forward else "</input>"
        start_idx = response.find(tag) + len(tag)
        end_idx = response.find(end_tag)
        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            logger.debug(f"No valid {tag} tags found in response")
            return ""
        answer = response[start_idx:end_idx].strip()
        logger.debug(f"Extracted answer (first 200 chars): {answer[:200]}...")
        return answer
    except Exception as e:
        logger.error(
            f"Error extracting answer: {e}, response (first 1000 chars): {response[:1000]}..."
        )
        return ""


def extract_feedback(response: str) -> str:
    logger.debug(
        f"Extracting feedback from response (first 1000 chars): {response[:1000]}..."
    )
    try:
        if not isinstance(response, str):
            logger.error(
                f"Feedback response is not a string: {type(response)}, value: {response}"
            )
            return ""
        start_marker = "Feedback:"
        start_idx = response.find(start_marker) + len(start_marker)
        if start_idx == -1:
            logger.debug("No 'Feedback:' header found in response")
            return ""
        # Assume feedback ends at next header (e.g., another "Header:") or end of string
        end_idx = response.find(":", start_idx)
        if end_idx == -1:
            end_idx = len(response)
        feedback = response[start_idx:end_idx].strip()
        logger.debug(f"Extracted feedback (first 200 chars): {feedback[:200]}...")
        return feedback
    except Exception as e:
        logger.error(
            f"Error extracting feedback: {e}, response (first 1000 chars): {response[:1000]}..."
        )
        return ""


def estimate_complexity(test_case: str, question: str, feedback: str) -> str:
    logger.debug(
        f"Estimating complexity for test_case (first 200 chars): {test_case[:200]}..., question (first 200 chars): {question[:200]}..., feedback (first 200 chars): {feedback[:200]}..."
    )
    try:
        operations = (
            len(test_case.split("\n")) + len(question.split()) + len(feedback.split())
        )
        complexity = (
            "Simple"
            if operations < 20
            else "Moderate"
            if operations < 40
            else "Complex"
        )
        logger.debug(f"Estimated complexity: {complexity} (operations={operations})")
        return complexity
    except Exception as e:
        logger.error(f"Error estimating complexity: {e}")
        return "Simple"


def compute_test_case_score(fwd_score: float, bwd_score: float) -> float:
    logger.debug(
        f"Computing test case score: forward={fwd_score}, backward={bwd_score}"
    )
    score = 0.6 * fwd_score + 0.4 * bwd_score
    logger.debug(f"Combined score: {score}")
    return score


def sanitize_input(text: str) -> str:
    """Sanitize input by escaping braces and ensuring it's a string."""
    if not isinstance(text, str):
        logger.warning(f"Input is not a string: {type(text)}, converting to string")
        text = str(text)
    return text.replace("{", "{{").replace("}", "}}")


def process_conversation(
    client: VLLMClient_batch, conversation: Dict[str, Any]
) -> Dict[str, Any]:
    conversation_id = conversation.get("id", "unknown")
    logger.info(f"Processing conversation {conversation_id}")

    # Log conversation structure
    logger.debug(f"Conversation keys: {list(conversation.keys())}")
    instruction = conversation.get("instruction", "")
    code = conversation.get("code", "")
    test_cases = conversation.get("test_cases", {})
    messages = conversation.get("messages", [])
    components = conversation.get("components", {}).get("test_cases_components", {})
    logger.debug(f"Instruction (first 200 chars): {instruction[:200]}...")
    logger.debug(f"Code (first 200 chars): {code[:200]}...")
    logger.debug(f"Test cases: {list(test_cases.keys())}")
    logger.debug(f"Messages count: {len(messages)}")
    logger.debug(f"Components test_cases_components: {list(components.keys())}")

    # Check for test case ID mismatch
    missing_test_cases = set(components.keys()) - set(test_cases.keys())
    if missing_test_cases:
        logger.warning(
            f"Test case IDs in components but not in test_cases: {missing_test_cases}"
        )

    # Prepare batch inputs for model
    system_prompts = []
    user_prompts = []
    qa_metadata = []

    # Limit to first 2 test cases for debugging
    test_case_count = 0
    for test_case_id, test_case_data in components.items():
        # if test_case_count >= 2:
        # logger.debug(f"Stopping after 2 test cases for debugging")
        # break
        test_case_count += 1
        logger.debug(f"Starting processing for test case {test_case_id}")
        try:
            test_case = test_cases.get(test_case_id, "")
            if not test_case:
                logger.warning(f"No test case found for ID {test_case_id}, skipping")
                continue
            forward_question = test_case_data.get("forward_question", "")
            backward_question = test_case_data.get("backward_question", "")
            forward_response = test_case_data.get("forward_response", "")
            backward_response = test_case_data.get("backward_response", "")
            forward_feedback_response = test_case_data.get(
                "forward_feedback_response", ""
            )
            backward_feedback_response = test_case_data.get(
                "backward_feedback_response", ""
            )
            logger.debug(
                f"Test case {test_case_id} data: forward_question (first 200 chars): {forward_question[:200]}..., backward_question (first 200 chars): {backward_question[:200]}..."
            )
            logger.debug(
                f"Forward response (first 1000 chars): {forward_response[:1000]}..."
            )
            logger.debug(
                f"Backward response (first 1000 chars): {backward_response[:1000]}..."
            )
            logger.debug(
                f"Forward feedback response (first 1000 chars): {forward_feedback_response[:1000]}..."
            )
            logger.debug(
                f"Backward feedback response (first 1000 chars): {backward_feedback_response[:1000]}..."
            )

            # Validate response types
            if not isinstance(forward_response, str):
                logger.error(
                    f"Forward response for test case {test_case_id} is not a string: {type(forward_response)}, value: {forward_response}"
                )
                continue
            if not isinstance(backward_response, str):
                logger.error(
                    f"Backward response for test case {test_case_id} is not a string: {type(backward_response)}, value: {backward_response}"
                )
                continue
            if not isinstance(forward_feedback_response, str):
                logger.error(
                    f"Forward feedback response for test case {test_case_id} is not a string: {type(forward_feedback_response)}, value: {forward_feedback_response}"
                )
                continue
            if not isinstance(backward_feedback_response, str):
                logger.error(
                    f"Backward feedback response for test case {test_case_id} is not a string: {type(backward_feedback_response)}, value: {backward_feedback_response}"
                )
                continue

            # Extract answer, reasoning, and feedback
            forward_answer = extract_answer(forward_response, is_forward=True)
            forward_reasoning = extract_reasoning(forward_response)
            backward_answer = extract_answer(backward_response, is_forward=False)
            backward_reasoning = extract_reasoning(backward_response)
            forward_feedback = extract_feedback(forward_feedback_response)
            backward_feedback = extract_feedback(backward_feedback_response)

            # Log extracted data
            logger.debug(
                f"Test case {test_case_id} forward: answer (first 200 chars): {forward_answer[:200]}..., reasoning (first 200 chars): {forward_reasoning[:200]}..., feedback (first 200 chars): {forward_feedback[:200]}..."
            )
            logger.debug(
                f"Test case {test_case_id} backward: answer (first 200 chars): {backward_answer[:200]}..., reasoning (first 200 chars): {backward_reasoning[:200]}..., feedback (first 200 chars): {backward_feedback[:200]}..."
            )

            # Estimate complexity (separately for forward and backward)
            forward_complexity = estimate_complexity(
                test_case, forward_question, forward_feedback
            )
            backward_complexity = estimate_complexity(
                test_case, backward_question, backward_feedback
            )

            # Prepare prompts for forward and backward QRAFs
            for is_forward, question, answer, reasoning, feedback, complexity in [
                (
                    True,
                    forward_question,
                    forward_answer,
                    forward_reasoning,
                    forward_feedback,
                    forward_complexity,
                ),
                (
                    False,
                    backward_question,
                    backward_answer,
                    backward_reasoning,
                    backward_feedback,
                    backward_complexity,
                ),
            ]:
                if not (question and answer and reasoning and feedback):
                    logger.warning(
                        f"Skipping QRAF for test case {test_case_id}, is_forward={is_forward}: missing data (question={bool(question)}, answer={bool(answer)}, reasoning={bool(reasoning)}, feedback={bool(feedback)})"
                    )
                    continue
                try:
                    logger.debug(
                        f"Formatting prompt for test case {test_case_id}, is_forward={is_forward}"
                    )
                    logger.debug(
                        f"Prompt inputs: instruction (first 200 chars): {instruction[:200]}..., code (first 200 chars): {code[:200]}..., test_case (first 200 chars): {test_case[:200]}..., question (first 200 chars): {question[:200]}..., answer (first 200 chars): {answer[:200]}..., reasoning (first 200 chars): {reasoning[:200]}..., feedback (first 200 chars): {feedback[:200]}..."
                    )
                    # Sanitize all inputs
                    sanitized_instruction = sanitize_input(instruction)
                    sanitized_code = sanitize_input(code)
                    sanitized_test_case = sanitize_input(test_case)
                    sanitized_question = sanitize_input(question)
                    sanitized_answer = sanitize_input(answer)
                    sanitized_reasoning = sanitize_input(reasoning)
                    sanitized_feedback = sanitize_input(feedback)
                    sanitized_is_backward = "True" if not is_forward else "False"
                    prompt = SCORING_PROMPT.format(
                        instruction=sanitized_instruction,
                        code=sanitized_code,
                        test_case=sanitized_test_case,
                        question=sanitized_question,
                        answer=sanitized_answer,
                        reasoning=sanitized_reasoning,
                        feedback=sanitized_feedback,
                        is_backward=sanitized_is_backward,
                    )
                    system_prompts.append(
                        "You are an expert in code analysis and evaluation with deep expertise in programming concepts and best practices. Your role is to rigorously and critically evaluate the provided question, answer, reasoning, and feedback set, identifying any flaws, ambiguities, or misalignments with the code’s intent. Assign scores with extreme precision, harshly penalizing vague questions, incorrect answers, unfocused reasoning, or irrelevant feedback to filter out low-quality content. For backward sets, ensure the reasoning follows a logical reverse rationale, correctly deducing the input from the output. Return the response strictly in the JSON format specified in the user prompt."
                    )
                    user_prompts.append(prompt)
                    qa_metadata.append(
                        {
                            "test_case_id": test_case_id,
                            "is_forward": is_forward,
                            "complexity": complexity,
                        }
                    )
                    logger.debug(
                        f"Added prompt for test case {test_case_id}, is_forward={is_forward}, prompt (first 1000 chars): {prompt[:1000]}..."
                    )
                except KeyError as e:
                    logger.error(
                        f"KeyError formatting prompt for test case {test_case_id}, is_forward={is_forward}: {e}\n{traceback.format_exc()}"
                    )
                    continue
                except ValueError as e:
                    logger.error(
                        f"ValueError formatting prompt for test case {test_case_id}, is_forward={is_forward}: {e}\n{traceback.format_exc()}"
                    )
                    continue
                except Exception as e:
                    logger.error(
                        f"Unexpected error formatting prompt for test case {test_case_id}, is_forward={is_forward}: {e}\n{traceback.format_exc()}"
                    )
                    continue
        except Exception as e:
            logger.error(
                f"Error processing test case {test_case_id}: {e}\n{traceback.format_exc()}"
            )
            continue

    # Query model in batch
    logger.debug(f"Sending {len(system_prompts)} prompts to model")
    try:
        responses = client.get_model_response(system_prompts, user_prompts)
        logger.debug(f"Received {len(responses)} responses")
        for i, response in enumerate(responses[:3]):
            logger.debug(f"Batch response {i} (first 1000 chars): {response[:1000]}...")
        if len(responses) > 3:
            logger.debug(f"Truncated {len(responses) - 3} additional responses")
    except Exception as e:
        logger.error(
            f"Model query failed for conversation {conversation_id}: {e}\n{traceback.format_exc()}"
        )
        return conversation

    # Process responses
    test_case_scores = {}
    for response, metadata in zip(responses, qa_metadata):
        test_case_id = metadata["test_case_id"]
        is_forward = metadata["is_forward"]
        logger.debug(
            f"Processing response for test case {test_case_id}, is_forward={is_forward}, response (first 1000 chars): {response[:1000]}..."
        )
        try:
            # Extract JSON from ```json\n``` delimiters
            json_start = response.find("```json\n") + len("```json\n")
            json_end = response.rfind("```")
            if json_start == -1 or json_end == -1 or json_start >= json_end:
                raise ValueError("No valid ```json``` delimiters found in response")
            json_str = response[json_start:json_end].strip()
            scores = json.loads(json_str)
            logger.debug(f"Parsed JSON scores: {scores}")
            # Validate scores
            ia, rq, rc, total = (
                scores["IA"],
                scores["RQ"],
                scores["RC"],
                scores["total"],
            )
            if not all(
                isinstance(s, (int, float)) and 0 <= s <= 10
                for s in [ia, rq, rc, total]
            ):
                raise ValueError(f"Invalid score values: {scores}")
            # Store scores in components
            if test_case_id not in components:
                components[test_case_id] = {}
            if is_forward:
                components[test_case_id]["forward_score"] = {
                    "IA": ia,
                    "RQ": rq,
                    "RC": rc,
                    "total": total,
                }
            else:
                components[test_case_id]["backward_score"] = {
                    "IA": ia,
                    "RQ": rq,
                    "RC": rc,
                    "total": total,
                }
            logger.debug(
                f"Stored scores for test case {test_case_id}, is_forward={is_forward}: {scores}"
            )
            # Accumulate for test case scoring
            if test_case_id not in test_case_scores:
                test_case_scores[test_case_id] = {"forward": None, "backward": None}
            if is_forward:
                test_case_scores[test_case_id]["forward"] = total
            else:
                test_case_scores[test_case_id]["backward"] = total
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(
                f"Invalid response for test case {test_case_id}, QRAF {'forward' if is_forward else 'backward'}: {e}, response (first 1000 chars): {response[:1000]}..."
            )
            continue

    # Compute test case scores and select best
    logger.debug(f"Test case scores: {test_case_scores}")
    best_score = -1
    best_test_case_index = None
    for idx, (test_case_id, scores) in enumerate(test_case_scores.items()):
        if scores["forward"] is None or scores["backward"] is None:
            logger.warning(
                f"Skipping test case {test_case_id}: missing forward or backward score"
            )
            continue
        combined_score = compute_test_case_score(scores["forward"], scores["backward"])
        test_case_scores[test_case_id]["combined"] = combined_score
        logger.debug(f"Test case {test_case_id} combined score: {combined_score}")
        if combined_score > best_score:
            best_score = combined_score
            best_test_case_index = idx

    # Update conversation
    conversation["components"]["test_cases_components"] = components
    conversation["best_test_case_index"] = (
        best_test_case_index if best_test_case_index is not None else -1
    )
    logger.debug(
        f"Best test case index: {best_test_case_index}, best score: {best_score}"
    )
    if best_test_case_index is None:
        logger.warning(f"No valid test case scores for conversation {conversation_id}")

    return conversation


def process_jsonl(
    input_file: str,
    output_file: str,
    model_name: str = "Qwen/Qwen2.5-Coder-14B-Instruct",
    max_conversations: int = 2,
):
    """Process a JSONL file, score Q/A pairs, and write updated JSONL with best test case index."""
    logger.info(f"Processing JSONL file: {input_file}")
    client = VLLMClient_batch(model_name=model_name)

    with open(input_file, "r", encoding="utf-8") as f_in, open(
        output_file, "w", encoding="utf-8"
    ) as f_out:
        for i, line in enumerate(f_in):
            if i >= max_conversations:
                logger.info(
                    f"Stopping after processing {max_conversations} conversations"
                )
                break
            try:
                conversation = json.loads(line.strip())
                logger.debug(
                    f"Parsed conversation {i + 1}: {list(conversation.keys())}"
                )
                updated_conversation = process_conversation(client, conversation)
                json.dump(updated_conversation, f_out)
                f_out.write("\n")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON line {i + 1}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing conversation {i + 1}: {e}")
                continue

    logger.info(f"Output written to {output_file}")


if __name__ == "__main__":
    input_jsonl = "input.jsonl"
    output_jsonl = "conversations_test_filteredQA.jsonl"
    process_jsonl(
        "conversations_test.jsonl",
        "conversations_test_filteredQA.jsonl",
        model_name="Qwen/Qwen2.5-Coder-14B-Instruct",
    )
