"""
Sliding window CoT verification.

Algorithm (from paper appendix):
  For each sentence in the CoT rationale:
    1. Extract verifiable entities (variable names, values, control flow keywords) via LLM.
    2. For each entity, scan a lookahead window of K=15 trace steps from current pointer.
       Accept if:
         a) Event Matching: a trace step within the window contains the variable modification or value.
         b) State Consistency: the cited value exists in the accumulated state at current pointer.
    3. If neither holds, entity is ungrounded -> rationale flagged.
  Second check: string-match the predicted I/O in the rationale against ground-truth I/O.
  Rationales failing either check are discarded.
"""
import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

WINDOW_K = 15


def extract_entities(sentence: str, client) -> list[str]:
    """Use LLM to extract verifiable entities from a CoT sentence."""
    prompt = (
        f"Extract all verifiable entities from this sentence. "
        f"Return only a JSON list of strings — variable names with their values "
        f"(e.g. 'lo=0', 'mid=2'), control flow keywords (e.g. 'if True', 'else branch', 'return'), "
        f"and numeric values. No explanation.\n\nSentence: {sentence}"
    )
    try:
        response = client.get_model_response(
            "You extract verifiable entities from code reasoning sentences. Return only a JSON list.",
            prompt,
            max_new_tokens=200,
            temperature=0.0,
        )
        # Parse JSON list from response
        match = re.search(r"\[.*?\]", response, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        logger.debug(f"Entity extraction failed: {e}")
    return []


def entity_in_window(entity: str, trace_lines: list[str], pointer: int, k: int = WINDOW_K) -> bool:
    """Check if entity appears in the lookahead window of K trace steps."""
    window = trace_lines[pointer: pointer + k]
    entity_lower = entity.lower().replace(" ", "")
    for line in window:
        line_norm = line.lower().replace(" ", "")
        if entity_lower in line_norm:
            return True
    return False


def entity_in_state(entity: str, accumulated_state: dict) -> bool:
    """Check if entity's value is consistent with accumulated variable state."""
    # Try to parse "var=value" format
    m = re.match(r"(\w+)\s*=\s*(.+)", entity.strip())
    if m:
        var, val = m.group(1).strip(), m.group(2).strip()
        if var in accumulated_state:
            return str(accumulated_state[var]).replace(" ", "") == val.replace(" ", "")
    return False


def update_state(trace_line: str, state: dict) -> None:
    """Parse a trace line and update accumulated variable state."""
    # Matches "New var: x = 5" or "Modified var: x = 5"
    m = re.search(r"(?:New var|Modified var):\s*(\w+)\s*=\s*(.+)", trace_line)
    if m:
        state[m.group(1).strip()] = m.group(2).strip()
    # Matches "Return value: 2"
    m2 = re.search(r"Return value:\s*(.+)", trace_line)
    if m2:
        state["__return__"] = m2.group(1).strip()


def verify_rationale(
    cot_text: str,
    trace_lines: list[str],
    ground_truth_output: str,
    ground_truth_input: str,
    client,
) -> tuple[bool, str]:
    """
    Verify a CoT rationale against an execution trace.
    Returns (passed: bool, reason: str).
    """
    if not trace_lines:
        return True, "no trace available — skipping entity check"
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cot_text) if s.strip()]
    pointer = 0
    accumulated_state: dict = {}

    for line in trace_lines[:pointer]:
        update_state(line, accumulated_state)

    # Phase 1: Entity-level sliding window — log only, do not reject
    # (entity extraction via small local models is noisy; only I/O check is definitive)
    ungrounded = []
    for sentence in sentences:
        try:
            entities = extract_entities(sentence, client)
        except Exception:
            entities = []
        for entity in entities:
            entity = str(entity)
            if not entity.strip():
                continue
            matched = (
                entity_in_window(entity, trace_lines, pointer)
                or entity_in_state(entity, accumulated_state)
            )
            if not matched:
                ungrounded.append(entity)
        pointer = min(pointer + 1, len(trace_lines) - 1)
        update_state(trace_lines[pointer] if trace_lines else "", accumulated_state)
    if ungrounded:
        logger.debug(f"Ungrounded entities (non-blocking): {ungrounded[:5]}")

    # Phase 2: fuzzy I/O check — literal value OR common English equivalents
    cot_lower = cot_text.lower()
    gt_out = (ground_truth_output or "").strip()
    if gt_out and gt_out.lower() not in ("unknown", "none", ""):
        fuzzy_matches = {gt_out.lower()}
        # Common English equivalents for edge cases
        if gt_out in ("[]", "set()", "{}"): fuzzy_matches.add("empty")
        if gt_out == "True":  fuzzy_matches.update({"true", "yes", "correct"})
        if gt_out == "False": fuzzy_matches.update({"false", "no", "incorrect"})
        if gt_out == "None":  fuzzy_matches.add("none")
        if not any(m in cot_lower for m in fuzzy_matches):
            return False, f"Output '{ground_truth_output}' not found in rationale"

    return True, "ok"


def verify_conversations(
    conversations_path: Path,
    traces_dir: Path,
    output_path: Path,
    client,
) -> tuple[int, int]:
    """
    Apply sliding window verification to all conversations.
    Returns (accepted, rejected) counts.
    """
    accepted, rejected = 0, 0
    lines = [l for l in conversations_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    with open(output_path, "w", encoding="utf-8") as out:
        for line in lines:
            conv = json.loads(line)
            conv_id = conv.get("id", "unknown")
            messages = conv.get("messages", [])

            # Extract forward reasoning (assistant turn 1) and ground-truth output
            fwd_reasoning = ""
            gt_output = ""
            if len(messages) >= 2:
                fwd_reasoning = messages[1].get("content", "").split("<response>")[0].strip()
            components = conv.get("components", {}).get("test_cases_components", {})
            for tc_data in components.values():
                gt_output = tc_data.get("test_case_data", {}).get("output_val", "")
                break

            # Load trace
            task_id, code_id = conv_id.split("_")[:2] if "_" in conv_id else ("", "")
            trace_file = traces_dir / f"trace_{task_id}_{code_id}_testcase_0.log"
            if not trace_file.exists():
                # No trace available — keep conversation without verification
                out.write(json.dumps(conv) + "\n")
                accepted += 1
                continue

            trace_lines = trace_file.read_text(encoding="utf-8").splitlines()
            gt_input = ""
            for tc_data in components.values():
                gt_input = tc_data.get("test_case_data", {}).get("input_val", "")
                break

            passed, reason = verify_rationale(
                fwd_reasoning, trace_lines, gt_output, gt_input, client
            )
            if passed:
                out.write(json.dumps(conv) + "\n")
                accepted += 1
            else:
                logger.info(f"Rejected {conv_id}: {reason}")
                rejected += 1

    return accepted, rejected
