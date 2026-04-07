import json
import logging
import os

from utils import Client  # Use original Client

from dataops_code_cot.utils import OpenAIClient as Client

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("trace_compression.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Enable vLLM debug logging
# vllm.logger.setLevel(logging.DEBUG)

# Compression prompt with continuity and <trace></trace> tags
COMPRESSION_PROMPT = """You are an expert in analyzing Python execution traces. Given a chunk of a detailed execution trace and summaries of previous chunks (if provided), generate a compressed snapshot that retains the essence of the execution flow, variable states, and key transitions, reducing the size significantly to within 32768 characters (50% of 65536). If this is not the first chunk, use the previous summaries to maintain continuity of variable states, function calls, loop behaviors, and execution flow, ensuring the snapshot is not fragmented and reflects connections between chunks. Follow these rules to create the snapshot:

1. **Retain Critical Events**:
   - Include function calls,

 returns, initial variable assignments (Starting var, New var), significant variable modifications, and return values.
   - Example: "call def compute(self, radius: float)", "New var: pi = 3.141592653589793", "return 0.0".
   - For subsequent chunks, note if variables (e.g., `area`, `memo`) or function states continue from previous summaries.

2. **Capture All Important Variable State Transitions**:
   - Include every significant variable state change (e.g., assignments, updates) with specific values and the logic or line where the change occurs (e.g., "area[0] = 3.14 at line 10 (pi * radius^2)", "memo[5] = 8 at line 15 (memo[i-1] + i)").
   - Do not omit state changes even if they occur within loops or conditionals; include the exact value and context (e.g., "x = 10 at line 12 (if y > 5)").
   - If a variable's state is partially updated in this chunk, note it (e.g., "area[0:5000] updated, continues in next chunk").
   - For continuity, cross-reference prior summaries to track ongoing variable states (e.g., "area: Continued from prior summary [0.0]*5000, updated [5000:10000] = 3.14").

3. **Summarize Repetitive Loops**:
   - Collapse loops with many iterations into a single summary, describing the iterating variable, range, and purpose (e.g., "Loop (lines 10-12): Set area[i] = pi * radius^2 for i=0 to 9999").
   - If updates are repetitive, summarize the pattern once (e.g., "All area[i] = 0.0 since radius = 0").
   - If specific state changes occur within the loop, list them explicitly (e.g., "memo[0] = 1, memo[1] = 2 at line 15").
   - For chunks, indicate if the loop continues from the previous chunk (e.g., "Continued loop from i=5000 to 9999").
   - If omitting duplicated loop details, include a brief connector (e.g., "Omitted repetitive iterations of i=1000 to 9999, same operation as i=0").

4. **Omit Redundant Line Executions**:
   - Exclude repetitive loop control lines (e.g., "line 13 for i in range(10000)") except for the first occurrence.
   - Keep lines with assignments, conditionals, or returns, especially those affecting variable states.
   - If omitting duplicated lines, add a brief connector to describe the omitted action (e.g., "Omitted repeated executions of line 14, updating area[i]").

5. **Handle Large or Uniform Data Structures**:
   - For large structures, show initial and final states, using ellipses for omitted entries (e.g., "area: list of 10,000 elements, all 0.0").
   - If continuing from previous chunks, update states based on prior summaries (e.g., "area: Updated from [0.0]*5000 to [0.0]*10000").
   - Include specific state changes within structures (e.g., "area[9999] = 3.14 at line 10").
   - If truncating repetitive structure updates, note the omitted action (e.g., "Omitted repetitive area[i] updates for i=1000 to 9999, same as i=0").

6. **Detect and Handle Trace Inconsistencies**:
   - Prioritize dominant operations and flag inconsistencies (e.g., "Note: ‘area +=’ for i>9995, likely an error").
   - Check previous summaries for context (e.g., "Inconsistent with prior summary where area[i] = ...").
   - Do not infer or manipulate values; reflect only what is explicitly present in the trace or summaries.

7. **Clarify Return Value**:
   - Include return values (e.g., "return 0.0").
   - Note discrepancies with variable states (e.g., "Returns 0.0, but area is a list").
   - For partial chunks, indicate if the return is pending (e.g., "No return in this chunk, execution continues").

8. **Aggregate Timing Information**:
   - Omit individual timestamps; include total elapsed time for the chunk without connectors for omitted timing details.
   - Example: "Chunk elapsed time: 00:00:01.0".

9. **Compress Conditional Branches**:
   - Summarize conditionals by noting the taken branch (e.g., "Checked radius < 0, not taken").
   - Include any variable state changes within conditionals (e.g., "x = 5 at line 8 (if radius > 0)").
   - Maintain conditional state from previous chunks if relevant.
   - If omitting repetitive conditional checks, add a connector (e.g., "Omitted repeated checks of radius < 0, same as first check").

10. **Preserve Context and Continuity**:
    - Use previous summaries to track ongoing variables, loops, or function calls (e.g., "Continuing from prior summary: memo = {{...}}, i=5000").
    - Ensure variable changes (e.g., `area[i]`) and state transitions (e.g., loop iterations) are consistent across chunks.
    - Include source path and starting variables for the first chunk; for later chunks, reference prior summaries.

11. **Retain Original Information Without Manipulation**:
    - Do not infer, extrapolate, or alter variable values or execution details beyond what is explicitly in the trace or prior summaries.
    - If information is ambiguous, note it (e.g., "Variable x state unclear, value not updated in this chunk").
    - Ensure all included state changes are directly traceable to the input trace.

12. **Handle Omitted Duplicated Information**:
    - For any truncation or elimination of duplicated information (except timing details), include a brief line to describe the omitted action as a connector (e.g., "Omitted repetitive assignments to x for i=100 to 999, same as i=0").
    - Do not add connectors for omitted timing details (e.g., individual timestamps).

13. **Output Format**:
    - Return the snapshot as plain text, mimicking the original trace’s style (e.g., "call", "New var", "return").
    - Use concise descriptions (e.g., "Loop (lines X-Y): [summary]").
    - Do not include Markdown or headings.
    - Strictly enclose the snapshot within <trace></trace> tags.
    - Ensure the snapshot is <32768 characters.

Previous Summaries (if any):
{previous_summaries}

Input Trace Chunk:
{raw_trace}

Output Snapshot:
<trace>
[Compressed snapshot]
</trace>
"""


def read_trace_file(trace_directory, filename):
    """Read the content of a trace file."""
    try:
        filepath = os.path.join(trace_directory, filename)
        if not os.path.exists(filepath):
            logger.warning(f"Trace file not found: {filepath}")
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading trace file {filename}: {str(e)}")
        return None


# Character limits
MAX_CHARS = 16384 * 4  # 65,536 chars for Phi-4
CHUNK_SIZE = 50000  # ~12,500 tokens, as original
MAX_CHUNK_SNAPSHOT_CHARS = MAX_CHARS * 0.5  # 32,768 chars
MAX_PROMPT_CHARS = 400000  # ~100,000 tokens, for stability
BATCH_SIZE = 10  # Number of traces per batch


def count_chars(text):
    """Count characters in the given text."""
    try:
        if not text:
            return 0
        return len(text)
    except Exception as e:
        logger.error(f"Error counting characters: {str(e)}")
        return 0


def collect_long_traces(jsonl_file, trace_directory, threshold_chars):
    """Collect all traces exceeding the threshold from the JSONL file."""
    traces_to_compress = []
    total_traces = 0
    skipped_traces = 0

    try:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    logger.debug(f"Processing line {line_num}")

                    task_id = str(entry.get("task_id", "unknown"))
                    code_id = str(entry.get("code_id", "unknown"))
                    test_cases = entry.get("passing_test_cases", [])

                    if (
                        not task_id
                        or not code_id
                        or task_id == "unknown"
                        or code_id == "unknown"
                    ):
                        logger.warning(
                            f"Skipping line {line_num}: Missing task_id or code_id, entry: {entry}"
                        )
                        continue

                    if not test_cases:
                        logger.warning(
                            f"Skipping line {line_num}: No test cases found for task_id {task_id}, code_id {code_id}"
                        )
                        continue

                    for testcase_num, test_case in enumerate(test_cases):
                        total_traces += 1
                        trace_id = f"{task_id}_{code_id}_testcase_{testcase_num}"
                        original_filename = (
                            f"trace_{task_id}_{code_id}_testcase_{testcase_num}.log"
                        )
                        logger.debug(
                            f"[{trace_id}] Trace filename: {original_filename}"
                        )

                        raw_trace = entry.get("raw_trace")
                        if (
                            not raw_trace
                            or raw_trace == "Execution timed out after 10 seconds"
                        ):
                            logger.debug(
                                f"[{trace_id}] No valid raw_trace in JSON for {original_filename}, attempting to read file"
                            )
                            raw_trace = read_trace_file(
                                trace_directory, original_filename
                            )

                        if not raw_trace:
                            logger.warning(
                                f"[{trace_id}] Skipping test case {testcase_num} for task_id {task_id}, code_id {code_id}: No valid trace found"
                            )
                            skipped_traces += 1
                            continue

                        char_count = count_chars(raw_trace)
                        logger.debug(
                            f"[{trace_id}] Trace character count: {char_count}"
                        )

                        if char_count > threshold_chars:
                            logger.info(
                                f"[{trace_id}] Trace exceeds threshold ({char_count} > {threshold_chars}) for {original_filename}, queuing for compression"
                            )
                            # Split trace into chunks upfront
                            chunks = [
                                raw_trace[i : i + CHUNK_SIZE]
                                for i in range(0, char_count, CHUNK_SIZE)
                            ]
                            traces_to_compress.append(
                                {
                                    "trace_id": trace_id,
                                    "raw_trace": raw_trace,
                                    "original_filename": original_filename,
                                    "chunks": chunks,
                                    "snapshots": {},  # To store snapshots for each chunk
                                }
                            )
                        else:
                            logger.debug(
                                f"[{trace_id}] Trace within threshold ({char_count} <= {threshold_chars}) for {original_filename}, no compression needed"
                            )

                except json.JSONDecodeError:
                    logger.error(f"Malformed JSON at line {line_num}: {line.strip()}")
                    continue
                except Exception as e:
                    logger.error(
                        f"Error processing line {line_num}, task_id {task_id}, code_id {code_id}: {str(e)}",
                        exc_info=True,
                    )
                    continue

    except FileNotFoundError:
        logger.error(f"JSONL file not found: {jsonl_file}")
        return [], total_traces, skipped_traces
    except Exception as e:
        logger.error(f"Error reading JSONL file {jsonl_file}: {str(e)}", exc_info=True)
        return [], total_traces, skipped_traces

    logger.info(
        f"Collected {len(traces_to_compress)} long traces from {total_traces} total traces, {skipped_traces} skipped"
    )
    return traces_to_compress, total_traces, skipped_traces


def batch_process_chunks(
    model_client, traces, chunk_idx, trace_summaries, system_prompt
):
    """Batch process the i-th chunk for each trace in the batch."""
    prompts = []
    trace_ids = []
    chunk_sizes = []

    for trace in traces:
        trace_id = trace["trace_id"]
        chunks = trace["chunks"]
        if chunk_idx <= len(chunks):
            chunk = chunks[chunk_idx - 1]
            previous_summaries = trace_summaries.get(trace_id, "")
            if len(previous_summaries) > MAX_CHUNK_SNAPSHOT_CHARS:
                previous_summaries = previous_summaries[-MAX_CHUNK_SNAPSHOT_CHARS:]
                logger.debug(
                    f"[{trace_id}] Truncated previous_summaries to {len(previous_summaries)} chars for chunk {chunk_idx}"
                )

            prompt = COMPRESSION_PROMPT.format(
                previous_summaries=previous_summaries, raw_trace=chunk
            )
            prompt_size = len(prompt)
            if prompt_size > MAX_PROMPT_CHARS:
                logger.warning(
                    f"[{trace_id}] Chunk {chunk_idx} prompt size {prompt_size} exceeds {MAX_PROMPT_CHARS}, truncating"
                )
                excess = prompt_size - MAX_PROMPT_CHARS
                chunk = chunk[:-excess]
                prompt = COMPRESSION_PROMPT.format(
                    previous_summaries=previous_summaries, raw_trace=chunk
                )
                logger.debug(
                    f"[{trace_id}] Truncated chunk {chunk_idx} to {len(chunk)} chars, new prompt size: {len(prompt)} chars"
                )

            prompts.append(prompt)
            trace_ids.append(trace_id)
            chunk_sizes.append(len(chunk))

    if not prompts:
        return [], []

    logger.info(f"Batching {len(prompts)} chunks for chunk position {chunk_idx}")
    try:
        snapshots = model_client.get_model_response(
            [system_prompt] * len(prompts), prompts, max_new_tokens=4096
        )
        results = []
        for trace_id, snapshot, chunk_size in zip(trace_ids, snapshots, chunk_sizes):
            logger.debug(
                f"[{trace_id}] Chunk {chunk_idx} snapshot: {snapshot[:100]}..."
            )
            if not (snapshot.startswith("<trace>") and snapshot.endswith("</trace>")):
                logger.warning(
                    f"[{trace_id}] Chunk {chunk_idx} snapshot missing <trace> tags, adding them"
                )
                snapshot = f"<trace>\n{snapshot.strip()}\n</trace>"

            snapshot_size = len(snapshot)
            if snapshot_size > MAX_CHUNK_SNAPSHOT_CHARS:
                logger.warning(
                    f"[{trace_id}] Chunk {chunk_idx} snapshot size {snapshot_size} exceeds {MAX_CHUNK_SNAPSHOT_CHARS}, resummarizing"
                )
                prompt = COMPRESSION_PROMPT.format(
                    previous_summaries="", raw_trace=snapshot
                )
                try:
                    snapshot = model_client.get_model_response(
                        system_prompt,
                        prompt,
                        max_new_tokens=MAX_CHUNK_SNAPSHOT_CHARS // 4,
                    )
                    if not (
                        snapshot.startswith("<trace>") and snapshot.endswith("</trace>")
                    ):
                        snapshot = f"<trace>\n{snapshot.strip()}\n</trace>"
                    snapshot_size = len(snapshot)
                    if snapshot_size > MAX_CHUNK_SNAPSHOT_CHARS:
                        logger.warning(
                            f"[{trace_id}] Resummarized chunk {chunk_idx} snapshot size {snapshot_size} still exceeds {MAX_CHUNK_SNAPSHOT_CHARS}, truncating"
                        )
                        snapshot = f"<trace>\n{snapshot[7 : MAX_CHUNK_SNAPSHOT_CHARS - 7]}\n</trace>"
                except Exception as e:
                    logger.error(
                        f"[{trace_id}] Error resummarizing chunk {chunk_idx}: {str(e)}",
                        exc_info=True,
                    )
                    snapshot = f"<trace>\nSkipped chunk {chunk_idx} due to processing error\n</trace>"

            results.append((trace_id, snapshot, chunk_size))
        return results, []
    except Exception as e:
        logger.error(
            f"Error batch processing chunk position {chunk_idx}: {str(e)}",
            exc_info=True,
        )
        failed = [
            (
                trace_id,
                f"<trace>\nSkipped chunk {chunk_idx} due to processing error\n</trace>",
                chunk_size,
            )
            for trace_id, chunk_size in zip(trace_ids, chunk_sizes)
        ]
        return [], failed


def compress_trace(model_client, trace_info, trace_summaries, system_prompt):
    """Compress a single trace using batched chunk processing."""
    try:
        trace_id = trace_info["trace_id"]
        raw_trace = trace_info["raw_trace"]
        original_filename = trace_info["original_filename"]
        _ = original_filename
        chunks = trace_info["chunks"]
        logger.debug(f"[{trace_id}] Processing trace with {len(chunks)} chunks")

        summaries = []
        for chunk_idx in range(1, len(chunks) + 1):
            snapshot = trace_info["snapshots"].get(chunk_idx)
            if snapshot:
                summaries.append(snapshot)
            else:
                logger.warning(
                    f"[{trace_id}] No snapshot for chunk {chunk_idx}, using placeholder"
                )
                snapshot = f"<trace>\nSkipped chunk {chunk_idx} due to processing error\n</trace>"
                summaries.append(snapshot)

        # Combine snapshots, carefully removing individual <trace> tags
        combined_parts = []
        for s in summaries:
            content = s.replace("<trace>", "").replace("</trace>", "").strip()
            if content:
                combined_parts.append(content)
        combined_trace = "\n".join(combined_parts)
        compressed_size = len(combined_trace)
        logger.debug(f"[{trace_id}] Combined snapshot size: {compressed_size} chars")

        # Check Phi-4 limit
        if compressed_size > MAX_CHARS:
            logger.warning(
                f"[{trace_id}] Compressed trace size {compressed_size} exceeds Phi-4 limit {MAX_CHARS}, attempting to summarize"
            )
            prompt = COMPRESSION_PROMPT.format(
                previous_summaries="", raw_trace=combined_trace
            )
            try:
                combined_trace = model_client.get_model_response(
                    system_prompt, prompt, max_new_tokens=MAX_CHARS // 4
                )
                if not (
                    combined_trace.startswith("<trace>")
                    and combined_trace.endswith("</trace>")
                ):
                    combined_trace = f"<trace>\n{combined_trace.strip()}\n</trace>"
                compressed_size = len(combined_trace)
                logger.debug(
                    f"[{trace_id}] Resummarized snapshot size: {compressed_size} chars"
                )
                if compressed_size > MAX_CHARS:
                    logger.warning(
                        f"[{trace_id}] Resummarized trace size {compressed_size} still exceeds Phi-4 limit {MAX_CHARS}, truncating"
                    )
                    combined_trace = (
                        f"<trace>\n{combined_trace[7 : MAX_CHARS - 7]}\n</trace>"
                    )
                    compressed_size = len(combined_trace)
            except Exception as e:
                logger.error(
                    f"[{trace_id}] Error resummarizing trace: {str(e)}", exc_info=True
                )
                return raw_trace, False

        # Wrap final trace in a single <trace></trace> pair
        compressed_trace = f"<trace>\n{combined_trace.strip()}\n</trace>"
        compressed_size = len(compressed_trace)

        # Final size check
        if compressed_size > MAX_CHARS:
            logger.warning(
                f"[{trace_id}] Final trace size {compressed_size} exceeds Phi-4 limit {MAX_CHARS}, truncating"
            )
            compressed_trace = (
                f"<trace>\n{compressed_trace[7 : MAX_CHARS - 7]}\n</trace>"
            )
            compressed_size = len(compressed_trace)

        # Verify size reduction
        original_size = count_chars(raw_trace)
        logger.debug(
            f"[{trace_id}] Original size: {original_size}, Compressed size: {compressed_size}, 50% threshold: {0.5 * original_size}"
        )
        if compressed_size > 0.5 * original_size:
            logger.warning(
                f"[{trace_id}] Compressed trace size {compressed_size} exceeds 50% of original {original_size}"
            )

        return compressed_trace, True
    except Exception as e:
        logger.error(f"[{trace_id}] Error compressing trace: {str(e)}", exc_info=True)
        return raw_trace, False


def save_compressed_trace(trace_directory, original_filename, compressed_trace):
    """Save the compressed trace to a new file."""
    try:
        compressed_filename = original_filename.replace("trace_", "compressed_trace_")
        compressed_filepath = os.path.join(trace_directory, compressed_filename)
        with open(compressed_filepath, "w", encoding="utf-8") as f:
            f.write(compressed_trace)
        logger.info(f"Saved compressed trace to {compressed_filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving compressed trace {compressed_filename}: {str(e)}")
        return False


def process_jsonl_file(jsonl_file, trace_directory, com_dir, max_chars=MAX_CHARS):
    """Process the .jsonl file, compress long traces with batching, and save them."""
    threshold_chars = max_chars * 0.6  # 39,321.6 chars
    os.makedirs(com_dir, exist_ok=True)
    logger.info(f"Ensured compressed trace directory exists: {com_dir}")

    # Step 1: Collect all long traces
    traces_to_compress, total_traces, skipped_traces = collect_long_traces(
        jsonl_file, trace_directory, threshold_chars
    )
    if not traces_to_compress:
        logger.info("No traces require compression")
        logger.info(
            f"Processed {total_traces} traces: 0 compressed, {skipped_traces} skipped ({(skipped_traces / total_traces * 100):.2f}% skipped)"
        )
        return

    # Step 2: Initialize VLLM client for compression
    try:
        model_client = Client()  # Configured for Granite
        logger.info("Initialized VLLM client for Granite")
    except Exception as e:
        logger.error(f"Error initializing VLLM client: {str(e)}", exc_info=True)
        logger.info(
            f"Processed {total_traces} traces: 0 compressed, {skipped_traces} skipped ({(skipped_traces / total_traces * 100):.2f}% skipped)"
        )
        return

    compressed_traces = 0
    system_prompt = "You are an expert in Python execution trace analysis."

    # Step 3: Batch process traces
    trace_batches = [
        traces_to_compress[i : i + BATCH_SIZE]
        for i in range(0, len(traces_to_compress), BATCH_SIZE)
    ]
    for batch_idx, batch in enumerate(trace_batches, 1):
        logger.info(
            f"Processing batch {batch_idx}/{len(trace_batches)} with {len(batch)} traces"
        )

        # Track summaries for each trace
        trace_summaries = {trace["trace_id"]: "" for trace in batch}

        # Determine maximum number of chunks in this batch
        max_chunks = max(len(trace["chunks"]) for trace in batch)
        logger.debug(f"Batch {batch_idx} has traces with up to {max_chunks} chunks")

        # Process chunks by position (1st chunks, 2nd chunks, etc.)
        for chunk_idx in range(1, max_chunks + 1):
            # Batch process the i-th chunks
            chunk_results, failed_chunks = batch_process_chunks(
                model_client, batch, chunk_idx, trace_summaries, system_prompt
            )

            # Store snapshots and update summaries
            for trace_id, snapshot, _ in chunk_results + failed_chunks:
                for trace in batch:
                    if trace["trace_id"] == trace_id:
                        trace["snapshots"][chunk_idx] = snapshot
                        trace_summaries[trace_id] += (
                            f"Chunk {chunk_idx} Summary:\n{snapshot}\n\n"
                        )
                        break

        # Compress and save each trace
        for trace in batch:
            compressed_trace, was_compressed = compress_trace(
                model_client, trace, trace_summaries, system_prompt
            )
            if was_compressed:
                if save_compressed_trace(
                    com_dir, trace["original_filename"], compressed_trace
                ):
                    compressed_traces += 1
                else:
                    skipped_traces += 1
            else:
                logger.warning(
                    f"[{trace['trace_id']}] Compression failed for {trace['original_filename']}, skipping save"
                )
                skipped_traces += 1

    logger.info(
        f"Processed {total_traces} traces: {compressed_traces} compressed, {skipped_traces} skipped ({(skipped_traces / total_traces * 100):.2f}% skipped)"
    )


if __name__ == "__main__":
    JSONL_FILE = "dual_exec_results.json"  # Adjust path
    TRACE_DIRECTORY = "data/pysnooper_trace_cleaned_v1"  # Adjust path
    COM_DIR = "data/pysnooper_comptrace_cleaned_v1"  # Adjust path
    MAX_CHARS = 16384 * 4  # 65,536 chars for Phi-4

    logger.info(
        f"Starting trace compression with jsonl_file={JSONL_FILE}, trace_dir={TRACE_DIRECTORY}, com_dir={COM_DIR}, max_chars={MAX_CHARS}"
    )
    process_jsonl_file(JSONL_FILE, TRACE_DIRECTORY, COM_DIR, max_chars=MAX_CHARS)
    logger.info("Trace compression completed")
