# Standard
import json
import re
from typing import Dict, List

import pytextrank
import spacy

_ = pytextrank
from concurrent.futures import ThreadPoolExecutor

# Third Party
from tqdm import tqdm


def parse_model_response(
    example: Dict, chunk: str, response: str, topic: str, subtopic: str, conc_id: int
) -> List[Dict]:
    """
    Parses a model response to extract complete JSON objects for concepts, descriptions, and examples.
    """
    parsed_concepts = []

    # Preprocess response to fix common JSON issues
    def preprocess_response(response: str) -> str:
        response = re.sub(r"//.*", "", response)  # Remove inline comments
        response = re.sub(r",\s*([\]}])", r"\1", response)  # Remove trailing commas
        return response

    response = preprocess_response(response)

    # Updated regex pattern to match JSON-like structures without recursive pattern
    json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"

    # Find all JSON blocks in the response
    json_blocks = re.findall(json_pattern, response, re.DOTALL)
    total_passed_blocks = 0
    for block in json_blocks:
        try:
            # Attempt to parse each JSON block
            concept_data = json.loads(block)
            # Extract fields if they exist
            concept = concept_data.get("Concept", "").strip()
            description = concept_data.get("Description", "").strip()
            examples = concept_data.get("Examples", [])

            # Ensure examples are properly formatted
            if isinstance(examples, str):
                # Handle case where examples might be a string
                examples = [examples.strip()]
            else:
                examples = [ex.strip() for ex in examples if ex and isinstance(ex, str)]

            # Append to parsed concepts
            parsed_concepts.append(
                {
                    "id": conc_id + total_passed_blocks,
                    "seed": chunk,
                    "topic": topic,
                    "subtopic": subtopic,
                    "concept": concept,
                    "description": description,
                    "examples": examples,
                }
            )

            total_passed_blocks += 1

        except json.JSONDecodeError:
            # If parsing fails, log the issue
            print("\n")

    return parsed_concepts


nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")


def extract_keywords(text):
    doc = nlp(text)
    keywords = []
    for phrase in doc._.phrases:
        keywords.append(
            {"text": phrase.text, "chunks": [chunk.text for chunk in phrase.chunks]}
        )
    return keywords


def extract_json_block(response):
    # Match the pseudo-JSON block within triple backticks or similar markers
    json_block_pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(json_block_pattern, response, re.DOTALL)

    if match:
        return match.group(1)  # Return the matched JSON-like block
    return None


def filter_and_refine_keywords(keywords: List[str], text: str, client) -> List[str]:
    filter_prompt = (
        f"Review these extracted keywords: {keywords}\n"
        f"1. Filter out any keyword texts unrelated to Python programming.\n"
        f"2. Identify keywords that contain special characters or appear as incomplete sentences. Extract valid Python-related concepts from them if possible.\n"
        f"3. Remove any keywords that remain invalid or incomplete after refinement.\n"
        f"Include: language features, concepts, libraries, frameworks, tools, maths concepts that can be demonstrated through python.\n"
        f"Exclude: General computing terms unrelated to Python.\n"
        f"Remove:\n"
        f"- Pure numbers or dates\n"
        f"- Book references or page numbers\n"
        f"- Non-technical and non-Python programming terms\n\n"
        f"Return only the refined list of keywords as list of concepts relevant to Python programming in the following format:\n"
        f"['concept1', 'concept2', 'concept3', ...]\n"
    )

    response = client.get_model_response(
        system_prompt="You are a Python expert skilled in identifying and refining programming-related keywords.",
        user_prompt=filter_prompt,
        max_new_tokens=200,
        temperature=0.5,
        top_k=50,
        top_p=1,
    )

    refined_keywords = [kw.strip() for kw in response.strip().split("\n") if kw.strip()]
    return refined_keywords


def generate_and_refine_concepts(
    keywords: List[str], text: str, client
) -> List[Dict[str, str]]:
    if isinstance(keywords, str):
        keyword_list = keywords.split("\n")
    else:
        keyword_list = keywords

    # Format keywords for prompt
    formatted_keywords = ", ".join(keyword_list)

    additional_concepts_prompt = (
        f"Given the following text chunk and the extracted Python programming keywords:\n\n"
        f"Text Chunk:\n{text}\n\n"
        f"Keywords:\n{keywords}\n\n"
        f"Identify any additional Python programming concepts that are present in the text but not yet captured in the list of keywords.\n"
        f"Focus on unique, meaningful concepts that provide deeper insights into Python programming fundamentals from basic to advanced concepts."
        f"Return the only list of additional concepts relevant to Python programming in the following format:\n"
        f"\n"
        f"['concept1', 'concept2', 'concept3', ...]\n"  # Example on a single line
    )

    additional_concepts_response = client.get_model_response(
        system_prompt="You are an expert at extracting programming concepts.",
        user_prompt=additional_concepts_prompt,
        max_new_tokens=200,
        temperature=0.5,
        top_k=50,
        top_p=1,
    )

    # Split additional concepts into list
    additional_concepts = [c.strip() for c in additional_concepts_response.split(",")]

    # Combine all concepts
    combined_concepts = keyword_list + additional_concepts

    detailed_concepts_prompt = (
        f"Given the text chunk, generate detailed descriptions and examples for these Python programming concepts: {list(combined_concepts)}\n\n"
        f"tExtract descriptions and 2-3 examples for each concept directly from the text chunk:\n\n"
        f"Text Chunk:\n{text}\n\n"
        f"Format your response as a JSON-like structure:\n"
        f"{{\n"
        f'  "Concept": "Concept Name",\n'
        f'  "Description": "Detailed description from text",\n'
        f'  "Examples": ["Example 1", "Example 2", "Example 3"]\n'
        f"}}"
    )

    concept_response = client.get_model_response(
        system_prompt="You are an expert at extracting programming concepts, associated descriptions and examples from a given text chunk.",
        user_prompt=detailed_concepts_prompt,
        max_new_tokens=2000,
        temperature=0.5,
        top_k=50,
        top_p=1,
    )

    return concept_response


# Process remaining text chunks
def generate_concepts(data, client):
    conc_id = 0
    concept_responses = []
    for topic_key, subtopics in tqdm(data.items(), desc="Processing Documents"):
        # print(f"\n### Processing Topic: {topic_key} ###")

        for subtopic_key, subtopic_value in subtopics.items():
            try:
                chunks = [
                    subtopic_value[i : i + 4000]
                    for i in range(0, len(subtopic_value), 4000)
                ]

                # for chunk in chunks:
                for chunk in tqdm(
                    chunks, desc=f"Processing Subtopic: {subtopic_key}", leave=False
                ):
                    # Extract keywords
                    keywords = extract_keywords(chunk)

                    # Filter and refine keywords
                    keyword_texts = [kw["text"] for kw in keywords]
                    refined_keywords = filter_and_refine_keywords(
                        keyword_texts, subtopic_value, client
                    )

                    # Generate and refine concepts
                    concept_response = generate_and_refine_concepts(
                        refined_keywords, subtopic_value, client
                    )

                    # Extract JSON block and parse the response
                    json_blocks = extract_json_block(concept_response)
                    if json_blocks:
                        concept_response = json_blocks

                    parsed_response = parse_model_response(
                        subtopic_value,
                        chunk,
                        concept_response,
                        topic_key,
                        subtopic_key,
                        conc_id,
                    )

                    conc_id += len(parsed_response)

                    print("\n")
                    concept_responses.extend(parsed_response)
                    # print(f"concept response {concept_responses}\n")
            except Exception:
                print("No responses\n")

    return concept_responses


def generate_and_refine_concepts_from_list(
    keywords: List[str], client
) -> List[Dict[str, str]]:
    if isinstance(keywords, str):
        keyword_list = keywords.split("\n")
    else:
        keyword_list = keywords

    concepts = keyword_list

    detailed_concepts_prompt = (
        f"Generate detailed descriptions and examples for these Python programming concepts: {list(concepts)}\n\n"
        f"Generate descriptions and 2-3 examples for each concept:\n\n"
        f"Format your response as a JSON-like structure:\n"
        f"{{\n"
        f'  "Concept": "Concept Name",\n'
        f'  "Description": "Detailed description from text",\n'
        f'  "Examples": ["Example 1", "Example 2", "Example 3"]\n'
        f"}}"
    )

    concept_response = client.get_model_response(
        system_prompt="You are an expert at extracting associated descriptions and examples from given programming concepts.",
        user_prompt=detailed_concepts_prompt,
        max_new_tokens=2500,
        temperature=0.5,
        top_k=50,
        top_p=1,
    )

    return concept_response


# Remove extra junk keywords from keyword list that LLM
# might have inserted
def remove_junk_keywords(keywords: List[str]) -> List[str]:
    junk_kws = [
        "To refine the list of keywords",
        "Filter out unrelated keywords",
        "Filter out any keyword",
        "Identify and refine keywords",
        "Remove invalid or incomplete",
    ]

    new_keywords = [
        kw for kw in keywords if not any(sub.lower() in kw.lower() for sub in junk_kws)
    ]

    return new_keywords


# Process concepts given as a list of concepts
def generate_concepts_from_list(data, client):
    conc_id = 0
    concept_responses = []
    chunk_size = 10

    print("\nProcessing concepts from list of concepts.")

    for topic_key, subtopics in tqdm(data.items(), desc="Processing Documents"):
        # print(f"\n### Processing Topic: {topic_key} ###")

        for subtopic_key, subtopic_value in subtopics.items():
            try:
                concept_lines = subtopic_value.strip().split("\n")
                chunks = [
                    concept_lines[i : i + chunk_size]
                    for i in range(0, len(concept_lines), chunk_size)
                ]
                # for chunk in chunks:
                for chunk in tqdm(
                    chunks, desc=f"Processing Subtopic: {subtopic_key}", leave=False
                ):
                    # Filter and refine keywords
                    refined_keywords = filter_and_refine_keywords(chunk, "", client)

                    refined_keywords = remove_junk_keywords(refined_keywords)

                    # Generate and refine concepts
                    concept_response = generate_and_refine_concepts_from_list(
                        refined_keywords, client
                    )

                    # Extract JSON block and parse the response
                    json_blocks = extract_json_block(concept_response)
                    if json_blocks:
                        concept_response = json_blocks

                    parsed_response = parse_model_response(
                        subtopic_value,
                        chunk,
                        concept_response,
                        topic_key,
                        subtopic_key,
                        conc_id,
                    )

                    conc_id += len(parsed_response)
                    # print("Parsed Response:", parsed_response)
                    print("\n")
                    concept_responses.extend(parsed_response)
            except Exception:
                print("\n")

    return concept_responses


def generate_concepts_concurrent(data, client):
    conc_id = 0
    concept_responses = []

    def process_subtopic(topic_key, subtopic_key, subtopic_value):
        """Process a single subtopic"""
        try:
            chunks = [
                subtopic_value[i : i + 4000]
                for i in range(0, len(subtopic_value), 4000)
            ]

            subtopic_responses = []
            subtopic_conc_id = 0

            for chunk in tqdm(
                chunks, desc=f"Processing Subtopic: {subtopic_key}", leave=False
            ):
                # Extract keywords
                keywords = extract_keywords(chunk)

                # Filter and refine keywords
                keyword_texts = [kw["text"] for kw in keywords]
                refined_keywords = filter_and_refine_keywords(
                    keyword_texts, subtopic_value, client
                )

                # Generate and refine concepts
                concept_response = generate_and_refine_concepts(
                    refined_keywords, subtopic_value, client
                )

                # Extract JSON block and parse the response
                json_blocks = extract_json_block(concept_response)
                if json_blocks:
                    concept_response = json_blocks

                parsed_response = parse_model_response(
                    subtopic_value,
                    chunk,
                    concept_response,
                    topic_key,
                    subtopic_key,
                    subtopic_conc_id,
                )

                subtopic_conc_id += len(parsed_response)
                subtopic_responses.extend(parsed_response)

            return subtopic_responses, subtopic_conc_id
        except Exception as e:
            print(f"Error processing subtopic {subtopic_key}: {e}")
            return [], 0

    # Prepare all subtopic tasks
    subtopic_tasks = []
    for topic_key, subtopics in data.items():
        for subtopic_key, subtopic_value in subtopics.items():
            subtopic_tasks.append((topic_key, subtopic_key, subtopic_value))

    # Process all subtopics concurrently using executor.map
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            tqdm(
                executor.map(lambda x: process_subtopic(*x), subtopic_tasks),
                total=len(subtopic_tasks),
                desc="Processing Subtopics",
            )
        )

    # Aggregate results
    for subtopic_responses, subtopic_conc_id in results:
        concept_responses.extend(subtopic_responses)
        conc_id += subtopic_conc_id

    return concept_responses
