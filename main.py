import io
from typing import List, Optional
import pynput
from PIL import ImageGrab, Image
import pydantic
import pync
import base64
from ollama import Client
import os
from dotenv import load_dotenv
import requests
import json
import pytesseract  # Added missing import for OCR

# Load environment variables from .env file
load_dotenv()

# Configuration
PLAY_LOCAL = False  # Set to True for local mode, False for online mode
DEBUG_LOGS = False  # Set to True to print raw OCR/API debug information
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # DeepSeek API key for online mode
DEEPSEEK_API_URL = (
    "https://api.deepseek.com/v1/chat/completions"  # DeepSeek API endpoint
)
DEEPSEEK_MODEL = "deepseek-chat"  # Model for online mode: 'deepseek-chat' (DeepSeek-V3, general-purpose, supports JSON output) or 'deepseek-reasoner' (DeepSeek-R1, best for programming/math/logic but does not support JSON output)
USE_JSON_RESPONSE_FORMAT = True  # Set to False to disable response_format={"type": "json_object"} if the model does not support it
# To set the API key, run: export DEEPSEEK_API_KEY="your_key" in the terminal
# or add it to a .env file with: DEEPSEEK_API_KEY=your_key
# Note: deepseek-reasoner does not support response_format={"type": "json_object"}. Using deepseek-chat instead.
# Online mode extracts text via OCR and sends it to the API.
# If image support is needed directly, confirm the model's capabilities.
# Ensure pytesseract and Tesseract OCR are installed:
# pip install pytesseract
# On macOS: brew install tesseract
# On Ubuntu: sudo apt-get install tesseract-ocr
# On Windows: Download and install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki


class MultipleChoiceResponse(pydantic.BaseModel):
    """
    Model containing the explanation of the question, reasoning behind the answer, and the answer itself.
    """

    explanation_of_question: str = pydantic.Field(
        ..., description="Explanation of the test question."
    )
    reasoning: str = pydantic.Field(..., description="Reasoning behind the answer.")
    is_single_answer: bool = pydantic.Field(
        ..., description="True if the question requires a single answer."
    )
    is_multiple_answer: bool = pydantic.Field(
        ..., description="True if multiple answers are allowed."
    )
    answer: List[int] = pydantic.Field(
        ...,
        description="Answer(s) to the question. Single value for single-answer (e.g., [3]), \
        multiple values for multi-answer (e.g., [2, 4, 5]).",
    )

    @pydantic.model_validator(mode="after")
    def validate_answer_conditions(self):
        self.answer = sorted(self.answer)
        if self.is_single_answer:
            if len(self.answer) != 1:
                raise ValueError(
                    "If 'is_single_answer' is True, 'answer' must contain exactly one item."
                )
            if self.is_multiple_answer:
                raise ValueError(
                    "'is_multiple_answer' must be False when 'is_single_answer' is True."
                )
        return self


class ProgrammingVersion(pydantic.BaseModel):
    """
    A single step in the progressive construction of a programming solution.
    """

    version_label: str = pydantic.Field(
        ..., description="Label for the version, e.g. 'Version 1 - Initial attempt'."
    )
    change_summary: str = pydantic.Field(
        ..., description="Short description of what changed from the previous version."
    )
    code: str = pydantic.Field(..., description="Code for this version.")
    known_issue: Optional[str] = pydantic.Field(
        None, description="Short description of the main issue still present."
    )
    is_correct: bool = pydantic.Field(
        ...,
        description="True only when this version is already correct. Usually false until the final version.",
    )


def validate_progressive_version_sequence(
    versions: List[ProgrammingVersion], field_name: str
) -> List[ProgrammingVersion]:
    """Allow only the last progressive version to be marked as correct."""
    if not 3 <= len(versions) <= 4:
        raise ValueError(f"'{field_name}' must contain 3 or 4 versions.")

    correct_indexes = [index for index, version in enumerate(versions) if version.is_correct]
    if len(correct_indexes) > 1:
        raise ValueError(
            f"'{field_name}' can mark at most one version as correct."
        )
    if correct_indexes and correct_indexes[0] != len(versions) - 1:
        raise ValueError(
            f"Only the last item in '{field_name}' can be marked as correct."
        )

    return versions


class ProgrammingProblemResponse(pydantic.BaseModel):
    """
    Model containing the analysis of a programming problem and its solution.
    """

    problem_description: str = pydantic.Field(
        ..., description="Detailed description of the programming problem."
    )
    required_output_format: Optional[str] = pydantic.Field(
        None, description="Required output format if specified."
    )
    required_function_name: Optional[str] = pydantic.Field(
        None, description="Required function name if specified."
    )
    programming_language: str = pydantic.Field(
        ...,
        description="Target programming language, default to Python if not deducible.",
    )
    progressive_versions: List[ProgrammingVersion] = pydantic.Field(
        ...,
        description="Three or four progressively improved versions of the solution.",
    )
    solution_code: str = pydantic.Field(
        ..., description="Final correct code solution to the problem."
    )
    explanation: str = pydantic.Field(
        ..., description="Explanation of the progression and the final solution approach."
    )

    @pydantic.model_validator(mode="after")
    def validate_progressive_versions(self):
        validate_progressive_version_sequence(
            self.progressive_versions, "progressive_versions"
        )
        return self


class GeneralProblemResponse(pydantic.BaseModel):
    """
    Model for online mode responses, handling various problem types.
    """

    problem_type: str = pydantic.Field(
        ...,
        description="Type of problem (e.g., multiple-choice, programming, math, logic, language).",
    )
    solution: str | List[int] = pydantic.Field(
        ...,
        description="Solution or answer(s). List of integers for multiple-choice, string for others.",
    )
    programming_language: Optional[str] = pydantic.Field(
        None, description="Target programming language for programming problems."
    )
    programming_versions: Optional[List[ProgrammingVersion]] = pydantic.Field(
        None,
        description="Three or four progressive versions for programming problems only.",
    )
    explanation: str = pydantic.Field(
        ..., description="Explanation of the solution or reasoning."
    )
    # Added fields for multiple-choice questions to align with local mode
    is_single_answer: Optional[bool] = pydantic.Field(
        None,
        description="True if the question requires a single answer (for multiple-choice only).",
    )
    is_multiple_answer: Optional[bool] = pydantic.Field(
        None,
        description="True if multiple answers are allowed (for multiple-choice only).",
    )

    @pydantic.model_validator(mode="after")
    def validate_programming_fields(self):
        if self.problem_type == "programming":
            if not isinstance(self.solution, str):
                raise ValueError("Programming solutions must be returned as a string.")
            if not self.programming_language:
                raise ValueError(
                    "Programming problems must include 'programming_language'."
                )
            if not self.programming_versions or not (
                3 <= len(self.programming_versions) <= 4
            ):
                raise ValueError(
                    "Programming problems must include 3 or 4 'programming_versions'."
                )
            validate_progressive_version_sequence(
                self.programming_versions, "programming_versions"
            )
        return self


def debug_print(message: str):
    """Print verbose debug information only when debugging is enabled."""
    if DEBUG_LOGS:
        print(message)


def normalize_multiline_text(text: str) -> str:
    """Convert escaped newlines to real ones and trim extra blank lines."""
    return text.replace("\\n", "\n").strip("\n")


def indent_block(text: str, spaces: int = 4) -> str:
    """Indent a multiline block for cleaner console output."""
    prefix = " " * spaces
    normalized = normalize_multiline_text(text)
    return "\n".join(f"{prefix}{line}" if line else "" for line in normalized.splitlines())


def print_programming_response(
    programming_language: str,
    progressive_versions: List[ProgrammingVersion],
    final_solution: str,
    explanation: str,
):
    """Print a programming response using clearly separated progressive sections."""
    print("----------------> Processing CODE QUESTION")
    print("=" * 72)
    print(f"Language: {programming_language}")
    print(
        "Progression: realistic human-style attempts, each one derived from the previous"
    )
    print("=" * 72)

    for index, version in enumerate(progressive_versions, start=1):
        print(f"\nVersion {index}: {version.version_label}")
        print(f"Change: {version.change_summary}")
        if version.is_correct:
            print("Status: correct draft")
        if version.known_issue:
            print(f"Known issue: {version.known_issue}")
        print("Code:")
        print(indent_block(version.code))

    print("\nFinal Correct Version")
    print("-" * 72)
    print(indent_block(final_solution))
    print("\nExplanation:")
    print(indent_block(explanation, spaces=2))
    print("=" * 72)


def is_trigger_key(key) -> bool:
    """Accept the main modifier keys used to trigger capture on macOS."""
    return key in {
        pynput.keyboard.Key.cmd,
        pynput.keyboard.Key.cmd_l,
        pynput.keyboard.Key.cmd_r,
        pynput.keyboard.Key.alt_l,
    }


def image_to_base64(image: Image) -> str:
    """Convert a PIL Image to a base64-encoded string."""
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    return base64.b64encode(img_bytes.getvalue()).decode("utf-8")


def get_multiple_choice_response(image: Image) -> MultipleChoiceResponse:
    """Analyze a multiple-choice question image using DeepSeek locally."""
    client = Client(host="http://localhost:11434")
    base64_image = image_to_base64(image)

    prompt = """You must respond ONLY with a valid JSON object in the following format, no other text:
    {
        "explanation_of_question": "Explanation of the test question",
        "reasoning": "Your reasoning for the answer",
        "is_single_answer": true,
        "is_multiple_answer": false,
        "answer": [1]
    }
    Analyze the image and fill in appropriate values, maintaining this exact JSON structure."""

    response = client.generate(
        model="deepseek-coder-v2",
        prompt=prompt,
        images=[base64_image],
        format="json",
        options={"temperature": 0},
    )

    try:
        return MultipleChoiceResponse.model_validate_json(response.response)
    except Exception as e:
        print(f"Raw response: {response.response}")
        raise e


def get_programming_problem_response(
    image: Image, extracted_text: str
) -> ProgrammingProblemResponse:
    """Analyze a programming problem image using DeepSeek locally."""
    client = Client(host="http://localhost:11434")
    base64_image = image_to_base64(image)

    prompt = """You must respond ONLY with a valid JSON object in the following format, no other text:
    {
        "problem_description": "The programming problem asks...",
        "required_output_format": null,
        "required_function_name": null,
        "programming_language": "Python",
        "progressive_versions": [
            {
                "version_label": "Version 1 - Initial attempt",
                "change_summary": "Starts from the basic idea, but still contains a realistic mistake.",
                "code": "def solution():\\n    pass",
                "known_issue": "The logic is incomplete or wrong in one realistic way.",
                "is_correct": false
            },
            {
                "version_label": "Version 2 - Improved attempt",
                "change_summary": "Builds directly on version 1 and fixes part of the issue.",
                "code": "def solution():\\n    pass",
                "known_issue": "Still misses an edge case or requirement.",
                "is_correct": false
            },
            {
                "version_label": "Version 3 - Almost correct",
                "change_summary": "Builds directly on version 2 and is close to the final answer.",
                "code": "def solution():\\n    pass",
                "known_issue": "Minor remaining issue before the final correct version.",
                "is_correct": false
            }
        ],
        "solution_code": "def solution():\\n    pass",
        "explanation": "Explain the progression and why the final version is correct."
    }
    Instructions:
    - Return 3 or 4 progressive_versions depending on problem complexity.
    - Each version must be clearly based on the previous one, not a full rewrite.
    - Early versions must contain minor realistic human mistakes or missing edge cases.
    - The final solution_code must be correct.
    - Avoid over-optimized or overly polished AI-looking code.
    - Keep the style practical and human, not excessively clever.

    Extracted text:
    """
    prompt += extracted_text

    response = client.generate(
        model="deepseek-coder-v2",
        prompt=prompt,
        images=[base64_image],
        format="json",
        options={"temperature": 0},
    )

    try:
        return ProgrammingProblemResponse.model_validate_json(response.response)
    except Exception as e:
        print(f"Raw response: {response.response}")
        raise e


def get_general_problem_response(image: Image) -> GeneralProblemResponse:
    """Analyze an image using DeepSeek API online and deduce the problem type."""
    if not DEEPSEEK_API_KEY:
        raise ValueError(
            "DEEPSEEK_API_KEY is not set. Please set it in the environment or .env file."
        )

    # Extract text from the image using OCR (since deepseek-reasoner may not support images)
    extracted_text = extract_text_from_image(image)
    debug_print("\n======================================")
    debug_print(f"Extracted Text for Online Mode: {extracted_text}")
    debug_print("======================================\n")

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    prompt = f"""The following is text extracted from an image. Analyze it and deduce the problem type (e.g., multiple-choice, programming, math, logic).
    Respond ONLY with a valid JSON object in the following format, no other text:

    For multiple-choice questions:
    {{
        "problem_type": "multiple-choice",
        "solution": [1] or [1, 2] for multiple-choice,
        "programming_language": null,
        "explanation": "A concise explanation of the reasoning (1-2 sentences, similar to local mode).",
        "is_single_answer": true,
        "is_multiple_answer": false
    }}

    For other problem types (e.g., programming, math, logic):
    {{
        "problem_type": "programming | math | logic",
        "solution": "The code solution or the solution process or reasoning",
        "programming_language": null | "Python",
        "explanation": "Detailed explanation of the solution or reasoning",
        "is_single_answer": false,
        "is_multiple_answer": false,
        "programming_versions": [
            {{
                "version_label": "Version 1 - Initial attempt",
                "change_summary": "Basic idea with a realistic mistake.",
                "code": "def solution():\\n    pass",
                "known_issue": "Short description of the main issue.",
                "is_correct": false
            }},
            {{
                "version_label": "Version 2 - Improved attempt",
                "change_summary": "Builds directly on version 1.",
                "code": "def solution():\\n    pass",
                "known_issue": "Still has a remaining issue.",
                "is_correct": false
            }},
            {{
                "version_label": "Version 3 - Almost correct",
                "change_summary": "Builds directly on version 2 and is close to correct.",
                "code": "def solution():\\n    pass",
                "known_issue": "Minor remaining issue before the final answer.",
                "is_correct": false
            }}
        ]
    }}

    Instructions:
    - For multiple-choice questions, provide a list of integers for the answer(s) and keep the explanation concise (1-2 sentences, similar to local mode).
    - For programming questions, deduce the target programming language and write the final correct code in "solution".
    - For programming questions, include 3 or 4 items in "programming_versions".
    - Each programming version must be clearly based on the previous one.
    - The early programming versions must have minor realistic bugs, missing cases, or small requirement mistakes.
    - Keep the programming code reasonably simple and human-like, not overly optimized.
    - For non-programming problems, omit "programming_versions".
    - For math/logic questions, provide the solution as a string and include a detailed explanation.
    - Ensure the JSON structure matches the specified format exactly.

    Extracted Text:
    {extracted_text}
    """

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }

    # Add response_format only if USE_JSON_RESPONSE_FORMAT is True
    if USE_JSON_RESPONSE_FORMAT:
        payload["response_format"] = {"type": "json_object"}
        debug_print("Using response_format={'type': 'json_object'} in API request")
    else:
        debug_print("Skipping response_format in API request")

    # Debug: Log request details (mask API key for security)
    debug_print(f"Sending request to {DEEPSEEK_API_URL}")
    debug_print(f"API key present: {'Yes' if DEEPSEEK_API_KEY else 'No'}")
    debug_print(f"Model: {DEEPSEEK_MODEL}")

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        response.raise_for_status()

        # Debug: Print raw response body before parsing
        debug_print(f"Raw response body: {response.text}")

        try:
            json_response = response.json()
        except json.JSONDecodeError as e:
            debug_print(f"Failed to parse response as JSON: {e}")
            debug_print(f"Raw response body: {response.text}")
            raise

        # Debug: Print parsed JSON response
        debug_print(f"Parsed JSON response: {json_response}")

        # Check for expected structure
        if "choices" not in json_response or not json_response["choices"]:
            raise ValueError("Response does not contain 'choices' array.")
        if (
            "message" not in json_response["choices"][0]
            or "content" not in json_response["choices"][0]["message"]
        ):
            raise ValueError(
                "Response does not contain expected 'message' or 'content' fields."
            )

        # Parse the content as JSON if response_format was used, otherwise treat as a JSON string
        content = json_response["choices"][0]["message"]["content"]
        if USE_JSON_RESPONSE_FORMAT:
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError as e:
                    debug_print(f"Failed to parse content as JSON: {e}")
                    debug_print(f"Content: {content}")
                    raise
        else:
            # If response_format is not used, content should already be a JSON string
            try:
                content = json.loads(content)
            except json.JSONDecodeError as e:
                debug_print(f"Failed to parse content as JSON: {e}")
                debug_print(f"Content: {content}")
                raise

        return GeneralProblemResponse.model_validate(content)

    except requests.exceptions.HTTPError as e:
        error_response = response.json() if response.content else {}
        if error_response.get("error", {}).get("message") == "Model Not Exist":
            raise ValueError(
                f"Model '{DEEPSEEK_MODEL}' does not exist. Use 'deepseek-reasoner' (DeepSeek-R1) or 'deepseek-chat' (DeepSeek-V3)."
            ) from e
        debug_print(f"HTTP Error: {e}")
        debug_print(f"Response body: {response.text}")
        raise
    except Exception as e:
        debug_print(f"Error processing response: {e}")
        if "response" in locals():
            debug_print(f"Raw response body: {response.text}")
        raise


def extract_text_from_image(image: Image) -> str:
    """Extract text from an image using Tesseract OCR."""
    return pytesseract.image_to_string(image)


def save_debug_image(image: Image):
    """Save the captured image for debugging."""
    if not DEBUG_LOGS:
        return
    image_path = "screenshot.png"
    image.save(image_path)
    print(f"Image saved as: {image_path}")


def notify(message: str, title: str = "Gorilla Test 🦍"):
    """Send a system notification."""
    pync.Notifier.notify(message, title=title)


def on_press(key):
    """Handle keypress to process questions."""
    if is_trigger_key(key):
        notify("Processing question... 🤔")
        try:
            image = ImageGrab.grab()
            save_debug_image(image)

            if PLAY_LOCAL:
                # Local mode: Use OCR and keyword-based logic
                extracted_text = extract_text_from_image(image)
                debug_print("\n======================================")
                debug_print(f"Extracted Text: {extracted_text}")
                debug_print("======================================\n")

                if (
                    "choice" in extracted_text.lower()
                    or "select" in extracted_text.lower()
                ):
                    print("----------------> Processing MULTIPLE CHOICE QUESTION")
                    response = get_multiple_choice_response(image)
                    print(response)
                    answer = (
                        response.answer[0]
                        if response.is_single_answer
                        else ", ".join(map(str, response.answer))
                    )
                    notify(f"Answer: {answer}")
                elif (
                    "function" in extracted_text.lower()
                    or "code" in extracted_text.lower()
                ):
                    response = get_programming_problem_response(image, extracted_text)
                    print_programming_response(
                        response.programming_language,
                        response.progressive_versions,
                        response.solution_code,
                        response.explanation,
                    )
                    notify(f"Solution: {response.solution_code}")
                else:
                    notify("Could not determine question type")
            else:
                # Online mode: Let DeepSeek deduce the problem type using extracted text
                response = get_general_problem_response(image)
                # Align output format with local mode for multiple-choice questions
                if response.problem_type == "multiple-choice":
                    print("----------------> Processing MULTIPLE CHOICE QUESTION")
                    print(
                        f"explanation_of_question='The task is to identify the correct answer from multiple choices.' "
                        f"reasoning='{response.explanation}' "
                        f"is_single_answer={response.is_single_answer} "
                        f"is_multiple_answer={response.is_multiple_answer} "
                        f"answer={response.solution}"
                    )
                    answer = (
                        response.solution[0]
                        if response.is_single_answer
                        else ", ".join(map(str, response.solution))
                    )
                    notify(f"Answer: {answer}")
                elif response.problem_type == "programming":
                    print_programming_response(
                        response.programming_language,
                        response.programming_versions,
                        response.solution,
                        response.explanation,
                    )
                    notify(
                        f"Solution ({response.programming_language}):\n{response.solution}"
                    )
                elif response.problem_type in ("math", "logic"):
                    print(
                        f"----------------> Processing {response.problem_type.upper()} QUESTION"
                    )
                    print(f"problem_type: {response.problem_type}")
                    print("solution:")
                    print(indent_block(response.solution))
                    print(f"explanation: {response.explanation}")
                    print(f"is_single_answer: {response.is_single_answer}")
                    print(f"is_multiple_answer: {response.is_multiple_answer}")
                    notify(f"Solution ({response.problem_type}):\n{response.solution}")
                else:
                    print(response)
                    notify(f"Solution ({response.problem_type}):\n{response.solution}")

        except Exception as e:
            print(f"Error: {e}")
            notify("An error occurred")


def main():
    """Initialize the script and start listening for keypresses."""
    print("Listening for keypresses...")
    notify("Script started")
    with pynput.keyboard.Listener(on_press=on_press) as listener:
        listener.join()


if __name__ == "__main__":
    main()
