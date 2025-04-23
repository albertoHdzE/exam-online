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
        description="Answer(s) to the question. Single value for single-answer (e.g., [3]), multiple values for multi-answer (e.g., [2, 4, 5]).",
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
    solution_code: str = pydantic.Field(
        ..., description="Code solution to the problem."
    )
    explanation: str = pydantic.Field(
        ..., description="Explanation of the solution approach."
    )


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
        "solution_code": "def solution():\\n    pass",
        "explanation": "This solution works by..."
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
    print("\n======================================")
    print(f"Extracted Text for Online Mode: {extracted_text}")
    print("======================================\n")

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
        "is_multiple_answer": false
    }}

    Instructions:
    - For multiple-choice questions, provide a list of integers for the answer(s) and keep the explanation concise (1-2 sentences, similar to local mode).
    - For programming questions, deduce the target programming language and write the code that solve the problem.
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
        print("Using response_format={'type': 'json_object'} in API request")
    else:
        print("Skipping response_format in API request")

    # Debug: Log request details (mask API key for security)
    print(f"Sending request to {DEEPSEEK_API_URL}")
    print(f"API key present: {'Yes' if DEEPSEEK_API_KEY else 'No'}")
    print(f"Model: {DEEPSEEK_MODEL}")

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        response.raise_for_status()

        # Debug: Print raw response body before parsing
        print(f"Raw response body: {response.text}")

        try:
            json_response = response.json()
        except json.JSONDecodeError as e:
            print(f"Failed to parse response as JSON: {e}")
            print(f"Raw response body: {response.text}")
            raise

        # Debug: Print parsed JSON response
        print(f"Parsed JSON response: {json_response}")

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
                    print(f"Failed to parse content as JSON: {e}")
                    print(f"Content: {content}")
                    raise
        else:
            # If response_format is not used, content should already be a JSON string
            try:
                content = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"Failed to parse content as JSON: {e}")
                print(f"Content: {content}")
                raise

        return GeneralProblemResponse.model_validate(content)

    except requests.exceptions.HTTPError as e:
        error_response = response.json() if response.content else {}
        if error_response.get("error", {}).get("message") == "Model Not Exist":
            raise ValueError(
                f"Model '{DEEPSEEK_MODEL}' does not exist. Use 'deepseek-reasoner' (DeepSeek-R1) or 'deepseek-chat' (DeepSeek-V3)."
            ) from e
        print(f"HTTP Error: {e}")
        print(f"Response body: {response.text}")
        raise
    except Exception as e:
        print(f"Error processing response: {e}")
        if "response" in locals():
            print(f"Raw response body: {response.text}")
        raise


def extract_text_from_image(image: Image) -> str:
    """Extract text from an image using Tesseract OCR."""
    return pytesseract.image_to_string(image)


def save_debug_image(image: Image):
    """Save the captured image for debugging."""
    image_path = "screenshot.png"
    image.save(image_path)
    print(f"Image saved as: {image_path}")


def notify(message: str, title: str = "Gorilla Test 🦍"):
    """Send a system notification."""
    pync.Notifier.notify(message, title=title)


def on_press(key):
    """Handle keypress to process questions."""
    if key == pynput.keyboard.Key.alt_l:
        notify("Processing question... 🤔")
        try:
            image = ImageGrab.grab()
            save_debug_image(image)

            if PLAY_LOCAL:
                # Local mode: Use OCR and keyword-based logic
                extracted_text = extract_text_from_image(image)
                print("\n======================================")
                print(f"Extracted Text: {extracted_text}")
                print("======================================\n")

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
                    print("----------------> Processing CODE QUESTION")
                    response = get_programming_problem_response(image, extracted_text)
                    print(response)
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
                    print("----------------> Processing CODE QUESTION")
                    print(f"problem_type: {response.problem_type}")
                    print(f"programming_language: {response.programming_language}")
                    print("solution:")
                    # Replace escaped newlines with actual newlines for proper formatting
                    formatted_solution = response.solution.replace("\\n", "\n")
                    # Print the solution with a slight indent for readability
                    print(
                        "\n".join(
                            "    " + line for line in formatted_solution.splitlines()
                        )
                    )
                    print(f"explanation: {response.explanation}")
                    print(f"is_single_answer: {response.is_single_answer}")
                    print(f"is_multiple_answer: {response.is_multiple_answer}")
                    notify(
                        f"Solution ({response.programming_language}):\n{response.solution}"
                    )
                elif response.problem_type in ("math", "logic"):
                    print(
                        f"----------------> Processing {response.problem_type.upper()} QUESTION"
                    )
                    print(f"problem_type: {response.problem_type}")
                    print("solution:")
                    # Replace escaped newlines with actual newlines for math/logic solutions
                    formatted_solution = response.solution.replace("\\n", "\n")
                    # Print the solution with a slight indent for readability
                    print(
                        "\n".join(
                            "    " + line for line in formatted_solution.splitlines()
                        )
                    )
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
