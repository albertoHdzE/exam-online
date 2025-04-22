import io
from typing import List, Optional
import pynput
from PIL import ImageGrab, Image
import pydantic
import pync
import base64
from ollama import Client
import os
import pytesseract  # Import pytesseract


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
    answer: list[int] = pydantic.Field(
        ...,
        description="Answer(s) to the question, depending on its requirements. The number of checkboxes may vary. If the question requires selecting a single checkbox, provide a single value (e.g., [3]). If the question allows selecting multiple checkboxes, provide a list of the selected values (e.g., [2, 4, 5]).",
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
        None, description="Required output format if specified in the problem."
    )
    required_function_name: Optional[str] = pydantic.Field(
        None, description="Required function name if specified in the problem."
    )
    solution_code: str = pydantic.Field(
        ..., description="Python code solution to the problem."
    )
    explanation: str = pydantic.Field(
        ..., description="Explanation of the solution approach."
    )


def image_to_base64(image: Image) -> str:
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    return base64.b64encode(img_bytes.getvalue()).decode("utf-8")


def get_multiple_choice_response(image) -> MultipleChoiceResponse:
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
        format="json",  # Add this parameter to request JSON output
    )

    # Add error handling for JSON parsing
    try:
        return MultipleChoiceResponse.model_validate_json(response.response)
    except Exception as e:
        print(f"Raw response: {response.response}")
        raise e


def get_programming_problem_response(
    image, extracted_text
) -> ProgrammingProblemResponse:
    client = Client(host="http://localhost:11434")
    base64_image = image_to_base64(image)

    # Update the prompt to deduce the programming language
    prompt = """You must respond ONLY with a valid JSON object in the following format, no other text:
    {
        "problem_description": "The programming problem asks...",
        "required_output_format": null,
        "required_function_name": null,
        "programming_language": "Try to deduce the target programming language based on context and extracted text. If not possible, default to Python.",
        "solution_code": "def solution():\\n    pass",  # Ensure this is a valid string
        "explanation": "This solution works by..."
    }
    Analyze the image and fill in appropriate values, maintaining this exact JSON structure."""

    response = client.generate(
        model="deepseek-coder-v2",
        prompt=prompt,
        images=[base64_image],
        format="json",  # Add this parameter to request JSON output
    )

    # Add error handling for JSON parsing
    try:
        return ProgrammingProblemResponse.model_validate_json(response.response)
    except Exception as e:
        print(f"Raw response: {response.response}")
        raise e


def on_press(key):
    if key == pynput.keyboard.Key.alt_l:
        notify("Processing question... 🤔")
        try:
            image = ImageGrab.grab()
            # Debug: Show and save the captured image
            show_debug_image(image)
            try:
                # Extract text from the image
                extracted_text = extract_text_from_image(image)
                print(f"Extracted Text: {extracted_text}")

                # Determine the type of question based on extracted text
                if (
                    "choice" in extracted_text.lower()
                    or "select" in extracted_text.lower()
                ):
                    # Process as multiple choice
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
                    # Process as programming problem
                    response = get_programming_problem_response(image, extracted_text)
                    print(response)
                    notify(f"Solution: {response.solution_code}")
                else:
                    notify("Could not determine question type")
            except Exception as e:
                print(f"Error: {e}")
                notify("An error occurred")
        except Exception as e:
            print(f"Error: {e}")
            notify("An error occurred")


def show_debug_image(image: Image):
    """Display the captured image and save it"""
    # Save the image in the root directory
    image_path = "screenshot.png"
    image.save(image_path)
    # image.show()
    print(f"Image saved as: {image_path}")


def extract_text_from_image(image: Image) -> str:
    """Extract text from an image using Tesseract OCR."""
    return pytesseract.image_to_string(image)


def notify(message: str, title: str = "Gorilla Test 🦍"):
    pync.Notifier.notify(message, title=title)


def main():
    print("Listening for keypresses...")
    notify("Script started")
    with pynput.keyboard.Listener(on_press=on_press) as listener:
        listener.join()


if __name__ == "__main__":
    main()
