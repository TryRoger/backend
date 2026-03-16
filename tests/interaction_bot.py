"""
Simple Talking Chatbot using Gemini Interactions API (with image support)

Requirements:
    pip install google-genai

Setup:
    export GEMINI_API_KEY="your-api-key-here"

Usage:
    python interaction_bot.py [image_path]

    If an image path is provided, the first message will include the image
    with a request to describe it. Subsequent messages continue the
    conversation so you can ask follow-up questions about the image.
"""

import os
import sys
import time
import base64
import mimetypes
from google import genai


def main():
    # Initialize client (reads GEMINI_API_KEY from environment by default)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Please set the GEMINI_API_KEY environment variable.")
        print("  export GEMINI_API_KEY='your-api-key-here'")
        return

    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-flash"

    # Hardcoded image path
    image_path = os.path.join(os.path.dirname(__file__), "..", "images", "000e0352_20260316_221841.png")
    if not os.path.isfile(image_path):
        print(f"Warning: Image file not found: {image_path}")
        image_path = None

    print("=" * 50)
    print("  Gemini Chatbot (Interactions API)")
    if image_path:
        print(f"  Image loaded: {os.path.basename(image_path)}")
    print("  Type 'quit' or 'exit' to end the conversation.")
    print("=" * 50)
    print()

    previous_interaction_id = None

    # If an image was provided, send it with a describe prompt first
    if image_path:
        try:
            print("Sending image to Gemini for description...\n")
            mime_type, _ = mimetypes.guess_type(image_path)
            if mime_type is None:
                mime_type = "image/png"
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            start = time.time()
            interaction = client.interactions.create(
                model=model,
                input=[
                    {"type": "text", "text": "Describe this image in detail."},
                    {"type": "image", "data": image_data, "mime_type": mime_type},
                ],
            )
            elapsed = time.time() - start

            response_text = None
            for output in interaction.outputs:
                if output.type == "text":
                    response_text = output.text
                    break

            if response_text:
                print(f"Gemini: {response_text}\n")
            else:
                print(f"Gemini: {interaction.outputs[-1].text}\n")
            print(f"[Response time: {elapsed:.2f}s]")

            previous_interaction_id = interaction.id

        except Exception as e:
            print(f"\nError sending image: {e}\n")
            print("(Continuing without image context.)")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        try:
            # Build the request — pass previous_interaction_id for
            # stateful multi-turn conversation
            kwargs = {
                "model": model,
                "input": user_input,
            }
            if previous_interaction_id:
                kwargs["previous_interaction_id"] = previous_interaction_id

            start = time.time()
            interaction = client.interactions.create(**kwargs)
            elapsed = time.time() - start

            # Extract the text response from outputs
            response_text = None
            for output in interaction.outputs:
                if output.type == "text":
                    response_text = output.text
                    break

            if response_text:
                print(f"\nGemini: {response_text}\n")
            else:
                # Fallback: grab the last output's text attribute
                print(f"\nGemini: {interaction.outputs[-1].text}\n")
            print(f"[Response time: {elapsed:.2f}s]")

            # Save interaction ID for the next turn
            previous_interaction_id = interaction.id

        except Exception as e:
            print(f"\nError: {e}\n")
            print("(Conversation context has been reset.)")
            previous_interaction_id = None


if __name__ == "__main__":
    main()
