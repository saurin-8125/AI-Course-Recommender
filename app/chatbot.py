# AI Course Recommender - Chatbot
# This file contains the logic for the chatbot.

import openai

from .config import CHATBOT_CONFIG

# Set the OpenAI API key
openai.api_key = CHATBOT_CONFIG["api_key"]


def get_chatbot_response(user_message: str) -> str:
    """
    Get a response from the chatbot.
    """
    if not openai.api_key or openai.api_key == "YOUR_OPENAI_API_KEY":
        return "Please configure your OpenAI API key in app/config.py"
    try:
        response = openai.chat.completions.create(
            model=CHATBOT_CONFIG["model"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"
