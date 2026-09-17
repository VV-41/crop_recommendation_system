# chatbot.py
import os
import requests
from typing import Callable, Dict, Any, Optional

# Reads API key from environment (recommended)
# Set in .env as: OPENROUTER_API_KEY=your_key_her
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def _call_openrouter(prompt: str, model: str = "openai/gpt-4o-mini") -> str:
    """
    Calls OpenRouter chat completions.
    This is only used for allowed commands like /tips and /ask.
    """
    if not OPENROUTER_API_KEY:
        return "Server error: OPENROUTER_API_KEY missing. Add it in your .env file."

    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an agriculture assistant. "
                            "Reply in simple Hinglish. "
                            "Be practical, short steps, avoid unsafe chemical dosage claims. "
                            "If unsure, say so."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.4,
            },
            timeout=30,
        )

        data = r.json()
        if r.status_code != 200:
            return f"Chatbot API error ({r.status_code}): {data}"

        if "choices" not in data:
            return f"Chatbot API error: {data}"

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Chatbot error: {e}"


def help_text() -> str:
    return (
        "✅ Allowed commands:\n"
        "1) /help\n"
        "2) /recommend N P K temp humidity ph rainfall\n"
        "   Example: /recommend 90 40 40 25 70 6.5 120\n"
        "3) /tips <crop_name>\n"
        "   Example: /tips rice\n"
        "4) /ask <question>\n"
        "   Example: /ask What is NPK?\n"
    )


def parse_recommend_args(text: str) -> Optional[Dict[str, float]]:
    """
    Parses:
      /recommend N P K temp humidity ph rainfall
    Returns dict of floats or None if invalid.
    """
    parts = text.strip().split()
    if len(parts) != 8:
        return None

    try:
        return {
            "N": float(parts[1]),
            "P": float(parts[2]),
            "K": float(parts[3]),
            "temperature": float(parts[4]),
            "humidity": float(parts[5]),
            "ph": float(parts[6]),
            "rainfall": float(parts[7]),
        }
    except ValueError:
        return None


def handle_message(
    user_text: str,
    predict_fn: Optional[Callable[[Dict[str, float]], Any]] = None,
) -> str:
    """
    Command-based router:
    - Only responds to allowed commands.
    - Calls OpenRouter ONLY for /tips and /ask (optional).
    - Uses your ML model ONLY for /recommend.

    predict_fn:
      A function you pass from app.py that takes the parsed /recommend args
      dict ({N, P, K, temperature, humidity, ph, rainfall}) and returns a
      prediction (string or dict). app.py is responsible for mapping these
      short keys onto the full feature set the model actually needs.
    """
    text = (user_text or "").strip()
    if not text:
        return "Type /help to see commands."

    # 1) HELP
    if text == "/help":
        return help_text()

    # 2) RECOMMEND (ML prediction)
    if text.startswith("/recommend"):
        args = parse_recommend_args(text)
        if args is None:
            return "❌ Usage: /recommend N P K temp humidity ph rainfall\nExample: /recommend 90 40 40 25 70 6.5 120"

        if predict_fn is None:
            return (
                "⚠️ predict_fn not connected.\n"
                "In app.py, pass your model prediction function into handle_message()."
            )

        try:
            pred = predict_fn(args)
            if isinstance(pred, dict):
                crop = pred.get("prediction") or pred.get("crop")
                return f"✅ Recommended crop: {crop}" if crop else f"✅ Recommendation: {pred}"
            return f"✅ Recommended crop: {pred}"
        except Exception as e:
            return f"Prediction error: {e}"

    # 3) TIPS (LLM allowed)
    if text.startswith("/tips"):
        parts = text.split(maxsplit=1)
        if len(parts) != 2 or not parts[1].strip():
            return "❌ Usage: /tips <crop_name>\nExample: /tips rice"
        crop = parts[1].strip()

        prompt = (
            f"Give short farming tips for {crop}. "
            "Include: soil, water, season, common mistakes. "
            "Keep it short bullet points."
        )
        return _call_openrouter(prompt)

    # 4) ASK (LLM allowed, but still command gated)
    if text.startswith("/ask"):
        parts = text.split(maxsplit=1)
        if len(parts) != 2 or not parts[1].strip():
            return "❌ Usage: /ask <question>\nExample: /ask What is NPK?"
        question = parts[1].strip()
        return _call_openrouter(question)

    # Everything else is blocked
    return "❌ Unsupported command. Type /help."