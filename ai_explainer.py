"""
AntigravityAI — Claude-powered explanation engine for VisionAI
Built by Arush Kumar & Ayushi Shukla | github.com/arushkumar-aiml/visionai
"""

import os
import anthropic


class AntigravityAI:
    """
    Uses Claude API to generate plain-English explanations and fun facts
    for image classification results.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = None
        self.available = False

        if self.api_key:
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self.available = True
            except Exception:
                self.available = False

    def explain(self, predicted_class: str, confidence_score: float) -> dict:
        """
        Generate a plain-English explanation + fun fact for a classification result.

        Args:
            predicted_class: The name of the detected class (e.g. "cat", "rose")
            confidence_score: Float between 0 and 1 (e.g. 0.87 = 87%)

        Returns:
            dict with keys: explanation (str), fun_fact (str), confidence_label (str), success (bool)
        """
        confidence_pct = round(confidence_score * 100, 1)

        if not self.available:
            return self._fallback_response(predicted_class, confidence_pct)

        try:
            prompt = f"""You are AntigravityAI, a friendly and enthusiastic AI assistant integrated into an image classifier app.

An image was classified as: "{predicted_class}"
Confidence score: {confidence_pct}%

Please respond with EXACTLY this format (no extra text, no markdown):

EXPLANATION: [Write exactly 3 sentences in simple, friendly language explaining what a {predicted_class} is and what makes it recognizable visually. Make it engaging and accessible for all ages.]

FUN_FACT: [Write exactly 1 fascinating, surprising, or delightful fun fact about {predicted_class} that most people don't know. Start with "Did you know..."]

CONFIDENCE_LABEL: [Based on {confidence_pct}%, write one short phrase (5-10 words) explaining what this confidence level means in plain English. Examples: "Very confident — almost certain!", "Pretty sure, but worth a second look.", "Uncertain — the image may be ambiguous."]"""

            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )

            raw = message.content[0].text.strip()
            return self._parse_response(raw, predicted_class, confidence_pct)

        except anthropic.AuthenticationError:
            self.available = False
            return self._fallback_response(predicted_class, confidence_pct)
        except Exception as e:
            return self._fallback_response(predicted_class, confidence_pct, error=str(e))

    def _parse_response(self, raw: str, predicted_class: str, confidence_pct: float) -> dict:
        """Parse the structured Claude response into a clean dict."""
        result = {
            "explanation": "",
            "fun_fact": "",
            "confidence_label": "",
            "success": True,
        }

        lines = raw.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("EXPLANATION:"):
                result["explanation"] = line[len("EXPLANATION:"):].strip()
            elif line.startswith("FUN_FACT:"):
                result["fun_fact"] = line[len("FUN_FACT:"):].strip()
            elif line.startswith("CONFIDENCE_LABEL:"):
                result["confidence_label"] = line[len("CONFIDENCE_LABEL:"):].strip()

        # Fallback for any missing fields
        if not result["explanation"]:
            result["explanation"] = (
                f"The model detected a {predicted_class} in this image. "
                f"This category represents a specific visual pattern the model has learned to recognize. "
                f"The classifier uses deep learning to identify these features automatically."
            )
        if not result["fun_fact"]:
            result["fun_fact"] = f"Did you know... image classifiers can recognize {predicted_class} using millions of learned visual patterns!"
        if not result["confidence_label"]:
            result["confidence_label"] = self._default_confidence_label(confidence_pct)

        return result

    def _fallback_response(self, predicted_class: str, confidence_pct: float, error: str = None) -> dict:
        """Return a basic response when Claude API is unavailable."""
        return {
            "explanation": (
                f"The VisionAI model identified this image as a {predicted_class}. "
                f"This classification is based on deep learning patterns the model learned during training. "
                f"Add your Anthropic API key to get a detailed AI-powered explanation!"
            ),
            "fun_fact": f"Did you know... deep learning models can classify {predicted_class} images by learning from thousands of examples?",
            "confidence_label": self._default_confidence_label(confidence_pct),
            "success": False,
            "error": error,
        }

    @staticmethod
    def _default_confidence_label(confidence_pct: float) -> str:
        if confidence_pct >= 90:
            return "Very confident — almost certain!"
        elif confidence_pct >= 75:
            return "Confident — strong match."
        elif confidence_pct >= 55:
            return "Moderately confident — likely correct."
        elif confidence_pct >= 40:
            return "Uncertain — the image may be ambiguous."
        else:
            return "Low confidence — consider retraining or uploading a clearer image."