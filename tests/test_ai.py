import os
import tempfile
import unittest
from unittest.mock import Mock, patch


os.environ.setdefault("CONSIGLIERE_DATA_DIR", tempfile.mkdtemp(prefix="consigliere-tests-"))

from src import main


class OllamaTests(unittest.TestCase):
    @patch("src.main.requests.post")
    def test_ask_ai_uses_specialized_model_and_context(self, post: Mock) -> None:
        response = Mock()
        response.json.return_value = {"response": "  Prova i 20 metri.  "}
        post.return_value = response

        result = main.ask_ai("Cosa ascolto?")

        self.assertEqual(result, "Prova i 20 metri.")
        response.raise_for_status.assert_called_once_with()
        request = post.call_args
        self.assertEqual(request.kwargs["json"]["model"], main.OLLAMA_MODEL)
        self.assertEqual(request.kwargs["json"]["options"]["num_ctx"], 4096)
        self.assertEqual(request.kwargs["timeout"], 60)

    @patch("src.main.requests.post")
    def test_ask_ai_returns_none_for_empty_response(self, post: Mock) -> None:
        response = Mock()
        response.json.return_value = {"response": "  "}
        post.return_value = response

        self.assertIsNone(main.ask_ai("Cosa ascolto?"))

    @patch("src.main.requests.post")
    def test_ask_ai_returns_none_for_http_error(self, post: Mock) -> None:
        post.side_effect = main.requests.ConnectionError("offline")

        self.assertIsNone(main.ask_ai("Cosa ascolto?"))

    @patch("src.main.ask_ai", return_value=None)
    def test_alert_falls_back_to_deterministic_message(self, ask_ai: Mock) -> None:
        evaluation = {
            "score": 70,
            "level": "eccellenti",
            "details": ["Attività moderata"],
            "warnings": [],
            "opportunities": ["K-index basso"],
            "top_bands": [("20m", 8)],
            "total_activity": 8,
            "solar": {"k": 1.0, "sfi": 140.0},
        }

        alert = main.generate_smart_alert(evaluation, {}, {})

        self.assertEqual(alert["source"], "rules")
        self.assertIn("K-index basso", alert["message"])
        self.assertLessEqual(len(alert["message"]), 280)
        ask_ai.assert_called_once()


if __name__ == "__main__":
    unittest.main()
