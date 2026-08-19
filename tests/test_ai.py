import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import Mock, patch


os.environ.setdefault("CONSIGLIERE_DATA_DIR", tempfile.mkdtemp(prefix="consigliere-tests-"))

from src import main


class RagClientTests(unittest.TestCase):
    @patch("src.main.requests.post")
    def test_ask_rag_sends_instruction_and_current_data(self, post: Mock) -> None:
        response = Mock()
        response.json.return_value = {"response": "  Prova i 20 metri.  "}
        post.return_value = response

        result = main.ask_rag("Dammi un consiglio.", "K-index: 2; SFI: 130")

        self.assertEqual(result, "Prova i 20 metri.")
        response.raise_for_status.assert_called_once_with()
        request = post.call_args
        self.assertEqual(request.args[0], main.SWLBOT_RAG_URL)
        self.assertEqual(request.kwargs["json"], {
            "instruction": "Dammi un consiglio.",
            "current_data": "K-index: 2; SFI: 130",
        })
        self.assertEqual(request.kwargs["timeout"], main.SWLBOT_RAG_TIMEOUT)

    @patch("src.main.requests.post")
    def test_ask_rag_returns_none_for_empty_response(self, post: Mock) -> None:
        response = Mock()
        response.json.return_value = {"response": "  "}
        post.return_value = response

        self.assertIsNone(main.ask_rag("Cosa ascolto?", "K-index: 2"))

    @patch("src.main.requests.post")
    def test_ask_rag_returns_none_for_http_error(self, post: Mock) -> None:
        post.side_effect = main.requests.ConnectionError("offline")

        self.assertIsNone(main.ask_rag("Cosa ascolto?", "K-index: 2"))

    @patch("src.main.requests.post")
    def test_direct_llm_uses_local_qwen_chat_api(self, post: Mock) -> None:
        response = Mock()
        response.json.return_value = {"message": {"content": "  Condizioni favorevoli.  "}}
        post.return_value = response

        result = main.ask_direct_llm(
            "Dammi un consiglio.",
            "Bande più usate: {'40m': 8}\n"
            "VALUTAZIONE DETERMINISTICA DELL'APP (non contraddire):\n"
            "Sono presenti 10 spot POTA correnti su 20m.",
        )

        self.assertEqual(result, "Condizioni favorevoli.")
        request = post.call_args
        self.assertEqual(request.args[0], main.DIRECT_LLM_URL)
        self.assertEqual(request.kwargs["json"]["model"], "qwen3.5:4b")
        self.assertFalse(request.kwargs["json"]["think"])
        self.assertEqual(request.kwargs["timeout"], main.DIRECT_LLM_TIMEOUT)
        prompt = request.kwargs["json"]["messages"][1]["content"]
        self.assertIn("10 spot POTA correnti su 20m", prompt)
        self.assertNotIn("Bande più usate", prompt)

    @patch("src.main.ask_direct_llm")
    @patch("src.main.ask_rag", return_value=None)
    def test_ask_ai_reports_direct_llm_fallback(self, ask_rag: Mock, direct: Mock) -> None:
        direct.return_value = "Risposta Qwen"

        self.assertEqual(
            main.ask_ai("Dammi un consiglio.", "K-index: 2"),
            ("Risposta Qwen", "llm-direct"),
        )
        direct.assert_called_once()

    @patch("src.main.ask_direct_llm", return_value=None)
    @patch("src.main.ask_rag", return_value=None)
    def test_ask_ai_reports_rules_when_both_ai_services_fail(self, ask_rag: Mock, direct: Mock) -> None:
        self.assertEqual(main.ask_ai("Cosa ascolto?", "K-index: 2"), (None, "rules"))

    @patch("src.main.ask_ai", return_value=(None, "rules"))
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

    @patch("src.main.ask_ai", return_value=("Ascolta i 20 metri.", "swlbot-rag"))
    def test_alert_reports_rag_source(self, ask_ai: Mock) -> None:
        evaluation = {
            "score": 70,
            "level": "eccellenti",
            "details": [],
            "warnings": [],
            "opportunities": ["K-index basso"],
            "top_bands": [("20m", 8)],
            "total_activity": 8,
            "solar": {"k": 1.0, "sfi": 140.0},
        }

        alert = main.generate_smart_alert(evaluation, {}, {})

        self.assertEqual(alert["source"], "swlbot-rag")
        ask_ai.assert_called_once()

    @patch("src.main.ask_ai", return_value=("Condizioni favorevoli.", "llm-direct"))
    def test_alert_reports_direct_llm_source(self, ask_ai: Mock) -> None:
        evaluation = {
            "score": 70,
            "level": "eccellenti",
            "details": [],
            "warnings": [],
            "opportunities": ["K-index basso"],
            "top_bands": [("20m", 8)],
            "total_activity": 8,
            "solar": {"k": 1.0, "sfi": 140.0},
        }

        alert = main.generate_smart_alert(evaluation, {}, {})

        self.assertEqual(alert["source"], "llm-direct")

    def test_rule_advice_keeps_filtered_pota_band_separate_from_log_history(self) -> None:
        advice = main.generate_rule_based_advice(
            solar={"k_float": None, "sfi_float": None},
            pota={"count": 10, "band": "20m", "by_band": {"40m": 99}},
            bands={"80m": 7},
            modes={"FT8": 7},
            ora=20,
        )

        self.assertIn("10 spot POTA correnti su 20m", advice)
        self.assertNotIn("40m (99 spot)", advice)
        self.assertIn("storico", advice)
        self.assertIn("non l'attività POTA corrente", advice)

    @patch("src.main.requests.get")
    def test_all_band_pota_scan_counts_every_recognized_band(self, get: Mock) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = [
            {"frequency": "14074", "mode": "FT8", "activator": "A1", "name": "P1"},
            {"frequency": "7074", "mode": "FT8", "activator": "A2", "name": "P2"},
            {"frequency": "3573", "mode": "FT8", "activator": "A3", "name": "P3"},
            {"frequency": "500", "mode": "CW", "activator": "OUT", "name": "P4"},
        ]
        get.return_value = response

        result = main.get_pota_spots("ALL", "ALL", 0)

        self.assertEqual(result["count"], 3)
        self.assertEqual(result["by_band"], {"20m": 1, "40m": 1, "80m": 1})
        self.assertEqual(result["total_spots"], 4)
        get.assert_called_once()

    @patch("src.main.ask_ai", return_value=("Consiglio", "swlbot-rag"))
    @patch("src.main.get_pota_spots")
    @patch("src.main.get_solar_data")
    def test_ai_analyze_requests_complete_pota_activity(self, solar: Mock, pota: Mock, ask_ai: Mock) -> None:
        solar.return_value = {
            "k_index": 2.0,
            "k_float": 2.0,
            "sfi": 120.0,
            "sfi_float": 120.0,
        }
        pota.return_value = {
            "count": 18,
            "band": "ALL",
            "by_band": {"20m": 10, "40m": 5, "80m": 3},
        }
        db = Mock()
        db.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

        result = main.ai_analyze(db)

        self.assertEqual(result["source"], "swlbot-rag")
        pota.assert_called_once_with("ALL", "ALL", 0)
        current_data = ask_ai.call_args.args[1]
        self.assertIn("Spot totali con banda riconosciuta: 18", current_data)
        self.assertIn("'20m': 10", current_data)
        self.assertIn("('20m', 10)", current_data)
        self.assertIn("QSO totali nello storico: 0", current_data)

    def test_qso_history_filters_and_excludes_technical_records(self) -> None:
        db = main.SessionLocal()
        try:
            db.add_all([
                main.QSO(timestamp=datetime(2026, 8, 1, 12), frequency=14.074, mode="FT8", call_sign="TEST20", rst_received="-5", locator="JN65", notes="demo twenty"),
                main.QSO(timestamp=datetime(2026, 7, 1, 12), frequency=7.030, mode="CW", call_sign="TEST40", rst_received="579", locator="JN55", notes="demo forty"),
                main.QSO(timestamp=datetime(2026, 8, 2, 12), frequency=0.0, mode="PROP", call_sign="NOAA_DATA", notes="dato tecnico"),
            ])
            db.flush()

            result = main.qso_history(
                offset=0,
                limit=50,
                band="20m",
                mode="FT8",
                search="TEST",
                db=db,
            )

            self.assertEqual(result["total"], 1)
            self.assertEqual(result["items"][0]["call_sign"], "TEST20")
            self.assertEqual(result["items"][0]["band"], "20m")
        finally:
            db.rollback()
            db.close()


if __name__ == "__main__":
    unittest.main()
