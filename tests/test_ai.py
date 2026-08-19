import os
import socket
import sqlite3
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import requests
import uvicorn
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request


os.environ.setdefault("CONSIGLIERE_DATA_DIR", tempfile.mkdtemp(prefix="consigliere-tests-"))

from src import main


class RagClientTests(unittest.TestCase):
    def setUp(self) -> None:
        main.solar_data_cache.clear()
        main.pota_data_cache.clear()

    def test_maidenhead_locator_is_normalized_and_converted(self) -> None:
        self.assertEqual(main.normalize_locator(" jn65 "), "JN65")
        latitude, longitude = main.maidenhead_to_coordinates("JN65")
        self.assertAlmostEqual(latitude, 45.5)
        self.assertAlmostEqual(longitude, 13.0)

        with self.assertRaisesRegex(ValueError, "Locator non valido"):
            main.normalize_locator("TRIESTE")

    def test_distance_and_direction_are_calculated_locally(self) -> None:
        origin = main.maidenhead_to_coordinates("JN65")
        destination = main.maidenhead_to_coordinates("JN66")
        distance, bearing = main.distance_and_bearing(origin, destination)

        self.assertAlmostEqual(distance, 111.2, delta=1.0)
        self.assertEqual(main.compass_direction(bearing), "N")

    def test_operator_locator_is_saved_with_derived_coordinates(self) -> None:
        db = main.SessionLocal()
        try:
            result = main.save_operator_locator("jn65", db)
            stored = main.read_operator_locator(db)

            self.assertEqual(result["locator"], "JN65")
            self.assertEqual(stored["locator"], "JN65")
            self.assertAlmostEqual(stored["latitude"], 45.5)
            self.assertAlmostEqual(stored["longitude"], 13.0)
        finally:
            db.close()

    def test_invalid_operator_locator_is_rejected(self) -> None:
        db = main.SessionLocal()
        try:
            with self.assertRaises(main.HTTPException) as error:
                main.save_operator_locator("TRIESTE", db)
            self.assertEqual(error.exception.status_code, 400)
        finally:
            db.close()

    def test_invalid_stored_locator_is_ignored(self) -> None:
        db = main.SessionLocal()
        try:
            row = db.query(main.Settings).filter(
                main.Settings.key == "operator_locator"
            ).first()
            if row:
                row.value = "INVALID"
            else:
                db.add(main.Settings(key="operator_locator", value="INVALID"))
            db.commit()

            self.assertEqual(main.get_operator_locator(db), "")
        finally:
            row = db.query(main.Settings).filter(
                main.Settings.key == "operator_locator"
            ).first()
            if row:
                row.value = ""
                db.commit()
            db.close()

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

    def test_advice_guard_rejects_unsupported_frequency_and_propagation_claims(self) -> None:
        current_data = (
            "K-index: 1.7\nConteggi reali per banda: {'20m': 50}\n"
            "VALUTAZIONE DETERMINISTICA DELL'APP (non contraddire):\n"
            "Gli spot indicano attività osservata, non un'apertura garantita."
        )
        self.assertIn(
            "frequenza non presente",
            main.find_unsupported_advice_claim("Evita le frequenze sopra 15 MHz.", current_data),
        )
        self.assertIn(
            "deduzione non supportata",
            main.find_unsupported_advice_claim("La MUF e lo strato D limitano il DX.", current_data),
        )
        self.assertIsNone(main.find_unsupported_advice_claim(
            "Gli spot su 20m indicano attività osservata, non un'apertura garantita.",
            current_data,
        ))

    @patch("src.main.ask_direct_llm", return_value="Attività POTA osservata su 20m.")
    @patch("src.main.ask_rag", return_value="Evita le bande sopra 15 MHz per la MUF.")
    def test_ask_ai_discards_unsupported_rag_and_uses_safe_direct_fallback(
        self, ask_rag: Mock, direct: Mock
    ) -> None:
        result = main.ask_ai(
            "Consiglio breve",
            "Banda 20m\nVALUTAZIONE DETERMINISTICA DELL'APP (non contraddire):\n"
            "Attività POTA osservata su 20m.",
        )
        self.assertEqual(result, ("Attività POTA osservata su 20m.", "llm-direct-guarded"))
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

    def test_pota_activity_is_not_labeled_as_a_band_opening(self) -> None:
        evaluation = main.evaluate_conditions(
            {"k_float": 2.0, "sfi_float": 120.0},
            {"by_band": {"10m": 8, "6m": 1}},
        )

        combined = " ".join(evaluation["opportunities"] + evaluation["details"])
        self.assertIn("Attività POTA osservata sulle bande alte", combined)
        self.assertNotIn("Apertura bande alte", combined)
        self.assertEqual(evaluation["level"], "operatività elevata")

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

    @patch("src.main.requests.get")
    def test_pota_spots_are_enriched_from_operator_locator(self, get: Mock) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = [
            None,
            {
                "frequency": "14074",
                "mode": "FT8",
                "activator": "NORTH",
                "name": "Parco Nord",
                "grid4": "JN66",
            },
            {
                "frequency": "7074",
                "mode": "FT8",
                "activator": "NOGRID",
                "name": "Parco senza locator",
            },
        ]
        get.return_value = response

        result = main.get_pota_spots("ALL", "ALL", 0, operator_locator="JN65")

        self.assertEqual(result["located_count"], 1)
        nearest = result["nearest_spots"][0]
        self.assertEqual(nearest["call"], "NORTH")
        self.assertEqual(nearest["direction"], "N")
        self.assertAlmostEqual(nearest["distance_km"], 111, delta=1)

    def test_rule_advice_uses_only_calculated_geographic_data(self) -> None:
        advice = main.generate_rule_based_advice(
            solar={"k_float": None, "sfi_float": None},
            pota={
                "count": 1,
                "band": "ALL",
                "by_band": {"20m": 1},
                "operator_locator": "JN65",
                "nearest_spots": [{
                    "call": "A1", "band": "20m", "distance_km": 111,
                    "direction": "N", "grid": "JN66",
                }],
            },
            bands={},
            modes={},
            ora=12,
        )

        self.assertIn("A1 su 20m a 111 km verso N", advice)
        self.assertIn("calcolate dai locator, non stimate dall'IA", advice)

    @patch("src.main.get_operator_locator", return_value="JN65")
    @patch("src.main.ask_ai", return_value=("Consiglio", "swlbot-rag"))
    @patch("src.main.get_pota_spots")
    @patch("src.main.get_solar_data")
    @patch("src.main.get_qso_statistics", return_value=(0, {}, {}))
    def test_ai_analyze_requests_complete_pota_activity(
        self, statistics: Mock, solar: Mock, pota: Mock,
        ask_ai: Mock, operator_locator: Mock
    ) -> None:
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
            "located_count": 1,
            "nearest_spots": [{
                "call": "A1", "band": "20m", "distance_km": 111,
                "bearing_deg": 1, "direction": "N", "grid": "JN66",
            }],
        }
        result = main.ai_analyze(Mock())

        self.assertEqual(result["source"], "swlbot-rag")
        pota.assert_called_once_with("ALL", "ALL", 0, operator_locator="JN65")
        current_data = ask_ai.call_args.args[1]
        self.assertIn("QTH locator: JN65", current_data)
        self.assertIn("Spot totali con banda riconosciuta: 18", current_data)
        self.assertIn("'20m': 10", current_data)
        self.assertIn("('20m', 10)", current_data)
        self.assertIn("QSO totali nello storico: 0", current_data)
        self.assertIn("'distance_km': 111", current_data)
        instruction = ask_ai.call_args.args[0]
        self.assertIn("non qualità della propagazione", instruction)
        self.assertIn("non dedurre DX", instruction)

    def test_qso_validation_normalizes_and_rejects_invalid_values(self) -> None:
        values = main.validate_qso_fields(
            14.074, "ft4", " iz3abc/p ", "-07", "jn65er", " prova "
        )
        self.assertEqual(values["mode"], "FT4")
        self.assertEqual(values["call_sign"], "IZ3ABC/P")
        self.assertEqual(values["locator"], "JN65ER")
        with self.assertRaisesRegex(ValueError, "Frequenza"):
            main.validate_qso_fields(float("nan"), "FT8", "A1")
        with self.assertRaisesRegex(ValueError, "Modo"):
            main.validate_qso_fields(14.074, "HTML", "A1")
        with self.assertRaisesRegex(ValueError, "Nominativo"):
            main.validate_qso_fields(14.074, "FT8", "<script>")

    def test_statistics_cover_complete_manual_log(self) -> None:
        db = main.SessionLocal()
        try:
            db.add_all([
                main.QSO(frequency=14.074, mode="FT8", call_sign="STAT20A"),
                main.QSO(frequency=14.250, mode="SSB", call_sign="STAT20B"),
                main.QSO(frequency=7.030, mode="CW", call_sign="STAT40"),
                main.QSO(frequency=0, mode="PROP", call_sign="NOAA_DATA"),
            ])
            db.flush()
            total, bands, modes = main.get_qso_statistics(db)
            self.assertGreaterEqual(total, 3)
            self.assertGreaterEqual(bands["20m"], 2)
            self.assertGreaterEqual(modes["FT8"], 1)
        finally:
            db.rollback()
            db.close()

    @patch("src.main.requests.get")
    def test_pota_cache_avoids_duplicate_network_calls(self, get: Mock) -> None:
        response = Mock()
        response.json.return_value = [{"frequency": "14074", "mode": "FT8"}]
        get.return_value = response

        first = main.get_pota_spots("ALL", "ALL", 0)
        second = main.get_pota_spots("ALL", "ALL", 0)

        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        get.assert_called_once()

    @patch("src.main.requests.get")
    def test_pota_uses_stale_cache_when_refresh_fails(self, get: Mock) -> None:
        main.pota_data_cache.set([{"frequency": "14074", "mode": "FT8"}])
        get.side_effect = main.requests.ConnectionError("offline")

        result = main.get_pota_spots("ALL", "ALL", 0, force_refresh=True)

        self.assertTrue(result["cached"])
        self.assertTrue(result["stale"])
        self.assertEqual(result["count"], 1)

    def test_manual_qso_can_be_edited_and_deleted_but_technical_rows_cannot(self) -> None:
        engine = create_engine("sqlite:///:memory:")
        main.Base.metadata.create_all(engine)
        db = sessionmaker(bind=engine)()
        try:
            qso = main.QSO(frequency=14.074, mode="FT8", call_sign="EDIT1")
            technical = main.QSO(frequency=0, mode="PROP", call_sign="NOAA_DATA")
            db.add_all([qso, technical])
            db.commit()

            edited = main.edit_qso(
                qso.id, 7.030, "cw", "edit2", "579", "jn65", "ok", db
            )
            self.assertEqual(edited["band"], "40m")
            self.assertEqual(db.get(main.QSO, qso.id).call_sign, "EDIT2")
            with self.assertRaises(main.HTTPException) as error:
                main.delete_qso(technical.id, db)
            self.assertEqual(error.exception.status_code, 404)
            self.assertEqual(main.delete_qso(qso.id, db)["status"], "ok")
            self.assertIsNone(db.get(main.QSO, qso.id))
        finally:
            db.close()

    def test_csv_and_adif_export_only_manual_qsos(self) -> None:
        engine = create_engine("sqlite:///:memory:")
        main.Base.metadata.create_all(engine)
        db = sessionmaker(bind=engine)()
        try:
            db.add_all([
                main.QSO(
                    timestamp=datetime(2026, 8, 19, 12, 30), frequency=14.074,
                    mode="FT8", call_sign="EXPORT1", rst_received="-08",
                    locator="JN65", notes="test, virgola",
                ),
                main.QSO(frequency=0, mode="ALERT", call_sign="SMART_ALERT"),
            ])
            db.commit()
            csv_text = "".join(main.csv_rows(db))
            adif_text = "".join(main.adif_rows(db))
            self.assertIn("EXPORT1", csv_text)
            self.assertNotIn("SMART_ALERT", csv_text)
            self.assertIn("<CALL:7>EXPORT1", adif_text)
            self.assertIn("<EOR>", adif_text)
            self.assertNotIn("SMART_ALERT", adif_text)
        finally:
            db.close()

    def test_database_backup_is_valid_and_contains_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.db"
            connection = sqlite3.connect(source)
            connection.execute("CREATE TABLE sample (value TEXT)")
            connection.execute("INSERT INTO sample VALUES ('presente')")
            connection.commit()
            connection.close()
            with patch.object(main, "_db_path", source):
                backup = main.create_database_backup()
            try:
                copied = sqlite3.connect(backup)
                value = copied.execute("SELECT value FROM sample").fetchone()[0]
                copied.close()
                self.assertEqual(value, "presente")
            finally:
                backup.unlink(missing_ok=True)

    def test_cross_site_writes_are_rejected_by_origin_check(self) -> None:
        def request(headers):
            encoded = [(key.lower().encode(), value.encode()) for key, value in headers.items()]
            return Request({"type": "http", "method": "POST", "headers": encoded})

        self.assertTrue(main.request_has_safe_origin(request({"host": "localhost:8080"})))
        self.assertTrue(main.request_has_safe_origin(request({
            "host": "localhost:8080", "origin": "http://localhost:8080"
        })))
        self.assertFalse(main.request_has_safe_origin(request({
            "host": "localhost:8080", "origin": "https://example.com"
        })))
        self.assertFalse(main.request_has_safe_origin(request({
            "host": "localhost:8080", "sec-fetch-site": "cross-site"
        })))

    def test_old_database_schema_remains_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.db"
            connection = sqlite3.connect(path)
            connection.execute(
                "CREATE TABLE qsos (id INTEGER PRIMARY KEY, timestamp DATETIME, "
                "frequency FLOAT, mode VARCHAR, call_sign VARCHAR, rst_received VARCHAR, "
                "rst_sent VARCHAR, locator VARCHAR, notes TEXT)"
            )
            connection.execute(
                "INSERT INTO qsos (frequency, mode, call_sign) VALUES (14.074, 'FT8', 'OLD1')"
            )
            connection.commit()
            connection.close()
            old_engine = create_engine(f"sqlite:///{path}")
            main.Base.metadata.create_all(old_engine)
            main.Index(
                "ix_qsos_frequency_timestamp", main.QSO.frequency, main.QSO.timestamp
            ).create(old_engine, checkfirst=True)
            db = sessionmaker(bind=old_engine)()
            try:
                self.assertEqual(db.query(main.QSO).one().call_sign, "OLD1")
                self.assertIsNotNone(db.query(main.Settings).all())
            finally:
                db.close()

    def test_http_smoke_flow_dashboard_qso_exports_backup_and_security(self) -> None:
        try:
            listener = socket.socket()
        except PermissionError:
            self.skipTest("Il sandbox non consente socket locali")
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(main.app, log_level="error"))
        thread = threading.Thread(
            target=server.run, kwargs={"sockets": [listener]}, daemon=True
        )
        thread.start()
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.01)
        self.assertTrue(server.started)

        base_url = f"http://127.0.0.1:{port}"
        qso_id = None
        try:
            dashboard = requests.get(base_url + "/", timeout=5)
            self.assertEqual(dashboard.status_code, 200)
            self.assertIn("Consigliere di Stazione", dashboard.text)
            openapi = requests.get(base_url + "/openapi.json", timeout=5).json()
            self.assertEqual(openapi["info"]["version"], main.APP_VERSION)

            created = requests.post(base_url + "/add", data={
                "frequency": "14.074", "mode": "FT8", "call_sign": "HTTPTEST",
                "rst_received": "-09", "locator": "JN65", "notes": "smoke",
            }, timeout=5)
            self.assertEqual(created.status_code, 200)
            qso_id = created.json()["id"]

            history = requests.get(
                base_url + "/qso/history", params={"search": "HTTPTEST"}, timeout=5
            ).json()
            self.assertGreaterEqual(history["total"], 1)
            edited = requests.post(base_url + f"/qso/{qso_id}/edit", data={
                "frequency": "7.030", "mode": "CW", "call_sign": "HTTPTEST",
                "rst_received": "579", "locator": "JN65", "notes": "modificato",
            }, timeout=5)
            self.assertEqual(edited.status_code, 200)
            self.assertEqual(edited.json()["band"], "40m")

            self.assertIn("HTTPTEST", requests.get(base_url + "/export/qso.csv", timeout=5).text)
            self.assertIn("<CALL:8>HTTPTEST", requests.get(base_url + "/export/qso.adi", timeout=5).text)
            backup = requests.get(base_url + "/backup/database", timeout=5)
            self.assertEqual(backup.status_code, 200)
            self.assertTrue(backup.content.startswith(b"SQLite format 3\x00"))

            blocked = requests.post(
                base_url + "/ui/trigger-check",
                headers={"Origin": "https://example.com"}, timeout=5,
            )
            self.assertEqual(blocked.status_code, 403)
        finally:
            if qso_id is not None:
                requests.post(base_url + f"/qso/{qso_id}/delete", timeout=5)
            server.should_exit = True
            thread.join(timeout=5)
            listener.close()
            self.assertFalse(thread.is_alive())

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
