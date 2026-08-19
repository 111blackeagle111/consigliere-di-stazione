# Consigliere di Stazione - Registro QSO intelligente per radioamatori
# Copyright (C) 2025  I6502TR (Andrea Maccafeo)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import csv
import io
import math
import os
import re
import sqlite3
import sys
import tempfile
import threading
from pathlib import Path
from urllib.parse import urlparse


APP_DATA_DIR = "ConsigliereDiStazione"
DATABASE_NAME = "swl_logs.db"
APP_HOST = os.getenv("CONSIGLIERE_HOST", "127.0.0.1")


def get_app_port() -> int:
    try:
        port = int(os.getenv("CONSIGLIERE_PORT", "8080"))
    except ValueError:
        return 8080
    return port if 1 <= port <= 65535 else 8080


APP_PORT = get_app_port()


def get_user_data_path() -> Path:
    """Return a writable, per-user directory for persistent application data."""
    override = os.getenv("CONSIGLIERE_DATA_DIR")
    if override:
        data_path = Path(override).expanduser()
    elif sys.platform == "win32":
        data_path = Path(os.getenv("LOCALAPPDATA", Path.home())) / APP_DATA_DIR
    else:
        data_path = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "consigliere-di-stazione"

    data_path.mkdir(parents=True, exist_ok=True)
    return data_path


def get_base_path() -> Path:
    """Path alle risorse bundled (templates). In PyInstaller punta a _MEIPASS."""
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).parent.parent


def get_app_version() -> str:
    version_file = get_base_path() / "VERSION"
    try:
        return version_file.read_text(encoding="utf-8").strip() or "dev"
    except OSError:
        return "dev"


APP_VERSION = get_app_version()


def get_data_path() -> Path:
    """Return the DB directory while preserving existing portable installations."""
    if os.getenv("CONSIGLIERE_DATA_DIR"):
        return get_user_data_path()

    if getattr(sys, 'frozen', False):
        legacy_path = Path(sys.executable).parent
        if (legacy_path / DATABASE_NAME).exists():
            return legacy_path
        return get_user_data_path()
    return Path(__file__).parent.parent


from fastapi import FastAPI, Depends, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, Index, case, func, or_
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from starlette.background import BackgroundTask
from datetime import datetime, timedelta, timezone
import requests
import json
import logging
import asyncio
import time
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# DATABASE SETUP
# ============================================================

Base = declarative_base()
_db_path = get_data_path() / DATABASE_NAME
engine = create_engine(f'sqlite:///{_db_path}', connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class QSO(Base):
    __tablename__ = "qsos"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.now)
    frequency = Column(Float, default=0.0)
    mode = Column(String, default="")
    call_sign = Column(String, default="")
    rst_received = Column(String, default="")
    rst_sent = Column(String, default="")
    locator = Column(String, default="")
    notes = Column(Text, default="")

class Settings(Base):
    __tablename__ = "settings"
    key = Column(String, primary_key=True)
    value = Column(String, default="")

# Crea tabelle
Base.metadata.create_all(bind=engine)
Index("ix_qsos_frequency_timestamp", QSO.frequency, QSO.timestamp).create(bind=engine, checkfirst=True)
Index("ix_qsos_mode_timestamp", QSO.mode, QSO.timestamp).create(bind=engine, checkfirst=True)

def get_callsign(db: Session) -> str:
    row = db.query(Settings).filter(Settings.key == "callsign").first()
    return row.value if row else "I6502TR"


def get_operator_locator(db: Session) -> str:
    row = db.query(Settings).filter(Settings.key == "operator_locator").first()
    if not row or not isinstance(row.value, str):
        return ""
    try:
        return normalize_locator(row.value)
    except ValueError:
        logger.warning("Locator operatore non valido ignorato: %r", row.value)
        return ""

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============================================================
# FASTAPI SETUP
# ============================================================

app = FastAPI(title="Consigliere di Stazione", version=APP_VERSION)

templates = Jinja2Templates(directory=str(get_base_path() / "templates"))


def request_has_safe_origin(request: Request) -> bool:
    """Reject browser cross-site writes while keeping CLI/local clients usable."""
    if request.headers.get("sec-fetch-site", "").lower() == "cross-site":
        return False
    origin = request.headers.get("origin")
    if not origin:
        return True
    return urlparse(origin).netloc.lower() == request.headers.get("host", "").lower()


@app.middleware("http")
async def protect_local_writes(request: Request, call_next):
    if request.method not in {"GET", "HEAD", "OPTIONS"} and not request_has_safe_origin(request):
        return JSONResponse(
            status_code=403,
            content={"detail": "Richiesta rifiutata: origine diversa dall'app locale"},
        )
    return await call_next(request)

# ============================================================
# FUNZIONI HELPER
# ============================================================

QSO_MODES = {"SSB", "CW", "FM", "AM", "FT8", "FT4", "PSK31", "RTTY", "JT65"}
BAND_RANGES = {
    "160m": (1.8, 2.0), "80m": (3.5, 4.0), "60m": (5.3, 5.4),
    "40m": (7.0, 7.3), "30m": (10.1, 10.15), "20m": (14.0, 14.35),
    "17m": (18.068, 18.168), "15m": (21.0, 21.45),
    "12m": (24.89, 24.99), "10m": (28.0, 29.7), "6m": (50.0, 54.0),
}


def get_band(freq_mhz: float) -> str:
    if freq_mhz == 0.0:
        return "PROP"
    for name, (min_f, max_f) in BAND_RANGES.items():
        if min_f <= freq_mhz <= max_f:
            return name
    return f"{freq_mhz:.3f}MHz"


def get_qso_statistics(db: Session) -> tuple[int, dict, dict]:
    """Aggregate the complete manual log in SQLite without loading every QSO."""
    band_case = case(
        *[
            (QSO.frequency.between(minimum, maximum), name)
            for name, (minimum, maximum) in BAND_RANGES.items()
        ],
        else_="Altra",
    )
    base_filter = QSO.frequency > 0
    total = db.query(func.count(QSO.id)).filter(base_filter).scalar() or 0
    bands = {
        name: count
        for name, count in db.query(band_case, func.count(QSO.id))
        .filter(base_filter)
        .group_by(band_case)
        .all()
    }
    modes = {
        mode: count
        for mode, count in db.query(QSO.mode, func.count(QSO.id))
        .filter(base_filter)
        .group_by(QSO.mode)
        .all()
    }
    return total, bands, modes


MAIDENHEAD_PATTERN = re.compile(r"^[A-R]{2}[0-9]{2}(?:[A-X]{2})?$")


def normalize_locator(locator: str) -> str:
    """Normalize and validate a 4- or 6-character Maidenhead locator."""
    normalized = locator.strip().upper()
    if normalized and not MAIDENHEAD_PATTERN.fullmatch(normalized):
        raise ValueError("Locator non valido: usa 4 o 6 caratteri, per esempio JN65 o JN65ER")
    return normalized


CALLSIGN_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9/.-]{0,19}$")


def validate_qso_fields(
    frequency: float,
    mode: str,
    call_sign: str,
    rst_received: str = "",
    locator: str = "",
    notes: str = "",
) -> dict:
    if not math.isfinite(frequency) or not 0.001 <= frequency <= 300000:
        raise ValueError("Frequenza non valida: inserisci un valore in MHz tra 0.001 e 300000")
    normalized_mode = mode.strip().upper()
    if normalized_mode not in QSO_MODES:
        raise ValueError("Modo non valido")
    normalized_call = call_sign.strip().upper()
    if not CALLSIGN_PATTERN.fullmatch(normalized_call):
        raise ValueError("Nominativo non valido: massimo 20 caratteri, lettere, numeri, /, . o -")
    normalized_rst = rst_received.strip().upper()
    if len(normalized_rst) > 10:
        raise ValueError("RST troppo lungo: massimo 10 caratteri")
    normalized_locator = normalize_locator(locator)
    normalized_notes = notes.strip()
    if len(normalized_notes) > 5000:
        raise ValueError("Note troppo lunghe: massimo 5000 caratteri")
    return {
        "frequency": float(frequency),
        "mode": normalized_mode,
        "call_sign": normalized_call,
        "rst_received": normalized_rst,
        "locator": normalized_locator,
        "notes": normalized_notes,
    }


def maidenhead_to_coordinates(locator: str) -> tuple[float, float]:
    """Return the center latitude/longitude of a Maidenhead grid square."""
    locator = normalize_locator(locator)
    if not locator:
        raise ValueError("Locator mancante")

    longitude = (ord(locator[0]) - ord("A")) * 20 - 180
    latitude = (ord(locator[1]) - ord("A")) * 10 - 90
    longitude += int(locator[2]) * 2
    latitude += int(locator[3])

    if len(locator) == 6:
        longitude += (ord(locator[4]) - ord("A")) * (5 / 60) + (2.5 / 60)
        latitude += (ord(locator[5]) - ord("A")) * (2.5 / 60) + (1.25 / 60)
    else:
        longitude += 1
        latitude += 0.5

    return latitude, longitude


def distance_and_bearing(
    origin: tuple[float, float], destination: tuple[float, float]
) -> tuple[float, float]:
    """Calculate great-circle distance in km and initial bearing in degrees."""
    lat1, lon1 = (math.radians(value) for value in origin)
    lat2, lon2 = (math.radians(value) for value in destination)
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    haversine = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    haversine = min(1.0, max(0.0, haversine))
    distance = 6371.0088 * 2 * math.atan2(math.sqrt(haversine), math.sqrt(1 - haversine))
    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    bearing = (math.degrees(math.atan2(x, y)) + 360) % 360
    return distance, bearing


def compass_direction(bearing: float) -> str:
    directions = ("N", "NE", "E", "SE", "S", "SO", "O", "NO")
    return directions[int((bearing + 22.5) // 45) % 8]


def nearest_spots_summary(pota_data: dict, limit: int = 3) -> str:
    parts = []
    for spot in pota_data.get("nearest_spots", [])[:limit]:
        parts.append(
            f"{spot.get('call', '???')} su {spot.get('band', '?')} a "
            f"{spot.get('distance_km', '?')} km verso {spot.get('direction', '?')} "
            f"(locator {spot.get('grid', '?')})"
        )
    return "; ".join(parts)

SWLBOT_RAG_URL = os.getenv("SWLBOT_RAG_URL", "http://127.0.0.1:8081/api/advice")
SWLBOT_RAG_TIMEOUT = float(os.getenv("SWLBOT_RAG_TIMEOUT", "90"))
DIRECT_LLM_URL = os.getenv("DIRECT_LLM_URL", "http://127.0.0.1:11434/api/chat")
DIRECT_LLM_MODEL = os.getenv("DIRECT_LLM_MODEL", "qwen3.5:4b")
DIRECT_LLM_TIMEOUT = float(os.getenv("DIRECT_LLM_TIMEOUT", "90"))


def ask_rag(instruction: str, current_data: str) -> Optional[str]:
    try:
        response = requests.post(
            SWLBOT_RAG_URL,
            json={
                "instruction": instruction,
                "current_data": current_data,
            },
            timeout=SWLBOT_RAG_TIMEOUT,
        )
        response.raise_for_status()
        response_text = response.json().get("response")
        if not isinstance(response_text, str) or not response_text.strip():
            raise ValueError("risposta swlbot RAG vuota o non valida")
        return response_text.strip()
    except (requests.RequestException, ValueError, TypeError) as exc:
        logger.warning("swlbot RAG non disponibile: %s", exc)
        return None


def ask_direct_llm(instruction: str, current_data: str) -> Optional[str]:
    """Fallback locale senza retrieval: riformula le regole, non crea analisi nuove."""
    system = (
        "Sei un redattore tecnico italiano. Devi soltanto riformulare la VALUTAZIONE "
        "DETERMINISTICA fornita dall'app, senza aggiungere nuove conclusioni. Non introdurre "
        "bande, frequenze, stazioni, aperture, direzioni, orari, strumenti o relazioni causali "
        "che non siano gia' scritti nella valutazione. Non dedurre rumore dalla quantita' di "
        "spot e non scambiare le bande storicamente usate dall'operatore per attivita' radio "
        "corrente. Non presumere che l'utente trasmetta: parla di ascolto salvo indicazione "
        "esplicita contraria. Se la valutazione e' insufficiente, dichiaralo. Segui soltanto i vincoli "
        "di lunghezza e stile dell'istruzione."
    )
    marker = "VALUTAZIONE DETERMINISTICA DELL'APP (non contraddire):"
    _, separator, deterministic_guidance = current_data.partition(marker)
    if separator:
        source_text = deterministic_guidance.strip()
    else:
        source_text = "Dati insufficienti per formulare una valutazione deterministica."
    prompt = (
        f"VINCOLI DI STILE:\n{instruction}\n\n"
        f"VALUTAZIONE DETERMINISTICA DA RIFORMULARE:\n{source_text}"
    )
    try:
        response = requests.post(
            DIRECT_LLM_URL,
            json={
                "model": DIRECT_LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": 300},
            },
            timeout=DIRECT_LLM_TIMEOUT,
        )
        response.raise_for_status()
        response_text = response.json().get("message", {}).get("content")
        if not isinstance(response_text, str) or not response_text.strip():
            raise ValueError("risposta LLM diretto vuota o non valida")
        return response_text.strip()
    except (requests.RequestException, ValueError, TypeError) as exc:
        logger.warning("LLM diretto non disponibile: %s", exc)
        return None


UNSUPPORTED_LIVE_CLAIMS = (
    r"\bmuf\b", r"strato\s+d", r"assorb", r"\brumore\b", r"\bdx\b",
    r"\bskip\b", r"\bfading\b", r"greyline",
    r"propagazione\s+(?:è|e')\s+(?:stabile|ottimale|eccellente|garantita)",
    r"evita(?:re)?\s+(?:le\s+|i\s+)?bande",
)
FREQUENCY_CLAIM_PATTERN = re.compile(
    r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:mhz|khz|hz)\b", re.IGNORECASE
)


def find_unsupported_advice_claim(response_text: str, current_data: str) -> Optional[str]:
    """Return the first real-time claim not grounded in the supplied live data."""
    response_lower = response_text.lower()
    data_lower = current_data.lower()
    for frequency in FREQUENCY_CLAIM_PATTERN.findall(response_text):
        normalized = re.sub(r"\s+", "", frequency.lower().replace(",", "."))
        normalized_data = re.sub(r"\s+", "", data_lower.replace(",", "."))
        if normalized not in normalized_data:
            return f"frequenza non presente nei dati correnti: {frequency}"
    for pattern in UNSUPPORTED_LIVE_CLAIMS:
        match = re.search(pattern, response_lower)
        if match and not re.search(pattern, data_lower):
            return f"deduzione non supportata: {match.group(0)}"
    return None


def ask_ai(instruction: str, current_data: str) -> tuple[Optional[str], str]:
    rag_rejected = False
    direct_rejected = False
    rag_response = ask_rag(instruction, current_data)
    if rag_response:
        unsupported = find_unsupported_advice_claim(rag_response, current_data)
        if not unsupported:
            return rag_response, "swlbot-rag"
        logger.warning("Risposta swlbot RAG scartata: %s", unsupported)
        rag_rejected = True

    direct_response = ask_direct_llm(instruction, current_data)
    if direct_response:
        unsupported = find_unsupported_advice_claim(direct_response, current_data)
        if not unsupported:
            return direct_response, "llm-direct-guarded" if rag_rejected else "llm-direct"
        logger.warning("Risposta LLM diretto scartata: %s", unsupported)
        direct_rejected = True

    return None, "rules-guarded" if rag_rejected or direct_rejected else "rules"

# ============================================================
# ALERT SYSTEM
# ============================================================

class AlertCache:
    def __init__(self):
        self.last_alerts = {}
        self.cooldown_minutes = 30
    
    def should_send(self, alert_type: str, content_hash: str) -> bool:
        now = datetime.now()
        if alert_type in self.last_alerts:
            last_time, last_hash = self.last_alerts[alert_type]
            if now - last_time < timedelta(minutes=self.cooldown_minutes):
                if last_hash == content_hash:
                    return False
        return True
    
    def update(self, alert_type: str, content_hash: str):
        self.last_alerts[alert_type] = (datetime.now(), content_hash)

alert_cache = AlertCache()

class SolarThresholds:
    K_INDEX_GOOD = 3.0
    K_INDEX_FAIR = 5.0
    SFI_GOOD = 100
    SFI_FAIR = 70


class TimedCache:
    def __init__(self, ttl_seconds: int):
        self.ttl_seconds = ttl_seconds
        self._value = None
        self._stored_at = 0.0
        self._lock = threading.Lock()

    def get(self, allow_stale: bool = False):
        with self._lock:
            if self._value is None:
                return None
            age = max(0.0, time.monotonic() - self._stored_at)
            if not allow_stale and age >= self.ttl_seconds:
                return None
            return self._value, age

    def set(self, value) -> None:
        with self._lock:
            self._value = value
            self._stored_at = time.monotonic()

    def clear(self) -> None:
        with self._lock:
            self._value = None
            self._stored_at = 0.0


solar_data_cache = TimedCache(60)
pota_data_cache = TimedCache(30)

# ============================================================
# DATI NOAA (ROBUSTO CON RETRY)
# ============================================================

def _fetch_solar_data():
    try:
        url_k = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json"
        logger.info("Fetching NOAA K-index...")
        resp_k = requests.get(url_k, timeout=30)
        resp_k.raise_for_status()
        data_k = resp_k.json()

        url_sfi = "https://services.swpc.noaa.gov/json/f107_cm_flux.json"
        logger.info("Fetching NOAA SFI...")
        resp_sfi = requests.get(url_sfi, timeout=30)
        resp_sfi.raise_for_status()
        data_sfi = resp_sfi.json()
        
        observed_values = [x for x in data_k if x.get("observed") == "observed"]
        if observed_values:
            last_k = observed_values[-1]
            k_val = last_k["kp"]
            k_time = last_k["time_tag"]
        else:
            last_k = data_k[-1] if data_k else None
            k_val = last_k["kp"] if last_k else "N/A"
            k_time = last_k["time_tag"] if last_k else None
        
        if data_sfi and len(data_sfi) > 0:
            last_sfi = data_sfi[0]
            sfi_val = last_sfi.get('flux', 'N/A')
            sfi_time = last_sfi.get('time_tag')
            if isinstance(sfi_val, (int, float)):
                sfi_val = float(sfi_val)
        else:
            sfi_val = "N/A"
            sfi_time = None
        
        logger.info(f"NOAA OK - K:{k_val}, SFI:{sfi_val}")
        
        return {
            "status": "ok",
            "k_index": k_val,
            "k_float": float(k_val) if k_val not in ["N/A", None] else None,
            "k_time": k_time,
            "sfi": sfi_val,
            "sfi_float": float(sfi_val) if sfi_val not in ["N/A", None] else None,
            "sfi_time": sfi_time,
            "timestamp": datetime.now().astimezone().isoformat(),
            "source": "NOAA SWPC"
        }
        
    except requests.exceptions.Timeout:
        logger.error("NOAA Timeout")
        return {"status": "error", "error": "Timeout - NOAA non risponde", "k_index": "N/A", "sfi": "N/A", "k_float": None, "sfi_float": None}
    except requests.exceptions.ConnectionError:
        logger.error("NOAA Connection Error")
        return {"status": "error", "error": "Errore connessione/DNS", "k_index": "N/A", "sfi": "N/A", "k_float": None, "sfi_float": None}
    except Exception as e:
        logger.error(f"NOAA Error type={type(e).__name__} args={e.args}: {e}")
        return {"status": "error", "error": str(e), "k_index": "N/A", "sfi": "N/A", "k_float": None, "sfi_float": None}


def get_solar_data(force_refresh: bool = False):
    cached = None if force_refresh else solar_data_cache.get()
    if cached:
        value, age = cached
        return {**value, "cached": True, "cache_age_seconds": round(age, 1), "stale": False}

    result = _fetch_solar_data()
    if result.get("status") == "ok":
        solar_data_cache.set(result)
        return {**result, "cached": False, "cache_age_seconds": 0.0, "stale": False}

    stale = solar_data_cache.get(allow_stale=True)
    if stale:
        value, age = stale
        return {
            **value,
            "cached": True,
            "cache_age_seconds": round(age, 1),
            "stale": True,
            "warning": result.get("error", "Aggiornamento NOAA non disponibile"),
        }
    return result

# ============================================================
# POTA DATA
# ============================================================

def get_pota_payload(force_refresh: bool = False) -> tuple[list, dict]:
    cached = None if force_refresh else pota_data_cache.get()
    if cached:
        value, age = cached
        return value, {
            "cached": True,
            "cache_age_seconds": round(age, 1),
            "data_timestamp": (datetime.now().astimezone() - timedelta(seconds=age)).isoformat(),
            "stale": False,
        }

    try:
        response = requests.get("https://api.pota.app/spot/activator", timeout=15)
        response.raise_for_status()
        value = response.json()
        if not isinstance(value, list):
            raise ValueError("Risposta POTA non valida")
        pota_data_cache.set(value)
        return value, {
            "cached": False,
            "cache_age_seconds": 0.0,
            "data_timestamp": datetime.now().astimezone().isoformat(),
            "stale": False,
        }
    except (requests.RequestException, ValueError, TypeError) as exc:
        stale = pota_data_cache.get(allow_stale=True)
        if stale:
            value, age = stale
            return value, {
                "cached": True,
                "cache_age_seconds": round(age, 1),
                "data_timestamp": (datetime.now().astimezone() - timedelta(seconds=age)).isoformat(),
                "stale": True,
                "warning": str(exc),
            }
        raise


def get_pota_spots(
    band="20m", mode="ALL", limit=50, operator_locator="", force_refresh=False
):
    try:
        all_spots, cache_info = get_pota_payload(force_refresh=force_refresh)
        
        normalized_locator = normalize_locator(operator_locator) if operator_locator else ""
        operator_coordinates = (
            maidenhead_to_coordinates(normalized_locator) if normalized_locator else None
        )
        band_counts = {}
        filtered = []
        located_spots = []
        
        for spot in all_spots:
            try:
                freq_raw = spot.get("frequency") or spot.get("freq", 0)
                freq = float(freq_raw)
                if freq > 1000:
                    freq = freq / 1000
                
                spot_band = None
                for b_name, (min_f, max_f) in BAND_RANGES.items():
                    if min_f <= freq <= max_f:
                        spot_band = b_name
                        band_counts[b_name] = band_counts.get(b_name, 0) + 1
                        break

                if not spot_band:
                    continue

                spot_mode = str(spot.get("mode", "")).upper()
                grid = str(spot.get("grid6") or spot.get("grid4") or "").strip().upper()
                item = {
                    "call": str(spot.get("activator") or "???"),
                    "freq": f"{freq:.3f}",
                    "mode": spot_mode or "??",
                    "band": spot_band,
                    "park": str(spot.get("name") or "Unknown")[:60],
                    "grid": grid,
                    "time": str(spot.get("spotTime"))[11:16] if spot.get("spotTime") else "??",
                }

                if operator_coordinates and grid:
                    try:
                        destination = maidenhead_to_coordinates(grid)
                        distance, bearing = distance_and_bearing(operator_coordinates, destination)
                        item.update({
                            "distance_km": round(distance),
                            "bearing_deg": round(bearing) % 360,
                            "direction": compass_direction(bearing),
                        })
                        located_spots.append(item.copy())
                    except ValueError:
                        pass

                if band != "ALL" and spot_band == band.lower():
                    if mode == "ALL" or spot_mode == mode.upper():
                        filtered.append(item)
            except (AttributeError, TypeError, ValueError, IndexError):
                continue

        nearest_spots = sorted(
            located_spots, key=lambda item: item["distance_km"]
        )[:5]
        visible_spots = filtered[:limit] if band != "ALL" and limit > 0 else []

        return {
            "spots": visible_spots,
            "count": len(filtered) if band != "ALL" else sum(band_counts.values()),
            "by_band": band_counts,
            "total_spots": len(all_spots),
            "band": band,
            "mode": mode,
            "operator_locator": normalized_locator,
            "located_count": len(located_spots),
            "nearest_spots": nearest_spots,
            **cache_info,
        }
        
    except Exception as e:
        logger.error(f"Errore POTA: {e}")
        return {
            "error": str(e), "spots": [], "count": 0, "by_band": {},
            "nearest_spots": [], "located_count": 0,
        }

# ============================================================
# EVALUATION & AI
# ============================================================

def evaluate_conditions(solar_data: dict, pota_data: dict) -> dict:
    score = 0
    details = []
    warnings = []
    opportunities = []
    
    k = solar_data.get("k_float")
    sfi = solar_data.get("sfi_float")
    
    if k is not None:
        if k < SolarThresholds.K_INDEX_GOOD:
            score += 40
            opportunities.append(f"Campo geomagnetico quieto (K {k})")
        elif k < SolarThresholds.K_INDEX_FAIR:
            score += 20
            details.append(f"Campo geomagnetico moderato (K {k})")
        else:
            score -= 20
            warnings.append(f"K-index disturbato ({k})")
    
    if sfi is not None:
        if sfi > SolarThresholds.SFI_GOOD:
            score += 40
            opportunities.append(f"SFI alto ({sfi})")
        elif sfi > SolarThresholds.SFI_FAIR:
            score += 20
            details.append(f"SFI discreto ({sfi})")
        else:
            score -= 10
            warnings.append(f"SFI basso ({sfi})")
    
    by_band = pota_data.get("by_band", {})
    total_activity = sum(by_band.values())
    top_bands = sorted(by_band.items(), key=lambda x: x[1], reverse=True)[:3]
    
    if total_activity > 50:
        score += 20
        opportunities.append(f"Alta attività POTA ({total_activity} spot)")
    elif total_activity > 20:
        score += 10
        details.append(f"Attività moderata ({total_activity} spot)")
    
    if by_band.get("10m", 0) > 5 or by_band.get("6m", 0) > 0:
        opportunities.append("Attività POTA osservata sulle bande alte")
        score += 15

    level = (
        "operatività elevata" if score >= 70 else
        "operatività buona" if score >= 50 else
        "operatività moderata" if score >= 30 else
        "operatività limitata"
    )
    
    return {
        "score": min(100, max(0, score)),
        "level": level,
        "details": details,
        "warnings": warnings,
        "opportunities": opportunities,
        "top_bands": top_bands,
        "total_activity": total_activity,
        "solar": {"k": k, "sfi": sfi}
    }

def generate_smart_alert(eval_data: dict, solar_data: dict, pota_data: dict) -> dict:
    score = eval_data["score"]
    level = eval_data["level"]
    
    if score < 30 and not eval_data["opportunities"]:
        return None
    
    ora = datetime.now().hour
    periodo = "mattina" if 6 <= ora < 12 else "pomeriggio" if 12 <= ora < 18 else "sera" if 18 <= ora < 23 else "notte"
    
    instruction = (
        "Genera un breve messaggio di alert sulle condizioni di propagazione attuali. "
        "Usa un tono professionale ma appassionato, italiano corretto, massimo 3 frasi "
        "e 280 caratteri. Non indicare frequenze o stazioni specifiche che non compaiono "
        "nei dati correnti. Gli spot POTA indicano attività, non dimostrano aperture, DX, "
        "qualità della propagazione o basso rumore. Non convertire i nomi delle bande in "
        "frequenze numeriche e non presentare le condizioni come ottimali o garantite."
    )
    geo_summary = nearest_spots_summary(pota_data, limit=2)
    fallback_parts = eval_data["opportunities"] + eval_data["details"] + eval_data["warnings"]
    if geo_summary:
        fallback_parts = fallback_parts[:2] + [
            f"Dal QTH {pota_data.get('operator_locator')}: {geo_summary}."
        ]
    rule_guidance = " ".join(fallback_parts[:3]) or f"Condizioni {level}: score {score}/100."
    rule_guidance = rule_guidance[:280]
    current_data = f"""
- Condizioni: {level.upper()} (score {score}/100)
- K-index: {eval_data['solar']['k']}
- SFI: {eval_data['solar']['sfi']}
- Attività POTA: {eval_data['total_activity']} spot
- Bande top: {[b[0] for b in eval_data['top_bands'][:2]]}
- QTH operatore: {pota_data.get('operator_locator') or 'Non impostato'}
- Spot POTA localizzati più vicini: {geo_summary or 'Non disponibili'}
- Opportunità: {', '.join(eval_data['opportunities']) if eval_data['opportunities'] else 'Nessuna'}
- Avvertenze: {', '.join(eval_data['warnings']) if eval_data['warnings'] else 'Nessuna'}
- Ora: {periodo}

VALUTAZIONE DETERMINISTICA DELL'APP (non contraddire):
{rule_guidance}
""".strip()

    ai_response, source = ask_ai(instruction, current_data)
    if not ai_response:
        ai_response = rule_guidance
        if source not in {"rules", "rules-guarded"}:
            source = "rules"
    
    return {
        "message": ai_response.strip(),
        "source": source,
        "level": level,
        "score": score,
        "timestamp": datetime.now().isoformat(),
        "data": eval_data
    }

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def dashboard(request: Request, db: Session = Depends(get_db)):
    logs = db.query(QSO).filter(QSO.frequency > 0).order_by(QSO.timestamp.desc()).limit(10).all()
    total, bands, _ = get_qso_statistics(db)
    
    solar_logs = db.query(QSO).filter(QSO.mode == "PROP").order_by(QSO.timestamp.desc()).first()
    latest_alert = db.query(QSO).filter(QSO.mode == "ALERT").order_by(QSO.timestamp.desc()).first()
    
    return templates.TemplateResponse(request, "index.html", {
        "logs": logs,
        "total": total,
        "bands": bands,
        "solar_data": solar_logs.notes if solar_logs else "Non aggiornato",
        "latest_alert": latest_alert.notes if latest_alert else None,
        "callsign": get_callsign(db),
        "operator_locator": get_operator_locator(db),
    })


@app.get("/qso/history")
def qso_history(
    offset: int = 0,
    limit: int = 50,
    band: str = "",
    mode: str = "",
    search: str = "",
    db: Session = Depends(get_db),
):
    """Storico QSO completo, caricato su richiesta e paginato."""
    offset = max(0, offset)
    limit = min(100, max(10, limit))
    query = db.query(QSO).filter(QSO.frequency > 0)

    if band in BAND_RANGES:
        minimum, maximum = BAND_RANGES[band]
        query = query.filter(QSO.frequency >= minimum, QSO.frequency <= maximum)
    if mode:
        query = query.filter(QSO.mode == mode.upper())
    if search.strip():
        term = f"%{search.strip()}%"
        query = query.filter(or_(
            QSO.call_sign.ilike(term),
            QSO.locator.ilike(term),
            QSO.notes.ilike(term),
        ))

    total = query.count()
    logs = query.order_by(QSO.timestamp.desc()).offset(offset).limit(limit).all()
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": [
            {
                "id": log.id,
                "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                "frequency": log.frequency,
                "band": get_band(log.frequency),
                "mode": log.mode,
                "call_sign": log.call_sign,
                "rst_received": log.rst_received,
                "locator": log.locator,
                "notes": log.notes,
            }
            for log in logs
        ],
    }


def local_timestamp_to_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.astimezone(timezone.utc)
    return value.astimezone(timezone.utc)


def csv_rows(db: Session):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "id", "timestamp_locale", "timestamp_utc", "frequenza_mhz", "banda",
        "modo", "nominativo", "rst_ricevuto", "locator", "note",
    ])
    yield "\ufeff" + output.getvalue()
    for qso in (
        db.query(QSO)
        .filter(QSO.frequency > 0)
        .order_by(QSO.timestamp.asc())
        .yield_per(500)
    ):
        output.seek(0)
        output.truncate(0)
        utc_value = local_timestamp_to_utc(qso.timestamp)
        writer.writerow([
            qso.id,
            qso.timestamp.isoformat() if qso.timestamp else "",
            utc_value.isoformat().replace("+00:00", "Z") if utc_value else "",
            f"{qso.frequency:.6f}".rstrip("0").rstrip("."),
            get_band(qso.frequency),
            qso.mode,
            qso.call_sign,
            qso.rst_received,
            qso.locator,
            qso.notes,
        ])
        yield output.getvalue()


def adif_field(name: str, value) -> str:
    text = "" if value is None else str(value)
    return f"<{name}:{len(text)}>{text}" if text else ""


def adif_rows(db: Session):
    yield (
        "Generated by Consigliere di Stazione\r\n"
        f"<ADIF_VER:5>3.1.4{adif_field('PROGRAMID', 'Consigliere di Stazione')}<EOH>\r\n"
    )
    for qso in (
        db.query(QSO)
        .filter(QSO.frequency > 0)
        .order_by(QSO.timestamp.asc())
        .yield_per(500)
    ):
        utc_value = local_timestamp_to_utc(qso.timestamp)
        fields = [
            adif_field("CALL", qso.call_sign),
            adif_field("QSO_DATE", utc_value.strftime("%Y%m%d") if utc_value else ""),
            adif_field("TIME_ON", utc_value.strftime("%H%M%S") if utc_value else ""),
            adif_field("BAND", get_band(qso.frequency) if get_band(qso.frequency) in BAND_RANGES else ""),
            adif_field("MODE", qso.mode),
            adif_field("FREQ", f"{qso.frequency:.6f}".rstrip("0").rstrip(".")),
            adif_field("RST_RCVD", qso.rst_received),
            adif_field("GRIDSQUARE", qso.locator),
            adif_field("COMMENT", qso.notes),
        ]
        yield "".join(field for field in fields if field) + "<EOR>\r\n"


@app.get("/export/qso.csv")
def export_qso_csv(db: Session = Depends(get_db)):
    filename = f"consigliere-qso-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    return StreamingResponse(
        csv_rows(db),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/export/qso.adi")
def export_qso_adif(db: Session = Depends(get_db)):
    filename = f"consigliere-qso-{datetime.now().strftime('%Y%m%d-%H%M%S')}.adi"
    return StreamingResponse(
        adif_rows(db),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def create_database_backup() -> Path:
    file_descriptor, backup_name = tempfile.mkstemp(
        prefix="consigliere-backup-", suffix=".db"
    )
    os.close(file_descriptor)
    backup_path = Path(backup_name)
    try:
        with sqlite3.connect(_db_path) as source, sqlite3.connect(backup_path) as destination:
            source.backup(destination)
        return backup_path
    except Exception:
        backup_path.unlink(missing_ok=True)
        raise


@app.get("/backup/database")
def download_database_backup():
    backup_path = create_database_backup()
    filename = f"swl_logs-backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}.db"
    return FileResponse(
        backup_path,
        filename=filename,
        media_type="application/vnd.sqlite3",
        background=BackgroundTask(backup_path.unlink, missing_ok=True),
    )

@app.post("/add")
def add_qso(frequency: float = Form(...), mode: str = Form(...), 
            call_sign: str = Form(...), rst_received: str = Form(""),
            locator: str = Form(""), notes: str = Form(""),
            db: Session = Depends(get_db)):
    try:
        values = validate_qso_fields(
            frequency, mode, call_sign, rst_received, locator, notes
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    qso = QSO(**values, timestamp=datetime.now())
    db.add(qso)
    db.commit()
    return {"status": "ok", "id": qso.id, "band": get_band(values["frequency"])}


@app.post("/qso/{qso_id}/edit")
def edit_qso(
    qso_id: int,
    frequency: float = Form(...),
    mode: str = Form(...),
    call_sign: str = Form(...),
    rst_received: str = Form(""),
    locator: str = Form(""),
    notes: str = Form(""),
    db: Session = Depends(get_db),
):
    qso = db.query(QSO).filter(QSO.id == qso_id, QSO.frequency > 0).first()
    if not qso:
        raise HTTPException(status_code=404, detail="QSO non trovato")
    try:
        values = validate_qso_fields(
            frequency, mode, call_sign, rst_received, locator, notes
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    for field, value in values.items():
        setattr(qso, field, value)
    db.commit()
    return {"status": "ok", "id": qso.id, "band": get_band(qso.frequency)}


@app.post("/qso/{qso_id}/delete")
def delete_qso(qso_id: int, db: Session = Depends(get_db)):
    qso = db.query(QSO).filter(QSO.id == qso_id, QSO.frequency > 0).first()
    if not qso:
        raise HTTPException(status_code=404, detail="QSO non trovato")
    db.delete(qso)
    db.commit()
    return {"status": "ok", "id": qso_id}

@app.get("/fetch/solar")
def fetch_solar(db: Session = Depends(get_db)):
    solar = get_solar_data()
    
    if solar.get("status") != "ok":
        return {"status": "error", "message": solar.get("error")}
    
    note = f"K: {solar.get('k_index')} | SFI: {solar.get('sfi')} | "
    interpretation = "Dati aggiornati"
    
    try:
        k = float(solar.get('k_index', 0))
        if k < 3:
            note += "Campo geomagnetico quieto"
            interpretation = (
                f"K-index {k}: campo geomagnetico quieto; da solo non indica "
                "quali bande siano aperte."
            )
        elif k < 5:
            note += "Campo geomagnetico moderato"
            interpretation = f"K-index {k}: attività geomagnetica moderata."
        else:
            note += "Campo geomagnetico disturbato"
            interpretation = f"K-index {k}: campo geomagnetico disturbato; verifica i segnali reali."
    except:
        note += "Dati aggiornati"
    
    latest_prop = (
        db.query(QSO)
        .filter(QSO.mode == "PROP", QSO.call_sign == "NOAA_DATA")
        .order_by(QSO.timestamp.desc())
        .first()
    )
    saved = not (
        solar.get("cached")
        and latest_prop
        and latest_prop.notes == note
        and latest_prop.timestamp
        and datetime.now() - latest_prop.timestamp < timedelta(minutes=5)
    )
    if saved:
        db.add(QSO(
            frequency=0.0,
            mode="PROP",
            call_sign="NOAA_DATA",
            rst_received=str(solar.get('k_index', '')),
            notes=note,
            timestamp=datetime.now(),
        ))
        db.commit()
    
    return {
        "status": "ok", "data": solar, "note": note,
        "interpretation": interpretation, "saved": saved,
    }

@app.get("/fetch/dxspots")
def fetch_dxspots(band: str = "20m", mode: str = "ALL", db: Session = Depends(get_db)):
    result = get_pota_spots(
        band=band,
        mode=mode,
        limit=10,
        operator_locator=get_operator_locator(db),
    )
    
    return {
        "status": "ok" if "error" not in result else "error",
        "band": band,
        "mode": mode if mode != "ALL" else "Tutti",
        "spots_found": result.get("count", 0),
        "spots": result.get("spots", []),
        "activity_by_band": result.get("by_band", {}),
        "operator_locator": result.get("operator_locator", ""),
        "located_count": result.get("located_count", 0),
        "cached": result.get("cached", False),
        "cache_age_seconds": result.get("cache_age_seconds", 0),
        "stale": result.get("stale", False),
        "warning": result.get("warning"),
        "source": "POTA.app",
        "data_timestamp": result.get("data_timestamp"),
        "response_timestamp": datetime.now().astimezone().isoformat(),
        "timestamp": result.get("data_timestamp") or datetime.now().astimezone().isoformat(),
    }

@app.get("/alert/check")
def check_alert_auto(db: Session = Depends(get_db), force: bool = False):
    solar = get_solar_data()
    if solar.get("status") != "ok":
        return {"status": "error", "message": "Dati solar non disponibili", "error_detail": solar.get("error")}
    
    pota = get_pota_spots(
        band="ALL", limit=0, operator_locator=get_operator_locator(db)
    )
    evaluation = evaluate_conditions(solar, pota)
    
    if evaluation["score"] < 40 and not force:
        return {
            "status": "quiet",
            "message": "Condizioni non particolarmente favorevoli",
            "score": evaluation["score"],
            "details": evaluation
        }
    
    alert = generate_smart_alert(evaluation, solar, pota)
    
    if not alert:
        return {"status": "no_alert", "reason": "Condizioni non rilevanti"}
    
    content_hash = f"{alert['level']}_{evaluation['score']}_{str(evaluation['top_bands'])}"
    if not force and not alert_cache.should_send("propagation", content_hash):
        return {"status": "cached", "message": "Alert simile già inviato", "last": str(alert_cache.last_alerts.get("propagation", [None])[0])}
    
    alert_cache.update("propagation", content_hash)
    
    alert_qso = QSO(
        frequency=0.0,
        mode="ALERT",
        call_sign="SYSTEM_AI",
        rst_received=str(evaluation["score"]),
        notes=f"[{alert['level'].upper()}] {alert['message']}...",
        timestamp=datetime.now()
    )
    db.add(alert_qso)
    db.commit()
    
    return {"status": "alert_generated", "alert": alert, "evaluation": evaluation}

@app.get("/alert/status")
def alert_status(db: Session = Depends(get_db)):
    solar = get_solar_data()
    pota = get_pota_spots(
        band="ALL", limit=0, operator_locator=get_operator_locator(db)
    )
    evaluation = evaluate_conditions(solar, pota)
    
    return {
        "current_conditions": evaluation,
        "solar": {"k": solar.get("k_index"), "sfi": solar.get("sfi"), "status": solar.get("status")},
        "activity_summary": pota.get("by_band", {}),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/alert/history")
def alert_history(limit: int = 10, db: Session = Depends(get_db)):
    alerts = db.query(QSO).filter(QSO.mode == "ALERT").order_by(QSO.timestamp.desc()).limit(limit).all()
    
    return {
        "alerts": [
            {
                "time": a.timestamp.isoformat() if a.timestamp else None,
                "level": a.rst_received,
                "message": a.notes,
                "ago": str(datetime.now() - a.timestamp).split('.')[0] if a.timestamp else "N/A"
            }
            for a in alerts
        ]
    }

@app.post("/ui/trigger-check")
def trigger_check_ui(db: Session = Depends(get_db)):
    """
    Endpoint POST per il bottone "Controlla Ora" dalla UI
    Forza la generazione dell'alert
    """
    return check_alert_auto(db=db, force=True)

@app.get("/ui/alert-stream")
async def alert_stream():
    async def event_generator():
        while True:
            try:
                solar = get_solar_data()
                pota = get_pota_spots(band="ALL", limit=0)
                evaluation = evaluate_conditions(solar, pota)
                
                data = {
                    "type": "status",
                    "time": datetime.now().isoformat(),
                    "score": evaluation["score"],
                    "level": evaluation["level"],
                    "solar": {
                        "k": solar.get("k_index", "N/A"),
                        "sfi": solar.get("sfi", "N/A"),
                        "status": solar.get("status", "unknown")
                    },
                    "activity": evaluation["top_bands"][:3]
                }
                
                yield f"data: {json.dumps(data)}\n\n"
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"SSE Error: {e}")
                await asyncio.sleep(10)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

# ============================================================
# DEBUG ENDPOINTS
# ============================================================

@app.get("/debug/noaa-test")
def test_noaa_connection():
    start = time.time()
    result = get_solar_data(force_refresh=True)
    elapsed = round(time.time() - start, 2)
    
    return {
        "test_timestamp": datetime.now().isoformat(),
        "response_time_ms": elapsed * 1000,
        "noaa_status": result.get("status"),
        "data": result if result.get("status") == "ok" else None,
        "error_details": result.get("error") if result.get("status") == "error" else None,
        "urls_tested": {
            "k_index": "https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json",
            "sfi": "https://services.swpc.noaa.gov/json/f107_cm_flux.json"
        }
    }

@app.get("/debug/pota-raw")
def debug_pota_raw(limit: int = 3):
    try:
        url = "https://api.pota.app/spot/activator"
        response = requests.get(url, timeout=10)
        data = response.json()
        sample = data[:limit] if data else []
        
        return {
            "status": "ok",
            "total_spots": len(data),
            "sample_structure": sample,
            "hint": "Controlla campi: frequency/freq, mode, activator, name, grid4/grid6"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
def generate_rule_based_advice(solar: dict, pota: dict, bands: dict, modes: dict, ora: int) -> str:
    k = solar.get("k_float")
    sfi = solar.get("sfi_float")
    periodo = "mattina" if 6 <= ora < 12 else "pomeriggio" if 12 <= ora < 18 else "sera" if 18 <= ora < 22 else "notte"
    consigli = []

    if k is not None:
        if k < 3:
            consigli.append(f"K-index basso ({k:.1f}): campo geomagnetico quieto; riduce il rischio di disturbi ma, da solo, non determina quali bande siano aperte.")
        elif k < 5:
            consigli.append(f"K-index moderato ({k:.1f}): condizioni potenzialmente variabili; verificare i segnali e gli spot reali prima di scegliere la banda.")
        else:
            consigli.append(f"K-index alto ({k:.1f}): possibili disturbi geomagnetici, soprattutto sui percorsi ad alte latitudini; monitorare i segnali reali.")

    if sfi is not None:
        if sfi > 150:
            consigli.append(f"Solar Flux molto alto ({sfi:.0f}): aumenta la probabilità di propagazione sulle bande HF alte, senza garantire aperture o direzioni specifiche.")
        elif sfi > 100:
            consigli.append(f"Solar Flux buono ({sfi:.0f}): può favorire le bande da 20m a 10m; confermare l'apertura con segnali e spot correnti.")
        elif sfi > 70:
            consigli.append(f"Solar Flux nella norma ({sfi:.0f}): non indica da solo una banda migliore; confrontare 20m e 40m con i dati correnti.")
        else:
            consigli.append(f"Solar Flux basso ({sfi:.0f}): le bande HF alte possono essere meno favorite; verificare 40m e 80m senza considerarle garantite.")

    pota_count = pota.get("count", 0)
    selected_band = pota.get("band")
    by_band = pota.get("by_band", {})
    top_bands = sorted(by_band.items(), key=lambda x: x[1], reverse=True)[:2]
    pota_advice = ""
    if pota_count > 0 and selected_band and selected_band != "ALL":
        pota_advice = (
            f"Sono presenti {pota_count} spot POTA correnti su {selected_band}: "
            "questa è attività osservata, non una misura della qualità di propagazione o ricezione."
        )
    elif top_bands:
        band_str = " e ".join(f"{b[0]} ({b[1]} spot)" for b in top_bands)
        pota_advice = f"Gli spot POTA correnti risultano più numerosi su {band_str}; usare questi conteggi come indicazione di attività, non di propagazione garantita."
    elif pota_count == 0 and ora >= 22:
        pota_advice = f"Ora di {periodo}: nessuno spot POTA trovato nel filtro corrente; il dato non permette di dedurre l'attività sulle altre bande."

    geo_summary = nearest_spots_summary(pota, limit=2)
    if geo_summary:
        geographic_advice = (
            f"Dal QTH {pota.get('operator_locator')}, tra gli spot dotati di locator i più vicini sono: "
            f"{geo_summary}. Distanze e direzioni sono calcolate dai locator, non stimate dall'IA."
        )
        pota_advice = f"{pota_advice} {geographic_advice}".strip()
    if pota_advice:
        consigli.append(pota_advice)

    if bands:
        banda_preferita = max(bands, key=bands.get)
        consigli.append(f"Nel tuo storico la banda più registrata è {banda_preferita} ({bands[banda_preferita]} QSO); questo descrive le tue abitudini, non l'attività POTA corrente.")

    return "\n".join(f"• {c}" for c in consigli[:4]) if consigli else "Dati insufficienti per generare consigli."


@app.get("/ai/analyze")
def ai_analyze(db: Session = Depends(get_db)):
    total_qsos, bands, modes = get_qso_statistics(db)

    solar = get_solar_data()
    # Una sola lettura dell'endpoint POTA contiene gli spot di tutte le frequenze.
    # Li raggruppiamo per banda per dare al consiglio una vista completa, senza
    # ripetere la stessa chiamata HTTP una volta per ogni banda.
    operator_locator = get_operator_locator(db)
    pota = get_pota_spots("ALL", "ALL", 0, operator_locator=operator_locator)
    pota_by_band = pota.get("by_band", {})
    pota_top_bands = sorted(pota_by_band.items(), key=lambda item: item[1], reverse=True)[:5]
    pota_nearest = pota.get("nearest_spots", [])

    local_now = datetime.now().astimezone()
    ora = local_now.hour
    periodo = "mattina" if 6 <= ora < 12 else "pomeriggio" if 12 <= ora < 18 else "sera" if 18 <= ora < 22 else "notte"
    rule_guidance = generate_rule_based_advice(solar, pota, bands, modes, ora)

    instruction = (
        "Analizza i dati operativi reali e fornisci 3 consigli pratici specifici per "
        "queste condizioni. Massimo 4 righe, tono professionale ma diretto. Non presentare "
        "frequenze o stazioni tratte dal corpus come se fossero attive adesso. Gli spot POTA "
        "misurano attività, non qualità della propagazione. Non definire la propagazione "
        "ottimale o garantita e non dedurre DX, rumore o aperture dai soli conteggi. Non "
        "convertire nomi di banda come 20m o 40m in frequenze numeriche se tali frequenze "
        "non compaiono esplicitamente nei dati correnti."
    )
    current_data = f"""
STATISTICHE OPERATORE:
- QTH locator: {operator_locator or 'Non impostato'}
- Bande più usate: {bands}
- Modi preferiti: {modes}
- QSO totali nello storico: {total_qsos}

CONDIZIONI SOLAR (NOAA):
- K-index: {solar.get('k_index', 'N/A')}
- SFI: {solar.get('sfi', 'N/A')}

ATTIVITA' POTA CORRENTE (tutte le bande):
- Spot totali con banda riconosciuta: {pota.get('count', 0)}
- Conteggi reali per banda: {pota_by_band}
- Bande con più spot: {pota_top_bands}
- Spot con locator utilizzabile: {pota.get('located_count', 0)}
- Spot localizzati più vicini al QTH: {pota_nearest}

ORARIO: {periodo} (locale, UTC{local_now.strftime('%z')[:3]}:{local_now.strftime('%z')[3:]})

VALUTAZIONE DETERMINISTICA DELL'APP (non contraddire):
{rule_guidance}
""".strip()

    response_text, source = ask_ai(instruction, current_data)
    if not response_text:
        response_text = rule_guidance
        if source not in {"rules", "rules-guarded"}:
            source = "rules"

    model = (
        "swlbot-rag" if source == "swlbot-rag" else
        DIRECT_LLM_MODEL if source.startswith("llm-direct") else None
    )
    return {"response": response_text, "source": source, "model": model}

@app.get("/settings/callsign")
def read_callsign(db: Session = Depends(get_db)):
    return {"callsign": get_callsign(db)}

@app.post("/settings/callsign")
def save_callsign(callsign: str = Form(...), db: Session = Depends(get_db)):
    normalized = callsign.strip().upper()
    if not CALLSIGN_PATTERN.fullmatch(normalized):
        raise HTTPException(
            status_code=400,
            detail="Nominativo non valido: massimo 20 caratteri, lettere, numeri, /, . o -",
        )
    row = db.query(Settings).filter(Settings.key == "callsign").first()
    if row:
        row.value = normalized
    else:
        db.add(Settings(key="callsign", value=normalized))
    db.commit()
    return {"callsign": normalized}


@app.get("/settings/locator")
def read_operator_locator(db: Session = Depends(get_db)):
    locator = get_operator_locator(db)
    if not locator:
        return {"locator": "", "latitude": None, "longitude": None}
    latitude, longitude = maidenhead_to_coordinates(locator)
    return {
        "locator": locator,
        "latitude": round(latitude, 5),
        "longitude": round(longitude, 5),
    }


@app.post("/settings/locator")
def save_operator_locator(locator: str = Form(""), db: Session = Depends(get_db)):
    try:
        normalized = normalize_locator(locator)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    row = db.query(Settings).filter(Settings.key == "operator_locator").first()
    if row:
        row.value = normalized
    else:
        db.add(Settings(key="operator_locator", value=normalized))
    db.commit()

    if not normalized:
        return {"locator": "", "latitude": None, "longitude": None}
    latitude, longitude = maidenhead_to_coordinates(normalized)
    return {
        "locator": normalized,
        "latitude": round(latitude, 5),
        "longitude": round(longitude, 5),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
