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

import os
from pathlib import Path
from fastapi import FastAPI, Depends, Request, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
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
engine = create_engine('sqlite:///swl_logs.db', connect_args={"check_same_thread": False})
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

# Crea tabelle
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============================================================
# FASTAPI SETUP
# ============================================================

app = FastAPI(title="SWL-Log AI")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

LAT = 45.46
LNG = 12.35

# ============================================================
# FUNZIONI HELPER
# ============================================================

def get_band(freq_mhz: float) -> str:
    if freq_mhz == 0.0:
        return "PROP"
    if freq_mhz > 1000:
        freq_mhz = freq_mhz / 1000
    
    bands = [
        (1.8, 2.0, "160m"), (3.5, 4.0, "80m"), (5.3, 5.4, "60m"),
        (7.0, 7.3, "40m"), (10.1, 10.15, "30m"), (14.0, 14.35, "20m"),
        (18.068, 18.168, "17m"), (21.0, 21.45, "15m"), 
        (24.89, 24.99, "12m"), (28.0, 29.7, "10m"), (50.0, 54.0, "6m")
    ]
    
    for min_f, max_f, name in bands:
        if min_f <= freq_mhz <= max_f:
            return name
    return f"{freq_mhz:.3f}MHz"

def ask_ai(prompt: str) -> str:
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2:3b", "prompt": prompt, "stream": False},
            timeout=30
        )
        return response.json().get("response", "Errore AI")
    except Exception as e:
        return f"AI non disponibile: {str(e)}"

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

# ============================================================
# DATI NOAA (ROBUSTO CON RETRY)
# ============================================================

def get_solar_data():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    
    try:
        url_k = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json"
        logger.info(f"Fetching NOAA K-index...")
        resp_k = session.get(url_k, timeout=(10, 30))
        resp_k.raise_for_status()
        data_k = resp_k.json()
        
        url_sfi = "https://services.swpc.noaa.gov/json/f107_cm_flux.json"
        logger.info(f"Fetching NOAA SFI...")
        resp_sfi = session.get(url_sfi, timeout=(10, 30))
        resp_sfi.raise_for_status()
        data_sfi = resp_sfi.json()
        
        observed_values = [x for x in data_k if len(x) > 2 and x[2] == "observed"]
        if observed_values:
            last_k = observed_values[-1]
            k_val = last_k[1]
            k_time = last_k[0]
        else:
            last_k = data_k[-1] if data_k else None
            k_val = last_k[1] if last_k else "N/A"
            k_time = last_k[0] if last_k else None
        
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
            "timestamp": datetime.now().isoformat(),
            "source": "NOAA SWPC"
        }
        
    except requests.exceptions.Timeout:
        logger.error("NOAA Timeout")
        return {"status": "error", "error": "Timeout - NOAA non risponde", "k_index": "N/A", "sfi": "N/A", "k_float": None, "sfi_float": None}
    except requests.exceptions.ConnectionError:
        logger.error("NOAA Connection Error")
        return {"status": "error", "error": "Errore connessione/DNS", "k_index": "N/A", "sfi": "N/A", "k_float": None, "sfi_float": None}
    except Exception as e:
        logger.error(f"NOAA Error: {e}")
        return {"status": "error", "error": str(e), "k_index": "N/A", "sfi": "N/A", "k_float": None, "sfi_float": None}

# ============================================================
# POTA DATA
# ============================================================

def get_pota_spots(band="20m", mode="ALL", limit=50):
    try:
        url = "https://api.pota.app/spot/activator"
        response = requests.get(url, timeout=15)
        
        if response.status_code != 200:
            return {"error": f"HTTP {response.status_code}", "spots": [], "count": 0, "by_band": {}}
        
        all_spots = response.json()
        band_ranges = {
            "160m": (1.8, 2.0), "80m": (3.5, 4.0), "60m": (5.3, 5.4),
            "40m": (7.0, 7.3), "30m": (10.1, 10.15), "20m": (14.0, 14.35),
            "17m": (18.068, 18.168), "15m": (21.0, 21.45), 
            "12m": (24.89, 24.99), "10m": (28.0, 29.7), "6m": (50.0, 54.0)
        }
        
        band_counts = {}
        filtered = []
        
        for spot in all_spots:
            try:
                freq_raw = spot.get("frequency") or spot.get("freq", 0)
                freq = float(freq_raw)
                if freq > 1000:
                    freq = freq / 1000
                
                spot_band = None
                for b_name, (min_f, max_f) in band_ranges.items():
                    if min_f <= freq <= max_f:
                        spot_band = b_name
                        band_counts[b_name] = band_counts.get(b_name, 0) + 1
                        break
                
                if band != "ALL":
                    search_band = band.lower()
                    if spot_band == search_band:
                        spot_mode = str(spot.get("mode", "")).upper()
                        if mode == "ALL" or spot_mode == mode.upper():
                            filtered.append({
                                "call": spot.get("activator", "???"),
                                "freq": f"{freq:.3f}",
                                "mode": spot_mode or "??",
                                "band": spot_band,
                                "park": spot.get("name", "Unknown")[:30],
                                "grid": spot.get("grid4", ""),
                                "time": spot.get("spotTime", "")[11:16] if spot.get("spotTime") else "??"
                            })
            except:
                continue
            
            if band != "ALL" and len(filtered) >= limit:
                break
        
        return {
            "spots": filtered[:limit] if band != "ALL" else [],
            "count": len(filtered) if band != "ALL" else 0,
            "by_band": band_counts,
            "total_spots": len(all_spots),
            "band": band,
            "mode": mode
        }
        
    except Exception as e:
        logger.error(f"Errore POTA: {e}")
        return {"error": str(e), "spots": [], "count": 0, "by_band": {}}

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
            opportunities.append(f"K-index eccellente ({k})")
        elif k < SolarThresholds.K_INDEX_FAIR:
            score += 20
            details.append(f"K-index discreto ({k})")
        else:
            score -= 20
            warnings.append(f"K-index disturbato ({k})")
    
    if sfi is not None:
        if sfi > SolarThresholds.SFI_GOOD:
            score += 40
            opportunities.append(f"SFI eccellente ({sfi})")
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
        opportunities.append("Apertura bande alte")
        score += 15
    
    return {
        "score": min(100, max(0, score)),
        "level": "eccellenti" if score >= 70 else "buone" if score >= 50 else "discrete" if score >= 30 else "scarse",
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
    
    prompt = f"""Sei un assistente radioamatore. Genera un breve messaggio alert (max 3 frasi) per condizioni propagazione attuali.

DATI:
- Condizioni: {level.upper()} (score {score}/100)
- K-index: {eval_data['solar']['k']}
- SFI: {eval_data['solar']['sfi']}
- Attività POTA: {eval_data['total_activity']} spot
- Bande top: {[b[0] for b in eval_data['top_bands'][:2]]}
- Opportunità: {', '.join(eval_data['opportunities']) if eval_data['opportunities'] else 'Nessuna'}
- Avvertenze: {', '.join(eval_data['warnings']) if eval_data['warnings'] else 'Nessuna'}
- Ora: {periodo}

Istruzioni: Tono professionale ma appassionato, italiano corretto, max 280 caratteri."""

    ai_response = ask_ai(prompt)
    
    return {
        "message": ai_response.strip(),
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
    logs = db.query(QSO).order_by(QSO.timestamp.desc()).limit(10).all()
    total = db.query(QSO).filter(QSO.frequency > 0).count()
    
    bands = {}
    for log in db.query(QSO).filter(QSO.frequency > 0).all():
        band = get_band(log.frequency)
        bands[band] = bands.get(band, 0) + 1
    
    solar_logs = db.query(QSO).filter(QSO.mode == "PROP").order_by(QSO.timestamp.desc()).first()
    latest_alert = db.query(QSO).filter(QSO.mode == "ALERT").order_by(QSO.timestamp.desc()).first()
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "logs": logs,
        "total": total,
        "bands": bands,
        "solar_data": solar_logs.notes if solar_logs else "Non aggiornato",
        "latest_alert": latest_alert.notes if latest_alert else None
    })

@app.post("/add")
def add_qso(frequency: float = Form(...), mode: str = Form(...), 
            call_sign: str = Form(...), rst_received: str = Form(""),
            locator: str = Form(""), notes: str = Form(""),
            db: Session = Depends(get_db)):
    
    qso = QSO(
        frequency=float(frequency),
        mode=mode.upper(),
        call_sign=call_sign.upper(),
        rst_received=rst_received,
        locator=locator.upper() if locator else "",
        notes=notes,
        timestamp=datetime.now()
    )
    db.add(qso)
    db.commit()
    return {"status": "ok", "id": qso.id, "band": get_band(float(frequency))}

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
            note += "Eccellente"
            interpretation = f"K-index {k}: Condizioni eccellenti."
        elif k < 5:
            note += "Buona"
        else:
            note += "Disturbata"
            interpretation = f"K-index {k}: Condizioni disturbate."
    except:
        note += "Dati aggiornati"
    
    prop_qso = QSO(
        frequency=0.0,
        mode="PROP",
        call_sign="NOAA_DATA",
        rst_received=str(solar.get('k_index', '')),
        notes=note,
        timestamp=datetime.now()
    )
    db.add(prop_qso)
    db.commit()
    
    return {"status": "ok", "data": solar, "note": note, "interpretation": interpretation}

@app.get("/fetch/dxspots")
def fetch_dxspots(band: str = "20m", mode: str = "ALL", db: Session = Depends(get_db)):
    result = get_pota_spots(band=band, mode=mode, limit=10)
    
    return {
        "status": "ok" if "error" not in result else "error",
        "band": band,
        "mode": mode if mode != "ALL" else "Tutti",
        "spots_found": result.get("count", 0),
        "spots": result.get("spots", []),
        "activity_by_band": result.get("by_band", {}),
        "source": "POTA.app",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/alert/check")
def check_alert_auto(db: Session = Depends(get_db), force: bool = False):
    solar = get_solar_data()
    if solar.get("status") != "ok":
        return {"status": "error", "message": "Dati solar non disponibili", "error_detail": solar.get("error")}
    
    pota = get_pota_spots(band="ALL", limit=0)
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
def alert_status():
    solar = get_solar_data()
    pota = get_pota_spots(band="ALL", limit=0)
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
    result = get_solar_data()
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
            "hint": "Controlla campi: frequency/freq, mode, activator, name"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
@app.get("/ai/analyze")
def ai_analyze(db: Session = Depends(get_db)):
    """
    Analisi AI dei log + condizioni attuali
    """
    # Recupera ultimi 20 QSO reali
    logs = db.query(QSO).filter(QSO.frequency > 0).order_by(QSO.timestamp.desc()).limit(20).all()
    
    # Statistiche bande/modi dai log
    modes = {}
    bands = {}
    for log in logs:
        modes[log.mode] = modes.get(log.mode, 0) + 1
        band = get_band(log.frequency)
        bands[band] = bands.get(band, 0) + 1
    
    # Dati real-time
    solar = get_solar_data()
    pota = get_pota_spots("20m", "", 5)
    
    # Costruisci prompt
    ora = datetime.utcnow().hour
    periodo = "mattina" if 6 <= ora < 12 else "pomeriggio" if 12 <= ora < 18 else "sera" if 18 <= ora < 22 else "notte"
    
    prompt = f"""Sei un consulente radioamatore esperto (IV3ZEW). Analizza questi dati operativi reali:

STATISTICHE OPERATORE:
- Bande più usate: {bands}
- Modi preferiti: {modes}
- QSO recenti: {len(logs)}

CONDIZIONI SOLAR (NOAA):
- K-index: {solar.get('k_index', 'N/A')}
- SFI: {solar.get('sfi', 'N/A')}

ATTIVITA' POTA (reale):
- Spot attuali su 20m: {pota.get('count', 0)} stazioni

ORARIO: {periodo} (UTC)

Dammi 3 consigli pratici specifici per queste condizioni. Max 4 righe. Tono professionale ma diretto."""

    response_text = ask_ai(prompt)
    
    return {"response": response_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
