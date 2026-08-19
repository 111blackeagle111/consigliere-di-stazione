#!/usr/bin/env python3
"""Inserisce uno storico QSO sintetico e riproducibile di sei mesi."""

import argparse
import random
import shutil
import sqlite3
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path


DEMO_TAG = "[DEMO-6M]"
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB = BASE_DIR / "swl_logs.db"

BAND_FREQUENCIES = {
    "80m": {"FT8": 3.573, "FT4": 3.575, "CW": 3.550, "SSB": 3.750},
    "40m": {"FT8": 7.074, "FT4": 7.0475, "CW": 7.030, "SSB": 7.100},
    "20m": {"FT8": 14.074, "FT4": 14.080, "CW": 14.050, "SSB": 14.200},
    "17m": {"FT8": 18.100, "FT4": 18.104, "CW": 18.080, "SSB": 18.130},
    "15m": {"FT8": 21.074, "FT4": 21.140, "CW": 21.050, "SSB": 21.250},
    "10m": {"FT8": 28.074, "FT4": 28.180, "CW": 28.050, "SSB": 28.500},
}

WINTER_WEIGHTS = {"80m": 20, "40m": 35, "20m": 30, "17m": 8, "15m": 5, "10m": 2}
SUMMER_WEIGHTS = {"80m": 8, "40m": 22, "20m": 38, "17m": 14, "15m": 12, "10m": 6}
MODE_WEIGHTS = {"FT8": 45, "SSB": 25, "CW": 20, "FT4": 10}
LOCATORS = ["JN45", "JN55", "JN65", "JN66", "JN75", "JN76", "JO31", "JO40", "JO60", "IN80"]
CALL_PREFIXES = ["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "IV3", "IZ0", "DL1", "F4", "EA3", "OE3", "HB9", "9A1", "S51", "SP5", "OK1", "PA3", "ON4", "G4", "M0"]


def weighted_choice(rng: random.Random, weights: dict[str, int]) -> str:
    return rng.choices(list(weights), weights=list(weights.values()), k=1)[0]


def signal_report(rng: random.Random, mode: str) -> str:
    if mode in {"FT8", "FT4"}:
        return str(rng.randint(-18, 5))
    if mode == "CW":
        return rng.choice(["539", "559", "579", "599"])
    return rng.choice(["53", "55", "57", "59"])


def synthetic_callsign(rng: random.Random) -> str:
    suffix = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(3))
    return f"{rng.choice(CALL_PREFIXES)}{suffix}"


def build_rows(count: int, now: datetime) -> list[tuple]:
    rng = random.Random(6502)
    rows = []
    for index in range(count):
        timestamp = now - timedelta(
            days=rng.randrange(180),
            hours=rng.randrange(24),
            minutes=rng.randrange(60),
        )
        seasonal_weights = WINTER_WEIGHTS if timestamp.month in {11, 12, 1, 2, 3, 4} else SUMMER_WEIGHTS
        band = weighted_choice(rng, seasonal_weights)
        mode = weighted_choice(rng, MODE_WEIGHTS)
        frequency = BAND_FREQUENCIES[band][mode]
        rst = signal_report(rng, mode)
        rows.append((
            timestamp.isoformat(sep=" "),
            frequency,
            mode,
            synthetic_callsign(rng),
            rst,
            rst,
            rng.choice(LOCATORS),
            f"{DEMO_TAG} ascolto sintetico {band} {mode}",
        ))
    return sorted(rows, key=lambda row: row[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--count", type=int, default=240)
    parser.add_argument("--replace-demo", action="store_true")
    args = parser.parse_args()

    if args.count < 1:
        raise SystemExit("--count deve essere maggiore di zero")
    if not args.db.exists():
        raise SystemExit(f"Database non trovato: {args.db}")

    backup = args.db.with_name(
        f"{args.db.name}.backup-before-demo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    shutil.copy2(args.db, backup)

    with sqlite3.connect(args.db) as connection:
        existing_demo = connection.execute(
            "SELECT COUNT(*) FROM qsos WHERE notes LIKE ?", (f"{DEMO_TAG}%",)
        ).fetchone()[0]
        if existing_demo and not args.replace_demo:
            backup.unlink(missing_ok=True)
            raise SystemExit(
                f"Sono gia' presenti {existing_demo} QSO demo; usa --replace-demo per rigenerarli."
            )
        if existing_demo:
            connection.execute("DELETE FROM qsos WHERE notes LIKE ?", (f"{DEMO_TAG}%",))

        rows = build_rows(args.count, datetime.now())
        connection.executemany(
            """
            INSERT INTO qsos
                (timestamp, frequency, mode, call_sign, rst_received, rst_sent, locator, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()

    bands = Counter()
    modes = Counter(row[2] for row in rows)
    for row in rows:
        frequency = row[1]
        band = min(BAND_FREQUENCIES, key=lambda name: abs(frequency - BAND_FREQUENCIES[name][row[2]]))
        bands[band] += 1

    print(f"Backup: {backup}")
    print(f"Inseriti: {len(rows)} QSO demo")
    print(f"Periodo: {rows[0][0]} -> {rows[-1][0]}")
    print(f"Bande: {dict(bands.most_common())}")
    print(f"Modi: {dict(modes.most_common())}")


if __name__ == "__main__":
    main()
