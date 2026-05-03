#!/usr/bin/env python3
"""
sqlite_oracle.py — Run BIRD ground-truth SQL queries against their SQLite DBs
and return the expected row sets. Used by simulated_exec_v2.py as the
ground-truth oracle.

Only 5 of 11 BIRD SQLite DBs are present in this checkout:
  - california_schools
  - card_games
  - debit_card_specializing
  - european_football_2
  - formula_1

The remaining 6 (codebase_community, financial, student_club, superhero,
thrombosis_prediction, toxicology) are not available. Queries against those
DBs will return None (indeterminate).

Usage:
    from eval.scripts.sqlite_oracle import SQLiteOracle
    oracle = SQLiteOracle()
    rows = oracle.execute(db_id="formula_1", sql="SELECT ...")
    # rows is a list of tuples, or None if DB missing
"""

import sqlite3
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BIRD_DB_DIR = PROJECT_ROOT / "data/bird-benchmark/dev_20240627/dev_databases"

# DBs we know are present and have non-zero sized files.
# Any DB not in this set gets None back.
AVAILABLE_DBS = {
    "california_schools",
    "card_games",
    "codebase_community",
    "debit_card_specializing",
    "european_football_2",
    "financial",
    "formula_1",
    "student_club",
    "superhero",
    "thrombosis_prediction",
    "toxicology",
}


class SQLiteOracle:
    """Opens BIRD SQLite DBs on demand. Caches connections for reuse."""

    def __init__(self):
        self._conns: dict[str, sqlite3.Connection] = {}

    def _path_for(self, db_id: str) -> Optional[Path]:
        p = BIRD_DB_DIR / db_id / f"{db_id}.sqlite"
        if not p.exists() or p.stat().st_size == 0:
            return None
        return p

    def _conn(self, db_id: str) -> Optional[sqlite3.Connection]:
        if db_id in self._conns:
            return self._conns[db_id]
        p = self._path_for(db_id)
        if p is None:
            return None
        c = sqlite3.connect(str(p))
        c.text_factory = str
        self._conns[db_id] = c
        return c

    def available(self, db_id: str) -> bool:
        """Is this DB's SQLite file on disk?"""
        return db_id in AVAILABLE_DBS and self._path_for(db_id) is not None

    def execute(self, db_id: str, sql: str, timeout: float = 5.0) -> Optional[list]:
        """Run a SQL query against the DB. Returns list of rows, or None if
        the DB isn't available or the query errors.
        """
        c = self._conn(db_id)
        if c is None:
            return None
        c.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
        try:
            cur = c.cursor()
            cur.execute(sql)
            return cur.fetchall()
        except sqlite3.Error:
            return None

    def close(self):
        for c in self._conns.values():
            try:
                c.close()
            except Exception:
                pass
        self._conns.clear()


if __name__ == "__main__":
    import json
    dev_path = PROJECT_ROOT / "data/bird-benchmark/dev_20240627/dev.json"
    dev = json.loads(dev_path.read_text())
    oracle = SQLiteOracle()

    # Stats: how many queries per DB, how many we can execute
    from collections import Counter, defaultdict
    db_count = Counter()
    db_executable = Counter()
    db_errors = defaultdict(int)

    for q in dev:
        db_id = q["db_id"]
        db_count[db_id] += 1
        if not oracle.available(db_id):
            continue
        rows = oracle.execute(db_id, q["SQL"])
        if rows is not None:
            db_executable[db_id] += 1
        else:
            db_errors[db_id] += 1

    print(f"{'DB':<30s} {'total':>6s} {'exec':>6s} {'errors':>7s}")
    print("-" * 52)
    for db_id in sorted(db_count.keys()):
        total = db_count[db_id]
        ex = db_executable[db_id]
        er = db_errors[db_id]
        avail = "OK" if oracle.available(db_id) else "MISSING"
        print(f"{db_id:<30s} {total:>6d} {ex:>6d} {er:>7d}  [{avail}]")

    total_ex = sum(db_executable.values())
    print(f"\nTotal executable: {total_ex} / {sum(db_count.values())}")
    oracle.close()
