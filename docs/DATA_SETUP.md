# Data Setup

The pipeline requires three datasets. None are included in the repo (too large + licensing).
Place them in a `data/` directory at the project root.

## Required Files

```
data/
├── weaviate-gorilla.json              # Weaviate benchmark (315 queries with AST)
├── bird-benchmark/
│   ├── bird-collections.json          # BIRD schemas in Weaviate format (143 collections)
│   └── dev_20240627/
│       ├── dev.json                   # 1,534 BIRD dev queries
│       └── dev_databases/             # 11 SQLite databases
│           ├── california_schools/california_schools.sqlite
│           ├── card_games/card_games.sqlite
│           ├── codebase_community/codebase_community.sqlite
│           ├── debit_card_specializing/debit_card_specializing.sqlite
│           ├── european_football_2/european_football_2.sqlite
│           ├── financial/financial.sqlite
│           ├── formula_1/formula_1.sqlite
│           ├── student_club/student_club.sqlite
│           ├── superhero/superhero.sqlite
│           ├── thrombosis_prediction/thrombosis_prediction.sqlite
│           └── toxicology/toxicology.sqlite
└── bird-processor/
    └── bird-to-weaviate.json          # Alternative BIRD→Weaviate conversion
```

Optional extras (for multi-domain experiments):
```
data/
├── retail-world-weaviate.json
├── movie-3-weaviate.json
├── student-loan-weaviate.json
├── chicago-crime-weaviate.json
└── university-weaviate.json
```

## How to Download

### BIRD Benchmark
Official release from the BIRD team:
- Dev set (1,534 queries + 11 DBs): https://bird-bench.github.io/
- Direct download: Look for `dev_20240627.zip` on the BIRD website

Unzip into `data/bird-benchmark/`.

### Weaviate Gorilla
Original benchmark from Weaviate:
- https://github.com/weaviate/Gorilla-Benchmark

Download `weaviate-gorilla.json` and place at `data/weaviate-gorilla.json`.

### BIRD-to-Weaviate Collections
This is the adapted schema format we use. The file `bird-collections.json` was generated
by converting BIRD CREATE TABLE definitions into Weaviate-style property descriptions.

If you need to regenerate it from scratch, see the conversion logic in the research notes
(ask the authors).

## Verification

After setup, verify the pipeline can load the data:

```bash
./venv/bin/python3 -c "
from src.mcp.server import MCPServer
server = MCPServer.from_multiple_sources(
    'data/weaviate-gorilla.json',
    'data/bird-benchmark/bird-collections.json'
)
print(f'Loaded {len(server.schemas)} collections')
# Expected: ~143 collections
"
```

For BIRD execution accuracy evaluation, also verify SQLite databases are accessible:

```bash
./venv/bin/python3 -c "
import sqlite3
from pathlib import Path
for db in ['california_schools', 'card_games', 'superhero']:
    p = Path(f'data/bird-benchmark/dev_20240627/dev_databases/{db}/{db}.sqlite')
    assert p.exists(), f'Missing: {p}'
    conn = sqlite3.connect(str(p))
    tables = conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()
    print(f'{db}: {len(tables)} tables')
"
```
