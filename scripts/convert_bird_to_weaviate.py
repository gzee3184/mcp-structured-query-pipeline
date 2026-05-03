#!/usr/bin/env python3
"""
Convert BIRD databases to Weaviate format using dev_tables.json metadata.
Does NOT require sqlite files — uses the schema metadata directly.

Enrichment strategies:
1. Column-Name Synthesis: Expands column names into natural language descriptions
2. Question-Driven: Extracts vocabulary from dev.json questions for each table
"""

import json
import re
from pathlib import Path
from collections import defaultdict

# Paths relative to the repo root — this script lives in scripts/
REPO_ROOT = Path(__file__).parent.parent
BASE_DIR = REPO_ROOT / "data" / "bird-benchmark" / "dev_20240627"
TABLES_FILE = BASE_DIR / "dev_tables.json"
DEV_FILE = BASE_DIR / "dev.json"
OUTPUT_FILE = REPO_ROOT / "data" / "bird-benchmark" / "bird-collections.json"

# BIRD column type to Weaviate type mapping
TYPE_MAP = {
    "integer": "int",
    "real": "number",
    "text": "text",
    "date": "text",
    "boolean": "boolean",
    "number": "number",
    "time": "text",
    "datetime": "text",
    "blob": "text",
}

# ── Enrichment #1: Column-Name Synthesis ────────────────────────────────

ABBREVIATIONS = {
    "id": "identifier", "dob": "date of birth", "qty": "quantity",
    "amt": "amount", "desc": "description", "num": "number",
    "addr": "address", "dept": "department", "org": "organization",
    "ref": "reference", "lat": "latitude", "lng": "longitude",
    "alt": "altitude", "url": "web address", "pos": "position",
    "img": "image", "src": "source", "dst": "destination",
    "pct": "percentage", "avg": "average", "max": "maximum",
    "min": "minimum", "cnt": "count", "grp": "group",
    "msg": "message", "lbl": "label", "val": "value",
    "cat": "category", "typ": "type", "stat": "status",
    "idx": "index", "seq": "sequence", "lvl": "level",
}


def expand_column_name(col: str) -> str:
    """Expand a column name into readable words.
    
    'customerID' -> 'customer ID'
    'birth_date' -> 'birth date'
    'GOT' -> 'GOT'
    """
    # Split on underscores
    parts = col.replace("_", " ").strip().split()
    expanded = []
    for part in parts:
        # Split camelCase
        sub_parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)', part)
        if not sub_parts:
            sub_parts = [part]
        for sp in sub_parts:
            low = sp.lower()
            expanded.append(ABBREVIATIONS.get(low, low))
    return " ".join(expanded)


def synthesize_description_from_columns(db_id: str, table_name: str, 
                                         friendly_name: str, columns: list) -> str:
    """Generate a rich description from column names."""
    # Expand all column names
    col_phrases = [expand_column_name(c["name"]) for c in columns]
    
    # Build a natural-language description
    # First make the table name more readable
    table_readable = friendly_name.replace("_", " ").strip()
    db_readable = db_id.replace("_", " ").strip()
    
    desc = f"Records from the {db_readable} database, {table_readable} table. "
    desc += f"Contains data about {table_readable.lower()}"
    
    # Add key column phrases (deduplicate, skip pure IDs)
    meaningful = []
    seen = set()
    for phrase in col_phrases:
        low = phrase.lower().strip()
        if low in seen or low == "identifier" or len(low) < 2:
            continue
        seen.add(low)
        # Skip if it's just "<table>_id" pattern
        if low.endswith("identifier") and len(low.split()) <= 2:
            continue
        meaningful.append(phrase)
    
    if meaningful:
        desc += f" including {', '.join(meaningful[:12])}"
    desc += "."
    
    return desc


# ── Enrichment #2: Question-Driven Vocabulary ───────────────────────────

def build_question_vocabulary(dev_file: Path) -> dict:
    """Extract key vocabulary from dev.json questions, grouped by (db_id, table).
    
    Returns: { (db_id, table_name_lower): set_of_key_phrases }
    """
    if not dev_file.exists():
        return {}
    
    all_q = json.loads(dev_file.read_text())
    vocab = defaultdict(set)
    
    # Common stopwords to filter out
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
        'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
        'neither', 'each', 'every', 'all', 'any', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'only', 'own', 'same', 'than',
        'too', 'very', 'just', 'because', 'as', 'until', 'while', 'of',
        'at', 'by', 'for', 'with', 'about', 'against', 'between', 'through',
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
        'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
        'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
        'your', 'yours', 'yourself', 'he', 'him', 'his', 'himself', 'she',
        'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
        'their', 'theirs', 'themselves', 'many', 'much', 'list', 'give',
        'find', 'show', 'tell', 'get', 'make', 'number', 'total', 'count',
        'among', 'please', 'also', 'like', 'well', 'still', 'even',
        'however', 'per', 'since', 'into', 'within', 'without', 'whether',
    }
    
    for q in all_q:
        db_id = q["db_id"]
        sql = q["SQL"]
        question = q["question"]
        
        # Skip JOINs — they reference multiple tables, harder to attribute
        if "JOIN" in sql.upper():
            continue
        
        # Extract which table this query targets
        match = re.search(r'FROM\s+(\w+)', sql, re.IGNORECASE)
        if not match:
            continue
        table = match.group(1).lower()
        
        # Extract meaningful words from the question (3+ chars, not stopwords)
        words = re.findall(r'[a-zA-Z]{3,}', question.lower())
        meaningful = [w for w in words if w not in stopwords]
        
        # Add individual words
        for word in meaningful:
            vocab[(db_id, table)].add(word)
        
        # Also extract 2-word phrases for richer context
        for i in range(len(words) - 1):
            if words[i] not in stopwords and words[i+1] not in stopwords:
                vocab[(db_id, table)].add(f"{words[i]} {words[i+1]}")
    
    return vocab


# ── Main Conversion ─────────────────────────────────────────────────────

def convert_table_name(name: str) -> str:
    """Convert SQL table name to PascalCase collection name."""
    parts = name.replace("-", "_").split("_")
    return "".join(p.capitalize() for p in parts)


def convert_database(db_entry: dict, question_vocab: dict) -> list:
    """Convert a single database entry from dev_tables.json to Weaviate collections."""
    db_id = db_entry["db_id"]
    table_names = db_entry["table_names_original"]
    # Use clean column names (semantic) instead of original (cryptic)
    # e.g., 'average salary' instead of 'A11' for Financial
    column_entries = db_entry["column_names"] 
    column_types = db_entry["column_types"]
    
    # Group columns by table index
    tables = {}
    for i, (table_idx, col_name) in enumerate(column_entries):
        if table_idx == -1:
            continue
        if table_idx not in tables:
            tables[table_idx] = []
        tables[table_idx].append({
            "name": col_name,
            "type": TYPE_MAP.get(column_types[i].lower(), "text")
        })
    
    collections = []
    for table_idx, columns in tables.items():
        table_name = table_names[table_idx]
        collection_name = f"{convert_table_name(db_id)}{convert_table_name(table_name)}"
        
        friendly_name = db_entry["table_names"][table_idx] if table_idx < len(db_entry["table_names"]) else table_name
        
        # MCPServer format: properties as a LIST of dicts
        properties = []
        for col in columns:
            properties.append({
                "name": col["name"],
                "data_type": [col["type"]],
                "description": f"{expand_column_name(col['name'])} field"
            })
        
        # ── Enrichment #1: Column-name synthesis ──
        description = synthesize_description_from_columns(
            db_id, table_name, friendly_name, columns
        )
        
        # ── Enrichment #2: Question-driven vocabulary ──
        table_key = (db_id, table_name.lower())
        q_vocab = question_vocab.get(table_key, set())
        if q_vocab:
            # Pick top terms (longest phrases first — more specific)
            sorted_terms = sorted(q_vocab, key=len, reverse=True)
            top_terms = sorted_terms[:20]
            description += f" Commonly queried for: {', '.join(top_terms)}."
        
        # ── Superhero Disambiguation ──
        if db_id == 'superhero':
            if table_name == 'attribute':
                collection_name = 'SuperheroGenericAttribute'
                description += " (Generic attributes like logical, behavioural, etc. For physical traits see SuperheroSuperhero)"
            elif table_name == 'superhero':
                description += " Includes physical traits like height, weight, eye colour, hair colour, skin colour."

        collections.append({
            "name": collection_name,
            "properties": properties,
            "envisioned_use_case_overview": description
        })
    
    return collections


def main():
    tables_data = json.loads(TABLES_FILE.read_text())
    
    # Build question vocabulary from dev.json
    print("Building question vocabulary from dev.json...")
    question_vocab = build_question_vocabulary(DEV_FILE)
    print(f"  Extracted vocabulary for {len(question_vocab)} (db, table) pairs")
    
    existing_dbs = {'california_schools', 'card_games', 'debit_card_specializing',
                    'european_football_2', 'formula_1'}
    
    all_collections = []
    new_count = 0
    
    for db_entry in tables_data:
        db_id = db_entry["db_id"]
        try:
            collections = convert_database(db_entry, question_vocab)
            all_collections.extend(collections)
            marker = "★ NEW" if db_id not in existing_dbs else "  existing"
            if db_id not in existing_dbs:
                new_count += len(collections)
            print(f"  {marker} {db_id}: {len(collections)} collections")
        except Exception as e:
            print(f"  ✗ {db_id}: {e}")
    
    output = {
        "weaviate_collections": all_collections,
        "queries": []
    }
    
    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    print(f"\nTotal: {len(all_collections)} collections ({new_count} new) saved to {OUTPUT_FILE.name}")
    
    # Show sample enriched descriptions
    print("\n── Sample Enriched Descriptions ──")
    for coll in all_collections:
        if coll["name"] in ["FinancialLoan", "ToxicologyAtom", "StudentClubZipCode", 
                             "CodebaseCommunityComments", "ThrombosisPredictionLaboratory"]:
            print(f"\n  {coll['name']}:")
            desc = coll["envisioned_use_case_overview"]
            # Wrap long lines
            if len(desc) > 100:
                print(f"    {desc[:100]}")
                print(f"    {desc[100:]}")
            else:
                print(f"    {desc}")


if __name__ == "__main__":
    main()
