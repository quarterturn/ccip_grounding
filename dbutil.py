"""Shared DB helpers for CCIP grounding tools (v2 schema).

v2: images(path PK, name) + image_tags(path, char).
All tools must refuse to run against legacy v1 databases.
"""
import sqlite3


class LegacySchemaError(SystemExit):
    pass


def open_v2(db_path):
    """Open a grounding DB, ensuring it uses the v2 schema."""
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='images'"
    ).fetchone()
    if row is None:
        conn.close()
        raise LegacySchemaError(f"Error: no 'images' table in {db_path}")
    if "characters" in (row[0] or "").lower():
        conn.close()
        raise LegacySchemaError(
            f"Error: {db_path} uses the legacy v1 schema. "
            f"Run: python3 migrate_v2.py --db {db_path} --dry-run")
    return conn


def distinct_chars(conn):
    """All character tags, exact values, sorted."""
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT char FROM image_tags ORDER BY char")]


def paths_for_char(conn, char):
    """Exact-match paths tagged with char (no substring bleed)."""
    return [r[0] for r in conn.execute(
        "SELECT path FROM image_tags WHERE char = ? ORDER BY path", (char,))]
