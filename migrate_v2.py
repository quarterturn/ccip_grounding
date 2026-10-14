#!/usr/bin/env python3
"""
Migrate a CCIP grounding database from legacy v1 schema to v2.

v1: images(name PK, path, characters TEXT)  -- comma-separated tags, basename key
v2: images(path PK, name) + image_tags(path, char)  -- path key, exact-match tags

The legacy table is preserved as images_legacy. Always run with --dry-run
first to see what would change.
"""
import argparse
import collections
import sqlite3
import sys


def main():
    ap = argparse.ArgumentParser(
        description="Migrate a v1 grounding DB to the v2 path-keyed schema (in place).")
    ap.add_argument("--db", required=True, help="Path to the .sql database to migrate")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would happen; write nothing")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()

    row = cur.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='images'"
    ).fetchone()
    if row is None:
        sys.exit(f"ERROR: no 'images' table in {args.db}")
    if "characters" not in (row[0] or "").lower():
        print("Database already uses the v2 schema. Nothing to do.")
        return

    if cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='images_legacy'").fetchone():
        sys.exit("ERROR: images_legacy already exists; refusing to overwrite. "
                 "Inspect and drop it manually first.")

    rows = cur.execute("SELECT name, path, characters FROM images").fetchall()

    # ---- analyse v1 content (no writes) ----
    tags_by_path = collections.defaultdict(set)
    no_path = []
    for name, path, characters in rows:
        p = (path or "").strip()
        if not p:
            no_path.append(name)
            continue
        for c in (characters or "").split(","):
            c = c.strip()
            if c:
                tags_by_path[p].add(c)

    v1_images = len(rows)
    v2_images = len(tags_by_path) + (1 if no_path else 0)  # untagged paths below
    # paths present in v1 but with empty tags:
    tagged_paths = set(tags_by_path)
    untagged_paths = [p for (_n, p, _c) in rows
                      if (p or "").strip() and p.strip() not in tagged_paths]
    v2_images = len(tagged_paths | {p.strip() for p in untagged_paths})
    v2_tags = sum(len(s) for s in tags_by_path.values())
    chars = collections.Counter()
    for s in tags_by_path.values():
        for c in s:
            chars[c] += 1

    print(f"v1 rows:            {v1_images}")
    print(f"v2 images:          {v2_images}")
    print(f"v2 tag rows:        {v2_tags}")
    print(f"characters:         {len(chars)}")
    for c, n in sorted(chars.items()):
        print(f"  {c}: {n}")
    if no_path:
        print(f"WARNING: {len(no_path)} rows have no path and will be dropped "
              f"(kept in images_legacy): {no_path[:5]}")

    if args.dry_run:
        print("\nDry run: nothing written.")
        return

    # ---- migrate ----
    with conn:
        cur.execute("CREATE TABLE images_new (path TEXT PRIMARY KEY, name TEXT)")
        cur.execute(
            "CREATE TABLE image_tags_new ("
            "path TEXT NOT NULL, char TEXT NOT NULL, "
            "PRIMARY KEY (path, char))")
        for name, path, characters in rows:
            p = (path or "").strip()
            if not p:
                continue
            cur.execute("INSERT OR IGNORE INTO images_new (path, name) VALUES (?, ?)",
                        (p, name))
            for c in (characters or "").split(","):
                c = c.strip()
                if c:
                    cur.execute(
                        "INSERT OR IGNORE INTO image_tags_new (path, char) VALUES (?, ?)",
                        (p, c))
        cur.execute("ALTER TABLE images RENAME TO images_legacy")
        cur.execute("ALTER TABLE images_new RENAME TO images")
        cur.execute("ALTER TABLE image_tags_new RENAME TO image_tags")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_tags_char ON image_tags(char)")

    # ---- verify ----
    new_images = cur.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    new_tags = cur.execute("SELECT COUNT(*) FROM image_tags").fetchone()[0]
    ok = (new_images == v2_images and new_tags == v2_tags)
    print(f"\nMigrated: images={new_images} (expected {v2_images}), "
          f"tags={new_tags} (expected {v2_tags})")
    if not ok:
        sys.exit("ERROR: post-migration counts do not match! "
                 "Legacy data is intact in images_legacy — investigate before use.")
    print("Legacy data preserved in table 'images_legacy'.")


if __name__ == "__main__":
    main()
