#!/usr/bin/env python3
"""
Database Pruning Utility for CCIP Grounding (v2 schema).
Removes entries from the identity database if the corresponding image file 
no longer exists on the filesystem.
"""
import argparse
import os
from tqdm import tqdm
from dbutil import open_v2


def prune_database(db_path):
    if not os.path.exists(db_path):
        print(f"Error: Database file {db_path} not found.")
        return

    conn = open_v2(db_path)
    try:
        rows = conn.execute("SELECT path FROM images").fetchall()
        if not rows:
            print("No images found in the database to verify.")
            return

        print(f"Checking {len(rows)} images in {db_path}...")

        to_delete = [p for (p,) in tqdm(rows) if not os.path.exists(p)]

        if not to_delete:
            print("All images verified. No pruning necessary.")
            return

        print(f"Pruning {len(to_delete)} missing files from database...")
        with conn:
            conn.executemany("DELETE FROM image_tags WHERE path = ?",
                             [(p,) for p in to_delete])
            conn.executemany("DELETE FROM images WHERE path = ?",
                             [(p,) for p in to_delete])

        print(f"Successfully pruned {len(to_delete)} entries.")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Prune missing images from the CCIP identity database.")
    parser.add_argument("db", help="Path to the .sql database file")
    args = parser.parse_args()

    prune_database(args.db)


if __name__ == "__main__":
    main()
