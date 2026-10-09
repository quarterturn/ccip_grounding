#!/usr/bin/env python3
"""
Database Pruning Utility for CCIP Grounding.
Removes entries from the identity database if the corresponding image file 
no longer exists on the filesystem.
"""
import argparse
import sqlite3
import os
from tqdm import tqdm

def prune_database(db_path):
    if not os.path.exists(db_path):
        print(f"Error: Database file {db_path} not found.")
        return

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Fetch all image records
        cursor.execute("SELECT name, path FROM images")
        rows = cursor.fetchall()
        
        if not rows:
            print("No images found in the database to verify.")
            return

        print(f"Checking {len(rows)} images in {db_path}...")
        
        to_delete = []
        for name, path in tqdm(rows):
            if not os.path.exists(path):
                to_delete.append(name)

        if not to_delete:
            print("All images verified. No pruning necessary.")
            return

        # Remove missing files from DB
        print(f"Pruning {len(to_delete)} missing files from database...")
        
        # Using a transaction for speed
        cursor.execute("BEGIN TRANSACTION")
        cursor.executemany("DELETE FROM images WHERE name = ?", [(name,) for name in to_delete])
        conn.commit()
        
        print(f"Successfully pruned {len(to_delete)} entries.")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

def main():
    parser = argparse.ArgumentParser(description="Prune missing images from the CCIP identity database.")
    parser.add_argument("db", help="Path to the .sql database file")
    args = parser.parse_args()

    prune_database(args.db)

if __name__ == "__main__":
    main()
