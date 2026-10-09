#!/usr/bin/env python3
"""
CCIP Hard Cap Pruner.
Ensures a strict maximum number of images per character in the database.
If a character exceeds the limit, images are randomly pruned until the limit is met.
Note: Pruning an image containing multiple characters will reduce the count for all those characters.
"""
import argparse
import sqlite3
import random
import os
from tqdm import tqdm

def hard_cap_database(db_path, limit):
    if not os.path.exists(db_path):
        print(f"Error: Database file {db_path} not found.")
        return

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get a list of all characters and their current counts
        # We need to parse the comma-separated strings
        cursor.execute("SELECT characters FROM images")
        all_rows = cursor.fetchall()
        
        char_counts = {}
        all_images = [] # list of (name, characters)
        
        for row in all_rows:
            chars_str = row[0]
            if chars_str:
                chars = [c.strip() for c in chars_str.split(',') if c.strip()]
                for c in chars:
                    char_counts[c] = char_counts.get(c, 0) + 1
                # We only store the image if it actually has characters
                all_images.append((row[0], chars))

        # Identify characters that are over the limit
        over_limit = [char for char, count in char_counts.items() if count > limit]
        
        if not over_limit:
            print("No characters exceed the limit. No pruning necessary.")
            return

        print(f"Characters over limit ({limit}):")
        for char in over_limit:
            print(f"  - {char}: {char_counts[char]}")

        # Pruning loop: Continue pruning until no character is over the limit
        # We use a while loop because pruning one image may affect multiple characters
        pruned_count = 0
        while True:
            # Recalculate counts
            current_counts = {}
            current_images = []
            
            # We need to re-query the DB because we are deleting in place
            cursor.execute("SELECT name, characters FROM images")
            rows = cursor.fetchall()
            for row in rows:
                chars_str = row[1]
                if chars_str:
                    chars = [c.strip() for c in chars_str.split(',') if c.strip()]
                    for c in chars:
                        current_counts[c] = current_counts.get(c, 0) + 1
                    current_images.append((row[0], chars))
            
            # Find current violators
            violators = [char for char, count in current_counts.items() if count > limit]
            if not violators:
                break
            
            # Pick a random violator to prune
            target_char = random.choice(violators)
            
            # Find all images containing this violator
            candidates = [img for img in current_images if target_char in img[1]]
            
            # Randomly pick one image to delete
            to_delete_name = random.choice(candidates)[0]
            
            cursor.execute("DELETE FROM images WHERE name = ?", (to_delete_name,))
            pruned_count += 1

        conn.commit()
        print(f"Pruning complete. Removed {pruned_count} images to satisfy the limit of {limit}.")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

def main():
    parser = argparse.ArgumentParser(description="Apply a hard cap to the number of images per character.")
    parser.add_argument("db", help="Path to the .sql database file")
    parser.add_argument("--limit", type=int, default=100, help="Max images per character (default: 100)")
    args = parser.parse_args()

    hard_cap_database(args.db, args.limit)

if __name__ == "__main__":
    main()
