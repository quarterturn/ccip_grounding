#!/usr/bin/env python3
"""
CCIP Hard Cap Pruner (v2 schema).
Ensures a strict maximum number of images per character in the database.
If a character exceeds the limit, images are randomly pruned until the limit
is met. Note: pruning an image containing multiple characters will reduce the
count for all those characters.
"""
import argparse
import collections
import os
import random
from dbutil import open_v2


def char_counts(conn):
    return dict(conn.execute(
        "SELECT char, COUNT(*) FROM image_tags GROUP BY char"))


def hard_cap_database(db_path, limit, seed=None):
    if not os.path.exists(db_path):
        print(f"Error: Database file {db_path} not found.")
        return

    rng = random.Random(seed)
    conn = open_v2(db_path)
    try:
        counts = char_counts(conn)
        over_limit = sorted(c for c, n in counts.items() if n > limit)

        if not over_limit:
            print("No characters exceed the limit. No pruning necessary.")
            return

        print(f"Characters over limit ({limit}):")
        for char in over_limit:
            print(f"  - {char}: {counts[char]}")

        pruned = 0
        while over_limit:
            target_char = rng.choice(over_limit)
            placeholders = ",".join("?" * len(over_limit))
            # Prefer images tagged ONLY with violating characters, so pruning
            # one character doesn't starve healthy ones below the cap.
            candidates = conn.execute(
                f"""
                SELECT t.path FROM image_tags t
                WHERE t.char = ?
                  AND NOT EXISTS (
                        SELECT 1 FROM image_tags o
                        WHERE o.path = t.path AND o.char != ?
                          AND o.char IN ({placeholders})
                  )
                """,
                (target_char, target_char, *over_limit)).fetchall()
            if not candidates:
                candidates = conn.execute(
                    "SELECT path FROM image_tags WHERE char = ?",
                    (target_char,)).fetchall()
            path = rng.choice(candidates)[0]
            with conn:
                conn.execute("DELETE FROM image_tags WHERE path = ?", (path,))
                conn.execute("DELETE FROM images WHERE path = ?", (path,))
            pruned += 1

            counts = char_counts(conn)
            over_limit = sorted(c for c, n in counts.items() if n > limit)

        print(f"Pruning complete. Removed {pruned} images to satisfy the limit of {limit}.")
        final = char_counts(conn)
        worst = max(final.values()) if final else 0
        print(f"Final max count per character: {worst} (limit {limit})")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Apply a hard cap to the number of images per character.")
    parser.add_argument("db", help="Path to the .sql database file")
    parser.add_argument("--limit", type=int, default=100, help="Max images per character (default: 100)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible pruning")
    args = parser.parse_args()

    hard_cap_database(args.db, args.limit, args.seed)


if __name__ == "__main__":
    main()
