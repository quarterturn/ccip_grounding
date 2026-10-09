#!/usr/bin/env python3
"""
CCIP Audit Sheet Generator.
Automatically creates contact sheets for all characters found in a grounding database.
"""
import argparse
import sqlite3
import os
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

def create_sheet(db_conn, char, output_dir, thumb_size=320):
    cursor = db_conn.cursor()
    
    # Find images tagged with this character
    cursor.execute('SELECT path FROM images WHERE characters LIKE ?', (f'%{char}%',))
    rows = cursor.fetchall()
    
    paths = [r[0] for r in rows]
    num_imgs = len(paths)
    if num_imgs == 0:
        return False

    # Calculate grid size
    cols = math.ceil(math.sqrt(num_imgs))
    rows_grid = math.ceil(num_imgs / cols)
    
    # Constant dimensions
    text_height = 30
    cell_w = thumb_size
    cell_h = thumb_size + text_height
    
    canvas_w = cols * cell_w
    canvas_h = rows_grid * cell_h
    canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, path in enumerate(paths):
        try:
            if not os.path.exists(path):
                continue
            with Image.open(path) as img:
                img = img.convert('RGB')
                img.thumbnail((thumb_size, thumb_size))
                
                col = i % cols
                row = i // cols
                x_offset = col * cell_w
                y_offset = row * cell_h
                
                off_x = (cell_w - img.width) // 2
                off_y = (thumb_size - img.height) // 2
                canvas.paste(img, (x_offset + off_x, y_offset + off_y))
                
                filename = os.path.basename(path)
                if len(filename) > 25:
                    filename = filename[:22] + '...'
                
                text_x = x_offset + (cell_w - draw.textlength(filename, font=font)) // 2 if font else x_offset + 5
                text_y = y_offset + thumb_size + 5
                draw.text((text_x, text_y), filename, fill=(0, 0, 0), font=font)
                
        except Exception:
            continue

    out_file = Path(output_dir) / f'audit_{char}.jpg'
    canvas.save(out_file, 'JPEG', quality=85)
    return True

def main():
    parser = argparse.ArgumentParser(description='Create contact sheets for all characters in a CCIP database.')
    parser.add_argument('--db', required=True, help='Path to the .sql database file')
    parser.add_argument('--output', default='output/audit_sheets', help='Directory to save contact sheets')
    parser.add_argument('--thumb-size', type=int, default=320, help='Width/Height of each cell')
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Error: Database {args.db} not found.")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(args.db)
    cursor = conn.cursor()
    
    # Get all unique characters from the comma-separated characters column
    cursor.execute("SELECT DISTINCT characters FROM images WHERE characters != ''")
    char_rows = cursor.fetchall()
    
    all_chars = set()
    for row in char_rows:
        for c in row[0].split(','):
            if c.strip():
                all_chars.add(c.strip())
    
    chars = sorted(list(all_chars))
    print(f"Found {len(chars)} characters. Generating sheets in {output_dir}...")

    for char in tqdm(chars, desc="Creating Sheets"):
        create_sheet(conn, char, output_dir, args.thumb_size)

    conn.close()
    print("Done!")

if __name__ == '__main__':
    main()
