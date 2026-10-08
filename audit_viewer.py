import argparse
import sqlite3
import os
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def main():
    ap = argparse.ArgumentParser(description='Create a contact sheet for auditing character grounding.')
    ap.add_argument('--title', required=True, help='Title of the show (used for DB name)')
    ap.add_argument('--char', required=True, help='Character name to audit')
    ap.add_argument('--output', default='output')
    ap.add_argument('--thumb-size', type=int, default=320, help='Width/Height of each cell')
    args = ap.parse_args()

    base = Path('/home/alex/Documents/ccip_grounding')
    db_path = base / f'{args.title}.sql'
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        print(f'Error: Database {db_path} not found.')
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Find images tagged with this character
    cursor.execute('SELECT path FROM images WHERE characters LIKE ?', (f'%{args.char}%',))
    rows = cursor.fetchall()
    conn.close()

    paths = [r[0] for r in rows]
    num_imgs = len(paths)
    if num_imgs == 0:
        print(f'No images found for character: {args.char}')
        return

    print(f'Creating contact sheet for {args.char} ({num_imgs} images)...')

    # Calculate grid size
    cols = math.ceil(math.sqrt(num_imgs))
    rows = math.ceil(num_imgs / cols)
    
    # Add extra height for the filename text underneath each image
    text_height = 30
    cell_w = args.thumb_size
    cell_h = args.thumb_size + text_height
    
    canvas_w = cols * cell_w
    canvas_h = rows * cell_h
    canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # Try to load a default font
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
                img.thumbnail((args.thumb_size, args.thumb_size))
                
                # Positioning
                col = i % cols
                row = i // cols
                x_offset = col * cell_w
                y_offset = row * cell_h
                
                # Center the thumbnail in the top part of the cell
                off_x = (cell_w - img.width) // 2
                off_y = (args.thumb_size - img.height) // 2
                canvas.paste(img, (x_offset + off_x, y_offset + off_y))
                
                # Draw filename underneath
                filename = os.path.basename(path)
                # Truncate filename if it's too long for the cell
                if len(filename) > 25:
                    filename = filename[:22] + '...'
                
                text_x = x_offset + (cell_w - draw.textlength(filename, font=font)) // 2 if font else x_offset + 5
                text_y = y_offset + args.thumb_size + 5
                draw.text((text_x, text_y), filename, fill=(0, 0, 0), font=font)
                
        except Exception as e:
            print(f'Skipping {path} due to error: {e}')

    out_file = output_dir / f'audit_{args.char}.jpg'
    canvas.save(out_file, 'JPEG', quality=85)
    print(f'Audit sheet saved to: {out_file}')

if __name__ == '__main__':
    main()
