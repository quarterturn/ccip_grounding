import os
import argparse
import subprocess
from pathlib import Path

# configurations
BASE_DIR = Path('/home/alex/Documents/ccip_grounding')
REF_DIR = BASE_DIR / 'reference'
INPUT_DIR = BASE_DIR / 'input' 
VENV_PYTHON = BASE_DIR / 'venv' / 'bin' / 'python3'
SCRIPT_PATH = BASE_DIR / 'ccip_grounder.py'
TITLE = 'mushoku_tensei_3'

def main():
    parser = argparse.ArgumentParser(description='Orchestrate CCIP grounding for all characters.')
    parser.add_argument('--threshold', type=float, default=None, help='CCIP diff threshold (overrides default if provided)')
    args = parser.parse_args()

    if not REF_DIR.exists():
        print(f'Error: Reference directory {REF_DIR} not found.')
        return

    # Get alphabetical list of character directories
    chars = [d.name for d in REF_DIR.iterdir() if d.is_dir()]
    chars.sort()
    print(f'Found {len(chars)} characters to ground: {chars}')
    if args.threshold:
        print(f'Using a tightened threshold of: {args.threshold}')

    for char in chars:
        separator = '=' * 60
        print(f'\n{separator}\n>>> Grounding character: {char}\n{separator}')
        
        # Construct command
        cmd = [
            str(VENV_PYTHON),
            str(SCRIPT_PATH),
            '--title', TITLE,
            '--reference-dir', f'reference/{char}',
            '--input', str(INPUT_DIR),
            '--ccip-region', 'head',
        ]
        
        if args.threshold is not None:
            cmd.extend(['--threshold', str(args.threshold)])
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f'Error grounding {char}: {e}')
            continue

    print('\n' + '=' * 60)
    print('SUCCESS: All characters have been processed!')
    print(f'The identity index has been updated in {TITLE}.sql')
    print('=' * 60)

if __name__ == '__main__':
    main()
