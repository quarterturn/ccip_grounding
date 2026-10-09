#!/usr/bin/env python3
import os
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# configurations
BASE_DIR = Path('/home/alex/Documents/ccip_grounding')
REF_DIR = BASE_DIR / 'reference'
INPUT_DIR = BASE_DIR / 'input' 
VENV_PYTHON = BASE_DIR / 'venv' / 'bin' / 'python3'
SCRIPT_PATH = BASE_DIR / 'ccip_grounder.py'
TITLE = 'mushoku_tensei_3'

def run_grounding(char, threshold, gpu_id):
    """Worker function to run grounding for a single character on a specific GPU."""
    separator = '=' * 60
    print(f'\n{separator}\n>>> [GPU {gpu_id}] Grounding character: {char}\n{separator}')
    
    # Construct command
    cmd = [
        str(VENV_PYTHON),
        str(SCRIPT_PATH),
        '--title', TITLE,
        '--reference-dir', f'reference/{char}',
        '--input', str(INPUT_DIR),
        '--ccip-region', 'head',
    ]
    
    if threshold is not None:
        cmd.extend(['--threshold', str(threshold)])
    
    # Use CUDA_VISIBLE_DEVICES to pin to a specific GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    try:
        subprocess.run(cmd, check=True, env=env)
        return f'SUCCESS: {char} processed on GPU {gpu_id}'
    except subprocess.CalledProcessError as e:
        return f'ERROR: {char} failed on GPU {gpu_id}: {e}'

def main():
    parser = argparse.ArgumentParser(description='Orchestrate CCIP grounding for all characters using Multi-GPU.')
    parser.add_argument('--threshold', type=float, default=None, help='CCIP diff threshold')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='List of GPU IDs to use')
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
    
    num_gpus = len(args.gpus)
    print(f'Using {num_gpus} GPUs: {args.gpus}')

    # Prepare jobs: map each character to a GPU ID in round-robin fashion
    jobs = []
    for i, char in enumerate(chars):
        gpu_id = args.gpus[i % num_gpus]
        jobs.append((char, args.threshold, gpu_id))

    # Process in parallel
    # Max workers = number of GPUs since each process uses one’s full VRAM
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        # Use map to trigger them; results are returned as they finish if we use as_completed, 
        # but executor.map is simpler for a fixed list.
        results = list(executor.map(lambda p: run_grounding(*p), jobs))

    for res in results:
        print(res)

    print('\n' + '=' * 60)
    print('ALL CHARACTERS PROCESSED!')
    print(f'The identity index has been updated in {TITLE}.sql')
    print('=' * 60)

if __name__ == '__main__':
    main()
