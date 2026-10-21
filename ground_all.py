#!/usr/bin/env python3
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

def run_grounding(char, title, threshold, gpu_id, extra_args, python_path=None):
    """Worker function to run grounding for a single character on a specific GPU."""
    separator = '=' * 60
    print(f'\n{separator}\n>>> [GPU {gpu_id}] Grounding character: {char}\n{separator}')

    # Construct command
    cmd = [
        str(python_path or VENV_PYTHON),
        str(SCRIPT_PATH),
        '--title', title,
        '--reference-dir', f'reference/{char}',
        '--input', str(INPUT_DIR),
        '--ccip-region', 'head',
    ]

    if threshold is not None:
        cmd.extend(['--threshold', str(threshold)])

    if gpu_id is not None:
        cmd.extend(['--gpu', str(gpu_id)])

    cmd.extend(extra_args)

    try:
        subprocess.run(cmd, check=True)
        return f'SUCCESS: {char} processed on GPU {gpu_id}'
    except subprocess.CalledProcessError as e:
        return f'ERROR: {char} failed on GPU {gpu_id}: {e}'

def main():
    parser = argparse.ArgumentParser(description='Orchestrate CCIP grounding for all characters using Multi-GPU.')
    parser.add_argument('--title', default='mushoku_tensei_3',
                        help='Show title (selects the (title).sql database; default: mushoku_tensei_3)')
    parser.add_argument('--threshold', type=float, default=None, help='CCIP diff threshold')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0],
                        help='List of GPU IDs to distribute characters over (passed as --gpu to the grounder)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the planned character/GPU assignments without running anything')
    parser.add_argument('--python', default=None,
                        help='Interpreter for ccip_grounder.py (default: venv/bin/python3; '
                             'chibi iGPU runs use .venv-gpu/bin/python; env CCIP_OV_GPU=1 '
                             'is inherited by children)')
    parser.add_argument('extra', nargs=argparse.REMAINDER,
                        help='Extra arguments forwarded verbatim to ccip_grounder.py (put after --)')
    args = parser.parse_args()

    extra_args = args.extra
    if extra_args and extra_args[0] == '--':
        extra_args = extra_args[1:]

    if not REF_DIR.exists():
        print(f'Error: Reference directory {REF_DIR} not found.')
        return

    # Get alphabetical list of character directories
    chars = [d.name for d in REF_DIR.iterdir() if d.is_dir()]
    chars.sort()
    print(f'Found {len(chars)} characters to ground: {chars}')
    if args.threshold is not None:
        print(f'Using threshold: {args.threshold}')

    num_gpus = len(args.gpus)
    print(f'Using {num_gpus} GPUs: {args.gpus}')

    # Prepare jobs: map each character to a GPU ID in round-robin fashion
    jobs = []
    for i, char in enumerate(chars):
        gpu_id = args.gpus[i % num_gpus]
        jobs.append((char, args.title, args.threshold, gpu_id, extra_args, args.python))

    if args.dry_run:
        for char, title, threshold, gpu_id, extra, _py in jobs:
            t = f' --threshold {threshold}' if threshold is not None else ''
            print(f'[dry-run] {char} -> GPU {gpu_id}: ccip_grounder.py --title {title}{t} {" ".join(extra)}')
        print(f'[dry-run] {len(jobs)} jobs planned, nothing executed.')
        return

    # Process in parallel
    # Max workers = number of GPUs since each process uses one's full VRAM
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        # Submit all jobs to the pool without lambdas to avoid pickling errors
        futures = [executor.submit(run_grounding, *job) for job in jobs]

        # Gather results as they complete
        results = [f.result() for f in futures]

    for res in results:
        print(res)

    failures = [r for r in results if r.startswith('ERROR')]
    print('\n' + '=' * 60)
    if failures:
        print(f'{len(failures)} of {len(results)} characters FAILED — see above.')
    else:
        print('ALL CHARACTERS PROCESSED!')
    print(f'The identity index has been updated in {args.title}.sql')
    print('=' * 60)

if __name__ == '__main__':
    main()
