#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.3/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
~/Documents/ccip_grounding/venv/bin/python3 ~/Documents/ccip_grounding/ccip_grounder.py "$@"
