#!/bin/bash

command -v conda >/dev/null 2>&1 || { echo >&2 "ERROR: conda is not installed. Please install it before running this script. Aborting."; exit 1; }
conda create -y --name DAGprocessor python==3.7.7
conda install -f -y -q --name DAGprocessor -c conda-forge --file requirements.txt
