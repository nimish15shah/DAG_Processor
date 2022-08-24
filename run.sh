#!/bin/bash

set -e
depth=3
banks=64
regs=32
novcs=0
if [ "$1" = "noconda" ]; then
	echo "Not using Conda but a Python virtual environment"
else
	command -v conda >/dev/null 2>&1 || { echo >&2 "ERROR: conda is not installed. Please install it before running this script. Aborting."; exit 1; }
fi
command -v vcs >/dev/null 2>&1 || { echo >&2 "WARNING: Synopsys VCS is not installed. RTL simulation will not be run."; novcs=1;}

if [ "$1" = "noconda" ]; then
	source ./venv_DAGprocessor/bin/activate
	python --version
else
	# conda activate DAGprocessor
	echo ""
fi
echo "============================================="
echo "  Generating instructions for input DAGs"
echo "  This can take 3-4 hours"
echo "  Log is captured in ./run.log"
echo "============================================="
python -O main.py --tmode compile --depth ${depth} --banks ${banks} --regs ${regs} &>> ./run.log
echo "Done generating instructions"
if [ ${noconda} = 1 ]; then
	echo "Launching RTL simulations"
	cd ./hw/tb
	min_depth= ${depth} > 1 ? 2 : 1
	n_tree= ${banks} / (2**$depth)
	make compile TREE_DEPTH=$depth MIN_DEPTH=${min_depth} N_TREE=${n_tree} REG_BANK_DEPTH=${regs} >> ../../run.log
	make run NET=tretail          W_CONFLICT=150  >> ../../run.log
	make run NET=mnist            W_CONFLICT=50   >> ../../run.log
	make run NET=nltcs            W_CONFLICT=225  >> ../../run.log
	make run NET=msweb            W_CONFLICT=250  >> ../../run.log
	make run NET=msnbc            W_CONFLICT=125  >> ../../run.log
	make run NET=bnetflix         W_CONFLICT=275  >> ../../run.log
	make run NET=HB_bp_200        W_CONFLICT=225  >> ../../run.log
	make run NET=HB_west2021      W_CONFLICT=225  >> ../../run.log
	make run NET=MathWorks_Sieber W_CONFLICT=300  >> ../../run.log
	make run NET=HB_jagmesh4      W_CONFLICT=200  >> ../../run.log
	make run NET=Bai_rdb968      W_CONFLICT=175   >> ../../run.log
	make run NET=Bai_dw2048      W_CONFLICT=150   >> ../../run.log
	cd ../..
	echo "Done RTL simulations"
	echo "Plotting charts.."
	python main.py --tmode plot_charts >> ./run.log
	echo "Charts available at ./out/plots/"
else
	echo "Skipping RTL simulations as Synopsys VCS is not available."
fi
echo "Finished!"


