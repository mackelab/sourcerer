#!/bin/sh

cd ..

echo $1
echo $2

# Run for powers of 1/sqrt(2)
# Turn of decay for lambda=0
python3 notebooks/benchmark_simulator_script.py base.tag=run_0 base.folder=$1 +experiment=$2 source.fin_lambda=0.0 source.linear_decay_steps=0
python3 notebooks/benchmark_simulator_script.py base.tag=run_1 base.folder=$1 +experiment=$2 source.fin_lambda=0.011
python3 notebooks/benchmark_simulator_script.py base.tag=run_2 base.folder=$1 +experiment=$2 source.fin_lambda=0.015
python3 notebooks/benchmark_simulator_script.py base.tag=run_3 base.folder=$1 +experiment=$2 source.fin_lambda=0.022
python3 notebooks/benchmark_simulator_script.py base.tag=run_4 base.folder=$1 +experiment=$2 source.fin_lambda=0.031
python3 notebooks/benchmark_simulator_script.py base.tag=run_5 base.folder=$1 +experiment=$2 source.fin_lambda=0.044
python3 notebooks/benchmark_simulator_script.py base.tag=run_6 base.folder=$1 +experiment=$2 source.fin_lambda=0.062
python3 notebooks/benchmark_simulator_script.py base.tag=run_7 base.folder=$1 +experiment=$2 source.fin_lambda=0.088
python3 notebooks/benchmark_simulator_script.py base.tag=run_8 base.folder=$1 +experiment=$2 source.fin_lambda=0.125
python3 notebooks/benchmark_simulator_script.py base.tag=run_9 base.folder=$1 +experiment=$2 source.fin_lambda=0.176
python3 notebooks/benchmark_simulator_script.py base.tag=run_10 base.folder=$1 +experiment=$2 source.fin_lambda=0.25
python3 notebooks/benchmark_simulator_script.py base.tag=run_11 base.folder=$1 +experiment=$2 source.fin_lambda=0.353
python3 notebooks/benchmark_simulator_script.py base.tag=run_12 base.folder=$1 +experiment=$2 source.fin_lambda=0.5
python3 notebooks/benchmark_simulator_script.py base.tag=run_13 base.folder=$1 +experiment=$2 source.fin_lambda=0.707
python3 notebooks/benchmark_simulator_script.py base.tag=run_14 base.folder=$1 +experiment=$2 source.fin_lambda=1.0