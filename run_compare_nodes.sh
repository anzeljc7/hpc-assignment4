#!/bin/bash

GRID_SIZES=(128 512 1024 2048 4096)
PROCS=(16 32)
RUNS=5
STEPS=100

for N in "${GRID_SIZES[@]}"; do
    for P in "${PROCS[@]}"; do

        echo "Submitting 1-node: N=$N P=$P"
        sbatch \
            --export=ALL,N="$N",STEPS="$STEPS",RUNS="$RUNS",PROGRAM_NAME="2_new_mpi_1node" \
            --nodes=1 \
            --ntasks-per-node="$P" \
            run_benchmark.sh

        if [ "$P" -gt 1 ] && [ $((P % 2)) -eq 0 ]; then
            TPN=$((P / 2))

            echo "Submitting 2-nodes: N=$N P=$P, tasks-per-node=$TPN"
            sbatch \
                --export=ALL,N="$N",STEPS="$STEPS",RUNS="$RUNS",PROGRAM_NAME="2_new_mpi_2nodes" \
                --nodes=2 \
                --ntasks-per-node="$TPN" \
                run_benchmark.sh
        fi

    done
done