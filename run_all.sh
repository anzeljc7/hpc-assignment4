#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=lenia_mpi_all
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --output=slurm_temp_%j.log
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00

# LOAD MODULES
module load OpenMPI

# BUILD
make clean
make

# ==========================================
# NASTAVITVE MERITEV
# ==========================================

PROGRAM_NAME="2_new_mpi"

RUNS=1
STEPS=100

GRID_SIZES=(2048 4096)
PROCESS_COUNTS=(16 32)

MAX_PROCS=$SLURM_NTASKS

RESULTS_DIR="results/${PROGRAM_NAME}"
mkdir -p "$RESULTS_DIR"

ALL_RUNS_FILE="${RESULTS_DIR}/${PROGRAM_NAME}_all_runs.csv"
SUMMARY_FILE="${RESULTS_DIR}/${PROGRAM_NAME}_summary.csv"

echo "N;Procesi;Koraki;Zagon;Cas_s" > "$ALL_RUNS_FILE"
echo "N;Procesi;Koraki;Runs;Avg_s;Min_s;Max_s" > "$SUMMARY_FILE"

echo "=========================================="
echo "Lenia MPI meritve"
echo "=========================================="
echo "Program:              $PROGRAM_NAME"
echo "Grid sizes:           ${GRID_SIZES[*]}"
echo "Process counts:       ${PROCESS_COUNTS[*]}"
echo "Steps:                $STEPS"
echo "Runs per config:      $RUNS"
echo "Max allocated procs:  $MAX_PROCS"
echo "Results dir:          $RESULTS_DIR"
echo "=========================================="
echo ""

if [ ! -f "./lenia.out" ]; then
    echo "NAPAKA: ./lenia.out ne obstaja."
    exit 1
fi

for N in "${GRID_SIZES[@]}"; do
    for PROCS in "${PROCESS_COUNTS[@]}"; do

        if [ "$PROCS" -gt "$MAX_PROCS" ]; then
            echo "Preskakujem N=$N P=$PROCS, ker je dodeljenih samo $MAX_PROCS procesov."
            continue
        fi

        echo "=========================================="
        echo "Meritev: N=${N}x${N}, P=${PROCS}, steps=${STEPS}"
        echo "=========================================="

        CSV_FILE="${RESULTS_DIR}/${PROGRAM_NAME}_N${N}_P${PROCS}.csv"
        echo "N;Procesi;Koraki;Zagon;Cas_s" > "$CSV_FILE"

        for ((i=1; i<=RUNS; i++)); do

            echo " Zagon $i/$RUNS"

            program_output=$(mpirun --mca pml ob1 -np "$PROCS" ./lenia.out "$N" "$STEPS" 2>&1)

            time_s=$(echo "$program_output" | awk '/^Execution time:/ {print $3}')

            if [ -z "$time_s" ]; then
                echo " Napaka: časa nisem našel v izpisu za N=$N, P=$PROCS."
                echo "$program_output" | sed 's/^/ /'
                exit 1
            fi

            printf " --> Čas: %9s s\n" "$time_s"

            echo "$N;$PROCS;$STEPS;$i;$time_s" >> "$CSV_FILE"
            echo "$N;$PROCS;$STEPS;$i;$time_s" >> "$ALL_RUNS_FILE"

        done

        avg=$(awk -F';' 'NR>1 {sum += $5; count++} END {if (count > 0) printf "%.6f", sum / count}' "$CSV_FILE")
        min=$(awk -F';' 'NR==2 {min=$5} NR>1 && $5 < min {min=$5} END {printf "%.6f", min}' "$CSV_FILE")
        max=$(awk -F';' 'NR==2 {max=$5} NR>1 && $5 > max {max=$5} END {printf "%.6f", max}' "$CSV_FILE")

        echo "$N;$PROCS;$STEPS;$RUNS;$avg;$min;$max" >> "$SUMMARY_FILE"

        echo " -------------------------------------------------"
        printf " Povprečje za N=%-5s P=%-3s = %9s s\n" "$N" "$PROCS" "$avg"
        printf " Minimum   za N=%-5s P=%-3s = %9s s\n" "$N" "$PROCS" "$min"
        printf " Maksimum  za N=%-5s P=%-3s = %9s s\n" "$N" "$PROCS" "$max"
        echo " -------------------------------------------------"
        echo ""

    done
done

echo "=========================================="
echo "Vse meritve so končane."
echo "Vsi zagoni:  $ALL_RUNS_FILE"
echo "Povzetek:    $SUMMARY_FILE"
echo "=========================================="

# ==========================================
# PREIMENOVANJE SLURM LOG DATOTEKE
# ==========================================

FINAL_SLURM_LOG="${RESULTS_DIR}/${PROGRAM_NAME}_all_slurm_izpis_${SLURM_JOB_ID}.log"
mv "slurm_temp_${SLURM_JOB_ID}.log" "$FINAL_SLURM_LOG"