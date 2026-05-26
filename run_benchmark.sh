#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=lenia_mpi
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --output=slurm_temp_%j.log
#SBATCH --hint=nomultithread
#SBATCH --time=00:15:00

# LOAD MODULES
module load OpenMPI

# BUILD
make clean
make

# ==========================================
# NASTAVITVE MERITEV
# ==========================================

PROGRAM_NAME="1_row_wise" # Ime za datoteke in podmapo

RUNS=5                  # Število ponovitev za povprečje
N=128                   # TUKAJ ROČNO SPREMENIŠ VELIKOST (128, 512, 1024, 2048, 4096)
STEPS=100               # Število korakov simulacije

# Število MPI procesov pride iz Slurma.
# Spremeniš ga zgoraj pri #SBATCH --ntasks-per-node=...
PROCS=$SLURM_NTASKS

# Ustvarimo specifično podmapo znotraj results
RESULTS_DIR="results/${PROGRAM_NAME}"
mkdir -p "$RESULTS_DIR"

echo "Začenjam MPI meritve z $PROCS procesi..."
echo "Testiram mrežo: ${N}x${N}"
echo "Število korakov: ${STEPS}"

# Dinamično poimenovanje datotek znotraj podmape
CSV_FILE="${RESULTS_DIR}/${PROGRAM_NAME}_N${N}_P${PROCS}.csv"

# Priprava glave v CSV datoteki
echo "N;Procesi;Koraki;Zagon;Cas_s" > "$CSV_FILE"

# Zanka za ponovitve zagona
for ((i=1; i<=RUNS; i++)); do

    echo " Zagon $i/$RUNS"

    # Zaženemo MPI program in mu podamo N in STEPS.
    program_output=$(mpirun --mca pml ob1 -np "$PROCS" ./lenia.out "$N" "$STEPS" 2>&1)

    # Iz izpisa potegnemo samo številko pri "Execution time:"
    time_s=$(echo "$program_output" | awk '/^Execution time:/ {print $3}')

    if [ -z "$time_s" ]; then
        echo " Napaka: časa nisem našel v izpisu za N=$N, P=$PROCS."
        echo "$program_output" | sed 's/^/ /'
        exit 1
    fi

    printf " --> Čas: %9s s\n" "$time_s"

    # Zapis v CSV datoteko
    echo "$N;$PROCS;$STEPS;$i;$time_s" >> "$CSV_FILE"

done

# Izračun povprečja za trenutni N in P
avg=$(awk -F';' 'NR>1 {sum += $5; count++} END {if (count > 0) printf "%.6f", sum / count}' "$CSV_FILE")

echo " -------------------------------------------------"
printf " Povprečje za N=%-5s P=%-3s = %9s s\n" "$N" "$PROCS" "$avg"
echo " -------------------------------------------------"
echo ""

echo "Meritve za N=$N in P=$PROCS so končane! Rezultati so v mapi '$RESULTS_DIR/'."

# ==========================================
# PREIMENOVANJE SLURM LOG DATOTEKE
# ==========================================

FINAL_SLURM_LOG="${RESULTS_DIR}/${PROGRAM_NAME}_N${N}_P${PROCS}_slurm_izpis.log"
mv "slurm_temp_${SLURM_JOB_ID}.log" "$FINAL_SLURM_LOG"