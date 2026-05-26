#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <limits.h>
#include <mpi.h>

#include "lenia.h"

#define DEFAULT_N 128
#define DEFAULT_NUM_STEPS 100
#define DT 0.1
#define KERNEL_SIZE 26
#define NUM_ORBIUMS 2

static int parse_uint_arg(const char *arg, unsigned int *value)
{
    char *endptr = NULL;
    errno = 0;

    unsigned long parsed = strtoul(arg, &endptr, 10);

    if (errno != 0 || endptr == arg || *endptr != '\0' || parsed == 0 || parsed > UINT_MAX) {
        return 0;
    }

    *value = (unsigned int)parsed;
    return 1;
}

int main(int argc, char *argv[])
{
    int myid, procs;
    char node_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    MPI_Get_processor_name(node_name, &name_len);

    unsigned int n = DEFAULT_N;
    unsigned int num_steps = DEFAULT_NUM_STEPS;

    if (argc >= 2) {
        if (!parse_uint_arg(argv[1], &n)) {
            if (myid == 0) {
                fprintf(stderr, "Napaka: neveljavna velikost mreze: %s\n", argv[1]);
                fprintf(stderr, "Uporaba: %s [grid_size] [num_steps]\n", argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (argc >= 3) {
        if (!parse_uint_arg(argv[2], &num_steps)) {
            if (myid == 0) {
                fprintf(stderr, "Napaka: neveljavno stevilo korakov: %s\n", argv[2]);
                fprintf(stderr, "Uporaba: %s [grid_size] [num_steps]\n", argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (argc > 3) {
        if (myid == 0) {
            fprintf(stderr, "Napaka: prevec argumentov.\n");
            fprintf(stderr, "Uporaba: %s [grid_size] [num_steps]\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    struct orbium_coo orbiums[NUM_ORBIUMS] = {
        {0, n / 3, 0},
        {n / 3, 0, 180}
    };

    printf("Hello from process %d of %d in node %s\n", myid, procs, node_name);

    if (myid == 0) {
        printf("Grid size: %ux%u\n", n, n);
        printf("Steps: %u\n", num_steps);
        printf("MPI processes: %d\n", procs);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    double *world = evolve_lenia(
        n,
        n,
        num_steps,
        DT,
        KERNEL_SIZE,
        orbiums,
        NUM_ORBIUMS
    );

    MPI_Barrier(MPI_COMM_WORLD);
    double stop = MPI_Wtime();

    if (myid == 0) {
        printf("Execution time: %.6f\n", stop - start);
    }

    free(world);

    MPI_Finalize();
    return 0;
}