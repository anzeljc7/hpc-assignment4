#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "lenia.h"
#include "orbium.h"
#include "gifenc.h"

// Uncomment to generate gif animation
// #define GENERATE_GIF

#define EMPTY_THRESHOLD 1e-6

static inline double gauss(double x, double mu, double sigma)
{
    double diff = (x - mu) / sigma;
    return exp(-0.5 * diff * diff);
}

double growth_lenia(double u)
{
    const double mu = 0.15;
    const double sigma = 0.015;

    return -1.0 + 2.0 * gauss(u, mu, sigma);
}

static inline double clip01(double x)
{
    if (x < 0.0) return 0.0;
    if (x > 1.0) return 1.0;
    return x;
}

// Ker so odmiki pri kernelu vedno samo v območju [-radius, size + radius],
// ne rabimo počasnega modulo operatorja.
static inline unsigned int wrap_index_fast(int idx, unsigned int size)
{
    if (idx < 0) {
        idx += (int)size;
    } else if (idx >= (int)size) {
        idx -= (int)size;
    }

    return (unsigned int)idx;
}

double *generate_kernel(double *K, const unsigned int size)
{
    const double mu = 0.5;
    const double sigma = 0.15;
    const int r = (int)size / 2;

    double sum = 0.0;

    for (int y = -r; y < r; y++) {
        for (int x = -r; x < r; x++) {
            const unsigned int idx = (unsigned int)((y + r) * (int)size + (x + r));

            const double dx = (double)(1 + x);
            const double dy = (double)(1 + y);
            const double distance = sqrt(dx * dx + dy * dy) / (double)r;

            if (distance > 1.0) {
                K[idx] = 0.0;
            } else {
                K[idx] = gauss(distance, mu, sigma);
            }

            sum += K[idx];
        }
    }

    for (unsigned int i = 0; i < size * size; i++) {
        K[i] /= sum;
    }

    return K;
}

static unsigned int get_start_row(int rank, int procs, unsigned int rows)
{
    unsigned int base = rows / (unsigned int)procs;
    unsigned int rem = rows % (unsigned int)procs;

    return (unsigned int)rank * base + ((unsigned int)rank < rem ? (unsigned int)rank : rem);
}

static unsigned int get_local_rows(int rank, int procs, unsigned int rows)
{
    unsigned int base = rows / (unsigned int)procs;
    unsigned int rem = rows % (unsigned int)procs;

    return base + ((unsigned int)rank < rem ? 1u : 0u);
}

// Združena konvolucija + posodobitev.
// Ne shranjujemo več tmp_local matrike.
static void compute_next_local(
    double *next_local,
    const double *world,
    const double *kernel,
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int kernel_size,
    const unsigned int start_row,
    const unsigned int local_rows,
    const double dt
) {
    const int radius = (int)kernel_size / 2;

    for (unsigned int local_i = 0; local_i < local_rows; local_i++) {
        const unsigned int global_i = start_row + local_i;

        for (unsigned int j = 0; j < cols; j++) {
            double sum = 0.0;

            for (unsigned int ki = 0; ki < kernel_size; ki++) {
                const unsigned int ii =
                    wrap_index_fast((int)global_i - radius + (int)ki, rows);

                const double *world_row = &world[ii * cols];
                const double *kernel_row = &kernel[ki * kernel_size];

                for (unsigned int kj = 0; kj < kernel_size; kj++) {
                    const unsigned int jj =
                        wrap_index_fast((int)j - radius + (int)kj, cols);

                    sum += kernel_row[kj] * world_row[jj];
                }
            }

            const unsigned int local_idx = local_i * cols + j;
            const unsigned int global_idx = global_i * cols + j;

            double new_value;

            // Če ni aktivne okolice, je growth praktično -1,
            // zato preskočimo drag exp().
            if (sum < EMPTY_THRESHOLD) {
                new_value = world[global_idx] - dt;
            } else {
                new_value = world[global_idx] + dt * growth_lenia(sum);
            }

            next_local[local_idx] = clip01(new_value);
        }
    }
}

#ifdef GENERATE_GIF
static void add_gif_frame(ge_GIF *gif, const double *world, unsigned int rows, unsigned int cols)
{
    for (unsigned int i = 0; i < rows; i++) {
        for (unsigned int j = 0; j < cols; j++) {
            gif->frame[i * cols + j] = (unsigned char)(world[i * cols + j] * 255.0);
        }
    }

    ge_add_frame(gif, 5);
}
#endif

double *evolve_lenia(
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int steps,
    const double dt,
    const unsigned int kernel_size,
    const struct orbium_coo *orbiums,
    const unsigned int num_orbiums
) {
    int myid, procs;

    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    const unsigned int start_row = get_start_row(myid, procs, rows);
    const unsigned int local_rows = get_local_rows(myid, procs, rows);
    const unsigned int local_count = local_rows * cols;

    double *kernel = (double *)calloc(kernel_size * kernel_size, sizeof(double));
    double *world = (double *)calloc(rows * cols, sizeof(double));
    double *next_world = (double *)calloc(rows * cols, sizeof(double));
    double *next_local = (double *)calloc(local_count > 0 ? local_count : 1, sizeof(double));

    int *recv_counts = (int *)malloc((size_t)procs * sizeof(int));
    int *displs = (int *)malloc((size_t)procs * sizeof(int));

    if (
        kernel == NULL ||
        world == NULL ||
        next_world == NULL ||
        next_local == NULL ||
        recv_counts == NULL ||
        displs == NULL
    ) {
        fprintf(stderr, "Process %d: Napaka pri alokaciji pomnilnika.\n", myid);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int p = 0; p < procs; p++) {
        const unsigned int p_start = get_start_row(p, procs, rows);
        const unsigned int p_rows = get_local_rows(p, procs, rows);

        recv_counts[p] = (int)(p_rows * cols);
        displs[p] = (int)(p_start * cols);
    }

    generate_kernel(kernel, kernel_size);

    // Za basic MPI verzijo vsak proces pripravi enako začetno stanje.
    // Zato ne potrebujemo scatterja.
    for (unsigned int o = 0; o < num_orbiums; o++) {
        world = place_orbium(
            world,
            rows,
            cols,
            orbiums[o].row,
            orbiums[o].col,
            orbiums[o].angle
        );
    }

#ifdef GENERATE_GIF
    ge_GIF *gif = NULL;

    if (myid == 0) {
        gif = ge_new_gif(
            "lenia.gif",
            cols,
            rows,
            inferno_pallete,
            8,
            -1,
            0
        );

        add_gif_frame(gif, world, rows, cols);
    }
#endif

    for (unsigned int step = 0; step < steps; step++) {
        compute_next_local(
            next_local,
            world,
            kernel,
            rows,
            cols,
            kernel_size,
            start_row,
            local_rows,
            dt
        );

        MPI_Allgatherv(
            next_local,
            (int)local_count,
            MPI_DOUBLE,
            next_world,
            recv_counts,
            displs,
            MPI_DOUBLE,
            MPI_COMM_WORLD
        );

        // O(1) zamenjava kazalcev namesto kopiranja cele matrike.
        double *tmp = world;
        world = next_world;
        next_world = tmp;

#ifdef GENERATE_GIF
        if (myid == 0) {
            add_gif_frame(gif, world, rows, cols);
        }
#endif
    }

#ifdef GENERATE_GIF
    if (myid == 0 && gif != NULL) {
        ge_close_gif(gif);
    }
#endif

    free(kernel);
    free(next_world);
    free(next_local);
    free(recv_counts);
    free(displs);

    return world;
}