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

// Function to calculate Gaussian
static inline double gauss(double x, double mu, double sigma)
{
    return exp(-0.5 * pow((x - mu) / sigma, 2));
}

// Function for growth criteria
double growth_lenia(double u)
{
    double mu = 0.15;
    double sigma = 0.015;

    return -1 + 2 * gauss(u, mu, sigma);
}

// Clip value to [0, 1]
static inline double clip01(double x)
{
    if (x < 0.0) return 0.0;
    if (x > 1.0) return 1.0;
    return x;
}

// Wrap index for periodic boundary conditions
static inline unsigned int wrap_index(int idx, unsigned int size)
{
    int result = idx % (int)size;

    if (result < 0) {
        result += size;
    }

    return (unsigned int)result;
}

// Function to generate convolution kernel
double *generate_kernel(double *K, const unsigned int size)
{
    double mu = 0.5;
    double sigma = 0.15;
    int r = size / 2;
    double sum = 0.0;

    if (K != NULL) {
        for (int y = -r; y < r; y++) {
            for (int x = -r; x < r; x++) {
                double distance = sqrt((1 + x) * (1 + x) + (1 + y) * (1 + y)) / r;

                K[(y + r) * size + x + r] = gauss(distance, mu, sigma);

                if (distance > 1.0) {
                    K[(y + r) * size + x + r] = 0.0;
                }

                sum += K[(y + r) * size + x + r];
            }
        }

        // Normalize
        for (unsigned int y = 0; y < size; y++) {
            for (unsigned int x = 0; x < size; x++) {
                K[y * size + x] /= sum;
            }
        }
    }

    return K;
}

// Izracun zacetne vrstice za posamezen rank
static unsigned int get_start_row(int rank, int procs, unsigned int rows)
{
    unsigned int base = rows / procs;
    unsigned int rem = rows % procs;

    return rank * base + (rank < (int)rem ? rank : rem);
}

// Izracun stevila vrstic za posamezen rank
static unsigned int get_local_rows(int rank, int procs, unsigned int rows)
{
    unsigned int base = rows / procs;
    unsigned int rem = rows % procs;

    return base + (rank < (int)rem ? 1 : 0);
}

// Local convolution: proces izracuna samo svoje vrstice [start_row, start_row + local_rows)
static void convolve2d_local(
    double *local_result,
    const double *world,
    const double *kernel,
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int kernel_size,
    const unsigned int start_row,
    const unsigned int local_rows
) {
    int radius = kernel_size / 2;

    for (unsigned int local_i = 0; local_i < local_rows; local_i++) {
        unsigned int global_i = start_row + local_i;

        for (unsigned int j = 0; j < cols; j++) {
            double sum = 0.0;

            for (unsigned int ki = 0; ki < kernel_size; ki++) {
                unsigned int ii = wrap_index((int)global_i - radius + (int)ki, rows);

                for (unsigned int kj = 0; kj < kernel_size; kj++) {
                    unsigned int jj = wrap_index((int)j - radius + (int)kj, cols);

                    sum += kernel[ki * kernel_size + kj] * world[ii * cols + jj];
                }
            }

            local_result[local_i * cols + j] = sum;
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

// Function to evolve Lenia
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

    unsigned int start_row = get_start_row(myid, procs, rows);
    unsigned int local_rows = get_local_rows(myid, procs, rows);
    unsigned int local_count = local_rows * cols;

    // Allocate memory
    double *kernel = (double *)calloc(kernel_size * kernel_size, sizeof(double));
    double *world = (double *)calloc(rows * cols, sizeof(double));

    // Lokalna polja za trenutni proces
    double *tmp_local = (double *)calloc(local_count > 0 ? local_count : 1, sizeof(double));
    double *next_local = (double *)calloc(local_count > 0 ? local_count : 1, sizeof(double));

    int *recv_counts = (int *)malloc(procs * sizeof(int));
    int *displs = (int *)malloc(procs * sizeof(int));

    if (
        kernel == NULL ||
        world == NULL ||
        tmp_local == NULL ||
        next_local == NULL ||
        recv_counts == NULL ||
        displs == NULL
    ) {
        fprintf(stderr, "Process %d: Napaka pri alokaciji pomnilnika.\n", myid);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Prepare counts and displacements for MPI_Allgatherv
    for (int p = 0; p < procs; p++) {
        unsigned int p_start = get_start_row(p, procs, rows);
        unsigned int p_rows = get_local_rows(p, procs, rows);

        recv_counts[p] = (int)(p_rows * cols);
        displs[p] = (int)(p_start * cols);
    }

    // Generate convolution kernel
    kernel = generate_kernel(kernel, kernel_size);

    // Place orbiums.
    // Za osnovno verzijo jih postavimo na vseh procesih enako,
    // zato ne potrebujemo zacetnega Scatter.
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

    // Lenia simulation
    for (unsigned int step = 0; step < steps; step++) {

        // 1. Vsak proces izracuna konvolucijo samo za svoje vrstice
        convolve2d_local(
            tmp_local,
            world,
            kernel,
            rows,
            cols,
            kernel_size,
            start_row,
            local_rows
        );

        // 2. Vsak proces posodobi samo svoje vrstice
        for (unsigned int local_i = 0; local_i < local_rows; local_i++) {
            unsigned int global_i = start_row + local_i;

            for (unsigned int j = 0; j < cols; j++) {
                unsigned int local_idx = local_i * cols + j;
                unsigned int global_idx = global_i * cols + j;

                next_local[local_idx] =
                    clip01(world[global_idx] + dt * growth_lenia(tmp_local[local_idx]));
            }
        }

        // 3. Vsi procesi si izmenjajo izracunane vrstice.
        // Po tem ima vsak proces celoten world za naslednji korak.
        MPI_Allgatherv(
            next_local,
            (int)local_count,
            MPI_DOUBLE,
            world,
            recv_counts,
            displs,
            MPI_DOUBLE,
            MPI_COMM_WORLD
        );

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
    free(tmp_local);
    free(next_local);
    free(recv_counts);
    free(displs);

    return world;
}