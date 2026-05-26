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

#define EMPTY_THRESHOLD 1e-12

typedef struct {
    int dy;
    int dx;
    double weight;
} kernel_entry_t;

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

static inline unsigned int wrap_col_fast(int idx, unsigned int cols)
{
    if (idx < 0) {
        idx += (int)cols;
    } else if (idx >= (int)cols) {
        idx -= (int)cols;
    }

    return (unsigned int)idx;
}

static inline unsigned int wrap_row_general(int idx, unsigned int rows)
{
    while (idx < 0) {
        idx += (int)rows;
    }

    while (idx >= (int)rows) {
        idx -= (int)rows;
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

static unsigned int build_kernel_entries(
    kernel_entry_t *entries,
    const double *kernel,
    const unsigned int kernel_size
) {
    const int radius = (int)kernel_size / 2;
    unsigned int count = 0;

    /*
     * Originalna konvolucija je kernel brala obrnjeno:
     * ki = kernel_size - 1 ... 0
     * kj = kernel_size - 1 ... 0
     *
     * Zato tukaj obdržimo isti vrstni red uteži.
     */
    for (unsigned int kri = 0; kri < kernel_size; kri++) {
        for (unsigned int kcj = 0; kcj < kernel_size; kcj++) {
            const unsigned int ki = kernel_size - 1 - kri;
            const unsigned int kj = kernel_size - 1 - kcj;

            const double weight = kernel[ki * kernel_size + kj];

            if (weight != 0.0) {
                entries[count].dy = (int)kri - radius;
                entries[count].dx = (int)kcj - radius;
                entries[count].weight = weight;
                count++;
            }
        }
    }

    return count;
}

static unsigned int get_start_row(int rank, int procs, unsigned int rows)
{
    const unsigned int base = rows / (unsigned int)procs;
    const unsigned int rem = rows % (unsigned int)procs;

    return (unsigned int)rank * base + ((unsigned int)rank < rem ? (unsigned int)rank : rem);
}

static unsigned int get_local_rows(int rank, int procs, unsigned int rows)
{
    const unsigned int base = rows / (unsigned int)procs;
    const unsigned int rem = rows % (unsigned int)procs;

    return base + ((unsigned int)rank < rem ? 1u : 0u);
}

static void prepare_counts_displs(
    int *counts,
    int *displs,
    int procs,
    unsigned int rows,
    unsigned int cols
) {
    for (int p = 0; p < procs; p++) {
        const unsigned int p_start = get_start_row(p, procs, rows);
        const unsigned int p_rows = get_local_rows(p, procs, rows);

        counts[p] = (int)(p_rows * cols);
        displs[p] = (int)(p_start * cols);
    }
}

static void exchange_halos_direct(
    double *grid,
    unsigned int local_rows,
    unsigned int cols,
    unsigned int halo,
    int myid,
    int procs
) {
    if (halo == 0 || local_rows == 0) {
        return;
    }

    const int up = (myid - 1 + procs) % procs;
    const int down = (myid + 1) % procs;
    const int halo_count = (int)(halo * cols);

    if (procs == 1) {
        /*
         * Periodicni robovi znotraj enega procesa.
         * Top halo dobi zadnjih halo vrstic.
         * Bottom halo dobi prvih halo vrstic.
         */
        memcpy(
            &grid[0],
            &grid[local_rows * cols],
            (size_t)halo_count * sizeof(double)
        );

        memcpy(
            &grid[(halo + local_rows) * cols],
            &grid[halo * cols],
            (size_t)halo_count * sizeof(double)
        );

        return;
    }

    MPI_Status status;

    /*
     * 1) Svoje zgornje vrstice pošljemo zgornjemu sosedu.
     *    Od spodnjega soseda prejmemo spodnji halo.
     */
    MPI_Sendrecv(
        &grid[halo * cols],
        halo_count,
        MPI_DOUBLE,
        up,
        100,
        &grid[(halo + local_rows) * cols],
        halo_count,
        MPI_DOUBLE,
        down,
        100,
        MPI_COMM_WORLD,
        &status
    );

    /*
     * 2) Svoje spodnje vrstice pošljemo spodnjemu sosedu.
     *    Od zgornjega soseda prejmemo zgornji halo.
     */
    MPI_Sendrecv(
        &grid[(halo + local_rows - halo) * cols],
        halo_count,
        MPI_DOUBLE,
        down,
        101,
        &grid[0],
        halo_count,
        MPI_DOUBLE,
        up,
        101,
        MPI_COMM_WORLD,
        &status
    );
}

static void fill_local_from_full_world(
    double *local_grid,
    const double *full_world,
    unsigned int rows,
    unsigned int cols,
    unsigned int start_row,
    unsigned int local_rows,
    unsigned int halo
) {
    const unsigned int total_local_rows = local_rows + 2 * halo;

    for (unsigned int lr = 0; lr < total_local_rows; lr++) {
        const int global_row_raw = (int)start_row - (int)halo + (int)lr;
        const unsigned int global_row = wrap_row_general(global_row_raw, rows);

        memcpy(
            &local_grid[lr * cols],
            &full_world[global_row * cols],
            (size_t)cols * sizeof(double)
        );
    }
}

static void refresh_halos(
    double *local_grid,
    double *full_world_fallback,
    int use_allgather_fallback,
    const int *counts,
    const int *displs,
    unsigned int rows,
    unsigned int cols,
    unsigned int start_row,
    unsigned int local_rows,
    unsigned int halo,
    int myid,
    int procs
) {
    if (use_allgather_fallback) {
        /*
         * Fallback za primere, kjer je lokalni pas vrstic ožji od halo roba.
         * Primer: N=128 in P=32, kjer ima proces samo 4 vrstice,
         * halo pa je 13 vrstic.
         */
        MPI_Allgatherv(
            &local_grid[halo * cols],
            (int)(local_rows * cols),
            MPI_DOUBLE,
            full_world_fallback,
            counts,
            displs,
            MPI_DOUBLE,
            MPI_COMM_WORLD
        );

        fill_local_from_full_world(
            local_grid,
            full_world_fallback,
            rows,
            cols,
            start_row,
            local_rows,
            halo
        );
    } else {
        exchange_halos_direct(
            local_grid,
            local_rows,
            cols,
            halo,
            myid,
            procs
        );
    }
}

static void compute_next_local(
    double *next,
    const double *current,
    const kernel_entry_t *kernel_entries,
    unsigned int num_kernel_entries,
    unsigned int cols,
    unsigned int local_rows,
    unsigned int halo,
    double dt
) {
    for (unsigned int local_i = 0; local_i < local_rows; local_i++) {
        const unsigned int center_row = halo + local_i;

        for (unsigned int j = 0; j < cols; j++) {
            double sum = 0.0;

            for (unsigned int k = 0; k < num_kernel_entries; k++) {
                const int rr = (int)center_row + kernel_entries[k].dy;
                const unsigned int cc = wrap_col_fast((int)j + kernel_entries[k].dx, cols);

                sum += kernel_entries[k].weight * current[(unsigned int)rr * cols + cc];
            }

            const unsigned int idx = center_row * cols + j;
            double new_value;

            /*
             * Če je okolica prazna, je growth_lenia(sum) praktično -1.
             * Tako se izognemo dragemu exp() za večino prazne mreže.
             */
            if (sum <= EMPTY_THRESHOLD) {
                new_value = current[idx] - dt;
            } else {
                new_value = current[idx] + dt * growth_lenia(sum);
            }

            next[idx] = clip01(new_value);
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

    const unsigned int halo = kernel_size / 2;
    const unsigned int start_row = get_start_row(myid, procs, rows);
    const unsigned int local_rows = get_local_rows(myid, procs, rows);
    const unsigned int local_total_rows = local_rows + 2 * halo;
    const unsigned int local_count = local_rows * cols;

    /*
     * Direktni halo exchange deluje, kadar ima vsak proces vsaj halo vrstic.
     * Za majhne mreže in veliko procesov uporabimo fallback.
     */
    const int use_allgather_fallback = (local_rows < halo);

    double *kernel = (double *)calloc(kernel_size * kernel_size, sizeof(double));
    kernel_entry_t *kernel_entries =
        (kernel_entry_t *)malloc((size_t)kernel_size * kernel_size * sizeof(kernel_entry_t));

    double *current = (double *)calloc((size_t)local_total_rows * cols, sizeof(double));
    double *next = (double *)calloc((size_t)local_total_rows * cols, sizeof(double));

    int *counts = (int *)malloc((size_t)procs * sizeof(int));
    int *displs = (int *)malloc((size_t)procs * sizeof(int));

    double *full_world_fallback = NULL;

    if (use_allgather_fallback) {
        full_world_fallback = (double *)calloc((size_t)rows * cols, sizeof(double));
    }

    if (
        kernel == NULL ||
        kernel_entries == NULL ||
        current == NULL ||
        next == NULL ||
        counts == NULL ||
        displs == NULL ||
        (use_allgather_fallback && full_world_fallback == NULL)
    ) {
        fprintf(stderr, "Process %d: napaka pri alokaciji pomnilnika.\n", myid);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    prepare_counts_displs(counts, displs, procs, rows, cols);

    generate_kernel(kernel, kernel_size);
    const unsigned int num_kernel_entries =
        build_kernel_entries(kernel_entries, kernel, kernel_size);

    /*
     * Začetni svet naredi samo rank 0, potem ga razdeli po vrsticah.
     */
    double *initial_world = NULL;

    if (myid == 0) {
        initial_world = (double *)calloc((size_t)rows * cols, sizeof(double));

        if (initial_world == NULL) {
            fprintf(stderr, "Process 0: napaka pri alokaciji initial_world.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (unsigned int o = 0; o < num_orbiums; o++) {
            initial_world = place_orbium(
                initial_world,
                rows,
                cols,
                orbiums[o].row,
                orbiums[o].col,
                orbiums[o].angle
            );
        }
    }

    MPI_Scatterv(
        initial_world,
        counts,
        displs,
        MPI_DOUBLE,
        &current[halo * cols],
        (int)local_count,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    free(initial_world);

#ifdef GENERATE_GIF
    ge_GIF *gif = NULL;
    double *gif_world = NULL;

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

        gif_world = (double *)malloc((size_t)rows * cols * sizeof(double));

        if (gif_world == NULL) {
            fprintf(stderr, "Process 0: napaka pri alokaciji gif_world.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
#endif

    for (unsigned int step = 0; step < steps; step++) {
        /*
         * Pred vsakim korakom osvežimo halo robove.
         */
        refresh_halos(
            current,
            full_world_fallback,
            use_allgather_fallback,
            counts,
            displs,
            rows,
            cols,
            start_row,
            local_rows,
            halo,
            myid,
            procs
        );

        /*
         * Konvolucija + update v eni zanki.
         */
        compute_next_local(
            next,
            current,
            kernel_entries,
            num_kernel_entries,
            cols,
            local_rows,
            halo,
            dt
        );

        /*
         * Pointer swap namesto kopiranja matrike.
         */
        double *tmp = current;
        current = next;
        next = tmp;

#ifdef GENERATE_GIF
        MPI_Gatherv(
            &current[halo * cols],
            (int)local_count,
            MPI_DOUBLE,
            gif_world,
            counts,
            displs,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        if (myid == 0) {
            add_gif_frame(gif, gif_world, rows, cols);
        }
#endif
    }

#ifdef GENERATE_GIF
    if (myid == 0) {
        ge_close_gif(gif);
        free(gif_world);
    }
#endif

    /*
     * Končni rezultat zberemo samo na rank 0.
     * Na ostalih procesih funkcija vrne NULL, kar je varno za free(NULL).
     */
    double *result_world = NULL;

    if (myid == 0) {
        result_world = (double *)malloc((size_t)rows * cols * sizeof(double));

        if (result_world == NULL) {
            fprintf(stderr, "Process 0: napaka pri alokaciji result_world.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Gatherv(
        &current[halo * cols],
        (int)local_count,
        MPI_DOUBLE,
        result_world,
        counts,
        displs,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    free(kernel);
    free(kernel_entries);
    free(current);
    free(next);
    free(counts);
    free(displs);
    free(full_world_fallback);

    return result_world;
}