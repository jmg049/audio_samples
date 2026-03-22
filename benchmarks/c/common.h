#pragma once

#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* --------------------------------------------------------------------------
 * Timing
 * -------------------------------------------------------------------------- */

static inline long long get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* --------------------------------------------------------------------------
 * Sine wave generation  (matches audio_samples sine_wave::<f32>)
 * freq=440, amplitude=0.8
 * -------------------------------------------------------------------------- */

static inline void generate_sine(float *buf, int n, float freq, float amp, int sr) {
    for (int i = 0; i < n; i++)
        buf[i] = amp * sinf(2.0f * 3.14159265358979323846f * freq * (float)i / (float)sr);
}

/* interleaved stereo: [L0,R0, L1,R1, ...] — L and R are identical sine waves */
static inline void generate_stereo_interleaved(float *buf, int n_frames,
                                                float freq, float amp, int sr) {
    for (int i = 0; i < n_frames; i++) {
        float s = amp * sinf(2.0f * 3.14159265358979323846f * freq * (float)i / (float)sr);
        buf[2 * i]     = s;
        buf[2 * i + 1] = s;
    }
}

/* --------------------------------------------------------------------------
 * Statistics + CSV output
 * -------------------------------------------------------------------------- */

static int cmp_ll(const void *a, const void *b) {
    long long x = *(const long long *)a;
    long long y = *(const long long *)b;
    return (x > y) - (x < y);
}

static inline void print_csv_header(void) {
    printf("operation,implementation,duration_s,n_samples,iterations,warmup,"
           "min_us,mean_us,median_us,max_us,stddev_us\n");
}

/* times_ns array is sorted in place */
static void print_stats(const char *op, const char *impl,
                        int duration_s, int n_samples,
                        int iterations, int warmup,
                        long long *times_ns) {
    qsort(times_ns, (size_t)iterations, sizeof(long long), cmp_ll);

    long long mn  = times_ns[0];
    long long mx  = times_ns[iterations - 1];
    long long med = times_ns[iterations / 2];

    double sum = 0.0;
    for (int i = 0; i < iterations; i++) sum += (double)times_ns[i];
    double mean = sum / (double)iterations;

    double var = 0.0;
    for (int i = 0; i < iterations; i++) {
        double d = (double)times_ns[i] - mean;
        var += d * d;
    }
    double stddev = sqrt(var / (double)iterations);

    printf("%s,%s,%d,%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f\n",
           op, impl, duration_s, n_samples, iterations, warmup,
           (double)mn    / 1000.0,
           mean          / 1000.0,
           (double)med   / 1000.0,
           (double)mx    / 1000.0,
           stddev        / 1000.0);
}

/* --------------------------------------------------------------------------
 * Argument parsing
 * -------------------------------------------------------------------------- */

static inline int parse_int_arg(int argc, char **argv, const char *flag, int def) {
    for (int i = 1; i < argc - 1; i++)
        if (strcmp(argv[i], flag) == 0) return atoi(argv[i + 1]);
    return def;
}
