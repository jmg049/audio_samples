/*
 * volume.c — RMS + peak analysis benchmark (pure C, no FFmpeg library)
 *
 * FFmpeg has no in-library API for RMS/peak computation (volumedetect lives
 * only in the filter graph / CLI).  Both sides therefore use their language's
 * native idiom: a tight loop in C, rms()+peak() in Rust.  This is the purest
 * algorithmic comparison in the suite.
 *
 * Compile:
 *   gcc -O3 -march=native -ffast-math -std=c11 \
 *       volume.c -o volume -lm
 */

#include "common.h"

/* Both rms and peak in a single pass — same data access pattern as
   audio_samples, which makes two separate passes internally. */
static void compute_rms_peak(const float *buf, int n,
                              double *out_rms, float *out_peak) {
    double sum_sq = 0.0;
    float  peak   = 0.0f;
    for (int i = 0; i < n; i++) {
        float s = buf[i];
        float a = fabsf(s);
        sum_sq += (double)s * (double)s;
        if (a > peak) peak = a;
    }
    *out_rms  = sqrt(sum_sq / (double)n);
    *out_peak = peak;
}

/* Volatile sink — prevents DCE even with -O3 -ffast-math.
   Written outside the timing window so it does not skew measurements. */
static volatile double g_sink = 0.0;

static void bench(int duration_s, int iterations, int warmup, int sr) {
    int n = sr * duration_s;
    float *input = malloc((size_t)n * sizeof(float));
    generate_sine(input, n, 440.0f, 0.5f, sr);  /* amplitude 0.5 matches Rust side */

    long long *times = malloc((size_t)iterations * sizeof(long long));
    double rms = 0.0; float peak = 0.0f;

    for (int i = 0; i < warmup; i++) {
        compute_rms_peak(input, n, &rms, &peak);
        g_sink = rms + (double)peak;
    }

    for (int i = 0; i < iterations; i++) {
        long long t0 = get_time_ns();
        compute_rms_peak(input, n, &rms, &peak);
        long long t1 = get_time_ns();
        times[i] = t1 - t0;
        g_sink = rms + (double)peak;  /* consume results outside timing window */
    }

    char op[64];
    snprintf(op, sizeof(op), "rms_and_peak_%ds", duration_s);
    print_stats(op, "c_native", duration_s, n, iterations, warmup, times);

    free(times);
    free(input);
}

int main(int argc, char **argv) {
    int dur    = parse_int_arg(argc, argv, "--duration",   1);
    int iters  = parse_int_arg(argc, argv, "--iterations", 1000);
    int warmup = parse_int_arg(argc, argv, "--warmup",     100);

    print_csv_header();
    bench(dur, iters, warmup, 44100);

    return 0;
}
