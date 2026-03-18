/*
 * processing.c — scale, clip, and normalize benchmarks (pure C, no FFmpeg)
 *
 * FFmpeg has no direct in-library API for these operations; they require a
 * filter graph (volume, aformat, dynaudnorm).  Both sides use their language's
 * native idiom: tight loops in C, AudioProcessing methods in Rust.
 *
 * Methodology: each iteration copies the source buffer into a pre-allocated
 * work buffer (memcpy) and then operates in-place on the work buffer.  This
 * matches the Rust benchmark which calls audio.clone().op(), so both sides
 * measure "copy + compute" as the unit of work.
 *
 * Compile:
 *   gcc -O3 -march=native -ffast-math -std=c11 processing.c -o processing -lm
 */

#include "common.h"

static volatile float g_sink = 0.0f;

/* -------------------------------------------------------------------------
 * scale_by_half: multiply every sample by 0.5
 * ---------------------------------------------------------------------- */

static void do_scale(float *data, int n) {
    for (int i = 0; i < n; i++)
        data[i] *= 0.5f;
}

static void bench_scale(int duration_s, int iterations, int warmup, int sr) {
    int n = sr * duration_s;
    float *source = malloc((size_t)n * sizeof(float));
    float *work   = malloc((size_t)n * sizeof(float));
    generate_sine(source, n, 440.0f, 0.8f, sr);

    long long *times = malloc((size_t)iterations * sizeof(long long));

    for (int i = 0; i < warmup; i++) {
        memcpy(work, source, (size_t)n * sizeof(float));
        do_scale(work, n);
        g_sink = work[0];
    }

    for (int i = 0; i < iterations; i++) {
        long long t0 = get_time_ns();
        memcpy(work, source, (size_t)n * sizeof(float));
        do_scale(work, n);
        times[i] = get_time_ns() - t0;
        g_sink = work[0];
    }

    char op[64];
    snprintf(op, sizeof(op), "scale_by_half_%ds", duration_s);
    print_stats(op, "c_native", duration_s, n, iterations, warmup, times);

    free(times); free(work); free(source);
}

/* -------------------------------------------------------------------------
 * clip: clamp every sample to [-0.5, 0.5]
 * ---------------------------------------------------------------------- */

static void do_clip(float *data, int n) {
    for (int i = 0; i < n; i++) {
        float s = data[i];
        if      (s >  0.5f) s =  0.5f;
        else if (s < -0.5f) s = -0.5f;
        data[i] = s;
    }
}

static void bench_clip(int duration_s, int iterations, int warmup, int sr) {
    int n = sr * duration_s;
    float *source = malloc((size_t)n * sizeof(float));
    float *work   = malloc((size_t)n * sizeof(float));
    generate_sine(source, n, 440.0f, 0.8f, sr);

    long long *times = malloc((size_t)iterations * sizeof(long long));

    for (int i = 0; i < warmup; i++) {
        memcpy(work, source, (size_t)n * sizeof(float));
        do_clip(work, n);
        g_sink = work[0];
    }

    for (int i = 0; i < iterations; i++) {
        long long t0 = get_time_ns();
        memcpy(work, source, (size_t)n * sizeof(float));
        do_clip(work, n);
        times[i] = get_time_ns() - t0;
        g_sink = work[0];
    }

    char op[64];
    snprintf(op, sizeof(op), "clip_%ds", duration_s);
    print_stats(op, "c_native", duration_s, n, iterations, warmup, times);

    free(times); free(work); free(source);
}

/* -------------------------------------------------------------------------
 * normalize_peak: two-pass — find absolute peak, then divide every sample
 * ---------------------------------------------------------------------- */

static void do_normalize_peak(float *data, int n) {
    float peak = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(data[i]);
        if (a > peak) peak = a;
    }
    if (peak > 0.0f) {
        float inv = 1.0f / peak;
        for (int i = 0; i < n; i++)
            data[i] *= inv;
    }
}

static void bench_normalize(int duration_s, int iterations, int warmup, int sr) {
    int n = sr * duration_s;
    float *source = malloc((size_t)n * sizeof(float));
    float *work   = malloc((size_t)n * sizeof(float));
    generate_sine(source, n, 440.0f, 0.8f, sr);

    long long *times = malloc((size_t)iterations * sizeof(long long));

    for (int i = 0; i < warmup; i++) {
        memcpy(work, source, (size_t)n * sizeof(float));
        do_normalize_peak(work, n);
        g_sink = work[0];
    }

    for (int i = 0; i < iterations; i++) {
        long long t0 = get_time_ns();
        memcpy(work, source, (size_t)n * sizeof(float));
        do_normalize_peak(work, n);
        times[i] = get_time_ns() - t0;
        g_sink = work[0];
    }

    char op[64];
    snprintf(op, sizeof(op), "normalize_peak_%ds", duration_s);
    print_stats(op, "c_native", duration_s, n, iterations, warmup, times);

    free(times); free(work); free(source);
}

/* -------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------- */

int main(int argc, char **argv) {
    int dur    = parse_int_arg(argc, argv, "--duration",   1);
    int iters  = parse_int_arg(argc, argv, "--iterations", 1000);
    int warmup = parse_int_arg(argc, argv, "--warmup",     100);

    print_csv_header();
    bench_scale(dur, iters, warmup, 44100);
    bench_clip(dur, iters, warmup, 44100);
    bench_normalize(dur, iters, warmup, 44100);

    return 0;
}
