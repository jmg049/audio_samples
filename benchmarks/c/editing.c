/*
 * editing.c — trim, pad, and fade-in benchmarks (pure C, no FFmpeg)
 *
 * FFmpeg's filter graph has equivalents (atrim, apad, afade) but they carry
 * substantial setup overhead.  Both sides use their language's native idiom:
 * plain memory operations in C, AudioEditing methods in Rust.
 *
 * Methodology:
 *   trim / pad  — malloc + copy + free each iteration to match audio_samples,
 *                 which allocates a new owned buffer on every call.
 *   fade_in     — malloc + copy + in-place ramp multiply + free each iteration
 *                 to match audio_samples' clone-then-mutate pattern.
 *
 * noinline wrappers are used so the compiler cannot eliminate malloc/memcpy
 * sequences whose result is freed without an apparent read.
 *
 * Compile:
 *   gcc -O3 -march=native -ffast-math -std=c11 editing.c -o editing -lm
 */

#include "common.h"

/* -------------------------------------------------------------------------
 * trim_middle: extract the centre 50 % of the signal
 *   start = 0.25 × n,  length = 0.5 × n
 * ---------------------------------------------------------------------- */

__attribute__((noinline))
static float *do_trim(const float *in, int start, int len) {
    float *out = malloc((size_t)len * sizeof(float));
    memcpy(out, in + start, (size_t)len * sizeof(float));
    return out;
}

static void bench_trim(int duration_s, int iterations, int warmup, int sr) {
    int n       = sr * duration_s;
    int start   = n / 4;   /* 0.25 × n */
    int out_len = n / 2;   /* 0.75×n − 0.25×n */

    float *input = malloc((size_t)n * sizeof(float));
    generate_sine(input, n, 440.0f, 0.8f, sr);

    long long *times = malloc((size_t)iterations * sizeof(long long));

    for (int i = 0; i < warmup; i++)
        free(do_trim(input, start, out_len));

    for (int i = 0; i < iterations; i++) {
        long long t0 = get_time_ns();
        float *out = do_trim(input, start, out_len);
        times[i] = get_time_ns() - t0;
        free(out);
    }

    char op[64];
    snprintf(op, sizeof(op), "trim_middle_%ds", duration_s);
    print_stats(op, "c_native", duration_s, n, iterations, warmup, times);

    free(times); free(input);
}

/* -------------------------------------------------------------------------
 * pad_end: append 0.5 s of silence after the signal
 *   output length = n + sr/2 samples
 * ---------------------------------------------------------------------- */

__attribute__((noinline))
static float *do_pad(const float *in, int n, int pad_n) {
    int out_len = n + pad_n;
    float *out  = malloc((size_t)out_len * sizeof(float));
    memcpy(out, in, (size_t)n * sizeof(float));
    memset(out + n, 0, (size_t)pad_n * sizeof(float));
    return out;
}

static void bench_pad(int duration_s, int iterations, int warmup, int sr) {
    int n     = sr * duration_s;
    int pad_n = sr / 2;   /* 0.5 s of silence */

    float *input = malloc((size_t)n * sizeof(float));
    generate_sine(input, n, 440.0f, 0.8f, sr);

    long long *times = malloc((size_t)iterations * sizeof(long long));

    for (int i = 0; i < warmup; i++)
        free(do_pad(input, n, pad_n));

    for (int i = 0; i < iterations; i++) {
        long long t0 = get_time_ns();
        float *out = do_pad(input, n, pad_n);
        times[i] = get_time_ns() - t0;
        free(out);
    }

    char op[64];
    snprintf(op, sizeof(op), "pad_end_%ds", duration_s);
    print_stats(op, "c_native", duration_s, n, iterations, warmup, times);

    free(times); free(input);
}

/* -------------------------------------------------------------------------
 * fade_in: linear ramp over the first 0.5 s
 *   gain[i] = i / fade_samples  for i in [0, fade_samples)
 * ---------------------------------------------------------------------- */

__attribute__((noinline))
static float *do_fade_in(const float *in, int n, int fade_samples) {
    float *out = malloc((size_t)n * sizeof(float));
    memcpy(out, in, (size_t)n * sizeof(float));
    for (int j = 0; j < fade_samples; j++)
        out[j] *= (float)j / (float)fade_samples;
    return out;
}

static void bench_fade_in(int duration_s, int iterations, int warmup, int sr) {
    int n            = sr * duration_s;
    int fade_samples = sr / 2;   /* 0.5 s */
    if (fade_samples > n) fade_samples = n;

    float *input = malloc((size_t)n * sizeof(float));
    generate_sine(input, n, 440.0f, 0.8f, sr);

    long long *times = malloc((size_t)iterations * sizeof(long long));

    for (int i = 0; i < warmup; i++)
        free(do_fade_in(input, n, fade_samples));

    for (int i = 0; i < iterations; i++) {
        long long t0 = get_time_ns();
        float *out = do_fade_in(input, n, fade_samples);
        times[i] = get_time_ns() - t0;
        free(out);
    }

    char op[64];
    snprintf(op, sizeof(op), "fade_in_%ds", duration_s);
    print_stats(op, "c_native", duration_s, n, iterations, warmup, times);

    free(times); free(input);
}

/* -------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------- */

int main(int argc, char **argv) {
    int dur    = parse_int_arg(argc, argv, "--duration",   1);
    int iters  = parse_int_arg(argc, argv, "--iterations", 1000);
    int warmup = parse_int_arg(argc, argv, "--warmup",     100);

    print_csv_header();
    bench_trim(dur, iters, warmup, 44100);
    bench_pad(dur, iters, warmup, 44100);
    bench_fade_in(dur, iters, warmup, 44100);

    return 0;
}
