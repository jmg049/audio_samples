/*
 * format_conv.c — sample format conversion benchmarks using libswresample
 *
 * Covers both conversion directions:
 *   f32 → i16  (writing to disk / sending to hardware)
 *   i16 → f32  (reading from disk / receiving from hardware)
 *
 * Both sides scale [-1.0, 1.0] ↔ [-32768, 32767] with clipping.
 * No resampling or channel change — pure format conversion.
 *
 * The SwrContext is created once and reused; output is allocated and freed
 * each iteration to mirror audio_samples' to_format() which returns a new
 * owned buffer.
 *
 * Compile:
 *   gcc -O3 -march=native -ffast-math -std=c11 \
 *       $(pkg-config --cflags libswresample libavutil) \
 *       format_conv.c -o format_conv \
 *       $(pkg-config --libs libswresample libavutil)
 */

#include "common.h"
#include <libswresample/swresample.h>
#include <libavutil/channel_layout.h>
#include <libavutil/mem.h>
#include <libavutil/samplefmt.h>

/* -------------------------------------------------------------------------
 * f32 → i16
 * ---------------------------------------------------------------------- */

static void do_f32_to_i16(SwrContext *swr, const float *in, int n) {
    uint8_t *out = NULL;
    av_samples_alloc(&out, NULL, 1, n, AV_SAMPLE_FMT_S16, 0);
    swr_convert(swr, &out, n, (const uint8_t **)&in, n);
    av_freep(&out);
}

static void bench_f32_to_i16(int duration_s, int iterations, int warmup, int sr) {
    int n = sr * duration_s;
    float *input = malloc((size_t)n * sizeof(float));
    generate_sine(input, n, 440.0f, 0.8f, sr);

    SwrContext *swr = NULL;
    int ret = swr_alloc_set_opts2(
        &swr,
        &(AVChannelLayout)AV_CHANNEL_LAYOUT_MONO, AV_SAMPLE_FMT_S16, sr,
        &(AVChannelLayout)AV_CHANNEL_LAYOUT_MONO, AV_SAMPLE_FMT_FLT, sr,
        0, NULL);
    if (ret < 0 || !swr) { fprintf(stderr, "swr_alloc_set_opts2 failed\n"); exit(1); }
    ret = swr_init(swr);
    if (ret < 0) { fprintf(stderr, "swr_init failed\n"); exit(1); }

    long long *times = malloc((size_t)iterations * sizeof(long long));

    for (int i = 0; i < warmup; i++)
        do_f32_to_i16(swr, input, n);

    for (int i = 0; i < iterations; i++) {
        long long t0 = get_time_ns();
        do_f32_to_i16(swr, input, n);
        times[i] = get_time_ns() - t0;
    }

    char op[64];
    snprintf(op, sizeof(op), "format_f32_to_i16_%ds", duration_s);
    print_stats(op, "ffmpeg_swresample", duration_s, n, iterations, warmup, times);

    swr_free(&swr);
    free(times);
    free(input);
}

/* -------------------------------------------------------------------------
 * i16 → f32
 * ---------------------------------------------------------------------- */

static void do_i16_to_f32(SwrContext *swr, const int16_t *in, int n) {
    uint8_t *out = NULL;
    av_samples_alloc(&out, NULL, 1, n, AV_SAMPLE_FMT_FLT, 0);
    swr_convert(swr, &out, n, (const uint8_t **)&in, n);
    av_freep(&out);
}

static void bench_i16_to_f32(int duration_s, int iterations, int warmup, int sr) {
    int n = sr * duration_s;
    int16_t *input = malloc((size_t)n * sizeof(int16_t));
    /* Generate sine as i16 — amplitude 0.8 × 32767 ≈ 26214 */
    for (int i = 0; i < n; i++) {
        float s = 0.8f * sinf(2.0f * 3.14159265358979323846f * 440.0f * (float)i / (float)sr);
        input[i] = (int16_t)(s * 32767.0f);
    }

    SwrContext *swr = NULL;
    int ret = swr_alloc_set_opts2(
        &swr,
        &(AVChannelLayout)AV_CHANNEL_LAYOUT_MONO, AV_SAMPLE_FMT_FLT, sr,
        &(AVChannelLayout)AV_CHANNEL_LAYOUT_MONO, AV_SAMPLE_FMT_S16, sr,
        0, NULL);
    if (ret < 0 || !swr) { fprintf(stderr, "swr_alloc_set_opts2 failed\n"); exit(1); }
    ret = swr_init(swr);
    if (ret < 0) { fprintf(stderr, "swr_init failed\n"); exit(1); }

    long long *times = malloc((size_t)iterations * sizeof(long long));

    for (int i = 0; i < warmup; i++)
        do_i16_to_f32(swr, input, n);

    for (int i = 0; i < iterations; i++) {
        long long t0 = get_time_ns();
        do_i16_to_f32(swr, input, n);
        times[i] = get_time_ns() - t0;
    }

    char op[64];
    snprintf(op, sizeof(op), "format_i16_to_f32_%ds", duration_s);
    print_stats(op, "ffmpeg_swresample", duration_s, n, iterations, warmup, times);

    swr_free(&swr);
    free(times);
    free(input);
}

/* -------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------- */

int main(int argc, char **argv) {
    int dur    = parse_int_arg(argc, argv, "--duration",   1);
    int iters  = parse_int_arg(argc, argv, "--iterations", 1000);
    int warmup = parse_int_arg(argc, argv, "--warmup",     100);

    print_csv_header();
    bench_f32_to_i16(dur, iters, warmup, 44100);
    bench_i16_to_f32(dur, iters, warmup, 44100);

    return 0;
}
