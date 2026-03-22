/*
 * channels.c — channel layout conversion benchmarks using libswresample
 *
 * Covers both conversion directions:
 *   stereo → mono  (downmix: 0.5×L + 0.5×R, matching MonoConversionMethod::Average)
 *   mono → stereo  (upmix: L = R = input, matching duplicate_to_channels(2))
 *
 * The SwrContext is created once and reused across iterations.
 * Output buffers are allocated and freed each iteration.
 *
 * Compile:
 *   gcc -O3 -march=native -ffast-math -std=c11 \
 *       $(pkg-config --cflags libswresample libavutil) \
 *       channels.c -o channels \
 *       $(pkg-config --libs libswresample libavutil)
 */

#include "common.h"
#include <libswresample/swresample.h>
#include <libavutil/channel_layout.h>
#include <libavutil/mem.h>
#include <libavutil/samplefmt.h>

/* -------------------------------------------------------------------------
 * stereo → mono
 * ---------------------------------------------------------------------- */

static void do_stereo_to_mono(SwrContext *swr, const float *in_interleaved,
                               int n_frames) {
    uint8_t *out = NULL;
    av_samples_alloc(&out, NULL, 1, n_frames, AV_SAMPLE_FMT_FLT, 0);
    swr_convert(swr, &out, n_frames,
                (const uint8_t **)&in_interleaved, n_frames);
    av_freep(&out);
}

static void bench_stereo_to_mono(int duration_s, int iterations, int warmup, int sr) {
    int n_frames = sr * duration_s;
    /* Interleaved stereo [L0,R0, L1,R1, ...] */
    float *input = malloc((size_t)n_frames * 2 * sizeof(float));
    generate_stereo_interleaved(input, n_frames, 440.0f, 0.8f, sr);

    SwrContext *swr = NULL;
    int ret = swr_alloc_set_opts2(
        &swr,
        &(AVChannelLayout)AV_CHANNEL_LAYOUT_MONO,   AV_SAMPLE_FMT_FLT, sr,
        &(AVChannelLayout)AV_CHANNEL_LAYOUT_STEREO, AV_SAMPLE_FMT_FLT, sr,
        0, NULL);
    if (ret < 0 || !swr) { fprintf(stderr, "swr_alloc_set_opts2 failed\n"); exit(1); }
    ret = swr_init(swr);
    if (ret < 0) { fprintf(stderr, "swr_init failed\n"); exit(1); }

    long long *times = malloc((size_t)iterations * sizeof(long long));

    for (int i = 0; i < warmup; i++)
        do_stereo_to_mono(swr, input, n_frames);

    for (int i = 0; i < iterations; i++) {
        long long t0 = get_time_ns();
        do_stereo_to_mono(swr, input, n_frames);
        times[i] = get_time_ns() - t0;
    }

    char op[64];
    snprintf(op, sizeof(op), "stereo_to_mono_%ds", duration_s);
    print_stats(op, "ffmpeg_swresample", duration_s, n_frames,
                iterations, warmup, times);

    swr_free(&swr);
    free(times);
    free(input);
}

/* -------------------------------------------------------------------------
 * mono → stereo
 * ---------------------------------------------------------------------- */

static void do_mono_to_stereo(SwrContext *swr, const float *in, int n_frames) {
    /* Output: interleaved stereo — 2 channels × n_frames floats */
    uint8_t *out = NULL;
    av_samples_alloc(&out, NULL, 2, n_frames, AV_SAMPLE_FMT_FLT, 0);
    swr_convert(swr, &out, n_frames, (const uint8_t **)&in, n_frames);
    av_freep(&out);
}

static void bench_mono_to_stereo(int duration_s, int iterations, int warmup, int sr) {
    int n_frames = sr * duration_s;
    float *input = malloc((size_t)n_frames * sizeof(float));
    generate_sine(input, n_frames, 440.0f, 0.8f, sr);

    SwrContext *swr = NULL;
    int ret = swr_alloc_set_opts2(
        &swr,
        &(AVChannelLayout)AV_CHANNEL_LAYOUT_STEREO, AV_SAMPLE_FMT_FLT, sr,
        &(AVChannelLayout)AV_CHANNEL_LAYOUT_MONO,   AV_SAMPLE_FMT_FLT, sr,
        0, NULL);
    if (ret < 0 || !swr) { fprintf(stderr, "swr_alloc_set_opts2 failed\n"); exit(1); }
    ret = swr_init(swr);
    if (ret < 0) { fprintf(stderr, "swr_init failed\n"); exit(1); }

    long long *times = malloc((size_t)iterations * sizeof(long long));

    for (int i = 0; i < warmup; i++)
        do_mono_to_stereo(swr, input, n_frames);

    for (int i = 0; i < iterations; i++) {
        long long t0 = get_time_ns();
        do_mono_to_stereo(swr, input, n_frames);
        times[i] = get_time_ns() - t0;
    }

    char op[64];
    snprintf(op, sizeof(op), "mono_to_stereo_%ds", duration_s);
    print_stats(op, "ffmpeg_swresample", duration_s, n_frames,
                iterations, warmup, times);

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
    bench_stereo_to_mono(dur, iters, warmup, 44100);
    bench_mono_to_stereo(dur, iters, warmup, 44100);

    return 0;
}
