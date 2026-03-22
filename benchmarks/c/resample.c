/*
 * resample.c — libswresample resampling benchmark
 *
 * Compares FFmpeg's default resampler and its optional soxr engine.
 * The SwrContext is created once and reused across iterations (matching
 * audio_samples' thread-local resampler cache behaviour).  Output buffers
 * are allocated and freed each iteration to mirror audio_samples, which
 * returns a new allocation per call.
 *
 * Compile:
 *   gcc -O3 -march=native -ffast-math -std=c11 \
 *       $(pkg-config --cflags libswresample libavutil) \
 *       resample.c -o resample \
 *       $(pkg-config --libs libswresample libavutil)
 */

#include "common.h"
#include <libswresample/swresample.h>
#include <libavutil/channel_layout.h>
#include <libavutil/mem.h>
#include <libavutil/opt.h>
#include <libavutil/samplefmt.h>

static SwrContext *create_swr(int src_rate, int dst_rate, int use_soxr) {
    SwrContext *swr = NULL;
    int ret = swr_alloc_set_opts2(
        &swr,
        &(AVChannelLayout)AV_CHANNEL_LAYOUT_MONO, AV_SAMPLE_FMT_FLT, dst_rate,
        &(AVChannelLayout)AV_CHANNEL_LAYOUT_MONO, AV_SAMPLE_FMT_FLT, src_rate,
        0, NULL);
    if (ret < 0 || !swr) { fprintf(stderr, "swr_alloc_set_opts2 failed\n"); exit(1); }
    if (use_soxr)
        av_opt_set_int(swr, "resampler", SWR_ENGINE_SOXR, 0);
    ret = swr_init(swr);
    if (ret < 0) { fprintf(stderr, "swr_init failed\n"); exit(1); }
    return swr;
}

static void do_one(SwrContext *swr, const float *in, int n_in,
                   int src_rate, int dst_rate) {
    /* Upper bound including any delay already in the context */
    int n_out = (int)(((long long)n_in * dst_rate + src_rate - 1) / src_rate) + 64;
    uint8_t *out = NULL;
    av_samples_alloc(&out, NULL, 1, n_out, AV_SAMPLE_FMT_FLT, 0);
    swr_convert(swr, &out, n_out, (const uint8_t **)&in, n_in);
    av_freep(&out);
}

static void bench(int duration_s, int iterations, int warmup,
                  int src_rate, int dst_rate,
                  const char *op, const char *impl, int use_soxr) {
    int n_in = src_rate * duration_s;
    float *input = malloc((size_t)n_in * sizeof(float));
    generate_sine(input, n_in, 440.0f, 0.8f, src_rate);

    SwrContext *swr = create_swr(src_rate, dst_rate, use_soxr);
    long long *times = malloc((size_t)iterations * sizeof(long long));

    for (int i = 0; i < warmup; i++)
        do_one(swr, input, n_in, src_rate, dst_rate);

    for (int i = 0; i < iterations; i++) {
        long long t0 = get_time_ns();
        do_one(swr, input, n_in, src_rate, dst_rate);
        times[i] = get_time_ns() - t0;
    }

    print_stats(op, impl, duration_s, n_in, iterations, warmup, times);

    swr_free(&swr);
    free(times);
    free(input);
}

int main(int argc, char **argv) {
    int dur    = parse_int_arg(argc, argv, "--duration",   1);
    int iters  = parse_int_arg(argc, argv, "--iterations", 1000);
    int warmup = parse_int_arg(argc, argv, "--warmup",     100);

    print_csv_header();

    char op[64];
    snprintf(op, sizeof(op), "resample_44100_to_16000_%ds", dur);

    bench(dur, iters, warmup, 44100, 16000, op, "ffmpeg_default", 0);
    bench(dur, iters, warmup, 44100, 16000, op, "ffmpeg_soxr",    1);

    return 0;
}
