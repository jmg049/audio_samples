/*
 * filter.c — libavfilter IIR filter benchmark (lowpass + highpass)
 *
 * FFmpeg's `lowpass` / `highpass` filters are 1- or 2-pole Butterworth IIRs.
 * The maximum pole count is 2, so both sides use order 2.
 *
 * The entire filter graph is created and destroyed each iteration to match
 * audio_samples, which recomputes filter coefficients and resets IIR state
 * on every call to butterworth_lowpass() / butterworth_highpass().
 *
 * Compile:
 *   gcc -O3 -march=native -ffast-math -std=c11 \
 *       $(pkg-config --cflags libavfilter libavutil) \
 *       filter.c -o filter \
 *       $(pkg-config --libs libavfilter libavutil)
 */

#include "common.h"
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersrc.h>
#include <libavfilter/buffersink.h>
#include <libavutil/channel_layout.h>
#include <libavutil/frame.h>
#include <libavutil/mem.h>
#include <libavutil/samplefmt.h>

static int do_one(const float *in, int n, int sr,
                  const char *filter_name, const char *filter_args) {
    AVFilterGraph    *graph    = avfilter_graph_alloc();
    AVFilterContext  *src_ctx  = NULL;
    AVFilterContext  *filt_ctx = NULL;
    AVFilterContext  *sink_ctx = NULL;

    /* abuffer source — configure via args string (required for avfilter_graph_config
       to succeed; av_buffersrc_parameters_set alone leaves format unresolved). */
    char src_args[256];
    snprintf(src_args, sizeof(src_args),
             "sample_rate=%d:sample_fmt=%s:channel_layout=mono:time_base=1/%d",
             sr, av_get_sample_fmt_name(AV_SAMPLE_FMT_FLTP), sr);
    const AVFilter *abuf = avfilter_get_by_name("abuffer");
    int ret = avfilter_graph_create_filter(&src_ctx, abuf, "in", src_args, NULL, graph);
    if (ret < 0) goto fail;

    /* lowpass=f=<cutoff>:p=2  or  highpass=f=<cutoff>:p=2 */
    const AVFilter *filt = avfilter_get_by_name(filter_name);
    ret = avfilter_graph_create_filter(&filt_ctx, filt, "filt", filter_args, NULL, graph);
    if (ret < 0) goto fail;

    /* abuffersink */
    const AVFilter *abuffersink = avfilter_get_by_name("abuffersink");
    ret = avfilter_graph_create_filter(&sink_ctx, abuffersink, "out", NULL, NULL, graph);
    if (ret < 0) goto fail;

    /* Link: src -> filter -> sink */
    ret = avfilter_link(src_ctx, 0, filt_ctx, 0); if (ret < 0) goto fail;
    ret = avfilter_link(filt_ctx, 0, sink_ctx, 0); if (ret < 0) goto fail;

    ret = avfilter_graph_config(graph, NULL);
    if (ret < 0) goto fail;

    /* Build input frame — use FLTP (planar); for mono, packed == planar in memory */
    AVFrame *frame = av_frame_alloc();
    frame->format      = AV_SAMPLE_FMT_FLTP;
    frame->sample_rate = sr;
    frame->nb_samples  = n;
    av_channel_layout_copy(&frame->ch_layout, &(AVChannelLayout)AV_CHANNEL_LAYOUT_MONO);
    ret = av_frame_get_buffer(frame, 0);
    if (ret < 0) { av_frame_free(&frame); goto fail; }
    memcpy(frame->data[0], in, (size_t)n * sizeof(float));

    ret = av_buffersrc_add_frame_flags(src_ctx, frame, AV_BUFFERSRC_FLAG_KEEP_REF);
    av_frame_free(&frame);
    if (ret < 0) goto fail;

    /* Flush source */
    av_buffersrc_close(src_ctx, AV_NOPTS_VALUE, 0);

    /* Drain output (discard — we only measure throughput) */
    AVFrame *out = av_frame_alloc();
    while (av_buffersink_get_frame(sink_ctx, out) >= 0)
        av_frame_unref(out);
    av_frame_free(&out);

    avfilter_graph_free(&graph);
    return 0;

fail:
    avfilter_graph_free(&graph);
    return ret;
}

static void bench(int duration_s, int iterations, int warmup,
                  int sr, float cutoff_hz,
                  const char *filter_name, const char *op_prefix,
                  const char *impl_name) {
    int n = sr * duration_s;
    float *input = malloc((size_t)n * sizeof(float));
    generate_sine(input, n, 440.0f, 0.8f, sr);

    long long *times = malloc((size_t)iterations * sizeof(long long));

    char filt_args[64];
    snprintf(filt_args, sizeof(filt_args), "f=%.1f:p=2", (double)cutoff_hz);

    /* Verify the filter actually works before timing */
    {
        int r = do_one(input, n, sr, filter_name, filt_args);
        if (r < 0) {
            char errbuf[256];
            av_strerror(r, errbuf, sizeof(errbuf));
            fprintf(stderr, "filter setup failed (%s): %s\n", filter_name, errbuf);
            free(times); free(input);
            return;
        }
    }
    for (int i = 1; i < warmup; i++)
        do_one(input, n, sr, filter_name, filt_args);

    for (int i = 0; i < iterations; i++) {
        long long t0 = get_time_ns();
        do_one(input, n, sr, filter_name, filt_args);
        times[i] = get_time_ns() - t0;
    }

    char op[64];
    snprintf(op, sizeof(op), "%s_%ds", op_prefix, duration_s);
    print_stats(op, impl_name, duration_s, n, iterations, warmup, times);

    free(times);
    free(input);
}

int main(int argc, char **argv) {
    int dur    = parse_int_arg(argc, argv, "--duration",   1);
    int iters  = parse_int_arg(argc, argv, "--iterations", 200);
    int warmup = parse_int_arg(argc, argv, "--warmup",     20);

    print_csv_header();

    bench(dur, iters, warmup, 44100, 1000.0f,
          "lowpass",  "lowpass_1000hz_order2",  "ffmpeg_lowpass_order2");
    bench(dur, iters, warmup, 44100, 1000.0f,
          "highpass", "highpass_1000hz_order2", "ffmpeg_highpass_order2");

    return 0;
}
