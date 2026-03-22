CC      := gcc
CFLAGS  := -O3 -march=native -ffast-math -std=c11 -Wall -Wextra \
           $(shell pkg-config --cflags libswresample libavfilter libavutil)
SWR_LIBS := $(shell pkg-config --libs libswresample libavutil)
AVF_LIBS := $(shell pkg-config --libs libavfilter libavutil)

BUILD   := $(CURDIR)/build
SRCS    := c/resample.c c/filter.c c/volume.c c/channels.c c/format_conv.c c/processing.c c/editing.c
BINS    := $(patsubst c/%.c,$(BUILD)/%,$(SRCS))

.PHONY: all clean

all: $(BUILD) $(BINS)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/resample:   c/resample.c   c/common.h | $(BUILD)
	$(CC) $(CFLAGS) $< -o $@ $(SWR_LIBS) -lm

$(BUILD)/filter:     c/filter.c     c/common.h | $(BUILD)
	$(CC) $(CFLAGS) $< -o $@ $(AVF_LIBS) -lm

$(BUILD)/volume:     c/volume.c     c/common.h | $(BUILD)
	$(CC) $(CFLAGS) $< -o $@ -lm

$(BUILD)/channels:   c/channels.c   c/common.h | $(BUILD)
	$(CC) $(CFLAGS) $< -o $@ $(SWR_LIBS) -lm

$(BUILD)/format_conv: c/format_conv.c c/common.h | $(BUILD)
	$(CC) $(CFLAGS) $< -o $@ $(SWR_LIBS) -lm

$(BUILD)/processing: c/processing.c c/common.h | $(BUILD)
	$(CC) $(CFLAGS) $< -o $@ -lm

$(BUILD)/editing:    c/editing.c    c/common.h | $(BUILD)
	$(CC) $(CFLAGS) $< -o $@ -lm

clean:
	rm -rf $(BUILD)
