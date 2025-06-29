pub fn samples_like<T: AudioSample>(samples: &[T], sample_rate: usize, channels: usize) -> AudioSamples<T> {
    AudioSamples {
        samples: samples.to_vec(),
        sample_rate,
        channels,
    }
}

pub fn times_like<T: AudioSample>(samples: &[T], sample_rate: usize, channels: usize) -> AudioSamples<T> {
    AudioSamples {
        samples: samples.to_vec(),
        sample_rate,
        channels,
    }
}

