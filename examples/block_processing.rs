//! Process a long signal block by block, keeping filter state between blocks.
//!
//! A one-minute signal is low-pass filtered and its level measured in blocks of
//! 4096 samples, as a program reading a file or a live input would do. Only one
//! block is held at a time. The result equals filtering the whole signal at once.

#[cfg(not(all(feature = "iir-filtering", feature = "statistics")))]
fn main() {
    eprintln!("error: This example requires the `iir-filtering` and `statistics` features.");
    std::process::exit(1);
}

#[cfg(all(feature = "iir-filtering", feature = "statistics"))]
fn main() -> audio_samples::AudioSampleResult<()> {
    use audio_samples::operations::iir_filtering::SosFilter;
    use audio_samples::operations::types::IirFilterDesign;
    use audio_samples::{nzu, sample_rate, sine_wave};
    use std::time::Duration;

    const BLOCK: usize = 4096;
    let sr = sample_rate!(16000);

    // Stands in for a source that delivers samples a block at a time.
    let source = sine_wave::<f64>(440.0, Duration::from_secs(60), sr, 0.5);
    let samples = source.as_slice().expect("mono audio is contiguous");

    let design = IirFilterDesign::butterworth_lowpass(nzu!(4), 1_000.0);
    let mut filter = SosFilter::from_design(&design, f64::from(sr.get()))?;

    let mut buffer = vec![0.0f64; BLOCK];
    let mut filtered = 0usize;
    let mut loudest = 0.0f64;
    let mut last_block = Vec::new();
    for block in samples.chunks(BLOCK) {
        let out = &mut buffer[..block.len()];
        out.copy_from_slice(block);
        filter.process_block(out);
        let rms = (out.iter().map(|v| v * v).sum::<f64>() / out.len() as f64).sqrt();
        loudest = loudest.max(rms);
        filtered += out.len();
        last_block.clear();
        last_block.extend_from_slice(out);
    }
    println!(
        "{} blocks of {BLOCK} samples",
        samples.len().div_ceil(BLOCK)
    );
    println!("{filtered} samples filtered, highest block RMS {loudest:.4}");

    // The filter keeps its state between blocks, so the blocks join without a
    // seam: the end matches filtering the whole signal in one call.
    let mut whole = samples.to_vec();
    SosFilter::from_design(&design, f64::from(sr.get()))?.process_block(&mut whole);
    let tail = &whole[whole.len() - last_block.len()..];
    assert!(
        tail.iter().zip(&last_block).all(|(a, b)| a == b),
        "block-by-block filtering must equal filtering the whole signal"
    );
    assert_eq!(filtered, samples.len());
    println!("the last block equals the same samples of the whole-signal result");
    Ok(())
}
