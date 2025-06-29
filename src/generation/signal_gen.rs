use std::time::Duration;

// Signal generation | PLACEHOLDERS - RIGHT FUNCTIONS, PROBABLY WRONG SIGNATURES

pub fn clicks(duration_seconds: Duration, sample_rate: usize);
pub fn tone(frequency: f64, duration_seconds: Duration, sample_rate: usize) -> Vec<f64>;
pub fn chirp(
    start_frequency: f64,
    end_frequency: f64,
    duration_seconds: Duration,
    sample_rate: usize,
) -> Vec<f64>;