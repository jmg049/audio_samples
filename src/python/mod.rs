use crate::{operations::AudioSamplesOperations, AudioSample};
use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2};
mod operations;

/// Python wrapper for the [crate::ArraySamples] struct. 
/// Contains the a GIL-independent reference to numpy arrays stored on the python heap.
pub enum PyAudioSamples<T: AudioSample> {
    Mono(Py<PyArray1<T>>),
    MultiChannel(Py<PyArray2<T>>),
}

impl<T: AudioSample> AudioSamplesOperations<T> for PyAudioSamples<T> {
    fn normalize<T: AudioSample>(&mut self, min: T, max: T, method: Option<NormalizationMethod>) -> () {
        todo!()
    }

    fn scale<T: AudioSample>(&mut self, factor: T) -> () {
        todo!()
    }

    fn as_type<O: AudioSample>(&self) -> AudioSamples<O> {
        todo!()
    }

    fn to_type<O: AudioSample>(self) -> AudioSamples<O> {
        todo!()
    }

    fn peak<T: AudioSample>(&self) -> T {
        todo!()
    }

    fn rms<T: AudioSample>(&self) -> T {
        todo!()
    }

    fn min<T: AudioSample>(&self) -> T {
        todo!()
    }

    fn max<T: AudioSample>(&self) -> T {
        todo!()
    }

    fn variance<T: AudioSample>(&self) -> T {
        todo!()
    }

    fn zero_crossing_rate(&self) -> f64 {
        todo!()
    }

    fn cross_correlation<T: AudioSample>(&self, other: &Self, lag: usize) -> T {
        todo!()
    }

    fn autocorrelation<T: AudioSample>(&self, lag: usize) -> T {
        todo!()
    }

    fn zero_crossings<T: AudioSample>(&self) -> usize {
        todo!()
    }

    fn mu_compress<T: AudioSample>(&mut self, mu: T) -> () {
        todo!()
    }

    fn mu_expand<T: AudioSample>(&mut self, mu: T) -> () {
        todo!()
    }

    fn stft() {
        todo!()
    }

    fn istft() {
        todo!()
    }

    fn fft() {
        todo!()
    }

    fn ifft() {
        todo!()
    }

    fn spectrogram() {
        todo!()
    }

    fn mel_spectrogram() {
        todo!()
    }

    fn mfcc() {
        todo!()
    }

    fn chroma() {
        todo!()
    }

    fn gammatone_spectrogram() {
        todo!()
    }

    fn apply_window<T: AudioSample>(&mut self, window: &[T]) -> () {
        todo!()
    }

    fn apply_filter<T: AudioSample>(&mut self, filter: &[T]) -> () {
        todo!()
    }

    fn vad(&self, threshold: f64) -> AudioSamples<bool> {
        todo!()
    }

    fn resample<T: AudioSample>(&self, new_sample_rate: usize) -> AudioSamples<T> {
        todo!()
    }

    fn time_stretch<T: AudioSample>(&self, factor: f64) -> AudioSamples<T> {
        todo!()
    }

    fn pitch_shift<T: AudioSample>(&self, semitones: f64) -> AudioSamples<T> {
        todo!()
    }

    fn fade_in<T: AudioSample>(&mut self, duration: f64) -> () {
        todo!()
    }

    fn fade_out<T: AudioSample>(&mut self, duration: f64) -> () {
        todo!()
    }

    fn reverse<T: AudioSample>(&self) -> AudioSamples<T> {
        todo!()
    }

    fn trim<T: AudioSample>(&mut self, start: f64, end: f64) -> () {
        todo!()
    }

    fn pad<T: AudioSample>(&mut self, duration: f64, pad_value: T) -> () {
        todo!()
    }

    fn split<T: AudioSample>(&self, duration: f64) -> Vec<AudioSamples<T>> {
        todo!()
    }

    fn concatenate<T: AudioSample>(&self, others: &[AudioSamples<T>]) -> AudioSamples<T> {
        todo!()
    }

    fn mix<T: AudioSample>(&self, others: &[AudioSamples<T>]) -> AudioSamples<T> {
        todo!()
    }

    fn mono<T: AudioSample>(&self) -> AudioSamples<T> {
        todo!()
    }

    fn stereo<T: AudioSample>(&self) -> AudioSamples<T> {
        todo!()
    }

    fn into_channels<T: AudioSample>(&self, channels: usize) -> AudioSamples<T> {
        todo!()
    }
}