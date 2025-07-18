//! Utility module for applying audio processing operations in batches.
//! Designed to handle multiple audio samples efficiently, leveraging
//! parallel processing and other performance optimizations.
//! 
//! Any operation that can be applied to a single audio sample 
//! can be applied to a batch of samples using this module.