//! Utility functions for audio processing.
//!
//! This module provides a collection of utility functions that make common
//! audio processing tasks more convenient and intuitive.
//!
//! # Modules
//!
//! - [`comparison`] - Audio comparison and similarity utilities
//! - [`detection`] - Format detection and audio analysis utilities
//! - [`generation`] - Audio signal generation utilities

pub mod comparison;
pub mod detection;
pub mod generation;

// Re-export common utilities
pub use comparison::*;
pub use detection::*;
pub use generation::*;

// Re-export existing utils from the parent module
// Note: Cannot glob-import a module into itself, so we'll define the compatibility exports individually
// pub use super::utils_old::*;
