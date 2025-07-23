//! Audio source implementations for streaming.

pub mod file;
pub mod generator;

#[cfg(feature = "streaming")]
pub mod tcp;

#[cfg(feature = "streaming")]
pub mod udp;

// Re-export main source types
pub use file::FileStreamSource;
pub use generator::GeneratorSource;

#[cfg(feature = "streaming")]
pub use tcp::TcpStreamSource;

#[cfg(feature = "streaming")]
pub use udp::UdpStreamSource;
