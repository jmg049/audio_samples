//! TCP streaming source for network audio.

use crate::streaming::{
    error::{StreamError, StreamResult},
    traits::{AudioFormatInfo, AudioSource, ByteOrder, SourceMetrics},
};
use crate::{AudioSample, AudioSamples, ConvertTo};
use parking_lot::Mutex;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::{
    io::{AsyncReadExt, BufReader},
    net::TcpStream,
    time::timeout,
};

/// Configuration for TCP streaming source.
#[derive(Debug, Clone)]
pub struct TcpConfig {
    /// Remote server address to connect to
    pub address: SocketAddr,
    /// Connection timeout
    pub connect_timeout: Duration,
    /// Read timeout for each chunk
    pub read_timeout: Duration,
    /// Size of chunks to read from the stream
    pub chunk_size: usize,
    /// Buffer size for the TCP stream
    pub buffer_size: usize,
    /// Whether to automatically reconnect on errors
    pub auto_reconnect: bool,
    /// Maximum reconnection attempts
    pub max_reconnect_attempts: usize,
    /// Time to wait between reconnection attempts
    pub reconnect_delay: Duration,
    /// Expected audio format (if known)
    pub expected_format: Option<AudioFormatInfo>,
}

impl Default for TcpConfig {
    fn default() -> Self {
        Self {
            address: "127.0.0.1:8080".parse().unwrap(),
            connect_timeout: Duration::from_secs(10),
            read_timeout: Duration::from_millis(100),
            chunk_size: 1024,
            buffer_size: 8192,
            auto_reconnect: true,
            max_reconnect_attempts: 5,
            reconnect_delay: Duration::from_millis(1000),
            expected_format: None,
        }
    }
}

impl TcpConfig {
    /// Create configuration for low-latency streaming
    pub fn low_latency(address: SocketAddr) -> Self {
        Self {
            address,
            connect_timeout: Duration::from_secs(5),
            read_timeout: Duration::from_millis(10),
            chunk_size: 256,
            buffer_size: 2048,
            auto_reconnect: true,
            max_reconnect_attempts: 10,
            reconnect_delay: Duration::from_millis(100),
            expected_format: None,
        }
    }

    /// Create configuration for high-quality streaming
    pub fn high_quality(address: SocketAddr) -> Self {
        Self {
            address,
            connect_timeout: Duration::from_secs(15),
            read_timeout: Duration::from_millis(500),
            chunk_size: 4096,
            buffer_size: 16384,
            auto_reconnect: true,
            max_reconnect_attempts: 3,
            reconnect_delay: Duration::from_millis(2000),
            expected_format: None,
        }
    }
}

/// State of the TCP connection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TcpState {
    Disconnected,
    Connecting,
    Connected,
    Error,
}

/// A streaming audio source that reads from TCP network connections.
///
/// This source connects to a TCP server and reads raw audio data,
/// handling connection errors and automatic reconnection.
pub struct TcpStreamSource<T: AudioSample> {
    config: TcpConfig,
    state: TcpState,

    connection: Option<BufReader<TcpStream>>,

    format_info: AudioFormatInfo,
    metrics: Arc<Mutex<SourceMetrics>>,

    // Connection management
    connection_time: Option<Instant>,
    reconnect_attempts: usize,
    last_reconnect_time: Option<Instant>,

    // Data buffer for partial reads
    data_buffer: Vec<u8>,
    is_active: bool,

    phantom: std::marker::PhantomData<T>,
}

impl<T: AudioSample> TcpStreamSource<T>
where
    i16: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    /// Create a new TCP streaming source.
    pub fn new(config: TcpConfig) -> Self {
        let format_info = config.expected_format.clone().unwrap_or_else(|| {
            // Default format if none specified
            AudioFormatInfo {
                sample_rate: 44100,
                channels: 2,
                sample_format: std::any::type_name::<T>().to_string(),
                bits_per_sample: T::BITS as u8,
                is_signed: true,
                is_float: T::BITS == 32 && std::any::type_name::<T>().contains("f32")
                    || T::BITS == 64,
                byte_order: ByteOrder::Native,
            }
        });

        Self {
            config,
            state: TcpState::Disconnected,
            connection: None,
            format_info,
            metrics: Arc::new(Mutex::new(SourceMetrics::default())),
            connection_time: None,
            reconnect_attempts: 0,
            last_reconnect_time: None,
            data_buffer: Vec::new(),
            is_active: true,
            phantom: std::marker::PhantomData,
        }
    }

    /// Create a TCP source with default configuration for the given address.
    pub fn with_address(address: SocketAddr) -> Self {
        let mut config = TcpConfig::default();
        config.address = address;
        Self::new(config)
    }

    /// Create a low-latency TCP source.
    pub fn low_latency(address: SocketAddr) -> Self {
        Self::new(TcpConfig::low_latency(address))
    }

    /// Create a high-quality TCP source.
    pub fn high_quality(address: SocketAddr) -> Self {
        Self::new(TcpConfig::high_quality(address))
    }

    /// Set the expected audio format for the stream.
    pub fn set_format(&mut self, format: AudioFormatInfo) {
        self.format_info = format;
        if self.config.expected_format.is_none() {
            self.config.expected_format = Some(self.format_info.clone());
        }
    }

    /// Get the current connection state.
    pub fn connection_state(&self) -> TcpState {
        self.state
    }

    /// Get the number of reconnection attempts made.
    pub fn reconnection_attempts(&self) -> usize {
        self.reconnect_attempts
    }

    /// Establish connection to the TCP server.
    async fn connect(&mut self) -> StreamResult<()> {
        if matches!(self.state, TcpState::Connected) {
            return Ok(());
        }

        self.state = TcpState::Connecting;

        let connection_result = timeout(
            self.config.connect_timeout,
            TcpStream::connect(&self.config.address),
        )
        .await;

        match connection_result {
            Ok(Ok(stream)) => {
                // Configure the stream for optimal performance
                if let Err(e) = stream.set_nodelay(true) {
                    return Err(StreamError::Connection {
                        operation: "set_nodelay".to_string(),
                        source: Box::new(e),
                    });
                }

                let buf_reader = BufReader::with_capacity(self.config.buffer_size, stream);
                self.connection = Some(buf_reader);
                self.state = TcpState::Connected;
                self.connection_time = Some(Instant::now());
                self.reconnect_attempts = 0;

                Ok(())
            }
            Ok(Err(e)) => {
                self.state = TcpState::Error;
                Err(StreamError::Connection {
                    operation: "connect".to_string(),
                    source: Box::new(e),
                })
            }
            Err(_) => {
                self.state = TcpState::Error;
                Err(StreamError::Timeout {
                    operation: "connect".to_string(),
                    duration: self.config.connect_timeout,
                })
            }
        }
    }

    /// Read raw data from the TCP stream.
    async fn read_raw_data(&mut self, buf: &mut [u8]) -> StreamResult<usize> {
        if self.connection.is_none() {
            self.connect().await?;
        }

        let connection = self.connection.as_mut().unwrap();

        let read_result = timeout(self.config.read_timeout, connection.read(buf)).await;

        match read_result {
            Ok(Ok(bytes_read)) => {
                if bytes_read == 0 {
                    // Connection closed by remote
                    self.state = TcpState::Disconnected;
                    self.connection = None;
                    return Err(StreamError::Connection {
                        operation: "read".to_string(),
                        source: Box::new(std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            "Connection closed by remote",
                        )),
                    });
                }
                Ok(bytes_read)
            }
            Ok(Err(e)) => {
                self.state = TcpState::Error;
                self.connection = None;
                Err(StreamError::Connection {
                    operation: "read".to_string(),
                    source: Box::new(e),
                })
            }
            Err(_) => Err(StreamError::Timeout {
                operation: "read".to_string(),
                duration: self.config.read_timeout,
            }),
        }
    }

    /// Attempt to reconnect if enabled and within retry limits.
    async fn handle_connection_error(&mut self, error: &StreamError) -> StreamResult<()> {
        if !self.config.auto_reconnect {
            return Err(error.clone());
        }

        if self.reconnect_attempts >= self.config.max_reconnect_attempts {
            self.is_active = false;
            return Err(StreamError::Connection {
                operation: "reconnect".to_string(),
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    format!(
                        "Max reconnection attempts ({}) exceeded",
                        self.config.max_reconnect_attempts
                    ),
                )),
            });
        }

        // Wait before reconnecting if needed
        if let Some(last_attempt) = self.last_reconnect_time {
            let elapsed = last_attempt.elapsed();
            if elapsed < self.config.reconnect_delay {
                #[cfg(feature = "streaming")]
                tokio::time::sleep(self.config.reconnect_delay - elapsed).await;
            }
        }

        self.reconnect_attempts += 1;
        self.last_reconnect_time = Some(Instant::now());
        self.state = TcpState::Disconnected;
        self.connection = None;

        // Update metrics
        {
            let mut metrics = self.metrics.lock();
            metrics.underruns += 1;
        }

        // Try to reconnect
        self.connect().await
    }

    /// Convert raw bytes to audio samples.
    fn bytes_to_samples(&self, raw_data: &[u8]) -> StreamResult<Vec<T>> {
        let sample_size = std::mem::size_of::<T>();
        let num_samples = raw_data.len() / sample_size;

        if raw_data.len() % sample_size != 0 {
            return Err(StreamError::InvalidConfig(format!(
                "Data length {} not divisible by sample size {}",
                raw_data.len(),
                sample_size
            )));
        }

        let mut samples = Vec::with_capacity(num_samples);

        // Convert bytes to samples based on type
        for chunk in raw_data.chunks_exact(sample_size) {
            let sample = match T::BITS {
                16 => {
                    let bytes: [u8; 2] = chunk.try_into().map_err(|_| {
                        StreamError::InvalidConfig("Invalid 16-bit sample data".to_string())
                    })?;
                    let value = i16::from_ne_bytes(bytes);
                    value.convert_to().map_err(StreamError::Audio)?
                }
                32 => {
                    let bytes: [u8; 4] = chunk.try_into().map_err(|_| {
                        StreamError::InvalidConfig("Invalid 32-bit sample data".to_string())
                    })?;

                    if std::any::type_name::<T>().contains("f32") {
                        let value = f32::from_ne_bytes(bytes);
                        T::cast_from(value)
                    } else {
                        let value = i32::from_ne_bytes(bytes);
                        value.convert_to().map_err(StreamError::Audio)?
                    }
                }
                64 => {
                    let bytes: [u8; 8] = chunk.try_into().map_err(|_| {
                        StreamError::InvalidConfig("Invalid 64-bit sample data".to_string())
                    })?;
                    let value = f64::from_ne_bytes(bytes);
                    value.convert_to().map_err(StreamError::Audio)?
                }
                _ => {
                    return Err(StreamError::InvalidConfig(format!(
                        "Unsupported sample bit width: {}",
                        T::BITS
                    )));
                }
            };
            samples.push(sample);
        }

        Ok(samples)
    }
}

impl<T: AudioSample> AudioSource<T> for TcpStreamSource<T>
where
    i16: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    async fn next_chunk(&mut self) -> StreamResult<Option<AudioSamples<T>>> {
        if !self.is_active {
            return Ok(None);
        }

        let channels = self.format_info.channels;
        let chunk_samples = self.config.chunk_size;
        let sample_size = std::mem::size_of::<T>();
        let bytes_needed = chunk_samples * channels * sample_size;

        // Ensure we have enough buffered data
        while self.data_buffer.len() < bytes_needed {
            let mut temp_buffer = vec![0u8; self.config.buffer_size];

            match self.read_raw_data(&mut temp_buffer).await {
                Ok(bytes_read) => {
                    temp_buffer.truncate(bytes_read);
                    self.data_buffer.extend_from_slice(&temp_buffer);

                    // Update metrics
                    {
                        let mut metrics = self.metrics.lock();
                        metrics.bytes_delivered += bytes_read as u64;
                    }
                }
                Err(e) => {
                    // Try to handle the error with reconnection
                    if let Err(reconnect_error) = self.handle_connection_error(&e).await {
                        return Err(reconnect_error);
                    }
                    continue; // Try reading again after reconnect
                }
            }
        }

        // Extract the needed bytes from our buffer
        let chunk_bytes: Vec<u8> = self.data_buffer.drain(..bytes_needed).collect();

        // Convert bytes to samples
        let samples = self.bytes_to_samples(&chunk_bytes)?;

        // Create AudioSamples from the data
        let array = ndarray::Array2::from_shape_vec((chunk_samples, channels), samples)
            .map_err(|e| StreamError::InvalidConfig(e.to_string()))?;

        let audio_samples =
            AudioSamples::new_multi_channel(array, self.format_info.sample_rate as u32);

        // Update metrics
        {
            let mut metrics = self.metrics.lock();
            metrics.chunks_delivered += 1;
            metrics.average_chunk_size = (metrics.average_chunk_size + chunk_samples) / 2.max(1);
            metrics.current_buffer_level =
                self.data_buffer.len() as f64 / self.config.buffer_size as f64;
        }

        Ok(Some(audio_samples))
    }

    fn format_info(&self) -> AudioFormatInfo {
        self.format_info.clone()
    }

    fn is_active(&self) -> bool {
        self.is_active && !matches!(self.state, TcpState::Error)
    }

    async fn seek(&mut self, _position: Duration) -> StreamResult<Duration> {
        Err(StreamError::InvalidConfig(
            "Seeking not supported for TCP streams".to_string(),
        ))
    }

    fn duration(&self) -> Option<Duration> {
        None // TCP streams have unknown/infinite duration
    }

    fn position(&self) -> Option<Duration> {
        if let Some(start_time) = self.connection_time {
            Some(start_time.elapsed())
        } else {
            None
        }
    }

    fn metrics(&self) -> SourceMetrics {
        self.metrics.lock().clone()
    }

    fn set_buffer_size(&mut self, size: usize) {
        self.config.chunk_size = size;
        self.config.buffer_size = size * 8; // Keep 8x buffer for network buffering
    }
}
