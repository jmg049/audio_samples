//! UDP streaming source for network audio with packet management.

use crate::streaming::{
    error::{StreamError, StreamResult},
    traits::{AudioFormatInfo, AudioSource, ByteOrder, SourceMetrics},
};
use crate::{AudioSample, AudioSamples, ConvertTo};
use parking_lot::Mutex;
use std::collections::{BTreeMap, HashMap};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(feature = "streaming")]
use tokio::{net::UdpSocket, time::timeout};

/// Configuration for UDP streaming source.
#[derive(Debug, Clone)]
pub struct UdpConfig {
    /// Local address to bind to
    pub local_address: SocketAddr,
    /// Remote server address (for connected UDP)
    pub remote_address: Option<SocketAddr>,
    /// Whether to use multicast
    pub multicast: bool,
    /// Multicast group address (if using multicast)
    pub multicast_group: Option<SocketAddr>,
    /// Read timeout for each packet
    pub read_timeout: Duration,
    /// Size of chunks to assemble from packets
    pub chunk_size: usize,
    /// Maximum UDP packet size
    pub max_packet_size: usize,
    /// Whether to handle out-of-order packets
    pub handle_out_of_order: bool,
    /// Jitter buffer size (in packets)
    pub jitter_buffer_size: usize,
    /// Maximum time to wait for missing packets
    pub packet_timeout: Duration,
    /// Expected audio format (if known)
    pub expected_format: Option<AudioFormatInfo>,
}

impl Default for UdpConfig {
    fn default() -> Self {
        Self {
            local_address: "0.0.0.0:0".parse().unwrap(),
            remote_address: None,
            multicast: false,
            multicast_group: None,
            read_timeout: Duration::from_millis(50),
            chunk_size: 1024,
            max_packet_size: 1472, // Standard MTU minus headers
            handle_out_of_order: true,
            jitter_buffer_size: 10,
            packet_timeout: Duration::from_millis(100),
            expected_format: None,
        }
    }
}

impl UdpConfig {
    /// Create configuration for low-latency streaming
    pub fn low_latency(local_address: SocketAddr) -> Self {
        Self {
            local_address,
            read_timeout: Duration::from_millis(5),
            chunk_size: 256,
            max_packet_size: 512,
            handle_out_of_order: false, // Reduce latency
            jitter_buffer_size: 3,
            packet_timeout: Duration::from_millis(10),
            ..Default::default()
        }
    }

    /// Create configuration for high-quality streaming
    pub fn high_quality(local_address: SocketAddr) -> Self {
        Self {
            local_address,
            read_timeout: Duration::from_millis(100),
            chunk_size: 2048,
            max_packet_size: 1472,
            handle_out_of_order: true,
            jitter_buffer_size: 20,
            packet_timeout: Duration::from_millis(200),
            ..Default::default()
        }
    }

    /// Create multicast configuration
    pub fn multicast(local_address: SocketAddr, group: SocketAddr) -> Self {
        Self {
            local_address,
            multicast: true,
            multicast_group: Some(group),
            handle_out_of_order: true,
            jitter_buffer_size: 15,
            ..Default::default()
        }
    }
}

/// Audio packet with sequence information
#[derive(Debug, Clone)]
struct AudioPacket {
    sequence: u32,
    timestamp: u64,
    data: Vec<u8>,
    received_at: Instant,
}

impl AudioPacket {
    fn new(sequence: u32, timestamp: u64, data: Vec<u8>) -> Self {
        Self {
            sequence,
            timestamp,
            data,
            received_at: Instant::now(),
        }
    }
}

/// Manages packet ordering and reassembly
struct PacketBuffer {
    packets: BTreeMap<u32, AudioPacket>,
    next_expected_sequence: u32,
    max_buffer_size: usize,
    packet_timeout: Duration,
}

impl PacketBuffer {
    fn new(max_size: usize, timeout: Duration) -> Self {
        Self {
            packets: BTreeMap::new(),
            next_expected_sequence: 0,
            max_buffer_size: max_size,
            packet_timeout: timeout,
        }
    }

    /// Add a packet to the buffer
    fn add_packet(&mut self, packet: AudioPacket) -> bool {
        let sequence = packet.sequence;

        // Remove old packets to prevent buffer overflow
        if self.packets.len() >= self.max_buffer_size {
            self.cleanup_old_packets();
        }

        // Don't add duplicate or very old packets
        if self.packets.contains_key(&sequence)
            || sequence + (self.max_buffer_size as u32) < self.next_expected_sequence
        {
            return false;
        }

        self.packets.insert(sequence, packet);
        true
    }

    /// Try to get the next contiguous chunk of data
    fn get_next_chunk(&mut self, target_size: usize) -> Option<Vec<u8>> {
        let mut data = Vec::new();
        let mut sequences_to_remove = Vec::new();

        // Collect contiguous packets starting from next_expected_sequence
        while data.len() < target_size {
            if let Some(packet) = self.packets.get(&self.next_expected_sequence) {
                data.extend_from_slice(&packet.data);
                sequences_to_remove.push(self.next_expected_sequence);
                self.next_expected_sequence += 1;
            } else {
                // Check if we should wait or if packet is likely lost
                let now = Instant::now();
                let should_skip = self
                    .packets
                    .iter()
                    .find(|(seq, _)| **seq > self.next_expected_sequence)
                    .map(|(_, packet)| now.duration_since(packet.received_at) > self.packet_timeout)
                    .unwrap_or(false);

                if should_skip {
                    // Skip the missing packet
                    self.next_expected_sequence += 1;
                    continue;
                }
                break; // Wait for more packets
            }
        }

        // Remove consumed packets
        for seq in sequences_to_remove {
            self.packets.remove(&seq);
        }

        if data.is_empty() { None } else { Some(data) }
    }

    /// Remove packets that have timed out
    fn cleanup_old_packets(&mut self) {
        let now = Instant::now();
        let timeout = self.packet_timeout;

        self.packets
            .retain(|_, packet| now.duration_since(packet.received_at) <= timeout);
    }

    /// Get buffer statistics
    fn buffer_stats(&self) -> (usize, u32) {
        let buffer_size = self.packets.len();
        let missing_count = if let Some((&max_seq, _)) = self.packets.iter().next_back() {
            max_seq
                .saturating_sub(self.next_expected_sequence)
                .saturating_sub(buffer_size as u32 - 1)
        } else {
            0
        };
        (buffer_size, missing_count)
    }
}

/// State of the UDP connection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UdpState {
    Unbound,
    Bound,
    Error,
}

/// A streaming audio source that reads from UDP network packets.
///
/// This source binds to a UDP port and handles packet reassembly,
/// out-of-order delivery, and packet loss for audio streaming.
pub struct UdpStreamSource<T: AudioSample> {
    config: UdpConfig,
    state: UdpState,

    #[cfg(feature = "streaming")]
    socket: Option<UdpSocket>,

    #[cfg(not(feature = "streaming"))]
    socket: Option<()>,

    format_info: AudioFormatInfo,
    metrics: Arc<Mutex<SourceMetrics>>,

    // Packet management
    packet_buffer: PacketBuffer,
    data_buffer: Vec<u8>,

    // Statistics
    packets_received: u64,
    packets_lost: u64,
    packets_out_of_order: u64,
    bytes_received: u64,

    is_active: bool,
    start_time: Option<Instant>,

    phantom: std::marker::PhantomData<T>,
}

impl<T: AudioSample> UdpStreamSource<T> {
    /// Create a new UDP streaming source.
    pub fn new(config: UdpConfig) -> Self {
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

        let packet_buffer = PacketBuffer::new(config.jitter_buffer_size, config.packet_timeout);

        Self {
            config,
            state: UdpState::Unbound,
            socket: None,
            format_info,
            metrics: Arc::new(Mutex::new(SourceMetrics::default())),
            packet_buffer,
            data_buffer: Vec::new(),
            packets_received: 0,
            packets_lost: 0,
            packets_out_of_order: 0,
            bytes_received: 0,
            is_active: true,
            start_time: None,
            phantom: std::marker::PhantomData,
        }
    }

    /// Create a UDP source with default configuration for the given address.
    pub fn with_address(local_address: SocketAddr) -> Self {
        let mut config = UdpConfig::default();
        config.local_address = local_address;
        Self::new(config)
    }

    /// Create a low-latency UDP source.
    pub fn low_latency(local_address: SocketAddr) -> Self {
        Self::new(UdpConfig::low_latency(local_address))
    }

    /// Create a high-quality UDP source.
    pub fn high_quality(local_address: SocketAddr) -> Self {
        Self::new(UdpConfig::high_quality(local_address))
    }

    /// Create a multicast UDP source.
    pub fn multicast(local_address: SocketAddr, group: SocketAddr) -> Self {
        Self::new(UdpConfig::multicast(local_address, group))
    }

    /// Set the expected audio format for the stream.
    pub fn set_format(&mut self, format: AudioFormatInfo) {
        self.format_info = format;
        if self.config.expected_format.is_none() {
            self.config.expected_format = Some(self.format_info.clone());
        }
    }

    /// Get the current connection state.
    pub fn connection_state(&self) -> UdpState {
        self.state
    }

    /// Get packet statistics.
    pub fn packet_stats(&self) -> (u64, u64, u64) {
        (
            self.packets_received,
            self.packets_lost,
            self.packets_out_of_order,
        )
    }

    #[cfg(feature = "streaming")]
    /// Bind the UDP socket and setup multicast if needed.
    async fn bind_socket(&mut self) -> StreamResult<()> {
        if matches!(self.state, UdpState::Bound) {
            return Ok(());
        }

        let socket = UdpSocket::bind(&self.config.local_address)
            .await
            .map_err(|e| StreamError::Connection {
                operation: "bind".to_string(),
                source: Box::new(e),
            })?;

        // Setup multicast if configured
        if self.config.multicast {
            if let Some(group) = self.config.multicast_group {
                socket
                    .join_multicast_v4(
                        group.ip().try_into().map_err(|_| {
                            StreamError::InvalidConfig(
                                "Invalid multicast group address".to_string(),
                            )
                        })?,
                        self.config.local_address.ip().try_into().map_err(|_| {
                            StreamError::InvalidConfig(
                                "Invalid local address for multicast".to_string(),
                            )
                        })?,
                    )
                    .map_err(|e| StreamError::Connection {
                        operation: "join_multicast".to_string(),
                        source: Box::new(e),
                    })?;
            }
        }

        // Connect to remote if specified
        if let Some(remote) = self.config.remote_address {
            socket
                .connect(remote)
                .await
                .map_err(|e| StreamError::Connection {
                    operation: "connect".to_string(),
                    source: Box::new(e),
                })?;
        }

        self.socket = Some(socket);
        self.state = UdpState::Bound;
        self.start_time = Some(Instant::now());

        Ok(())
    }

    #[cfg(not(feature = "streaming"))]
    /// Bind the UDP socket (no-op without streaming feature).
    async fn bind_socket(&mut self) -> StreamResult<()> {
        Err(StreamError::InvalidConfig(
            "UDP streaming requires 'streaming' feature".to_string(),
        ))
    }

    #[cfg(feature = "streaming")]
    /// Receive and parse a UDP packet.
    async fn receive_packet(&mut self) -> StreamResult<Option<AudioPacket>> {
        if self.socket.is_none() {
            self.bind_socket().await?;
        }

        let socket = self.socket.as_ref().unwrap();
        let mut buffer = vec![0u8; self.config.max_packet_size];

        let recv_result = timeout(self.config.read_timeout, socket.recv(&mut buffer)).await;

        match recv_result {
            Ok(Ok(bytes_received)) => {
                buffer.truncate(bytes_received);
                self.bytes_received += bytes_received as u64;
                self.packets_received += 1;

                // Simple packet format: [sequence:4][timestamp:8][data...]
                if buffer.len() < 12 {
                    return Err(StreamError::InvalidConfig(
                        "Packet too small for header".to_string(),
                    ));
                }

                let sequence = u32::from_ne_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]);
                let timestamp = u64::from_ne_bytes([
                    buffer[4], buffer[5], buffer[6], buffer[7], buffer[8], buffer[9], buffer[10],
                    buffer[11],
                ]);
                let data = buffer[12..].to_vec();

                let packet = AudioPacket::new(sequence, timestamp, data);
                Ok(Some(packet))
            }
            Ok(Err(e)) => {
                self.state = UdpState::Error;
                Err(StreamError::Connection {
                    operation: "recv".to_string(),
                    source: Box::new(e),
                })
            }
            Err(_) => {
                // Timeout is expected for UDP - not necessarily an error
                Ok(None)
            }
        }
    }

    #[cfg(not(feature = "streaming"))]
    /// Receive and parse a UDP packet (no-op without streaming feature).
    async fn receive_packet(&mut self) -> StreamResult<Option<AudioPacket>> {
        Err(StreamError::InvalidConfig(
            "UDP streaming requires 'streaming' feature".to_string(),
        ))
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
                    value.convert_to::<T>().map_err(StreamError::Audio)?
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
                        value.convert_to::<T>().map_err(StreamError::Audio)?
                    }
                }
                64 => {
                    let bytes: [u8; 8] = chunk.try_into().map_err(|_| {
                        StreamError::InvalidConfig("Invalid 64-bit sample data".to_string())
                    })?;
                    let value = f64::from_ne_bytes(bytes);
                    value.convert_to::<T>().map_err(StreamError::Audio)?
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

impl<T: AudioSample> AudioSource<T> for UdpStreamSource<T> {
    async fn next_chunk(&mut self) -> StreamResult<Option<AudioSamples<T>>> {
        if !self.is_active {
            return Ok(None);
        }

        let channels = self.format_info.channels;
        let chunk_samples = self.config.chunk_size;
        let sample_size = std::mem::size_of::<T>();
        let bytes_needed = chunk_samples * channels * sample_size;

        // Try to receive packets and build up enough data
        while self.data_buffer.len() < bytes_needed {
            // Try to get data from packet buffer first
            if let Some(packet_data) = self
                .packet_buffer
                .get_next_chunk(bytes_needed - self.data_buffer.len())
            {
                self.data_buffer.extend_from_slice(&packet_data);
                continue;
            }

            // Try to receive more packets
            match self.receive_packet().await? {
                Some(packet) => {
                    if self.config.handle_out_of_order {
                        // Add to packet buffer for ordering
                        if self.packet_buffer.add_packet(packet) {
                            // Packet was successfully added
                        } else {
                            // Duplicate or very old packet
                            self.packets_lost += 1;
                        }
                    } else {
                        // Low-latency mode: use packets immediately
                        self.data_buffer.extend_from_slice(&packet.data);
                    }
                }
                None => {
                    // Timeout - check if we have any data to work with
                    if self.data_buffer.is_empty() {
                        // Update metrics for underrun
                        {
                            let mut metrics = self.metrics.lock();
                            metrics.underruns += 1;
                        }
                        continue;
                    } else {
                        // Use what we have, even if incomplete
                        break;
                    }
                }
            }

            // Clean up old packets periodically
            self.packet_buffer.cleanup_old_packets();
        }

        if self.data_buffer.is_empty() {
            return Ok(None);
        }

        // Take what we can get, up to the target size
        let available_bytes = self.data_buffer.len().min(bytes_needed);
        let available_samples = available_bytes / (sample_size * channels) * channels;
        let chunk_bytes: Vec<u8> = self
            .data_buffer
            .drain(..available_samples * sample_size)
            .collect();

        // Convert bytes to samples
        let samples = self.bytes_to_samples(&chunk_bytes)?;
        let actual_chunk_samples = samples.len() / channels;

        // Create AudioSamples from the data
        let array = ndarray::Array2::from_shape_vec((actual_chunk_samples, channels), samples)
            .map_err(|e| StreamError::InvalidConfig(e.to_string()))?;

        let audio_samples =
            AudioSamples::new(array, self.format_info.sample_rate).map_err(StreamError::Audio)?;

        // Update metrics
        {
            let mut metrics = self.metrics.lock();
            metrics.chunks_delivered += 1;
            metrics.bytes_delivered += (actual_chunk_samples * channels * sample_size) as u64;
            metrics.average_chunk_size =
                (metrics.average_chunk_size + actual_chunk_samples) / 2.max(1);

            let (buffer_size, missing_packets) = self.packet_buffer.buffer_stats();
            metrics.current_buffer_level =
                buffer_size as f64 / self.config.jitter_buffer_size as f64;

            // Update loss statistics
            if missing_packets > 0 {
                metrics.chunks_dropped += missing_packets as u64;
            }
        }

        Ok(Some(audio_samples))
    }

    fn format_info(&self) -> AudioFormatInfo {
        self.format_info.clone()
    }

    fn is_active(&self) -> bool {
        self.is_active && !matches!(self.state, UdpState::Error)
    }

    async fn seek(&mut self, _position: Duration) -> StreamResult<Duration> {
        Err(StreamError::InvalidConfig(
            "Seeking not supported for UDP streams".to_string(),
        ))
    }

    fn duration(&self) -> Option<Duration> {
        None // UDP streams have unknown/infinite duration
    }

    fn position(&self) -> Option<Duration> {
        if let Some(start_time) = self.start_time {
            Some(start_time.elapsed())
        } else {
            None
        }
    }

    fn metrics(&self) -> SourceMetrics {
        let mut metrics = self.metrics.lock().clone();

        // Add UDP-specific metrics
        if self.packets_received > 0 {
            let loss_rate = self.packets_lost as f64 / self.packets_received as f64;
            // Store loss rate in overruns field as a percentage
            metrics.overruns = (loss_rate * 100.0) as u64;
        }

        metrics
    }

    fn set_buffer_size(&mut self, size: usize) {
        self.config.chunk_size = size;
        self.config.jitter_buffer_size = (size / 256).max(3); // Adjust jitter buffer
    }
}
