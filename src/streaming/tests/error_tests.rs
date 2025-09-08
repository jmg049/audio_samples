//! Tests for streaming error handling and recovery.

use super::super::error::*;
use std::time::Duration;

#[test]
fn test_stream_error_creation() {
    let error = StreamError::buffer_underrun("Test underrun");
    assert!(error.is_recoverable());
    assert!(!error.is_fatal());
    assert!(error.to_string().contains("underrun"));

    let config_error = StreamError::InvalidConfig("Bad config".to_string());
    assert!(!config_error.is_recoverable());
    assert!(config_error.is_fatal());
}

#[test]
fn test_stream_error_recovery_classification() {
    // Recoverable errors
    assert!(StreamError::buffer_underrun("test").is_recoverable());
    assert!(StreamError::buffer_overrun("test").is_recoverable());
    assert!(
        StreamError::Timeout {
            operation: "test".to_string(),
            duration: Duration::from_millis(100)
        }
        .is_recoverable()
    );

    // Non-recoverable errors
    assert!(!StreamError::InvalidConfig("test".to_string()).is_recoverable());
    assert!(
        !StreamError::FormatMismatch {
            expected: "f32".to_string(),
            actual: "i16".to_string()
        }
        .is_recoverable()
    );
}

#[test]
fn test_stream_error_clone() {
    let original = StreamError::buffer_underrun("original message");
    let cloned = original.clone();

    // Should be functionally equivalent
    assert_eq!(original.is_recoverable(), cloned.is_recoverable());
    assert_eq!(original.to_string(), cloned.to_string());
}

#[test]
fn test_stream_error_from_audio_error() {
    use crate::error::AudioSampleError;

    let audio_error = AudioSampleError::ConversionError(
        "1.5".to_string(),
        "f32".to_string(),
        "i16".to_string(),
        "Out of range".to_string(),
    );

    let stream_error: StreamError = audio_error.into();

    match stream_error {
        StreamError::Audio(_) => {} // Expected
        _ => panic!("Should convert to Audio variant"),
    }
}

#[test]
fn test_stream_error_from_io_error() {
    use std::io::{Error as IoError, ErrorKind};

    let io_error = IoError::new(ErrorKind::ConnectionRefused, "Connection failed");
    let stream_error: StreamError = io_error.into();

    match stream_error {
        StreamError::Network(_) => {} // Expected
        _ => panic!("Should convert to Network variant"),
    }
}

#[test]
fn test_timeout_error_formatting() {
    let timeout_error = StreamError::timeout("connect", Duration::from_millis(5000));
    let error_string = timeout_error.to_string();

    assert!(error_string.contains("connect"));
    assert!(error_string.contains("5000"));
}

#[test]
fn test_protocol_error() {
    let protocol_error = StreamError::protocol("UDP", "Invalid packet format");
    let error_string = protocol_error.to_string();

    assert!(error_string.contains("UDP"));
    assert!(error_string.contains("Invalid packet format"));
}

#[test]
fn test_format_mismatch_error() {
    let format_error = StreamError::format_mismatch("stereo 48kHz", "mono 44.1kHz");

    assert!(!format_error.is_recoverable());

    let error_string = format_error.to_string();
    assert!(error_string.contains("stereo 48kHz"));
    assert!(error_string.contains("mono 44.1kHz"));
}

#[test]
fn test_resource_allocation_error() {
    let resource_error = StreamError::ResourceAllocation {
        resource: "memory",
        reason: "Out of memory".to_string(),
    };

    assert!(!resource_error.is_recoverable());

    let error_string = resource_error.to_string();
    assert!(error_string.contains("memory"));
    assert!(error_string.contains("Out of memory"));
}

#[test]
fn test_connection_error() {
    use std::io::{Error as IoError, ErrorKind};

    let io_error = IoError::new(ErrorKind::TimedOut, "Connection timed out");
    let connection_error = StreamError::Connection {
        operation: "handshake".to_string(),
        source: Box::new(io_error),
    };

    let error_string = connection_error.to_string();
    assert!(error_string.contains("handshake"));

    // Test clone functionality for Connection error
    let cloned_error = connection_error.clone();
    assert!(cloned_error.to_string().contains("handshake"));
}

#[test]
fn test_error_chain() {
    use crate::error::AudioSampleError;

    // Create a chain of errors
    let audio_error = AudioSampleError::ConversionError(
        "2.0".to_string(),
        "f64".to_string(),
        "i16".to_string(),
        "Value too large".to_string(),
    );

    let stream_error = StreamError::Audio(audio_error);

    // Should be able to access the underlying cause
    let error_string = stream_error.to_string();
    assert!(error_string.contains("Audio processing error"));
}

#[test]
fn test_error_debug_formatting() {
    let error = StreamError::InvalidConfig("Invalid sample rate: -1".to_string());
    let debug_string = format!("{:?}", error);

    // Debug format should contain more details
    assert!(debug_string.contains("InvalidConfig"));
    assert!(debug_string.contains("Invalid sample rate: -1"));
}

#[test]
fn test_stream_result_type() {
    // Test that StreamResult works as expected
    fn example_function() -> StreamResult<String> {
        Ok("success".to_string())
    }

    fn failing_function() -> StreamResult<String> {
        Err(StreamError::InvalidConfig("test error".to_string()))
    }

    assert!(example_function().is_ok());
    assert!(failing_function().is_err());

    match failing_function() {
        Ok(_) => panic!("Should have failed"),
        Err(e) => assert!(e.to_string().contains("test error")),
    }
}
