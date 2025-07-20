use audio_samples::*;
use approx_eq::assert_approx_eq;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_from_basic() {
        // Test i16::convert_from with f32 source
        let f32_source: f32 = 0.5;
        let i16_result: i16 = i16::convert_from(f32_source).unwrap();
        assert_eq!(i16_result, 16384); // 0.5 * 32767 rounded

        // Test f32::convert_from with i16 source
        let i16_source: i16 = 16384;
        let f32_result: f32 = f32::convert_from(i16_source).unwrap();
        assert_approx_eq!(f32_result as f64, 0.5, 1e-4);

        // Test with zero values
        let zero_f32: f32 = 0.0;
        let zero_i16: i16 = i16::convert_from(zero_f32).unwrap();
        assert_eq!(zero_i16, 0);

        let zero_i16_source: i16 = 0;
        let zero_f32_result: f32 = f32::convert_from(zero_i16_source).unwrap();
        assert_approx_eq!(zero_f32_result as f64, 0.0, 1e-10);

        println!("All convert_from tests passed!");
    }

    #[test]
    fn test_convert_from_vs_convert_to() {
        // Verify that convert_from gives the same result as convert_to
        let i16_value: i16 = 12345;
        
        // Using convert_to
        let f32_from_convert_to: f32 = i16_value.convert_to().unwrap();
        
        // Using convert_from
        let f32_from_convert_from: f32 = f32::convert_from(i16_value).unwrap();
        
        assert_eq!(f32_from_convert_to, f32_from_convert_from);
        
        println!("convert_from and convert_to produce identical results!");
    }
}

fn main() {
    println!("Testing convert_from functionality...");
    
    // Basic usage examples
    let i16_val: i16 = 1000;
    let f32_result: f32 = f32::convert_from(i16_val).unwrap();
    println!("Converted i16 {} to f32 {}", i16_val, f32_result);
    
    let f32_val: f32 = 0.75;
    let i16_result: i16 = i16::convert_from(f32_val).unwrap();
    println!("Converted f32 {} to i16 {}", f32_val, i16_result);
    
    // Test with different types
    let i32_val: i32 = 1000000;
    let f64_result: f64 = f64::convert_from(i32_val).unwrap();
    println!("Converted i32 {} to f64 {}", i32_val, f64_result);
    
    println!("All manual tests completed successfully!");
}