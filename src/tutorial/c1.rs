use candle_core::{DType, Device, Tensor};

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create tensors a and b
    let a = Tensor::arange(0.0, 6.0, &Device::Cpu)?.reshape(&[2, 3])?;
    let b = Tensor::arange(0.0, 12.0, &Device::Cpu)?.reshape(&[3, 4])?;

    // Perform matrix multiplication
    let c = a.matmul(&b)?;

    // Print the result
    println!("Result of matrix multiplication:\n{}", c);

    Ok(())
}