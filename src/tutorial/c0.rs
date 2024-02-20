use candle_core::{DType, Device, Tensor};
const WIDTH:usize= 784;
const DATASIZE:usize= 10000;

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create tensors a and b
    let a = Tensor::arange(0.0, 6.0, &Device::Cpu)?.reshape(&[2, 3])?;//0..6?
    let b = Tensor::arange(0.0, 12.0, &Device::Cpu)?.reshape(&[3, 4])?;//0..12?

    //행렬곱 
    let c = a.matmul(&b)?;
    println!("Result of matrix multiplication:\n{}", c);

    Ok(())
}