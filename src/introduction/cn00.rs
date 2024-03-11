use candle_core::{Tensor,Device};

pub fn main()->anyhow::Result<()>{
    let device = Device::Cpu;

    let x = Tensor::new(&[0., 1., 2., 3., 4., 5., 6.], &device)?;
    println!("{}",x);
    Ok(())
}