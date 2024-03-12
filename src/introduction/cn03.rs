use candle_core::{Tensor,Result,Device};

pub fn main()->candle_core::Result<()>{
    let device= Device::Cpu;
    let x_train= Tensor::new(&[[73.], [93.], [89.], [96.], [73.]], &device)?;

    println!("{:?}",x_train);

    Ok(())
}