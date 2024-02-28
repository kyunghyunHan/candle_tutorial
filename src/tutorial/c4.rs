use candle_core::{Device, Result, Tensor};
use candle_nn::{Linear, Module};

pub fn c4() ->Result<()>{
   
    //창조
    let device = Device::Cpu;
    //torch.Tensor([[1, 2], [3, 4]])
    let x = Tensor::new(&[[1f32,2.],[3.,4.]],&device)?;
    println!("{}",x);
    Ok(())
}
