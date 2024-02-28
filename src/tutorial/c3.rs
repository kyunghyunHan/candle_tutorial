use candle_core::{Device, Result, Tensor};
use candle_nn::{Linear, Module};


struct Model{
    first:Linear,
    second:Linear,
}

impl Model{
    fn forward(&self,image:&Tensor)-> Result<Tensor>{
        let x = self.first.forward(image)?;
        let x =x.relu()?;
        self.second.forward(&x)
    }
}

pub fn c3()->Result<()>{
    let device = Device::Cpu;
    let weight = Tensor::randn(0f32, 1.0, (100, 784), &device)?;
    let bias = Tensor::randn(0f32, 1.0, (100, ), &device)?;
    let first = Linear::new(weight, Some(bias));
    let weight = Tensor::randn(0f32, 1.0, (10, 100), &device)?;
    let bias = Tensor::randn(0f32, 1.0, (10, ), &device)?;
    let second = Linear::new(weight, Some(bias));
    let model = Model { first, second };

    let dummy_image = Tensor::randn(0f32, 1.0, (1, 784), &device)?;

    let digit = model.forward(&dummy_image)?;
    println!("Digit {digit:?} digit");
    Ok(())}