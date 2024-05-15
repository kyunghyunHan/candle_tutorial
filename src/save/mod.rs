use candle_core::{safetensors::save, Device, Result, Tensor};

pub fn main() -> Result<()> {
    c2()?;
    Ok(())
}

struct Linear {
    weight: Tensor,
    bias: Tensor,
}
impl Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.matmul(&self.weight)?;
        x.broadcast_add(&self.bias)
    }
}

struct Model {
    first: Linear,
    second: Linear,
}

impl Model {
    fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let x = self.first.forward(image)?;
        let x = x.relu()?;
        self.second.forward(&x)
    }
}
pub fn c2() -> Result<()> {
    // Use Device::new_cuda(0)?; to use the GPU.
    // Use Device::Cpu; to use the CPU.
    let device = Device::cuda_if_available(0)?;

    // Creating a dummy model
    let weight = Tensor::randn(0f32, 1.0, (784, 100), &device)?;
    let bias = Tensor::randn(0f32, 1.0, (100,), &device)?;
    let first = Linear { weight, bias };
    let weight = Tensor::randn(0f32, 1.0, (100, 10), &device)?;
    let bias = Tensor::randn(0f32, 1.0, (10,), &device)?;
    let second = Linear { weight, bias };
    let model = Model { first, second };

    let dummy_image = Tensor::randn(0f32, 1.0, (1, 784), &device)?;
    
    let digit = model.forward(&dummy_image)?;
    println!("Digit {digit:?} digit");
    Ok(())
}
