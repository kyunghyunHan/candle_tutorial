use candle_core::{Device, Result, Tensor,DType};
use candle_nn::{VarBuilder,VarMap,Optimizer,SGD};
const LEARNING_RATE:f64= 0.1;
pub fn main() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let x1_train = Tensor::new(&[[73.], [93.], [89.], [96.], [73.]], &device)?;
    let x2_train = Tensor::new(&[[80.], [88.], [91.], [98.], [66.]], &device)?;
    let x3_train = Tensor::new(&[[75.], [93.], [90.], [100.], [70.]], &device)?;
    let y_train = Tensor::new(&[[152.], [185.], [180.], [196.], [142.]], &device)?;

    let w1 = Tensor::zeros(1, candle_core::DType::F64, &device)?;
    let w2 = Tensor::zeros(1, candle_core::DType::F64, &device)?;
    let w3 = Tensor::zeros(1, candle_core::DType::F64, &device)?;
    let b = Tensor::zeros(1, candle_core::DType::F64, &device)?;
    
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F64, &device);
    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE)?;
    let nb_epoch= 1000;
    for epoch in 0..=nb_epoch{
        let a=( (x1_train.broadcast_matmul(&w1.unsqueeze(1)?)?).broadcast_add(&(x2_train).broadcast_matmul(&w2.unsqueeze(1)?)?)?).broadcast_add(&(x3_train.broadcast_matmul(&b.unsqueeze(1)?)?))?;
        let cost =((a-&y_train)?).broadcast_pow(&Tensor::new(2f64, &device)?)?.mean(0)?;
        println!("{}",cost);

    }
    Ok(())
}
