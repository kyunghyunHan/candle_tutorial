use candle_core::{Device, Result, Tensor,DType};
use candle_nn::{VarBuilder,VarMap,Optimizer,SGD};
const LEARNING_RATE:f64= 0.1;
pub fn main() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let x1_train = Tensor::new(&[[73.], [93.], [89.], [96.], [73.]], &device)?;
    let x2_train = Tensor::new(&[[80.], [88.], [91.], [98.], [66.]], &device)?;
    let x3_train = Tensor::new(&[[75.], [93.], [90.], [100.], [70.]], &device)?;
    let y_train = Tensor::new(&[[152.], [185.], [180.], [196.], [142.]], &device)?;


    
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F64, &device);
    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE)?;
    let nb_epoch= 2000;
    for epoch in 0..=nb_epoch{


    }
    Ok(())
}