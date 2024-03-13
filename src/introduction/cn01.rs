//단순선형회귀
use candle_core::{DType, Device, Tensor,D};
use candle_nn::{loss, Linear, Module, Optimizer, VarBuilder, VarMap, SGD,ops};
const LEARNING_RATE: f64 = 0.01;
const EPOCHS: usize = 2000;

struct MultiLevelPerceptron {
    ln1: Linear,
}

impl MultiLevelPerceptron {
    fn new(vs: VarBuilder) -> candle_core::Result<Self> {
        let ln1 = candle_nn::linear(1, 1, vs.pp("ln1"))?;
        Ok(Self { ln1})
    }
 
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
      self.ln1.forward(xs)
    }
}
#[tokio::main]
pub async fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let x_train = Tensor::new(&[[1.], [2.], [3.]], &device)?; // 입력 데이터를 1차원으로 변경
    let y_train = Tensor::new(&[[2.], [4.], [6.]], &device)?;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F64, &device);
    let model = MultiLevelPerceptron::new(vs.clone())?;
    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE)?;
    for epoch in 0..EPOCHS + 1 {
        let logits = model.forward(&x_train)?;
   
        let loss = loss::mse(&logits, &y_train)?;
      
        sgd.backward_step(&loss)?;
        if epoch % 100 == 0 {
            println!("Epoch {}/{} Cost : {:.6}",EPOCHS,epoch, loss.to_scalar::<f64>()?)
        }
    }
    Ok(())
}
 