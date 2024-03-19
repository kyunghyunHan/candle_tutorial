//로지스틱
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{loss, ops, Linear, Module, Optimizer, VarBuilder, VarMap, SGD};

use polars::datatypes::DataType;
const LEARNING_RATE: f64 = 1.;
const EPOCHS: usize = 1000;

struct MultiLevelPerceptron {
    ln1: Linear,
}

impl MultiLevelPerceptron {
    fn new(vs: VarBuilder) -> candle_core::Result<Self> {
        let ln1 = candle_nn::linear(2, 1, vs.pp("ln1"))?;
        Ok(Self { ln1 })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
       let xs= self.ln1.forward(xs)?;
       candle_nn::ops::sigmoid(&xs)
        
    }
}
#[tokio::main]
pub async fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let x_train = Tensor::new(
        &[[1., 2.], [2., 3.], [3., 1.], [4., 3.], [5., 3.], [6., 2.]],
        &device,
    )?;
    let y_train = Tensor::new(& [[0.], [0.], [0.], [1.], [1.], [1.]], &device)?;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F64, &device);
    let model = MultiLevelPerceptron::new(vs.clone())?;
    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE)?;
    for epoch in 0..EPOCHS + 1 {
        let hypothesis = model.forward(&x_train)?;
        let loss = loss::binary_cross_entropy_with_logit(&hypothesis, &y_train)?;
        sgd.backward_step(&loss)?;


    if epoch % 10 == 0 {
        let ge_t = Tensor::new(
            0.5,
            &device,
        )?;
        //>=
        let prediction = hypothesis.broadcast_ge(&ge_t)?;
        let correct_prediction = prediction.to_dtype(DType::F64)?.eq(&y_train)?;
        let accuracy = correct_prediction.to_dtype(DType::F64)?.sum_all()?.to_scalar::<f64>()?/correct_prediction.shape().dims()[0] as f64;
        println!("Epoch {}/{} cost:{} Accuracy{}% ",epoch,EPOCHS,loss.to_scalar::<f64>()?,accuracy*100.0)
       


        }
    }

    Ok(())
}
