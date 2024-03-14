//로지스틱
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{loss, ops, Linear, Module, Optimizer, VarBuilder, VarMap, SGD};
use candle_datasets::{Batcher};
const LEARNING_RATE: f64 = 1e-5;
const EPOCHS: usize = 2000;

struct MultiLevelPerceptron {
    ln1: Linear,
}

impl MultiLevelPerceptron {
    fn new(vs: VarBuilder) -> candle_core::Result<Self> {
        let ln1 = candle_nn::linear(3, 1, vs.pp("ln1"))?;
        Ok(Self { ln1 })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.ln1.forward(xs)
    }
}
#[tokio::main]
pub async fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let x_train = Tensor::new(
        &[
            [73., 80., 75.],
            [93., 88., 93.],
            [89., 91., 90.],
            [96., 98., 100.],
            [73., 66., 70.],
        ],
        &device,
    )?;
    let y_train = Tensor::new(&[[152.], [185.], [180.], [196.], [142.]], &device)?;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F64, &device);
    let model = MultiLevelPerceptron::new(vs.clone())?;
    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE)?;
    for epoch in 0..EPOCHS + 1 {
        let prediction = model.forward(&x_train)?;

        let loss = loss::mse(&prediction, &y_train)?;

        sgd.backward_step(&loss)?;


    if epoch % 100 == 0 {
            println!(
                "Epoch {}/{} Cost : {:.6}",
                EPOCHS,
                epoch,
                loss.to_scalar::<f64>()?
            )
        }
    }
    let new = Tensor::new(&[[73., 80., 75.]], &device)?;
    let pred_y =model.forward(&new)?;
    println!("{}",pred_y);

    Ok(())
}
