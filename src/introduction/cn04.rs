
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{loss, ops, Linear, Module, Optimizer, VarBuilder, VarMap, SGD};
const LEARNING_RATE: f64 = 1e-5;
const EPOCHS: usize = 20;
const BATCH_SIZE: usize = 2;

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
    // println!("{:?}",Tensor:: from_slice(&y_train.to_vec1::<f32>(),(2,3),&device));
    let varmap = VarMap::new();

    let vs = VarBuilder::from_varmap(&varmap, DType::F64, &device);
    let model = MultiLevelPerceptron::new(vs.clone())?;

    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE)?;
    let num_batches = 3 / BATCH_SIZE;
    println!("{}", 1);

    for epoch in 0..EPOCHS {
        let mut total_loss = 0.0;

        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * BATCH_SIZE;
            let end_idx = (batch_idx + 1) * BATCH_SIZE;
            println!("{}", 1);
            let mut batch_loss = 0.0;

            for i in start_idx..end_idx {
                let x_batch = x_train.get(i as usize)?.unsqueeze(0)?;

                let y_batch = y_train.get(i as usize)?.unsqueeze(1)?;

                let logits = model.forward(&x_batch)?;

                let loss = loss::mse(&logits, &y_batch)?;

                batch_loss += loss.to_scalar::<f64>()?;

                sgd.backward_step(&loss)?;
            }
        
        }
        println!("Epoch {}: Average loss: {}", epoch, total_loss / num_batches as f64);

    }
    Ok(())
}
