use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{loss, ops, Linear, Module, Optimizer, VarBuilder, VarMap};
use polars::prelude::*;

const IMAGE_DIM: usize = 784; //2차원 벡터
const RESULTS: usize = 10; //모델이; 예측하는 개수
const EPOCHS: usize = 10; //에폭
const LAYER1_OUT_SIZE: usize = 4; //첫번쨰 출력충의 출력뉴런 수
const LAYER2_OUT_SIZE: usize = 2; //2번쨰 츨략층의  출력 뉴런 수
const LEARNING_RATE: f64 = 0.05;

#[derive(Clone,Debug)]
pub struct Dataset {
    pub train_images: Tensor, //train data
    pub train_labels: Tensor,
    pub test_images: Tensor, //test data
    pub test_labels: Tensor,
}
impl Dataset {
    fn new() -> candle_core::Result<Self> {
        let train_samples = 42_000;
        let test_samples = 28_000;
        //데이터불러오기
        let train_df = CsvReader::from_path("./dataset/digit-recognizer/train.csv")
            .unwrap()
            .finish()
            .unwrap();
        let test_df = CsvReader::from_path("./dataset/digit-recognizer/test.csv")
            .unwrap()
            .finish()
            .unwrap();
        let submission_df = CsvReader::from_path("./dataset/digit-recognizer/sample_submission.csv")
            .unwrap()
            .finish()
            .unwrap();

        let labels = train_df
            .column("label")
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .map(|x| x as u8)
            .collect::<Vec<u8>>();
        let test_labels = submission_df
            .column("Label")
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .map(|x| x as u8)
            .collect::<Vec<u8>>();

        let train_labels = Tensor::from_vec(labels, (train_samples,), &Device::Cpu)?;
        let test_labels = Tensor::from_vec(test_labels, (test_samples,), &Device::Cpu)?;

        let x_test = test_df
            .to_ndarray::<Int64Type>(IndexOrder::Fortran)
            .unwrap();
        let mut test_buffer_images: Vec<u8> = Vec::with_capacity(test_samples * 784);
        for i in x_test {
            test_buffer_images.push(i as u8)
        }
        let test_images =
            (Tensor::from_vec(test_buffer_images, (test_samples, 784), &Device::Cpu)?
                .to_dtype(DType::F32)?
                / 255.)?;

        let x_train = train_df
            .drop("label")
            .unwrap()
            .to_ndarray::<Int64Type>(IndexOrder::Fortran)
            .unwrap();
        let mut train_buffer_images: Vec<u8> = Vec::with_capacity(train_samples * 784);
        for i in x_train {
            train_buffer_images.push(i as u8)
        }
        let train_images =
            (Tensor::from_vec(train_buffer_images, (train_samples, 784), &Device::Cpu)?
                .to_dtype(DType::F32)?
                / 255.)?;
        Ok(Self {
            train_images: train_images,
            train_labels: train_labels,
            test_images: test_images,
            test_labels: test_labels,
        })
    }
}
struct MultiLevelPerceptron {
    ln1: Linear,
    ln2: Linear, //은닉충
    ln3: Linear, //출력충
}
//3개 => 2개의 은닉충 1개의 출력충
impl MultiLevelPerceptron {
    fn new(vs: VarBuilder) -> candle_core::Result<Self> {
        let ln1 = candle_nn::linear(IMAGE_DIM, LAYER1_OUT_SIZE, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(LAYER1_OUT_SIZE, LAYER2_OUT_SIZE, vs.pp("ln2"))?;
        let ln3 = candle_nn::linear(LAYER2_OUT_SIZE, RESULTS + 1, vs.pp("ln3"))?;
        Ok(Self { ln1, ln2, ln3 })
    }
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = self.ln1.forward(xs)?;
        let xs = xs.relu()?;
        let xs = self.ln2.forward(&xs)?;
        let xs = xs.relu()?;
        self.ln3.forward(&xs)
    }
}

fn train(m: Dataset, dev: &Device) -> anyhow::Result<MultiLevelPerceptron> {
    let train_results = m.train_labels.to_device(dev)?; //디바이스
    let train_votes = m.train_images.to_device(dev)?;
    let varmap = VarMap::new(); //VarMap은 변수들을 관리하는 데 사용되는 자료 구조
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);
    let model = MultiLevelPerceptron::new(vs.clone())?;
    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE)?;
    let test_votes = m.test_images.to_device(dev)?;
    let test_results = m.test_labels.to_device(dev)?;
    let mut final_accuracy: f32 = 0.;
    for epoch in 1..EPOCHS + 1 {
        let logits = model.forward(&train_votes)?;
        let log_sm = ops::log_softmax(&logits, D::Minus1)?; //Minus1:가장 마지막 축
        let loss = loss::nll(&log_sm, &train_results)?; //손실함수
        sgd.backward_step(&loss)?; //역전파
        let test_logits = model.forward(&test_votes)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_results)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?; //정확도 계산
        let test_accuracy = sum_ok / test_results.dims1()? as f32;
        final_accuracy = 100. * test_accuracy;
        println!(
            "Epoch: {epoch:3} Train loss: {:8.5} Test accuracy: {:5.2}%",
            loss.to_scalar::<f32>()?,
            final_accuracy
        );
        if final_accuracy == 100.0 {
            break;
        }
    }
    if final_accuracy < 100.0 {
        Err(anyhow::Error::msg("The model is not trained well enough."))
    } else {
        Ok(model)
    }
}
#[tokio::main]
pub async fn model() -> anyhow::Result<()> {
    let dev = Device::cuda_if_available(0)?;
    let m = Dataset::new()?;
    let trained_model: MultiLevelPerceptron;
    loop {
        println!("Trying to train neural network.");
        match train(m.clone(), &dev) {
            Ok(model) => {
                trained_model = model;
                break;
            }
            Err(e) => {
                println!("Error: {}", e);
                continue;
            }
        }
    }
    //추정
    let real_world_votes: Vec<u32> = vec![13, 22];

    let tensor_test_votes =
        Tensor::from_vec(real_world_votes.clone(), (1, IMAGE_DIM), &dev)?.to_dtype(DType::F32)?;

    let final_result = trained_model.forward(&tensor_test_votes)?;

    let result = final_result
        .argmax(D::Minus1)?
        .to_dtype(DType::F32)?
        .get(0)
        .map(|x| x.to_scalar::<f32>())??;
    println!("real_life_votes: {:?}", real_world_votes);
    println!("neural_network_prediction_result: {:?}", result);
    Ok(())
}
