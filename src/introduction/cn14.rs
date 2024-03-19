// 필요한 라이브러리 가져오기
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{
    conv::conv2d, loss, Conv2d, Conv2dConfig, LayerNorm, Linear, Module, Optimizer, VarBuilder,
    VarMap, SGD,ops,
};
use rand::prelude::*;
use polars::prelude::*;


fn max_pool2d(xs: Tensor) -> candle_core::Result<Tensor> {
    let xs = xs.pad_with_same(D::Minus1, 0, 0)?;
    let xs = xs.pad_with_same(D::Minus2, 0, 0)?;
    xs.max_pool2d_with_stride(3, 2)
}
//로지스틱

const LEARNING_RATE: f64 = 1.;
const EPOCHS: usize = 1000;

#[derive(Clone, Debug)]
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
        let submission_df =
            CsvReader::from_path("./dataset/digit-recognizer/sample_submission.csv")
                .unwrap()
                .finish()
                .unwrap();

        let labels = train_df
            .column("label")
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .map(|x| x as u32)
            .collect::<Vec<u32>>();
        let test_labels = submission_df
            .column("Label")
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .map(|x| x as u32)
            .collect::<Vec<u32>>();

        let train_labels = Tensor::from_vec(labels, (train_samples,), &Device::Cpu)?;
        let test_labels = Tensor::from_vec(test_labels, (test_samples,), &Device::Cpu)?;

        let x_test = test_df
            .to_ndarray::<Int64Type>(IndexOrder::Fortran)
            .unwrap();
        let mut test_buffer_images: Vec<u32> = Vec::with_capacity(test_samples * 784);
        for i in x_test {
            test_buffer_images.push(i as u32)
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
        let mut train_buffer_images: Vec<u32> = Vec::with_capacity(train_samples * 784);
        for i in x_train {
            train_buffer_images.push(i as u32)
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
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    fc2: Linear,
}

impl MultiLevelPerceptron {
    fn new(vs: VarBuilder) -> candle_core::Result<Self> {
        let conv1 = candle_nn::conv2d(1, 32, 5, Default::default(), vs.pp("c1"))?;
        let conv2 = candle_nn::conv2d(32, 64, 5, Default::default(), vs.pp("c2"))?;
        let fc1 = candle_nn::linear(1024, 1024, vs.pp("fc1"))?;
        let fc2 = candle_nn::linear(1024, 10, vs.pp("fc2"))?;
        Ok(Self {
            conv1,
            conv2,
            fc1,
            fc2,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (batch_size, _image_dimension) = xs.dims2()?;

        xs.reshape((batch_size, 1, 28, 28))?
            .apply(&self.conv1)?
            .max_pool2d(2)?
            .apply(&self.conv2)?
            .max_pool2d(2)?
            .flatten_from(1)?
            .apply(&self.fc1)?
            .relu()?
            .apply(&self.fc2)
    }
}
struct TrainingArgs {
    learning_rate: f64,
    load: Option<String>,
    save: Option<String>,
    epochs: usize,
}
fn training_loop(
    dateset: Dataset,
    args: &TrainingArgs,
) -> anyhow::Result<()> {
    const BSIZE: usize = 64;

    let device = candle_core::Device::Cpu;

    // Train dataset
    let train_labels = dateset.train_labels;
    let train_images = dateset
        .train_images
        .to_device(&device)?;
    let train_labels = train_labels
        .to_dtype(DType::U32)?
        .to_device(&device)?;

    // Model
    let mut var_map = VarMap::new();
    let var_builder_args =
        VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let model = MultiLevelPerceptron::new(var_builder_args.clone())?;
    if let Some(load) = &args.load {
        println!("loading weights from {load}");
        var_map.load(load)?
    }

    // Optimizer
    let mut optimizer =
        candle_nn::AdamW::new_lr(var_map.all_vars(), args.learning_rate)?;

    // Test dataset
    let test_images = dateset
        .test_images
        .to_device(&device)?;
    let test_labels = dateset
        .test_labels
        .to_dtype(DType::U32)?
        .to_device(&device)?;

    let batches = train_images.dim(0)? / BSIZE;
    let mut batch_indices = (0..batches).collect::<Vec<usize>>();
    for epoch in 1..args.epochs {
        let mut sum_loss = 0f32;
        batch_indices.shuffle(&mut thread_rng());

        // Train phase
        for batch_index in batch_indices.iter() {
            let train_images =
                train_images.narrow(0, batch_index * BSIZE, BSIZE)?;
            let train_labels =
                train_labels.narrow(0, batch_index * BSIZE, BSIZE)?;
            let logits = model.forward(&train_images)?;
            let log_softmax = ops::log_softmax(&logits, D::Minus1)?;
            let loss = loss::nll(&log_softmax, &train_labels)?;
            optimizer.backward_step(&loss)?;
            sum_loss += loss.to_vec0::<f32>()?;
        }
        let avg_loss = sum_loss / batches as f32;

        // Test phase
        let test_logits = model.forward(&test_images)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        println!(
            "{epoch:4} train loss {:8.5} test acc: {:5.2}%",
            avg_loss,
            100. * test_accuracy
        );
    }

    if let Some(save) = &args.save {
        println!("saving trained weights in {save}");
        var_map.save(save)?
    }

    Ok(())
}
#[tokio::main]
pub async fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let m = Dataset::new()?;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F64, &device);
    let model = MultiLevelPerceptron::new(vs.clone())?;
    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE)?;

    let training_args = TrainingArgs {
        epochs: 10,
        learning_rate: 0.001,
        load: None,
        save: Some("./test/model.safetensors".to_string()),
    };
    training_loop(m, &training_args)?;
    Ok(())
}
