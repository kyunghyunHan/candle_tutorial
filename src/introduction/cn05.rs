use candle_core::{Device, Tensor,DType};
use candle_nn::{init, GRUConfig, Linear, VarBuilder, LSTM, RNN,LSTMConfig,VarMap,Optimizer};
use std::collections::HashMap;
#[derive(Debug)]
struct Net {
    layers: Vec<LSTM>,
    // fc: Linear,
}
impl Net {
    fn new(vs: VarBuilder) -> candle_core::Result<Self> {
        let config= LSTMConfig::default_no_bias();
        let vb = &vs.pp("lstm");
        let mut layers = vec![];
        for layer_idx in 0..2 {
            let config = candle_nn::LSTMConfig {
                layer_idx,
                ..Default::default()
            };
            let lstm = candle_nn::lstm(1, 5, config, vb.clone())?;
            layers.push(lstm)
        }
        Ok(Self { layers })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor>{
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            let states = layer.seq(&xs)?;
            xs = layer.states_to_tensor(&states)?;
        }
        Ok(xs)
    }
}
#[tokio::main]
pub async fn main() -> anyhow::Result<()> {
    let input_str = "apple";
    let label_str = "pple!";

    let mut char_vocab: Vec<char> = input_str.chars().chain(label_str.chars()).collect();
    char_vocab.sort();
    char_vocab.dedup();
    let vocab_size = char_vocab.len();
    println!("문자 집합의 크기: {}", vocab_size);
    let input_size = vocab_size; // 입력의 크기는 문자 집합의 크기
    let hidden_size = 5;
    let output_size = 5;
    let learning_rate = 0.1;

    let mut char_to_index: HashMap<char, usize> = HashMap::new();
    for (i, &c) in char_vocab.iter().enumerate() {
        char_to_index.insert(c, i);
    }
    println!("{:?}", char_to_index);

    let mut index_to_char: HashMap<usize, char> = HashMap::new();
    for (key, &value) in &char_to_index {
        index_to_char.insert(value, *key);
    }
    println!("{:?}", index_to_char);

    let x_data: Vec<usize> = input_str.chars().map(|c| char_to_index[&c]).collect();
    let y_data: Vec<u32> = label_str
        .chars()
        .map(|c| char_to_index[&c] as u32)
        .collect();

    println!("{:?}", x_data);
    println!("{:?}", y_data);

    let mut x_one_hot = Vec::new();
    for &x in &x_data {
        x_one_hot.push(one_hot_encode(x, vocab_size));
    }
    println!("{:?}", x_one_hot);
    let device = Device::Cpu;
    let x = Tensor::new(x_one_hot, &device)?.unsqueeze(0)?;
    let y = Tensor::new(y_data, &device)?.unsqueeze(0)?;
    println!("{}", x);
    println!("{}", y);
    let varmap = VarMap::new();
   
    let vs = VarBuilder::from_varmap(&varmap, DType::F64, &device);

    let net= Net::new(vs)?;
  
    println!("{:?}",net);
    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), learning_rate)?;
    for i in 0..100{
        let hypothesis = net.forward(&x)?;

    }
    Ok(())
}

fn one_hot_encode(index: usize, vocab_size: usize) -> Vec<f64> {
    let mut one_hot_row = vec![0.0; vocab_size];
    one_hot_row[index] = 1.0;
    one_hot_row
}
