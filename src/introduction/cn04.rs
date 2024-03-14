/*소프트맥스 */
use candle_core::{Tensor,Device};
use candle_nn::{Optimizer,ops};
#[tokio::main]
pub async fn main() -> anyhow::Result<()> {
    let  z = Tensor::new(&[1.,2.,3.], &Device::Cpu)?;
    let htpothesis=ops::softmax(&z, 0)?;
    println!("{}",htpothesis);
  

    Ok(())
}