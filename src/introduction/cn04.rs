/*소프트맥스 */
use candle_core::{Tensor,Device,DType};
use candle_nn::{Optimizer,ops};
#[tokio::main]
pub async fn main() -> anyhow::Result<()> {
    
    let  z = Tensor::new(&[1.,2.,3.], &Device::Cpu)?;
    let htpothesis=ops::softmax(&z, 0)?;
    println!("{}",htpothesis);
    

    println!("{}",htpothesis.sum(0)?);
    
    let z= Tensor::rand(1.0, 2.0, &[3,5], &Device::Cpu)?;
    let htpothesis=ops::softmax(&z, 0)?;
    println!("{}",htpothesis);
    Ok(())
}