use candle_core::{Device,Result,Tensor};
//모델정의
struct Model{
    first:Tensor,
    second:Tensor,
}
impl Model {
    fn forward(&self,image:&Tensor)->Result<Tensor>{
        let x= image.matmul(&self.first)?;
        let x =x.relu()?;
        x.matmul(&self.second)
    }
}

#[tokio::main]
pub async fn c1()->anyhow::Result<()> {
//    let device= Device::new_cuda(0)?;
    let device= Device::Cpu;
    
   let first= Tensor::randn(0f32,1.0,(784,100),&device)?;
   let second= Tensor::randn(0f32,1.0,(100,10),&device)?;

   let dummy_image = Tensor::randn(0f32,1.0,(1,784),&device)?;
  let  model= Model{first,second};
   let digit = model.forward(&dummy_image)?;
   
   println!("{}",digit);
Ok(())
}