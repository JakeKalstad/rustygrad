mod tensor;

use tensor::Tensor;
use ndarray_rand::{rand, RandomExt};
use ndarray_rand::rand_distr::Uniform;
use std::f64::consts::E;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

#[derive(Debug)]
struct NeuralNetwork {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    weights_ih: Tensor,
    weights_ho: Tensor,
    bias_h: Tensor,
    bias_o: Tensor,
}

impl NeuralNetwork {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let weights_ih = Tensor::new(vec![input_size, hidden_size]);
        let weights_ho = Tensor::new(vec![hidden_size, output_size]);
        let bias_h = Tensor::new(vec![1, hidden_size]);
        let bias_o = Tensor::new(vec![1, output_size]);

        NeuralNetwork {
            input_size,
            hidden_size,
            output_size,
            weights_ih,
            weights_ho,
            bias_h,
            bias_o,
        }
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let hidden = (input.matmul(&self.weights_ih)).add(&self.bias_h);
        let hidden_activated = hidden.apply(sigmoid);
        let output = hidden_activated.matmul(&self.weights_ho).add(&self.bias_o);
        output
    }
}

fn main() {
    let tensor1 = crate::tensor::Tensor::new(vec![2, 3]);
    let tensor2 = crate::tensor::Tensor::new(vec![2, 1]);
    let result1 = tensor1.add(&tensor2);
    tensor1.print();
    tensor2.print();
    result1.print();

    let tensor1 = crate::tensor::Tensor::new(vec![2, 3]);
    let tensor2 = crate::tensor::Tensor::new(vec![3, 2]);
    let result2 = tensor1.matmul(&tensor2);

    tensor1.print();
    tensor2.print();
    result2.print();

    let input_size = 2;
    let hidden_size = 3;
    let output_size = 1;

    let nn = NeuralNetwork::new(input_size, hidden_size, output_size);
    let input = Tensor::new(vec![1, input_size]);
    let output = nn.forward(&input);

    println!("In: {:?}\n Out:  {:?}\n", input, output);
    println!("NN: {:?}\n", nn);
}
