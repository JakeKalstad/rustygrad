use ndarray::{Array, ArrayD};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;

// Define a Tensor struct
#[derive(Debug)]
pub struct Tensor {
    data: ArrayD<f64>,
}

impl Tensor {
    // Create a new tensor with random values
    pub fn new(shape: Vec<usize>) -> Self {
        let data = ArrayD::random(shape, StandardNormal);
        Tensor { data }
    }

    pub fn apply(&self, func: fn(f64) -> f64) -> Tensor {
        let applied_data = self.data.mapv(func);
        Tensor { data: applied_data }
    }
    
    // Create a new tensor with zeros
    pub fn zeros(shape: Vec<usize>) -> Self {
        let data = Array::from_elem(shape, 0.0);
        Tensor { data }
    }

    // Create a new tensor with ones
    pub fn ones(shape: Vec<usize>) -> Self {
        let data = Array::from_elem(shape, 1.0);
        Tensor { data }
    }

    // Perform element-wise addition
    pub fn add(&self, other: &Tensor) -> Self {
        let result_data = &self.data + &other.data;
        Tensor { data: result_data }
    }

    // Perform element-wise multiplication
    pub fn multiply(&self, other: &Tensor) -> Self {
        let result_data = &self.data * &other.data;
        Tensor { data: result_data }
    }

    // Perform matrix multiplication
    pub fn matmul(&self, other: &Tensor) -> Self {
        let dim_lhs = self.data.shape().to_vec();
        let dim_rhs = other.data.shape().to_vec();

        let lhs = self.data
            .to_owned()
            .into_shape((dim_lhs[0], dim_lhs[1]))
            .expect("Invalid dimensions");

        let rhs = other.data
            .to_owned()
            .into_shape((dim_rhs[0], dim_rhs[1]))
            .expect("Invalid dimensions");

        let result_data = lhs.dot(&rhs);
        Tensor { data: result_data.into_dimensionality().expect("Dimensions!") }
    }

    // Print the tensor's data
    pub fn print(&self) {
        println!("{:?}", self.data);
    }
}
 
