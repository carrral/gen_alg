pub mod point;
pub mod space;

use super::genetic::traits::FitnessFunction;
use super::genetic::types::MultivaluedFloat;

struct Kmeans {}

// impl FitnessFunction<MultivaluedFloat> for Kmeans {

// // TODO: Should return Result<FitnessFunction>
// fn eval(&self, t: U) -> FitnessReturn;
// fn get_closure(&self) -> &Box<dyn Fn(U) -> FitnessReturn>;
// }
