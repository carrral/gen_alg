#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(unused_variables)]

pub mod genetic;
pub mod test_functions;

use genetic::algorithms::genetic_optimize;
use genetic::types::{FitnessReturn, MultivaluedFloat, MultivaluedInteger};
use genetic::Bounds;
use genetic::{implementations, OptimizeType, StopCondition};
use implementations::multi_valued::{MVICandidateList, RCCList};
use rand::{distributions::WeightedIndex, prelude::*, Rng};

fn f1(mvf: MultivaluedFloat) -> FitnessReturn {
    let x1: f32 = mvf.vars_value[0];
    let x2: f32 = mvf.vars_value[1];
    let x3: f32 = mvf.vars_value[2];

    return x1.powi(2) + x2.powi(2) + x3.powi(2);
}

fn f2(mvf: MultivaluedFloat) -> FitnessReturn {
    let x1: f32 = mvf.vars_value[0];
    let x2: f32 = mvf.vars_value[1];
    return 100.0 * (x2 - x1.powi(2)).powi(2) + (1.0 - x1.powi(2));
}

fn f3(mvf: MultivaluedFloat) -> FitnessReturn {
    let x: Vec<f32> = mvf.vars_value;
    let lower: f32 = -5.12;
    let upper: f32 = 5.12;

    let step_fn = |xi: &f32| xi.floor();

    let sum = x.iter().map(step_fn).sum();
    return sum;
}

fn f6(mvf: MultivaluedFloat) -> FitnessReturn {
    let x1: f32 = mvf.vars_value[0];
    let x2: f32 = mvf.vars_value[1];

    let sum_of_squares: f32 = x1.powi(2) + x2.powi(2);
    let sqrt = sum_of_squares.sqrt();
    let sin = sqrt.sin();

    let upper = sin.powi(2) - 0.5;
    let lower = (1.0 + 0.001 * sum_of_squares).powi(2);

    return 0.5 + upper / lower;
}

fn main() {
    let functions = [f1, f2, f3, f6];
    let n_vars = [3, 2, 5, 2];
    let tags = ["Función 1", "Función 2", "Función 3", "Función 6"];

    let mut multivaried_fn1 = RCCList::new(3);
    let mut multivaried_fn2 = RCCList::new(2);
    let mut multivaried_fn3 = RCCList::new(5);
    let mut multivaried_fn6 = RCCList::new(2);

    let bounds_fn1 = Bounds::new(
        MultivaluedFloat::new(3, vec![-5.12, -5.12, -5.12]),
        MultivaluedFloat::new(3, vec![5.12, 5.12, 5.12]),
    );

    let bounds_fn2 = Bounds::new(
        MultivaluedFloat::new(2, vec![-2.048, -2.048]),
        MultivaluedFloat::new(2, vec![2.048, 2.048]),
    );

    let bounds_fn3 = Bounds::new(
        MultivaluedFloat::new(5, vec![-5.12; 5]),
        MultivaluedFloat::new(5, vec![5.12; 5]),
    );

    let bounds_fn6 = Bounds::new(
        MultivaluedFloat::new(2, vec![-100.0, -100.0]),
        MultivaluedFloat::new(2, vec![100.0, 100.0]),
    );

    multivaried_fn1.set_bounds(bounds_fn1);
    multivaried_fn2.set_bounds(bounds_fn2);
    multivaried_fn3.set_bounds(bounds_fn3);
    multivaried_fn6.set_bounds(bounds_fn6);

    let mut candidates = [
        multivaried_fn1,
        multivaried_fn2,
        multivaried_fn3,
        multivaried_fn6,
    ];

    const PARAM_LENGTH: usize = 4;
    let initial_size = [16, 30, 50, 1000];
    let selected_per_round = [8, 15, 25, 500];
    let mating_pr = 0.6;
    let mut_pr = [0.2, 0.1, 0.07, 0.01];
    let opt_type = OptimizeType::MIN;
    let stop_condition = StopCondition::CYCLES(10);
    let debug = false;
    let show_fitness = false;
    let mut results: Vec<Vec<Result<(MultivaluedFloat, FitnessReturn), String>>> =
        vec![vec![]; functions.len()];

    for i in 0..functions.len() {
        let f = functions[i];
        println!("{}", tags[i]);
        for j in 0..PARAM_LENGTH {
            let _results = genetic_optimize(
                initial_size[j],
                selected_per_round[j],
                &mut candidates[i],
                f,
                mating_pr,
                mut_pr[j],
                &opt_type,
                &stop_condition,
                debug,
                show_fitness,
            );

            match &_results {
                Ok(v) => {
                    println!("Fitness: {}, Value: {}", v.1, v.0.to_string());
                }
                Err(err) => println!("Error: {}", err),
            }

            results[i].push(_results);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use genetic::implementations::multi_valued::MVICandidateList;
    use genetic::implementations::multi_valued::MultivaluedIntCandidate;
    use genetic::implementations::single_valued::IntegerCandidate;
    use genetic::implementations::single_valued::IntegerCandidateList;
    use genetic::utils;
    use genetic::InternalState;

    fn setup() -> IntegerCandidateList {
        let vals = ["0001", "0000", "0010", "0011"];
        let mut cl = IntegerCandidateList::default();
        for val in vals.iter() {
            let c = IntegerCandidate::new(String::from(*val));
            cl.candidates.push(c);
        }
        return cl;
    }

    #[test]
    fn test_implementations() {
        let c = IntegerCandidate::new(String::from("0010"));
        assert_eq!(c.get_integer_representation(), 2);
    }

    #[test]
    fn test_diagnostics() {
        let mut cl = setup();
        cl.eval_fitness(squared);
        let d = cl.get_diagnostics(&OptimizeType::MAX);
        assert_eq!(d.0, 9.0);
        assert_eq!(d.1, 3.5);
    }

    #[test]
    fn test_stop_condition() {
        let mut internal = InternalState::default();
        internal.max_achieved_fitness = 99.0;
        let stop = StopCondition::BOUND(100.0, 1.0);

        assert_eq!(internal.satisfies(&stop), true);

        internal.max_achieved_fitness = 98.9;
        assert_ne!(internal.satisfies(&stop), true);

        let stop = StopCondition::BOUND(100.0, 1.0);
        // internal.max_achieved_fitness = 102.0;
        // assert_eq!(internal.satisfies(&stop), true);
    }

    #[test]
    fn test_std_normal() {
        use rand::distributions::StandardNormal;
        use rand::rngs::SmallRng;
        use rand::FromEntropy;
        use utils::get_alpha;

        for i in 0..1000 {
            let val = get_alpha();
            assert!(val.abs() <= 1.0);
        }
    }

    #[test]
    fn test_pow() {
        let s: f32 = 500.0;
        assert_eq!(s * 10f32.powi(-3), 0.5);
    }

    #[test]
    fn test_parse_f32() {
        assert_eq!(
            utils::parse_f32(&"00111110001000000000000000000000".to_string()),
            0.15625,
        );

        assert_eq!(
            utils::parse_f32(&"11000101101010010111101011101101".to_string()),
            -5423.36572265625
        );
    }

    #[test]
    fn test_mvic() {
        let mvic = MultivaluedIntCandidate::new(3, "000000000000000001000000".to_string());
        let values = mvic.get_vars_from_bit_string();
        assert_eq!(values, [0, 0, 64]);
    }

    #[test]
    fn test_parse_signed_int() {
        assert_eq!(
            utils::bin_to_int(&String::from("11111111111111111111111111110101")),
            -11
        );
        assert_eq!(
            utils::bin_to_int(&String::from("11111111111111111111111110000001")),
            -127
        );
        assert_eq!(
            utils::bin_to_int(&String::from("00000000000000000000111110100000")),
            4000
        );
    }
}
