#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(unused_variables)]
#![allow(unused_assignments)]

pub mod genetic;
pub mod k_means;
pub mod plot;
pub mod test_functions;
use genetic::algorithms::genetic_optimize;
use genetic::types::{FitnessReturn, MultivaluedFloat};
use genetic::{implementations, OptimizeType, StopCondition};
use gnuplot::{AutoOption, AxesCommon, Caption, Color, Figure, PointSymbol};
use implementations::multi_valued::RCCList;
use k_means::space::Space;
use k_means::utils::gen_random_points;
use k_means::wrapper::ClusterList;
use k_means::Kmeans;
use matplotlib::{Env, Plot};
use plot::Plot2D;
use rand::{distributions::WeightedIndex, prelude::*, Rng};
use std::io::Write;
use test_functions::Rosenbrock;

fn main4() {
    write!(&mut std::io::stdout(), "{}", 'a');
    write!(&mut std::io::stdout(), "{}", 'b');
}
fn main() {
    // let points =  gen_random_points(
    let mut plot = Plot2D::new();
    plot.set_x_range(-100.0, 100.0, 10.0).unwrap();
    plot.set_y_range(-100.0, 100.0, 10.0).unwrap();
    plot.draw(&mut std::io::stdout()).unwrap();
}

fn main2() {
    let mut mvfl = RCCList::new(2);
    let lower_bound = MultivaluedFloat::new(2, vec![-2.0, -2.0]);
    let upper_bound = MultivaluedFloat::new(2, vec![2.0, 2.0]);
    let bounds = genetic::Bounds::new(lower_bound, upper_bound);
    mvfl.set_bounds(bounds);

    let results = genetic_optimize(
        1000,
        300,
        &mut mvfl,
        &Rosenbrock::new(),
        0.6,
        0.1,
        &OptimizeType::MIN,
        &StopCondition::CYCLES(100),
        true,
        true,
    );

    match results {
        Ok((v, fit)) => {
            println!("Resultado: {}, Fitness: {}", v.to_string(), fit);
        }
        Err(s) => {
            println!("Error: {}", s);
        }
    }
}

fn main3() {
    let random_points = gen_random_points(2, 20, (-500.0, 500.0));
    println!("{:?}", random_points);

    let space = Space::new(random_points, 2);

    let mut k_means = Kmeans::new(5, 2, space);

    let mut cluster_list = ClusterList::new(5, 2, k_means.get_space());
    let result = genetic_optimize(
        20,
        5,
        &mut cluster_list,
        &k_means,
        0.6,
        0.2,
        &OptimizeType::MIN,
        &StopCondition::CYCLES(10),
        true,
        false,
    );

    let space;

    match result {
        Ok((val, fit)) => space = k_means.cluster(val),
        Err(e) => println!("{}", e),
    }

    let figure = k_means.make_figure().unwrap().show();
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let mvic = MultivaluedIntCandidate::new(3, "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000".to_string());
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
