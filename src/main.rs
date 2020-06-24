#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(unused_variables)]
#![allow(unused_assignments)]

pub mod genetic;
pub mod k_means;
pub mod oil;
pub mod plot;
pub mod test_functions;

use genetic::algorithms::genetic_optimize;
use genetic::types::{FitnessReturn, MultivaluedFloat};
use genetic::{implementations, OptimizeType, StopCondition};
use implementations::multi_valued::RCCList;
use k_means::point::Point;
use k_means::space::Space;
use k_means::utils::gen_random_points;
use k_means::wrapper::ClusterList;
use k_means::Kmeans;
use plot::Plot2D;
use rand::{distributions::WeightedIndex, prelude::*, Rng};
use std::error::Error;
use std::io;
use std::io::Write;
use std::process;

use oil::OilField;

fn main() {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b',')
        .from_reader(io::stdin());

    let mut points: Vec<Point> = vec![];

    // for record in reader.records() {
    for record in reader.deserialize() {
        let oil_field: OilField = record.unwrap();
        let point = Point::new(vec![
            oil_field.x1,
            oil_field.x2,
            oil_field.x3,
            oil_field.x4,
            oil_field.x5,
            oil_field.x6,
            oil_field.x7,
            oil_field.x8,
        ]);
        points.push(point);
    }

    let clusters = 5;
    let mut space = Space::new(points, 8);
    let k_means = Kmeans::new(5, 8, space);
    let mut cluster_list = ClusterList::new(5, 8, k_means.get_space());
    let result = genetic_optimize(
        100,
        10,
        &mut cluster_list,
        &k_means,
        0.6,
        0.03,
        &OptimizeType::MIN,
        &StopCondition::CYCLES(300),
        false,
        true,
    );

    match result {
        Ok((v, fitness)) => {
            let centers = Kmeans::mvf_as_points(&v, 8, clusters);
            println!("{:#?}", centers);
        }
        Err(e) => println!("{}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use genetic::implementations::multi_valued::MultivaluedIntCandidate;
    use genetic::implementations::single_valued::IntegerCandidate;
    use genetic::implementations::single_valued::IntegerCandidateList;
    use genetic::utils;
    use genetic::InternalState;
    use plot::Plot2D;
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
    fn test_maps() {
        let a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        assert_eq!(a[0][0], 1);
        assert_eq!(a[0][1], 2);
        assert_eq!(a[2][0], 7);

        // let mapped = Plot2D::map(37, (0, 135), (-100, 100)).unwrap();
        // assert_eq!(-63, mapped);
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
