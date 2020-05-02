// #![allow(dead_code)]
#![allow(non_camel_case_types)]
// #![allow(unused_variables)]

pub mod genetic;

use genetic::types::*;
use genetic::{impls, traits, utils, OptimizeType, StopCondition};
use gnuplot::Figure;
use impls::multi_valued::RCCList;
use impls::single_valued::IntegerCandidateList;
use rand::{distributions::WeightedIndex, prelude::*, Rng};
use traits::CandidateList;

trait ShowGraph {
    fn show_graph(
        x_axis: &[usize],
        y_axis: &[FitnessReturn],
        title: &str,
        color: &str,
    ) -> Result<bool, std::io::Error>;
}

fn squared(x: isize) -> FitnessReturn {
    isize::pow(x, 2) as f32
}

fn rosenbrock_banana(mvf: MultivariedFloat) -> FitnessReturn {
    if mvf.n_vars != 2 {
        panic!(
            "Ésta función toma 2 parámetros, se recibieron {}",
            mvf.n_vars
        );
    }

    let x: f32 = mvf.vars_value[0];
    let y: f32 = mvf.vars_value[1];

    (1f32 - x).powi(2) + 100f32 * (y - x.powi(2)).powi(2)
}

fn multivalued_fn2(mvf: MultivariedFloat) -> FitnessReturn {
    if mvf.n_vars != 2 {
        panic!(
            "Ésta función toma 2 parámetros, se recibieron {}",
            mvf.n_vars
        );
    }

    let x: f32 = mvf.vars_value[0];
    let y: f32 = mvf.vars_value[1];
    -(x - 5.0).powi(2) - (y - 7.0).powi(2) + 5.0
}

fn main() {
    let mut l = IntegerCandidateList::default();
    let mut mvfl = RCCList::new(2);
    let lower_bound = MultivariedFloat::new(2, vec![-3.0, -3.0]);
    let upper_bound = MultivariedFloat::new(2, vec![3.0, 3.0]);
    let bounds = genetic::Bounds::new(lower_bound, upper_bound);
    mvfl.set_bounds(bounds);

    let results = basic_genetic_algorithm(
        16,
        4,
        &mut mvfl,
        rosenbrock_banana,
        0.6,
        0.01,
        &OptimizeType::MIN,
        &StopCondition::CYCLES(30),
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

// TODO: store historical best values, return fittest
fn basic_genetic_algorithm<T, U>(
    n: usize, // Tamaño de la población inicial
    selected_per_round: usize,
    candidates: &mut impl CandidateList<T, U>,
    fitness_fn: fn(U) -> FitnessReturn, // Función de adaptación
    mating_pr: f32,
    mut_pr: f32,
    opt: &OptimizeType, // MAX/MIN
    stop_cond: &StopCondition,
) -> Result<(U, FitnessReturn), String> {
    // @fitness: Fitness function
    // @opt: OptimizeType::MIN/OptimizeType::MAX

    // No sabe de una respuesta correcta, solo continúa hasta que stop_cond se cumple

    let mut results = Err("No calculations done".to_string());
    let mut fitness_over_time = Figure::new();
    let mut avg_fitness_over_time = Figure::new();

    let mut fitness_over_time_x: Vec<usize> = Default::default();
    let mut fitness_over_time_y: Vec<FitnessReturn> = Default::default();

    let mut avg_fitness_over_time_x: Vec<usize> = Default::default();
    let mut avg_fitness_over_time_y: Vec<FitnessReturn> = Default::default();
    let mut internal_state: genetic::InternalState = Default::default();

    utils::debug_msg(&*format!("Tamaño de la población: {}", n));
    utils::debug_msg(&*format!("Optimización: {}", &*opt.to_string()));
    utils::debug_msg(&*format!("Probabilidad de reproducción: {}", mating_pr));
    utils::debug_msg(&*format!("Probabilidad de mutación: {}", mut_pr));

    candidates.track_stop_cond(&stop_cond)?;

    // Generate initial round of candidates
    candidates.generate_initial_candidates(n);
    candidates.eval_fitness(fitness_fn);

    let tup = candidates.get_diagnostics(opt);
    // fitness_over_time_y.push(tup.0);
    // avg_fitness_over_time_y.push(tup.1);
    // fitness_over_time_x.push(0);
    // avg_fitness_over_time_x.push(0);
    let max_fitness;
    match candidates.get_results(opt) {
        (val, fitness) => {
            results = Ok((val, fitness));
            max_fitness = fitness;
        }
    };

    internal_state.update_values(max_fitness);

    loop {
        utils::debug_msg(&*format!("Generación {}:", internal_state.cycles));

        if internal_state.satisfies(stop_cond) {
            candidates.debug();
            utils::debug_msg("FIN\n\n");
            break;
        }

        candidates.debug();
        println!("\n");

        candidates.track_internal_state(&internal_state);

        candidates.mate(n, selected_per_round, mating_pr, opt);
        candidates.mutate_list(mut_pr, opt);

        candidates.eval_fitness(fitness_fn);

        // Update stats
        let tup = candidates.get_diagnostics(opt);
        let current_gen = internal_state.cycles;

        // fitness_over_time_y.push(tup.0);
        // avg_fitness_over_time_y.push(tup.1);
        // fitness_over_time_x.push(current_gen);
        // avg_fitness_over_time_x.push(current_gen);

        // Update internal state
        let max_fitness;
        match candidates.get_results(opt) {
            (val, fitness) => {
                results = Ok((val, fitness));
                max_fitness = fitness;
            }
        };

        internal_state.update_values(max_fitness);
    }

    // fitness_over_time.axes2d().lines(
    // &fitness_over_time_x,
    // &fitness_over_time_y,
    // &[Caption("Fitness / Tiempo"), Color("black")],
    // );

    // avg_fitness_over_time.axes2d().lines(
    // &avg_fitness_over_time_x,
    // &avg_fitness_over_time_y,
    // &[Caption("Fitness promedio / Tiempo"), Color("red")],
    // );

    // let f = fitness_over_time.show();
    // let g = avg_fitness_over_time.show();

    return results;
}

#[cfg(test)]
mod tests {
    use super::*;
    use genetic::impls::multi_valued::MVFCandidateList;
    use genetic::impls::multi_valued::MultivariedFloatCandidate;
    use genetic::impls::single_valued::IntegerCandidate;
    use genetic::impls::single_valued::IntegerCandidateList;
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
    fn test_impls() {
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
    fn test_get_f32_values() {
        let f1 = "01000000000100110011001100110011"; // 2.3
        let f2 = "11000010111111110101011100001010"; // -127.67
        let f3 = "01000010000000111110000101001000"; // 32.97
        let mut v = String::default();
        v.push_str(f1);
        v.push_str(f2);
        v.push_str(f3);

        let mvfc = MultivariedFloatCandidate::new(3, v);

        let vals = mvfc.get_vars_from_bit_string();
        assert_eq!(vals, vec![2.3, -127.67, 32.97])
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
}
