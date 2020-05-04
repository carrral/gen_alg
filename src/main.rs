#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(unused_variables)]

pub mod genetic;

use genetic::types::*;
use genetic::utils::debug_msg;
use genetic::{impls, traits, OptimizeType, StopCondition};
use gnuplot::{Caption, Color, Figure};
use impls::multi_valued::RCCList;
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

fn rosenbrock_banana(mvf: MultivaluedFloat) -> FitnessReturn {
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

fn multivalued_fn2(mvf: MultivaluedFloat) -> FitnessReturn {
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

fn multivalued_fn_i_3(mvi: MultivaluedInteger) -> FitnessReturn {
    if mvi.n_vars != 3 {
        panic!(
            "Invalid number of variables: expected 3, got {}",
            mvi.n_vars
        );
    }
    let x = mvi.vars_value[0] as f32;
    let y = mvi.vars_value[1] as f32;
    let z = mvi.vars_value[2] as f32;

    let f = x + y + z;
    return f;
}

fn main() {
    let mut mvfl = RCCList::new(2);
    let lower_bound = MultivaluedFloat::new(2, vec![-30.0, -30.0]);
    let upper_bound = MultivaluedFloat::new(2, vec![30.0, 30.0]);
    let bounds = genetic::Bounds::new(lower_bound, upper_bound);
    mvfl.set_bounds(bounds);

    let results = basic_genetic_algorithm(
        16,
        4,
        &mut mvfl,
        multivalued_fn2,
        0.6,
        0.01,
        &OptimizeType::MAX,
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

fn basic_genetic_algorithm<T, U: Clone>(
    n: usize, // Tamaño de la población inicial
    selected_per_round: usize,
    candidates: &mut impl CandidateList<T, U>,
    fitness_fn: fn(U) -> FitnessReturn, // Función de adaptación
    mating_pr: f32,
    mut_pr: f32,
    opt: &OptimizeType, // MAX/MIN
    stop_cond: &StopCondition,
    debug_value: bool,
    show_fitness_plot: bool,
) -> Result<(U, FitnessReturn), String> {
    // @fitness: Fitness function
    // @opt: OptimizeType::MIN/OptimizeType::MAX

    // No sabe de una respuesta correcta, solo continúa hasta que stop_cond se cumple

    let mut results = Err("No calculations done".to_string());
    let mut historic_best = Err("No calculations done".to_string());
    let mut historic_max_fitness;
    let mut fitness_over_time = Figure::new();
    let mut avg_fitness_over_time = Figure::new();

    let mut fitness_over_time_x: Vec<usize> = Default::default();
    let mut fitness_over_time_y: Vec<FitnessReturn> = Default::default();

    let mut avg_fitness_over_time_x: Vec<usize> = Default::default();
    let mut avg_fitness_over_time_y: Vec<FitnessReturn> = Default::default();
    let mut internal_state: genetic::InternalState = Default::default();

    debug_msg(format!("Tamaño de la población: {}", n));
    debug_msg(format!("Optimización: {}", &*opt.to_string()));
    debug_msg(format!("Probabilidad de reproducción: {}", mating_pr));
    debug_msg(format!("Probabilidad de mutación: {}", mut_pr));

    // Exits execution if stop condition doesn't match the candidate list requirements
    candidates.track_stop_cond(&stop_cond)?;

    // Generate initial round of candidates
    candidates.generate_initial_candidates(n);
    candidates.eval_fitness(fitness_fn);

    let tup = candidates.get_diagnostics(opt);
    fitness_over_time_y.push(tup.0);
    avg_fitness_over_time_y.push(tup.1);
    fitness_over_time_x.push(0);
    avg_fitness_over_time_x.push(0);
    let max_fitness;
    match candidates.get_results(opt) {
        (val, fitness) => {
            results = Ok((val.clone(), fitness));
            historic_max_fitness = fitness;
            historic_best = Ok((val, fitness));
            max_fitness = fitness;
        }
    };

    internal_state.update_values(max_fitness);

    loop {
        debug_msg(format!("Generación {}:", internal_state.cycles));

        if internal_state.satisfies(stop_cond) {
            candidates.debug(debug_value);
            debug_msg(String::from("FIN\n\n"));
            break;
        }

        candidates.mark_for_selection(opt, selected_per_round);
        candidates.debug(debug_value);
        println!("\n");

        candidates.track_internal_state(&internal_state);

        candidates.mate(n, selected_per_round, mating_pr, opt);
        candidates.mutate_list(mut_pr, opt);

        candidates.eval_fitness(fitness_fn);

        // Update stats
        let tup = candidates.get_diagnostics(opt);
        let current_gen = internal_state.cycles;

        fitness_over_time_y.push(tup.0);
        avg_fitness_over_time_y.push(tup.1);
        fitness_over_time_x.push(current_gen);
        avg_fitness_over_time_x.push(current_gen);

        // Update internal state
        let max_fitness;
        match candidates.get_results(opt) {
            (val, fitness) => {
                results = Ok((val.clone(), fitness));
                max_fitness = fitness;
            }
        };

        if historic_max_fitness < max_fitness {
            historic_max_fitness = max_fitness;
            historic_best = results.clone();
        }

        internal_state.update_values(max_fitness);
    }

    if show_fitness_plot {
        fitness_over_time.axes2d().lines(
            &fitness_over_time_x,
            &fitness_over_time_y,
            &[Caption("Fitness / Tiempo"), Color("black")],
        );

        avg_fitness_over_time.axes2d().lines(
            &avg_fitness_over_time_x,
            &avg_fitness_over_time_y,
            &[Caption("Fitness promedio / Tiempo"), Color("red")],
        );

        let f = fitness_over_time.show();
        match f {
            Ok(fig) => debug_msg(String::from("Éxito mostrando la gráfica de fitness")),
            Err(m) => debug_msg(String::from("Error al iniciar GNUPlot")),
        };
        let g = avg_fitness_over_time.show();

        match g {
            Ok(fig) => debug_msg(String::from(
                "Éxito mostrando la gráfica de fitness promedio",
            )),
            Err(m) => debug_msg(String::from("Error al iniciar GNUPlot")),
        };
    }

    return historic_best;
}

#[cfg(test)]
mod tests {
    use super::*;
    use genetic::impls::multi_valued::MVICandidateList;
    use genetic::impls::multi_valued::MultivaluedIntCandidate;
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
}
