use super::traits::{CandidateList, FitnessFunction};
use super::types::FitnessReturn;
use super::utils::debug_msg;
use super::{InternalState, OptimizeType, StopCondition};
use gnuplot::{Caption, Color, Figure};

pub fn genetic_optimize<'a, T, U: Clone + 'a>(
    n: usize, // Tamaño de la población inicial
    selected_per_round: usize,
    candidates: &mut impl CandidateList<'a, T, U>,
    fitness_fn: &impl FitnessFunction<'a, U>, // Función de adaptación
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
    let mut internal_state: InternalState = Default::default();

    debug_msg(format!("Tamaño de la población: {}", n), debug_value);
    debug_msg(format!("Optimización: {}", &*opt.to_string()), debug_value);
    debug_msg(
        format!("Probabilidad de reproducción: {}", mating_pr),
        debug_value,
    );
    debug_msg(format!("Probabilidad de mutación: {}", mut_pr), debug_value);

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
        debug_msg(
            format!("Generación {}:", internal_state.cycles),
            debug_value,
        );

        if internal_state.satisfies(stop_cond) {
            candidates.debug(debug_value);
            debug_msg(String::from("FIN\n\n"), debug_value);
            break;
        }

        candidates.mark_for_selection(opt, selected_per_round);
        candidates.debug(debug_value);

        if debug_value {
            println!("\n");
        }

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
            Ok(fig) => debug_msg(
                String::from("Éxito mostrando la gráfica de fitness"),
                debug_value,
            ),
            Err(m) => println!("{}", m),
        };
        let g = avg_fitness_over_time.show();

        match g {
            Ok(fig) => debug_msg(
                String::from("Éxito mostrando la gráfica de fitness promedio"),
                debug_value,
            ),
            Err(m) => debug_msg(String::from("Error al iniciar GNUPlot"), debug_value),
        };
    }

    return historic_best;
}
