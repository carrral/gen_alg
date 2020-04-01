// #![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(unused_variables)]

pub mod genetic_alg;

use genetic_alg::types::*;
use genetic_alg::{utils, Candidate, CandidateList, OptimizeType, StopCondition};
use gnuplot::{Caption, Color, Figure};
use rand::{distributions::WeightedIndex, prelude::*, Rng};

// ==-- IMPLEMENTATIONS --==

pub struct IntegerCandidateList {
    ind_size: usize,
    candidates: Vec<IntegerCandidate>,
}

impl IntegerCandidateList {
    fn get_n_fittest(&mut self, n: usize, opt_type: &OptimizeType) -> &[IntegerCandidate] {
        self.sort(opt_type);

        let start_i: usize = self.len() - n;

        &self.candidates[start_i..]
    }

    fn sort(&mut self, opt_type: &OptimizeType) {
        match *opt_type {
            // Ordena el vector en forma ascendente ->  [0,1,...,N]
            OptimizeType::MAX => self
                .candidates
                .sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap()),
            // Ordena el vector en forma descendente -> [N,...,1,0]
            OptimizeType::MIN => self
                .candidates
                .sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap()),
        };
    }
}

impl CandidateList<isize> for IntegerCandidateList {
    fn len(&self) -> usize {
        self.candidates.len()
    }

    // Generates an initial vector of Random Candidates and stores it in self
    fn generate_initial_candidates(&mut self, requested: usize) {
        for i in 0..requested {
            let mut s: String = String::new();

            for j in 0..self.ind_size {
                let r = utils::random_range(0, 2).to_string();
                s.push_str(&r);
            }
            let c = IntegerCandidate::new(s);
            self.candidates.push(c);
        }
    }

    // Returns (max_fitness, avg_fitness)
    fn get_diagnostics(&self, opt_type: &OptimizeType) -> (FitnessReturn, FitnessReturn) {
        let mut total_fitness = 0.0;
        let mut max_fitness = match *opt_type {
            OptimizeType::MAX => std::f64::MIN,
            OptimizeType::MIN => std::f64::MAX,
        };

        if self.candidates[0].get_fitness().is_none() {
            panic!("Fitness hasn't been calculated yet");
        }

        for candidate in &self.candidates {
            let fitness = match candidate.get_fitness() {
                Some(v) => v,
                None => 0.0, // Unreachable
            };
            total_fitness += fitness;
            match *opt_type {
                OptimizeType::MAX => {
                    if max_fitness < fitness {
                        max_fitness = fitness;
                    }
                }
                OptimizeType::MIN => {
                    if max_fitness > fitness {
                        max_fitness = fitness;
                    }
                }
            };
        }
        (max_fitness, total_fitness / self.len() as f64)
    }

    // Updates internal CandidateList
    fn mate(&mut self, n_out: usize, n_selected: usize, prob_rep: f64, opt_type: &OptimizeType) {
        //FIXME: Regresa Result si n_selected > n_out o n_selected > self.len
        use genetic_alg::utils::*;

        // @prob_rep: Probability of reproduction
        // @n_out: # of Candidates selected each round for reproduction
        let mut new_candidates: Vec<IntegerCandidate> = Default::default();

        let size = self.ind_size;

        // Select @n_out best candidates
        let best_candidates: &[IntegerCandidate] = self.get_n_fittest(n_selected, opt_type);

        // if DEBUG {
        // println!();
        // debug_msg("Seleccionados");
        // debug_candidates(&best_candidates);
        // }

        //Probar de la siguiente manera:
        // Para cada Ciclo, hasta que no se junten los N requeridos:
        //      Se tienen dos listas idénticas de candidatos, con un desfasamiento
        //      Ciclo 1:    Candidato 1 -> Candidato 2
        //                  Candidato 2 -> Candidato 3
        //                  Candidato 3 -> Candidato 1
        //
        //      Ciclo 2:    Candidato 1 -> Candidato 3
        //                  Candidato 2 -> Candidato 1
        //                  Candidato 3 -> Candidato 2

        let index: usize = 0;
        let mut offset: usize = 1;
        let k: usize = splitting_point(size, prob_rep);
        let j = 0;

        loop {
            let mut break_loop = false;
            for i in 0..n_selected {
                let offset_index = (i + offset) % n_selected;
                let current_offspring_vals = genetic_alg::functions::cross_strings(
                    &best_candidates[i].value,
                    &best_candidates[offset_index].value,
                    k,
                );
                let offspring = (
                    IntegerCandidate::new(String::from(current_offspring_vals.0)),
                    IntegerCandidate::new(String::from(current_offspring_vals.1)),
                );
                new_candidates.push(offspring.0);
                new_candidates.push(offspring.1);

                if new_candidates.len() >= n_out {
                    break_loop = true;
                    break;
                }

                // TODO: Reloj modulo para evitar out of bounds panic
            }

            if break_loop {
                break;
            }

            offset += 1;
        }

        self.candidates = new_candidates;
    }

    // Operates on the whole list with a given probability mut_pr
    fn mutate_list(&mut self, mut_pr: f64, opt: &OptimizeType) {
        let cointoss = [mut_pr, 1f64 - mut_pr];
        for candidate in &mut self.candidates {
            // result: 0 -> mutate, 1 -> do nothing
            let result = utils::roulette(&cointoss);
            match result {
                0 => candidate.mutate(opt),
                _ => continue,
            };
        }
    }

    //Evaluates fitness for the whole candidate list
    fn eval_fitness(&mut self, f: fn(isize) -> FitnessReturn) {
        for candidate in &mut self.candidates {
            candidate.eval_fitness(f);
        }
    }

    fn debug(&self) {
        for candidate in &self.candidates {
            candidate.debug();
        }
    }

    fn get_results(&mut self, opt_type: &OptimizeType) -> (isize, FitnessReturn) {
        self.sort(opt_type);
        let fittest = self.get_n_fittest(1, opt_type);
        let max_fitness = match fittest[0].get_fitness() {
            Some(v) => v,
            // TODO: Raise panic
            None => 0.0,
        };
        (fittest[0].get_integer_representation(), max_fitness)
    }
}

impl Default for IntegerCandidateList {
    fn default() -> Self {
        IntegerCandidateList {
            ind_size: 8,
            candidates: Default::default(),
        }
    }
}

#[derive(Clone)]
pub struct IntegerCandidate {
    pub value: String,
    pub fitness: Option<FitnessReturn>,
    pub mutated: bool,
    pub selected: bool,
}

impl IntegerCandidate {
    pub fn new(value: String) -> Self {
        IntegerCandidate {
            value,
            fitness: None,
            mutated: false,
            selected: false,
        }
    }

    fn get_integer_representation(&self) -> isize {
        let slice: &str = &*self.value;
        let self_int: isize = isize::from_str_radix(slice, 2).unwrap();
        return self_int;
    }

    fn len(&self) -> usize {
        self.value.len()
    }
}

impl Candidate<isize> for IntegerCandidate {
    fn eval_fitness(&mut self, f: fn(isize) -> FitnessReturn) -> FitnessReturn {
        let self_int = self.get_integer_representation();
        self.fitness = Some(f(self_int));
        f(self_int)
    }

    fn get_fitness(&self) -> Option<FitnessReturn> {
        self.fitness
    }

    fn to_string(&self) -> String {
        let fitness = match self.get_fitness() {
            Some(v) => v.to_string(),
            None => String::from("UNDEFINED"),
        };
        let s = String::from(format!(
            "IntegerCandidate{{val: {}, int: {}, fit:{}, mut: {}}}",
            self.value,
            self.get_integer_representation(),
            fitness,
            self.mutated
        ));

        return s;
    }

    fn debug(&self) {
        println!("{}", self.to_string());
    }

    fn mutate(&mut self, opt_type: &OptimizeType) {
        let mut mutated = String::new();

        let (unwanted_char, wanted) = match *opt_type {
            OptimizeType::MAX => ('0', '1'),
            OptimizeType::MIN => ('1', '0'),
        };
        let mut k: usize;
        let mut tries: usize = 0;
        loop {
            // TODO: Cambiar intento al azar por iterción izquierda->derecha, derecha -> izquierda
            k = utils::random_range(0, self.len() as isize) as usize;
            let char_array: Vec<char> = self.value.chars().collect();
            if char_array[k] == unwanted_char || tries > self.len() {
                break;
            }

            tries += 1;
        }

        let mut i: usize = 0;
        for c in self.value.chars() {
            let mutated_char = match i {
                a if a == k => wanted,
                _ => c,
            };
            mutated.push(mutated_char);
            i += 1;
        }

        self.value = mutated;
        self.mutated = true;
    }
}

impl Default for IntegerCandidate {
    fn default() -> Self {
        IntegerCandidate::new(String::new())
    }
}

// ==-- TRAITS --==

trait ShowGraph {
    fn show_graph(
        x_axis: &[usize],
        y_axis: &[FitnessReturn],
        title: &str,
        color: &str,
    ) -> Result<bool, std::io::Error>;
}

fn squared(x: isize) -> FitnessReturn {
    isize::pow(x, 2) as f64
}
fn main() {
    let mut l = IntegerCandidateList::default();

    let results = basic_genetic_algorithm(
        8,
        4,
        &mut l,
        squared,
        0.5,
        0.1,
        &OptimizeType::MAX,
        &StopCondition::BOUND(65025.0, 0.0),
    );
}

// ==-- UTILS --==

fn basic_genetic_algorithm<T>(
    n: usize, // Tamaño de la población inicial
    selected_per_round: usize,
    candidates: &mut impl CandidateList<T>,
    fitness_fn: fn(T) -> FitnessReturn, // Función de adaptación
    mating_pr: f64,
    mut_pr: f64,
    opt: &OptimizeType, // MAX/MIN
    stop_cond: &StopCondition,
) -> Result<T, &'static str> {
    // @fitness: Fitness function
    // @opt: OptimizeType::MIN/OptimizeType::MAX

    // No sabe de una respuesta correcta, solo continúa hasta que stop_cond se cumple

    let mut results: (Result<T, &'static str>, FitnessReturn) = (Err("No calculations done"), 0f64);
    let mut fitness_over_time = Figure::new();
    let mut avg_fitness_over_time = Figure::new();

    let mut fitness_over_time_x: Vec<usize> = Default::default();
    let mut fitness_over_time_y: Vec<FitnessReturn> = Default::default();

    let mut avg_fitness_over_time_x: Vec<usize> = Default::default();
    let mut avg_fitness_over_time_y: Vec<FitnessReturn> = Default::default();

    let mut internal_state: genetic_alg::InternalState = Default::default();

    utils::debug_msg(&*format!("Tamaño de la población: {}", n));
    utils::debug_msg(&*format!("Optimización: {}", &*opt.to_string()));
    utils::debug_msg(&*format!("Probabilidad de reproducciónr: {}", mating_pr));
    utils::debug_msg(&*format!("Probabilidad de mutación: {}", mut_pr));

    // Generate initial round of candidates
    candidates.generate_initial_candidates(n);
    candidates.eval_fitness(fitness_fn);

    let tup = candidates.get_diagnostics(opt);
    fitness_over_time_y.push(tup.0);
    avg_fitness_over_time_y.push(tup.1);
    fitness_over_time_x.push(0);
    avg_fitness_over_time_x.push(0);

    internal_state.update_values(tup.0);

    loop {
        utils::debug_msg(&*format!("Generación {}:", internal_state.cycles));

        if internal_state.satisfies(stop_cond) {
            candidates.debug();
            utils::debug_msg("FIN\n\n");
            break;
        }

        candidates.debug();

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
        results = match candidates.get_results(opt) {
            (val, fitness) => (Ok(val), fitness),
        };

        internal_state.update_values(results.1);
    }

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
    let g = avg_fitness_over_time.show();

    return results.0;
}

// fn debug_candidates<T>(candidates: &[impl Candidate<T>]) {
// for candidate in candidates {
// candidate.debug();
// }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use genetic_alg::InternalState;

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
    fn test_sort() {
        let ascendente = [0.0, 1.0, 4.0, 9.0];
        let descendente = [9.0, 4.0, 1.0, 0.0];

        let mut cl = setup();
        cl.eval_fitness(squared);
        let s = cl.get_n_fittest(4, &OptimizeType::MAX);

        for i in 0..4 {
            let x = match s[i].get_fitness() {
                Some(v) => v,
                None => 0.0,
            };
            assert_eq!(ascendente[i], x);
        }

        let s = cl.get_n_fittest(4, &OptimizeType::MIN);
        for i in 0..4 {
            let x = match s[i].get_fitness() {
                Some(v) => v,
                None => 0.0,
            };
            assert_eq!(descendente[i], x);
        }
    }

    #[test]
    fn test_stop_condition() {
        let mut internal = InternalState::default();
        internal.max_achieved_fitness = 99.0;
        let stop = StopCondition::BOUND(100.0, 1.0);

        assert_eq!(internal.satisfies(&stop), true);

        internal.max_achieved_fitness = 98.9;
        assert_ne!(internal.satisfies(&stop), true);

        internal.max_achieved_fitness = 102.0;
        assert_eq!(internal.satisfies(&stop), true);
    }
}
