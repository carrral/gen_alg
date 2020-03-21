#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(unused_variables)]

use genetic_alg::constants::*;
use genetic_alg::types::*;
use genetic_alg::{Candidate, CandidateList, IntegerCandidate, OptimizeType, StopCondition};
use gnuplot::{Caption, Color, Figure};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::Rng;
use std::fmt;

// ==-- IMPLEMENTATIONS --==

// ==-- TRAITS --==

trait ShowGraph {
    fn show_graph(
        x_axis: &[usize],
        y_axis: &[FitnessReturn],
        title: &str,
        color: &str,
    ) -> Result<bool, std::io::Error>;
}

fn main() {
    // basic_genetic_algorithm(
    // POP_SIZE,
    // fitness,
    // elitist_mating,
    // &OptimizeType::MIN,
    // &StopCondition::CYCLES(10),
    // );
}

// ==-- UTILS --==

mod utils {

    use super::*;

    pub fn bin_to_int(bin: &String) -> t_int {
        let r_int: t_int = isize::from_str_radix(bin, 2).unwrap();
        r_int
    }

    pub fn debug_msg(msg: &str) {
        println!("  => {} ", msg);
    }

    pub fn random_range(start: isize, finish: isize) -> isize {
        let mut rng = thread_rng();
        return rng.gen_range(start, finish);
    }

    pub fn splitting_point(n: usize, pr: f64) -> usize {
        let spf: f64 = pr * (n as f64);
        return spf as usize;
    }

    pub fn roulette(weights: &[FitnessReturn; POP_SIZE]) -> t_int {
        //TODO: Implementar generics
        // Regresa 0 <= valor < weights.len()

        let mut rng = thread_rng();
        let values = 0..weights.len();
        let weighted_dist = WeightedIndex::new(weights).unwrap();

        return weighted_dist.sample(&mut rng) as t_int;
    }
}

mod genetic_alg {

    use super::*;

    pub mod constants {
        // ==-- CONSTANTS --==
        pub const TEST_RANGE: isize = 10;
        pub const IND_SIZE: usize = 8; // Must be even
        pub const POP_SIZE: usize = 8;
        pub const GEN: usize = 10;
        pub const MUT_PR: f64 = 0.3;
        pub const REP_PR: f64 = 0.7;
        pub const SELECTED_EACH_ROUND: usize = 4;
        pub const DEBUG: bool = true;
    }

    pub mod functions {
        use super::constants::*;
        use super::types::*;
        use super::IntegerCandidate;

        pub fn mate(
            father: &IntegerCandidate,
            mother: &IntegerCandidate,
            k: usize,
        ) -> (IntegerCandidate, IntegerCandidate) {
            // TODO: Volverlo un trait?
            // FIXME: Tomar como parámetro un IntegerCandidate
            //Regresa una tupla de hijos

            let gnomes_father = (&father.value[0..k], &father.value[k..IND_SIZE]);
            let gnomes_mother = (&mother.value[0..k], &mother.value[k..IND_SIZE]);
            let mut sons: (IntegerCandidate, IntegerCandidate) = (
                IntegerCandidate::new(String::from(gnomes_father.0), 0 as FitnessReturn, false),
                IntegerCandidate::new(String::from(gnomes_mother.0), 0 as FitnessReturn, false),
            );

            sons.0.value.push_str(gnomes_mother.1);
            sons.1.value.push_str(gnomes_father.1);

            return sons;
        }
    }

    // ==-- STRUCTS --==
    #[derive(PartialEq, Eq)]
    pub enum OptimizeType {
        MAX,
        MIN,
    }

    pub enum StopCondition {
        CYCLES(usize),
        ERROR_MARGIN(f64),
        BOUND(types::FitnessReturn),
    }

    pub trait Candidate<T> {
        // Evalúa el valor de fitness de self.value, lo asigna en self.fitness
        // y lo regresa.
        fn eval_fitness(&self, f: fn(T) -> FitnessReturn) -> FitnessReturn;
        fn to_string(&self) -> String;
        fn get_fitness(&self) -> FitnessReturn;
        fn debug(&self);
    }

    pub trait FitnessFunction<T> {
        fn eval(&self, x: T) -> FitnessReturn;
    }

    struct uIntFitnessFunction;

    impl FitnessFunction<usize> for uIntFitnessFunction {
        fn eval(&self, x: usize) -> FitnessReturn {
            0.00
        }
    }

    pub trait CandidateList<T> {
        // Generates an initial vector of Random Candidates
        fn generate_initial_candidates(&self, requested: usize);

        // Returns (max_fitness, avg_fitness)
        fn get_diagnostics(&self) -> (FitnessReturn, FitnessReturn);

        // Updates internal CandidateList
        fn mate(&mut self, n_out: usize, n_selected: usize, prob_rep: f64, opt_type: &OptimizeType);

        // Operates on the whole list with a given probability mut_pr
        fn mutate_list(&self, mut_pr: f64, opt: &OptimizeType);

        //Evaluates fitness for the whole candidate list
        fn eval_fitness(&self, f: fn(T) -> FitnessReturn);

        // fn get_fittest(&self, opt_type: &OptimizeType) -> &dyn Candidate<T>;

        fn len(&self) -> usize;

        fn debug(&self);
    }

    pub struct IntegerCandidateList {
        ind_size: usize,
        candidates: Vec<IntegerCandidate>,
    }

    impl IntegerCandidateList {
        fn get_n_fittest(&mut self, n: usize, opt_type: &OptimizeType) -> &[IntegerCandidate] {
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

            let start_i: usize = self.len() - n;

            &self.candidates[start_i..]
        }
    }

    impl CandidateList<usize> for IntegerCandidateList {
        fn len(&self) -> usize {
            self.candidates.len()
        }

        // Generates an initial vector of Random Candidates
        fn generate_initial_candidates(&self, requested: usize) {}

        // Returns (max_fitness, avg_fitness)
        fn get_diagnostics(&self) -> (FitnessReturn, FitnessReturn) {
            (0.0, 0.0)
        }

        // Updates internal CandidateList
        fn mate(
            &mut self,
            n_out: usize,
            n_selected: usize,
            prob_rep: f64,
            opt_type: &OptimizeType,
        ) {
            use genetic_alg::functions::*;
            use utils::*;

            // @prob_rep: Probability of reproduction
            // @n_out: # of Candidates selected each round for reproduction
            let mut best_candidates: &[IntegerCandidate];
            let mut new_candidates: Vec<IntegerCandidate> = Default::default();

            // Select @n_out best candidates
            // Sort candidates by fitness (lowest first)

            match *opt_type {
                // Ordena el vector en forma ascendente ->  [0,1,...,N]
                OptimizeType::MAX => debug_msg("Maximizando"),
                // Ordena el vector en forma descendente -> [N,...,1,0]
                OptimizeType::MIN => debug_msg("Minimizando"),
            };

            debug_msg("Ordenados");
            self.debug();

            best_candidates = self.get_n_fittest(n_selected, opt_type);

            debug_msg("Seleccionados");
            debug_candidates(&best_candidates);

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
            let k: usize = splitting_point(IND_SIZE, prob_rep);

            loop {
                let mut break_loop = false;
                for i in 0..best_candidates.len() {
                    let offset_index = (i + offset) % n_selected;
                    let current_offspring =
                        mate(&best_candidates[i], &best_candidates[offset_index], k);
                    new_candidates.push(current_offspring.0);
                    new_candidates.push(current_offspring.1);
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
        fn mutate_list(&self, mut_pr: f64, opt: &OptimizeType) {}

        //Evaluates fitness for the whole candidate list
        fn eval_fitness(&self, f: fn(usize) -> FitnessReturn) {}

        fn debug(&self) {
            for candidate in &self.candidates {
                candidate.debug();
            }
        }
    }

    impl Default for IntegerCandidateList {
        fn default() -> Self {
            IntegerCandidateList {
                ind_size: 0,
                candidates: Default::default(),
            }
        }
    }

    #[derive(Clone)]
    pub struct IntegerCandidate {
        pub value: ParamType,
        pub fitness: Option<FitnessReturn>,
        pub mutated: bool,
        pub selected: bool,
        // pub fitness_fn: uIntFitnessFunction
    }

    impl IntegerCandidate {
        pub fn new(value: ParamType, fitness: FitnessReturn, mutated: bool) -> Self {
            IntegerCandidate {
                value,
                fitness: Some(fitness),
                mutated,
                selected: true,
            }
        }

        fn get_integer_representation(&self) -> isize {
            let slice: &str = &*self.value;
            let self_int: isize = isize::from_str_radix(slice, 2).unwrap();
            return self_int;
        }
    }

    impl Candidate<isize> for IntegerCandidate {
        fn eval_fitness(&self, f: fn(isize) -> FitnessReturn) -> FitnessReturn {
            let self_int = self.get_integer_representation();
            return f(self_int);
        }

        fn get_fitness(&self) -> FitnessReturn {
            match self.fitness {
                Some(v) => v,
                None => std::f64::MAX,
            }
        }

        fn to_string(&self) -> String {
            let s = String::from(format!(
                "IntegerCandidate{{val: {},fit: {},mut: {}}}",
                self.value,
                self.get_fitness(),
                self.mutated
            ));

            return s;
        }

        fn debug(&self) {}
    }

    impl Default for IntegerCandidate {
        fn default() -> Self {
            IntegerCandidate::new(String::new(), 0 as FitnessReturn, false)
        }
    }

    impl fmt::Debug for IntegerCandidate {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                f,
                "* ({:?},{:?},{:?},{:?})",
                self.value,
                utils::bin_to_int(&self.value),
                self.fitness,
                self.mutated
            )
        }
    }

    pub struct InternalState<'a> {
        pub cycles: usize,
        best_match: Option<&'a IntegerCandidate>,
        error_margin: f64,
    }

    impl<'a> InternalState<'a> {
        pub fn satisfies(&self, stop: &StopCondition) -> bool {
            // So if StopCondition is never met, it wont loop forever
            const HARD_STOP: usize = 10000;
            let satisfied = match *stop {
                StopCondition::CYCLES(value) => value == self.cycles,
                StopCondition::ERROR_MARGIN(value) => value <= self.error_margin,

                // TODO: Implementación de Bound con un margen de aceptabilidad
                StopCondition::BOUND(value) => true,
            };

            return satisfied || self.cycles > HARD_STOP;
        }

        pub fn update<T>(&self, candidate_list: &impl CandidateList<T>) {
            //TODO: Implement
        }
    }

    impl<'a> Default for InternalState<'a> {
        fn default() -> Self {
            InternalState {
                cycles: 0,
                best_match: None,
                error_margin: 0.0,
            }
        }
    }

    pub mod types {
        use super::*;
        // ==-- TYPES --==
        pub type MatingFnSignature =
            fn(&mut IntegerCandidateList, usize, usize, f64, &OptimizeType) -> IntegerCandidateList;
        // CHANGED: La referencia a IntegerCandidateList necesita ser mutable porque se reordena
        pub type FitnessFnSignature = fn(&ParamType) -> FitnessReturn;
        pub type ParamType = String;
        pub type FitnessReturn = f64;
        pub type t_int = isize;
        // pub type IntegerCandidateList = Vec<IntegerCandidate>;
    }
}

fn fitness(x: &ParamType) -> FitnessReturn {
    let x_int: isize = isize::from_str_radix(x, 2).unwrap();
    return x_int.pow(2) as FitnessReturn;
}

// fn generate_initial_candidates2(requested: usize, l: usize) -> IntegerCandidateList {
// let mut candidates: IntegerCandidateList = Vec::new();

// for _i in 0..requested {
// let mut current_cand_str = String::new();
// for _j in 0..l {
// let val = utils::random_range(0, 2);
// current_cand_str.push_str(&val.to_string());
// }

// let current_candidate: IntegerCandidate =
// IntegerCandidate::new(current_cand_str, 0 as FitnessReturn, false);
// candidates.push(current_candidate);
// }

// return candidates;
// }

fn mutate(candidate: &mut IntegerCandidate, mut_pr: f64, opt: &OptimizeType) {
    let mut rng = thread_rng();

    // Flip a weighted coin
    let weights: [f64; 2] = [mut_pr, 1f64 - mut_pr];
    // (Probability of mutating,  Probability of NOT mutating)

    let weighted_cointoss = WeightedIndex::new(weights.iter()).unwrap();

    let choice = weighted_cointoss.sample(&mut rng) as i8;

    let mut mutated: String;

    let mutate = {
        if choice == 0 {
            true
        } else {
            false
        }
    };

    if mutate {
        mutated = String::new();
        let unwanted_char = match opt {
            OptimizeType::MAX => '0',
            OptimizeType::MIN => '1',
        };
        let mut k: usize;
        let mut tries: usize = 0;
        loop {
            k = utils::random_range(0, IND_SIZE as isize) as usize;
            let char_array: Vec<char> = candidate.value.chars().collect();
            if char_array[k] == unwanted_char || tries > IND_SIZE {
                break;
            }

            tries += 1;
        }

        let mut i: usize = 0;
        let mut mutated_char: char;
        for c in candidate.value.chars() {
            if i == k {
                mutated_char = match opt {
                    OptimizeType::MAX => '1',
                    OptimizeType::MIN => '0',
                };
            } else {
                mutated_char = c;
            }
            mutated.push(mutated_char);
            i += 1;
        }

        candidate.value = mutated;
        candidate.mutated = true;
    } else {
        candidate.mutated = false;
    }
}

// fn roulette_mating(
// candidates: &mut IntegerCandidateList,
// selected: usize,
// prob_rep: f64,
// ) -> IntegerCandidateList {
// let new_candidates: IntegerCandidateList = Vec::new();

// // Obtener vector de pesos
// let mut weights: [FitnessReturn; POP_SIZE] = [0 as FitnessReturn; POP_SIZE];

// // FIXME: Nos podemos ahorrar este ciclo?
// for i in 0..POP_SIZE {
// weights[i] = candidates[i].get_fitness();
// }

// // Obtener POP_SIZE/2 madres y padres para reproducción
// let mut mothers: IntegerCandidateList = Default::default();
// let mut fathers: IntegerCandidateList = Default::default();

// for i in 0..POP_SIZE / 2 {
// // Obtener padres de acuerdo a pesos de candidatos
// // Seleccionar un índice al azar con peso
// let index_m = utils::roulette(&weights) as usize;
// let index_f = utils::roulette(&weights) as usize;

// // COPIAR los candidatos elegidos al azar a sus respectivos vectores
// let selected_father = candidates[index_f].clone();
// let selected_mother = candidates[index_m].clone();

// fathers.push(selected_father);
// mothers.push(selected_mother);
// }

// // Reproducir
// let mut new_candidates: IntegerCandidateList = Default::default();

// for i in 0..POP_SIZE / 2 {
// //Elegir punto de corte de acuerdo a prob_rep
// // FIXME: utilizar fn equivalent_index();
// let k: usize = utils::random_range(1, IND_SIZE as isize) as usize;

// let sons: (IntegerCandidate, IntegerCandidate) =
// genetic_alg::functions::mate(&fathers[i], &mothers[i], k);

// new_candidates.push(sons.0);
// new_candidates.push(sons.1);
// }

// // Para este punto ya tenemos POP_SIZE nuevos candidatos

// return new_candidates;
// }

// fn elitist_mating(
// candidates: &mut IntegerCandidateList,
// n_out: usize,
// n_selected: usize,
// prob_rep: f64,
// opt_type: &OptimizeType,
// ) -> IntegerCandidateList {
// use genetic_alg::functions::*;
// use utils::*;

// // @prob_rep: Probability of reproduction
// // @n_out: # of Candidates selected each round for reproduction
// let mut best_candidates: IntegerCandidateList = Vec::new();
// let mut new_candidates: IntegerCandidateList = Vec::new();

// // Select @n_out best candidates
// // Sort candidates by fitness (lowest first)
// match *opt_type {
// // Ordena el vector en forma ascendente ->  [0,1,...,N]
// OptimizeType::MAX => candidates.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap()),
// // Ordena el vector en forma descendente -> [N,...,1,0]
// OptimizeType::MIN => candidates.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap()),
// };

// match *opt_type {
// // Ordena el vector en forma ascendente ->  [0,1,...,N]
// OptimizeType::MAX => debug_msg("Maximizando"),
// // Ordena el vector en forma descendente -> [N,...,1,0]
// OptimizeType::MIN => debug_msg("Minimizando"),
// };

// debug_msg("Ordenados");
// candidates.debug();

// //Pop last @n_selected elements from vector and push to best_candidates
// for i in 0..n_selected {
// let current_candidate: IntegerCandidate = match candidates.pop() {
// Some(c) => c,

// //IntegerCandidate default constructor
// None => Default::default(),
// };
// best_candidates.push(current_candidate);
// }

// debug_msg("Seleccionados");
// debug_candidates(&best_candidates);

// //Probar de la siguiente manera:
// // Para cada Ciclo, hasta que no se junten los N requeridos:
// //      Se tienen dos listas idénticas de candidatos, con un desfasamiento
// //      Ciclo 1:    Candidato 1 -> Candidato 2
// //                  Candidato 2 -> Candidato 3
// //                  Candidato 3 -> Candidato 1
// //
// //      Ciclo 2:    Candidato 1 -> Candidato 3
// //                  Candidato 2 -> Candidato 1
// //                  Candidato 3 -> Candidato 2

// let index: usize = 0;
// let mut offset: usize = 1;
// let k: usize = splitting_point(IND_SIZE, prob_rep);

// loop {
// let mut break_loop = false;
// for i in 0..best_candidates.len() {
// let offset_index = (i + offset) % n_selected;
// let current_offspring = mate(&best_candidates[i], &best_candidates[offset_index], k);
// new_candidates.push(current_offspring.0);
// new_candidates.push(current_offspring.1);
// if new_candidates.len() >= n_out {
// break_loop = true;
// break;
// }

// // TODO: Reloj modulo para evitar out of bounds panic
// }

// if break_loop {
// break;
// }

// offset += 1;
// }

// return new_candidates;
// }

fn basic_genetic_algorithm<T>(
    n: usize, // Tamaño de la población inicial
    candidates: impl CandidateList<T>,
    fitness_fn: FitnessFnSignature,    // Función de adaptación
    mating_fn: fn(T) -> FitnessReturn, //Función de reproducción
    opt: &OptimizeType,                // MAX/MIN
    stop_cond: &StopCondition,
) {
    // @fitness: Fitness function
    // @opt: OptimizeType::MIN/OptimizeType::MAX

    // No sabe de una respuesta correcta, solo continúa hasta que stop_cond se cumple

    let mut fitness_over_time = Figure::new();
    let mut avg_fitness_over_time = Figure::new();

    let mut fitness_over_time_x: Vec<usize> = Default::default();
    let mut fitness_over_time_y: Vec<FitnessReturn> = Default::default();

    let mut avg_fitness_over_time_x: Vec<usize> = Default::default();
    let mut avg_fitness_over_time_y: Vec<FitnessReturn> = Default::default();

    let mut internal_state: genetic_alg::InternalState = Default::default();

    let mut total_fitness: FitnessReturn;

    // Generate initial round of candidates
    // TODO: Not generic
    candidates.generate_initial_candidates(n);

    println!("Probabilidad de mutación: {}", MUT_PR);

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
}

// fn debug_candidate(candidate: &IntegerCandidate) {
// println!(
// "     * ( {}, {}, {:.2}, {})",
// candidate.value,
// utils::bin_to_int(&candidate.value),
// candidate.fitness,
// candidate.mutated,
// );
// }

fn debug_candidates<T>(candidates: &[impl Candidate<T>]) {
    for candidate in candidates {
        candidate.debug();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordering_as_expected() {
        let mut candidates: IntegerCandidateList = Default::default();
        for i in 0..4 {
            let current_candidate = IntegerCandidate::new(String::from("00000000"), i, false);
            candidates.push(current_candidate);
        }
        candidates.sort_by(|a, b| a.fitness.cmp(&b.fitness));
        let mut a: [FitnessReturn; 4] = Default::default();
        for i in 0..4 {
            a[i] = candidates[i].fitness;
        }
        assert_eq!(a, [0, 1, 2, 3]);

        candidates.sort_by(|a, b| b.fitness.cmp(&a.fitness));
        debug_msg("Descendente");
        debug_candidates(&candidates);
        for i in 0..4 {
            a[i] = candidates[i].fitness;
        }
        assert_eq!(a, [3, 2, 1, 0]);
    }
}
