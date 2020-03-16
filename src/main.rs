#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(unused_variables)]

use genetic_alg::constants::*;
use genetic_alg::types::*;
use genetic_alg::{Candidate, OptimizeType, StopCondition};
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
    basic_genetic_algorithm(
        POP_SIZE,
        fitness,
        elitist_mating,
        &OptimizeType::MIN,
        &StopCondition::CYCLES,
    );

    // let mut candidates: CandidateList = Default::default();
    // for i in 0..4 {
    // let current_candidate = Candidate::new(String::from("00000000"), i as FitnessReturn, false);
    // candidates.push(current_candidate);
    // }
    // candidates.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
    // debug_msg("Ascendente");
    // debug_candidates(&candidates);
    // candidates.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
    // debug_msg("Descendente");
    // debug_candidates(&candidates);
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
        use super::types::*;
        use super::Candidate;

        pub fn mate(father: &Candidate, mother: &Candidate, k: usize) -> (Candidate, Candidate) {
            // TODO: Volverlo un trait?
            // FIXME: Tomar como parámetro un Candidate
            //Regresa una tupla de hijos

            let gnomes_father = (&father.value[0..k], &father.value[k..IND_SIZE]);
            let gnomes_mother = (&mother.value[0..k], &mother.value[k..IND_SIZE]);
            let mut sons: (Candidate, Candidate) = (
                Candidate::new(String::from(gnomes_father.0), 0 as FitnessReturn, false),
                Candidate::new(String::from(gnomes_mother.0), 0 as FitnessReturn, false),
            );

            sons.0.value.push_str(gnomes_mother.1);
            sons.1.value.push_str(gnomes_father.1);

            return sons;
        }

        fn mate2() {}
    }

    // ==-- STRUCTS --==
    #[derive(PartialEq, Eq)]
    pub enum OptimizeType {
        MAX,
        MIN,
    }

    pub enum StopCondition {
        CYCLES,
        ERR(f64),
        BOUND(types::FitnessReturn),
    }

    #[derive(Clone)]
    pub struct Candidate {
        pub value: ParamType,
        pub fitness: FitnessReturn,
        pub mutated: bool,
        pub selected: bool,
    }

    impl Candidate {
        pub fn new(value: ParamType, fitness: FitnessReturn, mutated: bool) -> Self {
            Candidate {
                value,
                fitness,
                mutated,
                selected: true,
            }
        }
    }

    impl Default for Candidate {
        fn default() -> Self {
            Candidate::new(String::new(), 0 as FitnessReturn, false)
        }
    }

    impl fmt::Debug for Candidate {
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

    pub mod types {
        use super::*;
        // ==-- TYPES --==
        pub type MatingFnSignature =
            fn(&mut CandidateList, usize, usize, f64, &OptimizeType) -> CandidateList;
        // CHANGED: La referencia a CandidateList necesita ser mutable porque se reordena
        pub type FitnessFnSignature = fn(&ParamType) -> FitnessReturn;
        pub type ParamType = String;
        pub type FitnessReturn = f64;
        pub type t_int = isize;
        pub type CandidateList = Vec<Candidate>;
    }
}

fn fitness(x: &ParamType) -> FitnessReturn {
    let x_int: isize = isize::from_str_radix(x, 2).unwrap();
    return x_int.pow(2) as FitnessReturn;
}

fn generate_initial_candidates(requested: usize, l: usize) -> CandidateList {
    let mut candidates: CandidateList = Vec::new();

    for _i in 0..requested {
        let mut current_cand_str = String::new();
        for _j in 0..l {
            let val = utils::random_range(0, 2);
            current_cand_str.push_str(&val.to_string());
        }

        let current_candidate: Candidate =
            Candidate::new(current_cand_str, 0 as FitnessReturn, false);
        candidates.push(current_candidate);
    }

    return candidates;
}

fn mutate(candidate: &mut Candidate, mut_pr: f64, opt: &OptimizeType) {
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

fn roulette_mating(
    candidates: &mut CandidateList,
    selected: usize,
    prob_rep: f64,
) -> CandidateList {
    let new_candidates: CandidateList = Vec::new();

    // Obtener vector de pesos
    let mut weights: [FitnessReturn; POP_SIZE] = [0 as FitnessReturn; POP_SIZE];

    // FIXME: Nos podemos ahorrar este ciclo?
    for i in 0..POP_SIZE {
        weights[i] = candidates[i].fitness;
    }

    // Obtener POP_SIZE/2 madres y padres para reproducción
    let mut mothers: CandidateList = Default::default();
    let mut fathers: CandidateList = Default::default();

    for i in 0..POP_SIZE / 2 {
        // Obtener padres de acuerdo a pesos de candidatos
        // Seleccionar un índice al azar con peso
        let index_m = utils::roulette(&weights) as usize;
        let index_f = utils::roulette(&weights) as usize;

        // COPIAR los candidatos elegidos al azar a sus respectivos vectores
        let selected_father = candidates[index_f].clone();
        let selected_mother = candidates[index_m].clone();

        fathers.push(selected_father);
        mothers.push(selected_mother);
    }

    // Reproducir
    let mut new_candidates: CandidateList = Default::default();

    for i in 0..POP_SIZE / 2 {
        //Elegir punto de corte de acuerdo a prob_rep
        // FIXME: utilizar fn equivalent_index();
        let k: usize = utils::random_range(1, IND_SIZE as isize) as usize;

        let sons: (Candidate, Candidate) =
            genetic_alg::functions::mate(&fathers[i], &mothers[i], k);

        new_candidates.push(sons.0);
        new_candidates.push(sons.1);
    }

    // Para este punto ya tenemos POP_SIZE nuevos candidatos

    return new_candidates;
}

fn elitist_mating(
    candidates: &mut CandidateList,
    n_out: usize,
    n_selected: usize,
    prob_rep: f64,
    opt_type: &OptimizeType,
) -> CandidateList {
    use genetic_alg::functions::*;
    use utils::*;

    // @prob_rep: Probability of reproduction
    // @n_out: # of Candidates selected each round for reproduction
    let mut best_candidates: CandidateList = Vec::new();
    let mut new_candidates: CandidateList = Vec::new();

    // Select @n_out best candidates
    // Sort candidates by fitness (lowest first)
    match *opt_type {
        // Ordena el vector en forma ascendente ->  [0,1,...,N]
        OptimizeType::MAX => candidates.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap()),
        // Ordena el vector en forma descendente -> [N,...,1,0]
        OptimizeType::MIN => candidates.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap()),
    };

    match *opt_type {
        // Ordena el vector en forma ascendente ->  [0,1,...,N]
        OptimizeType::MAX => debug_msg("Maximizando"),
        // Ordena el vector en forma descendente -> [N,...,1,0]
        OptimizeType::MIN => debug_msg("Minimizando"),
    };

    debug_msg("Ordenados");
    debug_candidates(candidates);

    //Pop last @n_selected elements from vector and push to best_candidates
    for i in 0..n_selected {
        let current_candidate: Candidate = match candidates.pop() {
            Some(c) => c,

            //Candidate default constructor
            None => Default::default(),
        };
        best_candidates.push(current_candidate);
    }

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
            let current_offspring = mate(&best_candidates[i], &best_candidates[offset_index], k);
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

    return new_candidates;
}

fn basic_genetic_algorithm(
    n: usize,                       // Tamaño de la población inicial
    fitness_fn: FitnessFnSignature, // Función de adaptación
    mating_fn: MatingFnSignature,   //Función de reproducción
    opt: &OptimizeType,             // MAX/MIN
    stop_cond: &StopCondition,
) {
    // @fitness: Fitness function
    // @opt: OptimizeType::MIN/OptimizeType::MAX

    // No sabe de una respuesta correcta, solo continúa hasta que alcanza GEN

    let mut fitness_over_time = Figure::new();
    let mut avg_fitness_over_time = Figure::new();

    let mut fitness_over_time_x: [usize; GEN] = [0; GEN];
    let mut fitness_over_time_y: [FitnessReturn; GEN] = Default::default();

    let mut avg_fitness_over_time_x: [usize; GEN] = [0; GEN];
    let mut avg_fitness_over_time_y: [FitnessReturn; GEN] = Default::default();

    for i in 0..GEN {
        // Initialize diagnostic vectors with arbitrary values
        fitness_over_time_x[i] = i;
        avg_fitness_over_time_x[i] = i;
    }

    let mut total_fitness: FitnessReturn;

    // Generate initial round of candidates
    // TODO: Not generic
    let mut candidates: CandidateList = generate_initial_candidates(n, IND_SIZE);

    println!("Probabilidad de mutación: {}", MUT_PR);

    for _gen in 0..GEN {
        total_fitness = 0 as FitnessReturn;

        let mut max_fitness: FitnessReturn = match *opt {
            OptimizeType::MAX => std::isize::MIN as FitnessReturn,
            OptimizeType::MIN => std::isize::MAX as FitnessReturn,
        };

        // Obtener fitness de cada candidato
        // TODO: Not generic
        for candidate in &mut candidates {
            candidate.fitness = fitness(&candidate.value);
            total_fitness += candidate.fitness;

            // Obtener máximo de cada generación
            max_fitness = {
                if *opt == OptimizeType::MAX {
                    if max_fitness < candidate.fitness {
                        candidate.fitness
                    } else {
                        max_fitness
                    }
                } else {
                    if max_fitness > candidate.fitness {
                        candidate.fitness
                    } else {
                        max_fitness
                    }
                }
            }
        }

        fitness_over_time_y[_gen] = max_fitness;
        avg_fitness_over_time_y[_gen] = total_fitness / (POP_SIZE as FitnessReturn);

        println!(
            "Fitness máximo de la generación {}°: {}",
            _gen + 1,
            max_fitness
        );
        println!(
            "Fitness promedio: {} ",
            total_fitness / (POP_SIZE as FitnessReturn)
        );
        println!("    Candidatos:");
        debug_candidates(&candidates);

        // Mating
        // TODO: Not generic, use CandidateList.mate() instead
        candidates = mating_fn(&mut candidates, POP_SIZE, SELECTED_EACH_ROUND, REP_PR, opt);

        // Mutation
        // TODO: Not generic, use CandidateList.mutate() instead
        for candidate in &mut candidates {
            mutate(candidate, MUT_PR, opt);
        }
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
}

fn debug_candidate(candidate: &Candidate) {
    println!(
        "     * ( {}, {}, {:.2}, {})",
        candidate.value,
        utils::bin_to_int(&candidate.value),
        candidate.fitness,
        candidate.mutated,
    );
}

fn debug_candidates(candidates: &CandidateList) {
    for candidate in candidates {
        debug_candidate(candidate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordering_as_expected() {
        let mut candidates: CandidateList = Default::default();
        for i in 0..4 {
            let current_candidate = Candidate::new(String::from("00000000"), i, false);
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
