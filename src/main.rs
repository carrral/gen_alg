#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(unused_variables)]

use colored::*;
use gnuplot::{Caption, Color, Figure};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::Rng;

// ==-- CONSTANTS --==
const TEST_RANGE: isize = 10;
const IND_SIZE: usize = 8; // Must be even
const POP_SIZE: usize = 8;
const GEN: usize = 30;
const MUT_PR: f64 = 0.3;
const REP_PR: f64 = 0.4;
const SELECTED_EACH_ROUND: usize = 4;
const DEBUG: bool = true;

// ==-- TYPES --==
type MatingFnSignature = fn(&mut CandidateList, usize, usize, f64, &OptimizeType) -> CandidateList;
// FIXME: La referencia a CandidateList no necesita ser mutable
type FitnessFnSignature = fn(&ParamType) -> FitnessReturn;
type ParamType = String;
type FitnessReturn = isize;
type t_int = isize;
type CandidateList = Vec<Candidate>;

// ==-- STRUCTS --==
#[derive(Debug, Eq, Ord, PartialEq, PartialOrd, Clone)]
// TODO: implementar métodos de ordenamiento de candidatos
struct Candidate {
    value: ParamType,
    fitness: FitnessReturn,
    mutated: bool,
    selected: bool,
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

#[derive(PartialEq, Eq)]
enum OptimizeType {
    MAX,
    MIN,
}

enum StopCondition {
    CYCLES,
    ERR(f64),
    BOUND(FitnessReturn),
}

fn main() {
    basic_genetic_algorithm(
        POP_SIZE,
        fitness,
        elitist_mating,
        &OptimizeType::MAX,
        &StopCondition::CYCLES,
    );
}

fn debug_candidate(c: &Candidate) {
    println!("({},{},{})", (*c).value, (*c).fitness, (*c).mutated);
}

fn random_range(start: isize, finish: isize) -> isize {
    let mut rng = thread_rng();
    return rng.gen_range(start, finish);
}

fn splitting_point(n: usize, pr: f64) -> usize {
    let spf: f64 = pr * (n as f64);
    return spf as usize;
}

fn roulette(weights: &[FitnessReturn; POP_SIZE]) -> t_int {
    //TODO: Implementar generics
    // Regresa 0 <= valor < weights.len()

    let mut rng = thread_rng();
    let values = 0..weights.len();
    let weighted_dist = WeightedIndex::new(weights).unwrap();

    return weighted_dist.sample(&mut rng) as t_int;
}

fn fitness(x: &ParamType) -> FitnessReturn {
    let x_int: isize = isize::from_str_radix(x, 2).unwrap();
    return x_int.pow(2) as FitnessReturn;
}

fn mate(father: &Candidate, mother: &Candidate, k: usize) -> (Candidate, Candidate) {
    // FIXME: Tomar como parámetro un Candidate
    //Regresa una tupla de hijos

    let gnomes_father = (&father.value[0..k], &father.value[k..IND_SIZE]);
    let gnomes_mother = (&mother.value[0..k], &mother.value[k..IND_SIZE]);
    let mut sons: (Candidate, Candidate) = (
        Candidate::new(String::from(gnomes_father.0), 0, false),
        Candidate::new(String::from(gnomes_mother.0), 0, false),
    );

    sons.0.value.push_str(gnomes_mother.1);
    sons.1.value.push_str(gnomes_father.1);

    return sons;
}

fn generate_initial_candidates() -> CandidateList {
    let mut candidates: CandidateList = Vec::new();

    for _i in 0..POP_SIZE {
        let mut current_cand_str = String::new();
        for _j in 0..IND_SIZE {
            let val = random_range(0, 2);
            current_cand_str.push_str(&val.to_string());
        }

        let current_candidate: Candidate = Candidate::new(current_cand_str, 0, false);
        candidates.push(current_candidate);
    }

    return candidates;
}

fn bin_to_int(bin: &String) -> t_int {
    let r_int: t_int = isize::from_str_radix(bin, 2).unwrap();
    r_int
}

fn mutate(candidate: &mut Candidate, mut_pr: f64, opt: &OptimizeType) {
    let mut rng = thread_rng();

    //Elegir si se va a mutar
    let weights: [f64; 2] = [mut_pr, 1f64 - mut_pr];
    // (Probabilidad de mutar,  Probabilidad de NO mutar)

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
        let unwanted_char = match *opt {
            OptimizeType::MAX => '0',
            OptimizeType::MIN => '1',
        };
        let mut k: usize;
        let mut tries: usize = 0;
        loop {
            k = random_range(0, IND_SIZE as isize) as usize;
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
                mutated_char = match *opt {
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
        let index_m = roulette(&weights) as usize;
        let index_f = roulette(&weights) as usize;

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
        let k: usize = random_range(1, IND_SIZE as isize) as usize;

        let sons: (Candidate, Candidate) = mate(&fathers[i], &mothers[i], k);

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
    // @prob_rep: Probability of reproduction
    // @n_out: # of Candidates selected each round for reproduction
    let mut best_candidates: CandidateList = Vec::new();
    let mut new_candidates: CandidateList = Vec::new();

    // Select @n_out best candidates
    // Sort candidates by fitness (lowest first)
    match *opt_type {
        // Ordena el vector en forma ascendente ->  [0,1,...,N]
        OptimizeType::MAX => candidates.sort_by(|a, b| a.fitness.cmp(&b.fitness)),
        // Ordena el vector en forma descendente -> [N,...,1,0]
        OptimizeType::MIN => candidates.sort_by(|a, b| b.fitness.cmp(&a.fitness)),
    };

    //Pop last @n_selected elements from vector and push to best_candidates
    for i in 0..n_selected {
        let current_candidate: Candidate = match candidates.pop() {
            Some(c) => c,

            //Candidate default constructor
            None => Default::default(),
        };
        best_candidates.push(current_candidate);
    }

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
        // Initialize vectors with arbitrary values
        fitness_over_time_x[i] = i;
        avg_fitness_over_time_x[i] = i;
    }

    let mut total_fitness: FitnessReturn;

    // Generate initial round of candidates
    let mut candidates: CandidateList = generate_initial_candidates();

    println!("Probabilidad de mutación: {}", MUT_PR);

    for _gen in 0..GEN {
        total_fitness = 0;

        let mut max_fitness: FitnessReturn = match *opt {
            OptimizeType::MAX => -1 as FitnessReturn,
            OptimizeType::MIN => 1000 as FitnessReturn,
        };

        // Obtener fitness de cada candidato
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
        avg_fitness_over_time_y[_gen] = total_fitness / (POP_SIZE as isize);

        println!(
            "Fitness máximo de la generación {}°: {}",
            _gen + 1,
            max_fitness
        );
        println!("Fitness promedio: {} ", total_fitness / (POP_SIZE as isize));
        println!("    Candidatos:");
        for candidate in &mut candidates {
            println!(
                "     * ( {}, {}, {}, {} )",
                candidate.value,
                bin_to_int(&candidate.value),
                fitness(&candidate.value),
                &candidate.mutated
            );
        }

        candidates = mating_fn(
            &mut candidates,
            POP_SIZE,
            SELECTED_EACH_ROUND,
            REP_PR,
            &OptimizeType::MAX,
        );

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

fn debugmsg(callee: &str, msg: &str) {
    if DEBUG {
        println!("           {}::{}", callee.yellow(), msg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordering_as_expected() {
        let mut candidates: CandidateList = Default::default();
        for i in 0..4 {
            let current_candidate = Candidate::new(String::new(), i, false);
            candidates.push(current_candidate);
        }
        candidates.sort_by(|a, b| a.fitness.cmp(&b.fitness));
        let mut a: [FitnessReturn; 4] = Default::default();
        for i in 0..4 {
            a[i] = candidates[i].fitness;
        }
        assert_eq!(a, [0, 1, 2, 3]);

        candidates.sort_by(|a, b| b.fitness.cmp(&a.fitness));
        for i in 0..4 {
            a[i] = candidates[i].fitness;
        }
        assert_eq!(a, [3, 2, 1, 0]);
    }
}
