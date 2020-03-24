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

    // TODO: Change function name
    pub fn cross_strings(father: &String, mother: &String, k: usize) -> (String, String) {
        // TODO: Volverlo un trait?
        // FIXME: Tomar como parámetro un IntegerCandidate
        //Regresa una tupla de hijos

        let gnomes_father = (&father[0..k], &father[k..IND_SIZE]);
        let gnomes_mother = (&mother[0..k], &mother[k..IND_SIZE]);
        let mut sons: (String, String) =
            (String::from(gnomes_father.0), String::from(gnomes_mother.0));

        sons.0.push_str(gnomes_mother.1);
        sons.1.push_str(gnomes_father.1);

        return sons;
    }
}

// ==-- STRUCTS --==
#[derive(PartialEq, Eq)]
pub enum OptimizeType {
    MAX,
    MIN,
}

impl OptimizeType {
    pub fn to_string(&self) -> String {
        let s = match *self {
            OptimizeType::MAX => "MAX",
            OptimizeType::MIN => "MIN",
        };
        return format!("OptimizeType::{}", s);
    }
}

pub enum StopCondition {
    CYCLES(usize),
    ERROR_MARGIN(f64),
    BOUND(types::FitnessReturn, f64),
}

pub trait Candidate<T> {
    // Evalúa el valor de fitness de self.value, lo asigna en self.fitness
    // y lo regresa.
    fn eval_fitness(&mut self, f: fn(T) -> FitnessReturn) -> FitnessReturn;
    fn to_string(&self) -> String;
    fn get_fitness(&self) -> Option<FitnessReturn>;
    fn debug(&self);
    fn mutate(&mut self, opt_type: &OptimizeType);
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
    fn generate_initial_candidates(&mut self, requested: usize);

    // Returns (max_fitness, avg_fitness)
    fn get_diagnostics(&self, opt_type: &OptimizeType) -> (FitnessReturn, FitnessReturn);

    // Updates internal CandidateList
    fn mate(&mut self, n_out: usize, n_selected: usize, prob_rep: f64, opt_type: &OptimizeType);

    // Operates on the whole list with a given probability mut_pr
    fn mutate_list(&mut self, mut_pr: f64, opt: &OptimizeType);

    //Evaluates fitness for the whole candidate list
    fn eval_fitness(&mut self, f: fn(T) -> FitnessReturn);

    // fn get_fittest(&self, opt_type: &OptimizeType) -> &dyn Candidate<T>;

    fn len(&self) -> usize;

    fn debug(&self);

    // Regresa el mejor resultado encontrado en una generación
    fn get_results(&mut self, opt_type: &OptimizeType) -> (T, FitnessReturn);
}

pub struct InternalState {
    pub cycles: usize,
    pub max_achieved_fitness: FitnessReturn,
    error_margin: f64,
}

impl InternalState {
    pub fn satisfies(&self, stop: &StopCondition) -> bool {
        // So if StopCondition is never met, it wont loop forever
        const HARD_STOP: usize = 10000;
        let satisfied = match *stop {
            StopCondition::CYCLES(value) => value == self.cycles,
            StopCondition::ERROR_MARGIN(value) => value <= self.error_margin,

            StopCondition::BOUND(value, radix) => {
                (value - self.max_achieved_fitness).abs() <= radix
            }
        };

        return satisfied || self.cycles > HARD_STOP;
    }

    pub fn update_values(&mut self, max_fitness: FitnessReturn) {
        self.max_achieved_fitness = max_fitness;
        self.cycles += 1;
    }
}

impl Default for InternalState {
    fn default() -> Self {
        InternalState {
            cycles: 0,
            max_achieved_fitness: 0.0,
            error_margin: 0.0,
        }
    }
}

pub mod types {
    // ==-- TYPES --==
    // CHANGED: La referencia a IntegerCandidateList necesita ser mutable porque se reordena
    pub type FitnessFnSignature = fn(&String) -> FitnessReturn;
    pub type FitnessReturn = f64;
    pub type t_int = isize;
    // pub type IntegerCandidateList = Vec<IntegerCandidate>;
}

pub mod utils {

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

    pub fn roulette(weights: &[f64]) -> t_int {
        //TODO: Implementar generics
        // Regresa 0 <= valor < weights.len()

        let mut rng = thread_rng();
        let values = 0..weights.len();
        let weighted_dist = WeightedIndex::new(weights).unwrap();

        return weighted_dist.sample(&mut rng) as t_int;
    }
}
