use super::*;

pub mod constants {
    // ==-- CONSTANTS --==
    pub const TEST_RANGE: isize = 10;
    pub const IND_SIZE: usize = 8; // Must be even
    pub const POP_SIZE: usize = 8;
    pub const GEN: usize = 10;
    pub const MUT_PR: f32 = 0.3;
    pub const REP_PR: f32 = 0.7;
    pub const SELECTED_EACH_ROUND: usize = 4;
    pub const DEBUG: bool = true;
}

pub mod functions {

    // TODO: Change function name
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
    ///Stops until internal cycle count is met
    CYCLES(usize),

    ///Stops until error margin is uniform among candidates
    ERROR_MARGIN(f32),

    /// Stops until the internal max fitness value is within a certain radix of specified bound
    BOUND(types::FitnessReturn, f32),
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

pub trait CandidateList<T> {
    // Generates an initial vector of Random Candidates
    fn generate_initial_candidates(&mut self, requested: usize);

    // Returns (max_fitness, avg_fitness)
    fn get_diagnostics(&self, opt_type: &OptimizeType) -> (FitnessReturn, FitnessReturn);

    // Updates internal CandidateList
    fn mate(&mut self, n_out: usize, n_selected: usize, prob_rep: f32, opt_type: &OptimizeType);

    // Operates on the whole list with a given probability mut_pr
    fn mutate_list(&mut self, mut_pr: f32, opt: &OptimizeType);

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
    error_margin: f32,
}

impl InternalState {
    pub fn satisfies(&self, stop: &StopCondition) -> bool {
        // So if StopCondition is never met, it wont loop forever
        const HARD_STOP: usize = 10000;
        let satisfied = match *stop {
            StopCondition::CYCLES(value) => value == self.cycles,
            StopCondition::ERROR_MARGIN(value) => value <= self.error_margin,

            StopCondition::BOUND(value, radix) => {
                self.max_achieved_fitness >= value
                    || (value - self.max_achieved_fitness).abs() <= radix
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
    pub type FitnessReturn = f32;
    pub type t_int = isize;
    // pub type IntegerCandidateList = Vec<IntegerCandidate>;
}

pub mod utils {

    use super::*;

    pub fn bin_to_int(bin: &String) -> t_int {
        let r_int: t_int = isize::from_str_radix(bin, 2).unwrap();
        r_int
    }

    pub fn parse_f32(bin_str: &String) -> f32 {
        const FLOAT_LEN: usize = 32;
        const SIGN_LEN: usize = 1;
        const EXPONENT_LEN: usize = 8;
        const MANTISSA_LEN: usize = 23;

        let l = bin_str.chars().count();
        if l != FLOAT_LEN {
            panic!(format!(
                "Invalid string length: expected 32, instead got {}",
                l
            ));
        }

        let char_array: Vec<char> = bin_str.chars().collect();

        for c in &char_array {
            if *c != '0' || *c != '1' {
                panic!("Invalid values found while parsing bit string");
            }
        }

        let sign = match char_array[0] {
            '0' => 1,
            '1' => -1,
            _ => 1,
        };

        let exponent_str = &bin_str[SIGN_LEN..(SIGN_LEN + EXPONENT_LEN)];
        let mantissa_str =
            &bin_str[(SIGN_LEN + EXPONENT_LEN)..(SIGN_LEN + EXPONENT_LEN + MANTISSA_LEN)];

        // Exponent & mantissa extraction ok
        let exponent = i32::from_str_radix(exponent_str, 2).unwrap();

        let mantissa = {
            let mut s: f32 = 1.0;
            let mut i: i32 = -1;
            for c in mantissa_str.chars() {
                if c != '0' {
                    s += 2f32.powi(i);
                }
                i -= 1;
            }
            s
        };

        utils::debug_msg(&*format!("({},{})", mantissa_str, exponent_str));
        utils::debug_msg(&*format!("({},{},{})", sign, mantissa, exponent));

        // return sign * mantissa.pow(exponent) as f32;
        (sign as f32) * (mantissa as f32) * 2f32.powi(exponent - 127i32)
    }

    pub fn cross_strings(father: &String, mother: &String, k: usize) -> (String, String) {
        // TODO: Volverlo un trait?
        // FIXME: Tomar como parámetro un IntegerCandidate
        //Regresa una tupla de hijos

        let gnomes_father = (&father[..k], &father[k..]);
        let gnomes_mother = (&mother[..k], &mother[k..]);
        let mut sons: (String, String) =
            (String::from(gnomes_father.0), String::from(gnomes_mother.0));

        sons.0.push_str(gnomes_mother.1);
        sons.1.push_str(gnomes_father.1);

        return sons;
    }

    pub fn generate_random_bitstring(n: usize) -> String {
        let mut s: String = String::new();

        for _j in 0..n {
            let r = utils::random_range(0, 2).to_string();
            s.push_str(&r);
        }
        return s;
    }

    pub fn generic_mutate(s: &String, opt_type: &OptimizeType) -> String {
        let mut mutated = String::new();

        let (unwanted_char, wanted) = match *opt_type {
            OptimizeType::MAX => ('0', '1'),
            OptimizeType::MIN => ('1', '0'),
        };
        let mut k: usize;
        let mut tries: usize = 0;
        loop {
            // TODO: Cambiar intento al azar por iterción izquierda->derecha, derecha -> izquierda
            k = utils::random_range(0, s.len() as isize) as usize;
            let char_array: Vec<char> = s.chars().collect();
            if char_array[k] == unwanted_char || tries > s.len() {
                break;
            }

            tries += 1;
        }

        let mut i: usize = 0;
        for c in s.chars() {
            let mutated_char = match i {
                a if a == k => wanted,
                _ => c,
            };
            mutated.push(mutated_char);
            i += 1;
        }

        return mutated;
    }

    pub fn debug_msg(msg: &str) {
        println!("  => {} ", msg);
    }

    pub fn random_range(start: isize, finish: isize) -> isize {
        let mut rng = thread_rng();
        return rng.gen_range(start, finish);
    }

    pub fn splitting_point(n: usize, pr: f32) -> usize {
        let spf: f32 = pr * (n as f32);
        return spf as usize;
    }

    pub fn roulette(weights: &[f32]) -> t_int {
        //TODO: Implementar generics
        // Regresa 0 <= valor < weights.len()

        let mut rng = thread_rng();
        let weighted_dist = WeightedIndex::new(weights).unwrap();

        return weighted_dist.sample(&mut rng) as t_int;
    }
}
