use super::*;

pub mod algorithms;
pub mod implementations;
pub mod traits;
pub mod types;
pub mod utils;

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

pub struct Bounds<T> {
    lower: T,
    upper: T,
    strays: Option<Vec<T>>,
}

impl Bounds<MultivaluedFloat> {
    pub fn new(lower: MultivaluedFloat, upper: MultivaluedFloat) -> Self {
        if lower.n_vars != upper.n_vars {
            panic!("Dimmension mismatch , {} != {}", lower.n_vars, upper.n_vars);
        }
        Bounds {
            lower,
            upper,
            strays: None,
        }
    }

    pub fn in_range(&self, mvf: &MultivaluedFloat) -> bool {
        true
    }

    pub fn val_in_range(&self, v: f32, index: usize) -> bool {
        let upper = self.upper.get_vals()[index];
        let lower = self.lower.get_vals()[index];

        match &self.strays {
            Some(s) => {
                for var in s.iter() {
                    if v == var.get_vals()[index] {
                        return false;
                    }
                }
            }
            None => (),
        }

        return v >= lower && v <= upper;
    }

    pub fn get_nth_upper(&self, n: usize) -> f32 {
        self.upper.get_vals()[n]
    }

    pub fn get_nth_lower(&self, n: usize) -> f32 {
        self.lower.get_vals()[n]
    }
}

pub struct InternalState {
    pub cycles: usize,
    pub max_achieved_fitness: FitnessReturn,
    error_margin: f32,
}

impl InternalState {
    pub fn satisfies(&self, stop: &StopCondition) -> bool {
        // So if StopCondition is never met, it wont loop forever
        const HARD_STOP: usize = std::usize::MAX;
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
