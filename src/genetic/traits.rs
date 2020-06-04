use super::types::*;
use super::{InternalState, OptimizeType, StopCondition};

pub trait Candidate<T> {
    // Evalúa el valor de fitness de self.value, lo asigna en self.fitness
    // y lo regresa.
    fn eval_fitness(&mut self, f: &Box<dyn Fn(T) -> FitnessReturn>) -> FitnessReturn;
    fn to_string(&self) -> String;
    fn get_fitness(&self) -> Option<FitnessReturn>;
    fn debug(&self);
    fn mutate(&mut self, opt_type: &OptimizeType);
}
pub trait CandidateList<T, U> {
    /// @T: Type that will be held in internal candidate vector
    /// @U: Type that will be evaluated in the fitness function provided
    // Generates an initial vector of Random Candidates
    fn generate_initial_candidates(&mut self, requested: usize);

    // Returns (max_fitness, avg_fitness)
    fn get_diagnostics(&self, opt_type: &OptimizeType) -> (FitnessReturn, FitnessReturn);

    // Updates internal CandidateList
    fn mate(&mut self, n_out: usize, n_selected: usize, prob_rep: f32, opt_type: &OptimizeType);

    // Operates on the whole list with a given probability mut_pr
    fn mutate_list(&mut self, mut_pr: f32, opt: &OptimizeType);

    //Evaluates fitness for the whole candidate list
    fn eval_fitness(&mut self, f: &Box<dyn Fn(U) -> FitnessReturn>);

    // fn get_fittest(&self, opt_type: &OptimizeType) -> &dyn Candidate<T>;

    fn len(&self) -> usize;

    fn debug(&self, value: bool);

    // Regresa el mejor resultado encontrado en una generación
    fn get_results(&mut self, opt_type: &OptimizeType) -> (U, FitnessReturn);

    fn sort(&mut self, opt_type: &OptimizeType);

    /// For debugging purposes
    fn mark_for_selection(&mut self, opt_type: &OptimizeType, n_selected: usize);

    /// Gets info from @stop_cond at the beginning of the algorithm
    fn track_stop_cond(&mut self, stop_cond: &StopCondition) -> Result<bool, String>;

    /// Updates internal values from @internal_state every generation
    /// without taking ownership.
    fn track_internal_state(&mut self, internal_state: &InternalState);
}

pub trait FitnessFunction<U> {
    // TODO: Should return Result<FitnessFunction>
    fn get_closure(&self) -> &Box<dyn Fn(U) -> FitnessReturn>;
}
