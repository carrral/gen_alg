use super::super::genetic::implementations::multi_valued::{RCCList, RCCandidate};
use super::super::genetic::traits::{CandidateList, FitnessFunction};
use super::super::genetic::types::{FitnessReturn, MultivaluedFloat};
use super::super::genetic::utils::random_range;
use super::super::genetic::{InternalState, OptimizeType, StopCondition};
use super::point::Point;
use super::space::Space;

/// Wrapper struct for RCCList
pub struct ClusterList<'a> {
    ///This will perform the  heavy lifting except for a few functions
    list: RCCList,
    k: usize,
    dimmensions: usize,
    ///Space struct whose points will be used for initializing candidates
    space_ref: &'a Space,
}

impl<'a> ClusterList<'a> {
    // impl ClusterList {
    pub fn new(k: usize, dimmensions: usize, space_ref: &'a Space) -> Self {
        ClusterList {
            list: RCCList::new(k * dimmensions),
            k,
            dimmensions,
            space_ref,
        }
    }
}

impl<'a> CandidateList<'a, RCCandidate, MultivaluedFloat> for ClusterList<'a> {
    fn generate_initial_candidates(&mut self, requested: usize) {
        // Get k random points from space_ref requested times
        for i in 0..requested {
            let mut random_points_index: Vec<usize> = vec![];
            for j in 0..self.k {
                let random = random_range(0, self.space_ref.len() as isize);
                random_points_index.push(random as usize);
            }

            // Accumulate point's values into  a k * dimm vector
            let mut init_vector: Vec<f32> = Vec::with_capacity(self.k * self.dimmensions);
            for index in random_points_index.iter() {
                let selected_center: &Point = &self.space_ref.get_points()[*index];
                for value in selected_center.get_values() {
                    init_vector.push(*value);
                }
            }

            // At this point, the vector should be brand new and not related to the Points in Space
            // in any way.

            // Initialize candidate
            let new_candidate = RCCandidate::new(self.k * self.dimmensions, &init_vector);
            self.list.candidates.push(new_candidate);
        }
    }

    // Returns (max_fitness, avg_fitness)
    fn get_diagnostics(&self, opt_type: &OptimizeType) -> (FitnessReturn, FitnessReturn) {
        self.list.get_diagnostics(opt_type)
    }

    // Updates internal CandidateList
    fn mate(&mut self, n_out: usize, n_selected: usize, prob_rep: f32, opt_type: &OptimizeType) {
        self.list.mate(n_out, n_selected, prob_rep, opt_type);
    }

    // Operates on the whole list with a given probability mut_pr
    fn mutate_list(&mut self, mut_pr: f32, opt: &OptimizeType) {
        self.list.mutate_list(mut_pr, opt);
    }

    //Evaluates fitness for the whole candidate list
    fn eval_fitness(&mut self, f: &dyn FitnessFunction<'a, MultivaluedFloat>) {
        self.list.eval_fitness(f);
    }

    // fn get_fittest(&self, opt_type: &OptimizeType) -> &dyn Candidate<T>;

    fn len(&self) -> usize {
        self.list.len()
    }

    fn debug(&self, value: bool) {
        self.list.debug(value);
    }

    // Regresa el mejor resultado encontrado en una generación
    fn get_results(&mut self, opt_type: &OptimizeType) -> (MultivaluedFloat, FitnessReturn) {
        self.list.get_results(opt_type)
    }

    fn sort(&mut self, opt_type: &OptimizeType) {
        self.list.sort(opt_type);
    }

    /// For debugging purposes
    fn mark_for_selection(&mut self, opt_type: &OptimizeType, n_selected: usize) {
        self.list.mark_for_selection(opt_type, n_selected);
    }

    /// Gets info from @stop_cond at the beginning of the algorithm
    fn track_stop_cond(&mut self, stop_cond: &StopCondition) -> Result<bool, String> {
        self.list.track_stop_cond(stop_cond)
    }

    /// Updates internal values from @internal_state every generation
    /// without taking ownership.
    fn track_internal_state(&mut self, internal_state: &InternalState) {
        self.list.track_internal_state(internal_state);
    }
}
