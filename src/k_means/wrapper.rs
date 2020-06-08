use super::super::genetic::implementations::multi_valued::{RCCList, RCCandidate};
use super::super::genetic::traits::CandidateList;
use super::super::genetic::types::{FitnessReturn, MultivaluedFloat};
use super::super::genetic::{InternalState, OptimizeType, StopCondition};
use super::point::Point;
use super::Kmeans;

/// Wrapper struct for RCCList
// pub struct ClusterList<'a> {
pub struct ClusterList {
    ///This will perform the  heavy lifting except for a few functions
    list: RCCList,
    k: usize,
    // kmeans_struct: &'a Kmeans<'a>,
}

// impl<'a> ClusterList<'a> {
impl ClusterList {
    pub fn new(k: usize, dimmensions: usize) -> Self {
        ClusterList {
            list: RCCList::new(k * dimmensions),
            k,
            // kmeans_struct,
        }
    }
}

// impl<'a> CandidateList<RCCandidate, MultivaluedFloat> for ClusterList<'a> {
impl CandidateList<RCCandidate, MultivaluedFloat> for ClusterList {
    /// Does nothing, as the initial candidates have already been generated
    /// by Kmeans
    fn generate_initial_candidates(&mut self, requested: usize) {}

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
    fn eval_fitness(&mut self, f: &Box<dyn Fn(MultivaluedFloat) -> FitnessReturn>) {}

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

/// Wrapper for RCCandidate. Instead of instantiating Points for each fitness evaluation, the
/// Points it contains refer to the same slice during their lifetime so they dont need to be
/// updated.
struct ClusterCenterCandidate<'a> {
    rccandidate: RCCandidate,
    points: Vec<Point<'a>>,
}

impl<'a> ClusterCenterCandidate<'a> {
    // impl ClusterCenterCandidate {
    pub fn new(k: usize, dimmensions: usize, values: &'a mut Vec<f32>) -> Self {
        let rccandidate = RCCandidate::new(k * dimmensions, &values);

        // Slice values vector into k points  as (cluster centers)

        let mut slices = vec![];

        for i in 0..k {
            let slice = &values[k..(k + dimmensions)];
            slices.push(slice);
        }

        let mut points = vec![];

        for i in 0..slices.len() {
            let point = Point::new(slices[i]);
            points.push(point);
        }

        ClusterCenterCandidate {
            rccandidate,
            points,
        }
    }
}
