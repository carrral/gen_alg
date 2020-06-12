use super::super::genetic::implementations::multi_valued::{RCCList, RCCandidate};
use super::super::genetic::traits::{Candidate, CandidateList, FitnessFunction};
use super::super::genetic::types::{FitnessReturn, MultivaluedFloat};
use super::super::genetic::utils::random_range;
use super::super::genetic::{InternalState, OptimizeType, StopCondition};
use super::point::Point;
use super::space::Space;
use super::Kmeans;

/// Wrapper struct for RCCList
pub struct ClusterList<'a> {
    ///This will perform the  heavy lifting except for a few functions
    list: RCCList,
    k: usize,
    dimmensions: usize,
    ///Space struct whose points will be used for initializing candidates
    space_ref: &'a Space<'a>,
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
    // impl CandidateList<RCCandidate, MultivaluedFloat> for ClusterList {
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
                let selected_center: &Point = &self.space_ref.points[i];
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
    fn eval_fitness(&mut self, f: &FitnessFunction<'a, MultivaluedFloat>) {}

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

// Wrapper for RCCandidate. Instead of instantiating Points for each fitness evaluation, the
// Points it contains refer to the same slice during their lifetime so they dont need to be
// updated.
// struct ClusterCenterCandidate<'a> {
// /// Manages Candidate operations
// rccandidate: RCCandidate,

// /// Saves slices of RCCandidate as a Point Structure
// points: Vec<Point<'a>>,
// }

// impl<'a> ClusterCenterCandidate<'a> {
// // impl ClusterCenterCandidate {
// pub fn new(k: usize, dimmensions: usize, values: &'a Vec<f32>) -> Self {
// let rccandidate = RCCandidate::new(k * dimmensions, &values);

// // Slice values vector into k points  as (cluster centers)

// let mut slices = vec![];

// for i in 0..k {
// let slice = &values[k..(k + dimmensions)];
// slices.push(slice);
// }

// let mut points = vec![];

// for i in 0..slices.len() {
// let point = Point::new(slices[i]);
// points.push(point);
// }

// ClusterCenterCandidate {
// rccandidate,
// points,
// }
// }
// fn eval_fitness<'b>(
// &mut self,
// f: &'b Box<dyn Fn(&Vec<Point<'a>>) -> FitnessReturn>,
// ) -> FitnessReturn {
// // fn eval_fitness(&mut self, f: Box<dyn Fn(&Vec<Point<'a>>) -> FitnessReturn>) -> FitnessReturn {
// f(&self.points)
// // unimplemented!()
// }
// }

// // impl<'a> Candidate<&Vec<Point<'a>>> for ClusterCenterCandidate<'a> {
// // // Evalúa el valor de fitness de self.value, lo asigna en self.fitness
// // // y lo regresa.

// // fn to_string(&self) -> String {
// // let mut string = String::from("ClusterCenters(");
// // for point in &self.points {
// // let point_str = point.to_string();
// // string.push_str(&point_str);
// // }
// // string.push_str(")");

// // return string;
// // }
// // fn get_fitness(&self) -> Option<FitnessReturn> {
// // return self.rccandidate.get_fitness();
// // }
// // fn debug(&self) {
// // println!("{}", self.to_string());
// // }
// // fn mutate(&mut self, opt_type: &OptimizeType) {
// // self.rccandidate.mutate(opt_type);
// // }
// // }
