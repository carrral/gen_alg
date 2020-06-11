pub mod point;
pub mod space;
pub mod wrapper;

use super::genetic::implementations::multi_valued::RCCList;
use super::genetic::traits::FitnessFunction;
use super::genetic::types::FitnessReturn;
use super::genetic::types::MultivaluedFloat;
use super::genetic::utils::random_range;
use space::{Center, Space};
use wrapper::ClusterList;

struct Kmeans<'a> {
    fitness: Box<dyn Fn(MultivaluedFloat) -> FitnessReturn>,
    space: Space<'a>,
    k: usize,
    dimmensions: usize,
}

impl<'a> Kmeans<'a> {
    /// Initializes the k-means algorithm with k  clusters with
    /// k random cluster centers
    fn new(k: usize, dimmensions: usize, space: Space) {
        let f = |mvf: MultivaluedFloat| {};
    }

    pub fn vector_as_points(&self, mvf: MultivaluedFloat) {
        if mvf.n_vars != self.dimmensions * self.k {
            panic!("Mismatch in dimmensions");
        }
    }

    /// Returns a "flattened" K*Dimmension vector of random clusters
    /// for ClusterList candidate initialization
    pub fn init_random_clusters(&self) -> Vec<f32> {
        let centers_index =
            (0..self.k).map(|i| random_range(0, self.space.len() as isize) as usize);
        // Collects into an iterable of Cluster centers (points)
        let clusters = centers_index.map(|index| {
            // Get clone of Point  with given index as a starting point
            // for the algorithm
            let center = self.space.get_points()[index].get_values();
            return center;
        });

        // Flatten into a k*dimm vector
        let mut flattened_clusters: Vec<f32> = Default::default();

        clusters.for_each(|cluster| {
            cluster.iter().for_each(|val: &f32| {
                flattened_clusters.push(*val);
            });
        });

        return flattened_clusters;
    }
}

impl<'a> FitnessFunction<MultivaluedFloat> for Kmeans<'a> {
    // TODO: Should return Result<FitnessFunction>
    fn get_closure(&self) -> &Box<dyn Fn(MultivaluedFloat) -> FitnessReturn> {
        &self.fitness
    }
}
