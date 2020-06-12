pub mod point;
pub mod space;
pub mod wrapper;

use super::genetic::implementations::multi_valued::RCCList;
use super::genetic::traits::FitnessFunction;
use super::genetic::types::FitnessReturn;
use super::genetic::types::MultivaluedFloat;
use super::genetic::utils::random_range;
use point::Point;
use space::{Center, Space};
use wrapper::ClusterList;

struct Kmeans<'a> {
    fitness: Box<dyn Fn(MultivaluedFloat) -> FitnessReturn>,
    space: Space<'a>,
    k: usize,
    dimmensions: usize,
}

impl<'a> Kmeans<'a> {
    pub fn new(
        k: usize,
        dimmensions: usize,
        space: Space<'a>,
        fitness: Box<dyn Fn(MultivaluedFloat) -> FitnessReturn>,
    ) -> Self {
        Kmeans {
            fitness: Box::new(fitness),
            space,
            k,
            dimmensions,
        }
    }

    /// Structures a mvf as a list of points so dist_euclidian can be applied to each. Meant to be
    /// called as an auxiliary function.
    pub fn mvf_as_points(mvf: &MultivaluedFloat, dimmensions: usize, k: usize) -> Vec<Point> {
        if mvf.n_vars % dimmensions != k {
            panic!("Point could not be de-structured!");
        }

        let mut slices: Vec<&[f32]> = Default::default();

        for i in 0..k {
            let slice = &mvf.get_vals()[i..(k + dimmensions)];
            slices.push(slice);
        }

        let points: Vec<Point> = slices
            .iter()
            .map(|slice| Point::new(slice))
            .collect::<Vec<Point>>();

        return points;
    }
}

impl<'a> FitnessFunction<'a, MultivaluedFloat> for Kmeans<'a> {
    // TODO: Should return Result<FitnessFunction>

    fn eval(&self, mvf: MultivaluedFloat) -> FitnessReturn {
        let centers = Kmeans::mvf_as_points(&mvf, self.dimmensions, self.k);
        0.0
    }
}

fn distance_sum(
    mvf: MultivaluedFloat,
    dimmensions: usize,
    k: usize,
    space: &mut Space,
) -> FitnessReturn {
    let centers = Kmeans::mvf_as_points(&mvf, dimmensions, k);

    0.0
}
