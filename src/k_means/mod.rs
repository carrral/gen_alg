pub mod cluster;
pub mod point;
pub mod space;
pub mod utils;
pub mod wrapper;
use super::genetic::implementations::multi_valued::RCCList;

use super::genetic::traits::FitnessFunction;
use super::genetic::types::FitnessReturn;
use super::genetic::types::MultivaluedFloat;
use super::genetic::utils::debug_msg;
use super::genetic::utils::random_range;
use super::plot::Plot2D;
use cluster::{Cluster, PlotSettings};
use gnuplot::{AutoOption, AxesCommon, Caption, Color, Figure, PointSymbol};
use point::Point;
use space::{Center, Space};
use std::collections::HashMap;
use std::fmt;
use utils::{gen_id_list, gen_plot_settings};
use wrapper::ClusterList;

pub struct Kmeans {
    space: Space,
    k: usize,
    dimmensions: usize,
}

impl Kmeans {
    pub fn new(k: usize, dimmensions: usize, space: Space) -> Self {
        Kmeans {
            space,
            k,
            dimmensions,
        }
    }

    /// Structures a mvf as a list of points so dist_euclidian can be applied to each. Meant to be
    /// called as an auxiliary function.
    pub fn mvf_as_points(mvf: &MultivaluedFloat, dimmensions: usize, k: usize) -> Vec<Point> {
        if !(mvf.n_vars % dimmensions == 0 && mvf.n_vars / dimmensions == k) {
            panic!("Point could not be de-structured!");
        }

        let mut slices: Vec<&[f32]> = Default::default();

        for i in 0..k {
            let slice = &mvf.get_vals()[(i * dimmensions)..(i * dimmensions + dimmensions)];
            slices.push(slice);
        }

        let points: Vec<Point> = slices
            .iter()
            .map(|slice| Point::new(slice.to_vec()))
            .collect::<Vec<Point>>();

        return points;
    }

    /// Meant to be called after genetic_optimize.
    /// After the algorithm has found the actual result,
    /// this method will tag each point accordingly
    pub fn cluster<'a>(&'a mut self, mvf: MultivaluedFloat) -> Result<&'a Space, String> {
        let ids = gen_id_list(self.k);
        let mut unique_settings = gen_plot_settings(self.k)?;
        let mut centers = Kmeans::mvf_as_points(&mvf, self.dimmensions, self.k);

        // Tag each center

        centers
            .iter_mut()
            .enumerate()
            .zip(ids)
            .for_each(|((i, c), id)| {
                c.set_id(id);
            });

        // Find the closest custer center for each point and tag each point
        self.space.points.iter_mut().for_each(|point| {
            let mut _min_dist = std::f32::MAX;
            let mut closest: usize = 0;

            centers.iter().enumerate().for_each(|(i, center)| {
                let dist = center.dist_euclidian(point).unwrap();

                if dist < _min_dist {
                    _min_dist = dist;
                    closest = i;
                }
            });

            point.set_id(centers[closest].get_id());
        });

        self.space.set_tagged(true);

        // Initialize k cluster objects.
        let mut clusters: Vec<Cluster> = Default::default();

        for i in 0..centers.len() {
            // let point = centers.remove(i);
            let point = centers[i].to_owned();
            // println!("len: {}", centers.len());
            let mut cluster = Cluster::new(point);
            let settings = unique_settings[i].clone();
            cluster.set_plot_settings(settings);
            clusters.push(cluster);
        }

        let mut clusters_id_map: HashMap<usize, &mut Cluster> = HashMap::new();

        clusters.iter_mut().for_each(|cluster: &mut Cluster| {
            clusters_id_map.insert(cluster.get_cluster_id(), cluster);
        });

        // Iterate through points and  filter with id
        // Add to cluster
        for i in 0..self.space.len() {
            let point = self.space.points[i].to_owned();
            let option = clusters_id_map.get_mut(&point.get_id());
            match option {
                Some(cluster) => cluster.add_point(point.into()),
                None => panic!("This state should have not been reached!!"),
            }
        }

        self.get_space_mut().set_clusters(clusters);

        return Ok(self.get_space());
    }

    pub fn get_space_mut<'b>(&'b mut self) -> &'b mut Space {
        &mut self.space
    }

    pub fn get_space<'b>(&'b self) -> &'b Space {
        &self.space
    }

    pub fn make_plot<'a>(&'a mut self) -> Result<Plot2D, String> {
        let dimm = self.dimmensions;

        if dimm > 3 {
            return Err("Display impossible for dimmensions > 3".to_string());
        }

        let global_res: Result<Figure, String> = Err("Unexpected return occurred".to_string());

        let mut figure = Plot2D::new();
        let points = self.get_space_mut().into_ref_list();

        // Get bounds of flattened clusters
        let bounds;
        match Space::get_bounds(&points, dimm) {
            Ok(b) => bounds = b,
            Err(e) => return Err(e),
        }

        match self.get_space_mut().get_clusters() {
            Some(clusters) => {
                match dimm {
                    1 => return Err("Display of dim=1 not supported yet!".to_string()),
                    2 => {
                        // Set bounds in figure
                        let (upper_x, upper_y) = (bounds.get_nth_upper(0), bounds.get_nth_upper(1));
                        let (lower_x, lower_y) = (bounds.get_nth_lower(0), bounds.get_nth_lower(1));
                        figure.set_x_range(lower_x, upper_x, 1.0).unwrap();

                        figure.set_y_range(lower_y, upper_y, 1.0).unwrap();

                        clusters.iter().for_each(|c: &Cluster| {
                            // Get x values
                            // Get y values
                            // Get PlotSettings
                            let mut plot_char: char = ' ';

                            match c.get_plot_settings() {
                                Some(ps) => match ps {
                                    PlotSettings::Pair(c, h) => {
                                        plot_char = *c;
                                    }
                                },
                                None => unreachable!(),
                            }
                            //
                            c.get_points().iter().for_each(|p: &Point| {
                                let _x = p.nth_value(0);
                                let _y = p.nth_value(1);
                                figure.set_point((_x, _y), plot_char).unwrap();
                            });

                            // println!("{:?},{:?}", c, x);
                        });

                        return Ok(figure);
                    }
                    3 => return Err("3-D space not supported yet!".to_string()),
                    _ => unreachable!(),
                }
            }
            None => return Err("Space has not been tagged yet".to_string()),
        }
    }
}

impl<'a> FitnessFunction<'a, MultivaluedFloat> for Kmeans {
    // TODO: Should return Result<FitnessFunction>

    fn eval(&self, mvf: MultivaluedFloat) -> FitnessReturn {
        let centers = Kmeans::mvf_as_points(&mvf, self.dimmensions, self.k);

        // Find the closest custer center for each point

        let mut sum = 0.0;
        self.space.get_points().iter().for_each(|point| {
            let mut _min_dist = std::f32::MAX;

            centers.iter().for_each(|center: &Point| {
                let dist = center.dist_euclidian(point).unwrap();

                if dist < _min_dist {
                    _min_dist = dist;
                }
            });

            sum += _min_dist;
        });

        return sum;
    }
}
