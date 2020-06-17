use super::super::genetic::types::MultivaluedFloat;
use super::super::genetic::Bounds;
use super::cluster::Cluster;
use super::point::Point;

pub type Center<'a> = Point;
// pub type Center = Point;

// pub struct Cluster<'a> {
// center: Box<Center>,
// points: Vec<&'a Point>,
// }

// impl<'a> Cluster<'a> {
// pub fn new(center: Center) -> Self {
// Cluster {
// center,
// points: vec![],
// }
// }

// pub fn len(&self) -> usize {
// self.points.len()
// }

// pub fn get_center(&self) -> &Center {
// &self.center
// }

// pub fn add_point(&mut self, point: &'a Point) {
// self.points.push(point);
// }
// }

// pub struct Space {
pub struct Space {
    clusters: Option<Vec<Cluster>>,
    dimmensions: usize,
    pub points: Vec<Point>,

    /// Only set to true after all points have been tagged and clustered.
    tagged: bool,
}

impl Space {
    fn new(points: Vec<Point>, dimmensions: usize) -> Self {
        points.iter().for_each(|p| {
            if p.len() != dimmensions {
                panic!("All points must be of dimmension {}", dimmensions);
            }
        });

        Space {
            clusters: None,
            dimmensions,
            points,
            tagged: false,
        }
    }

    /// Finds the highest and lowest values of a point vector reference
    /// and gets its floor and ceiling.
    /// i.e: if (min,max) = (2, 456) this will yield (0,1000).
    /// As the method can be equally called over a flattened Cluster list or
    /// an untouched Point list from a Space instance, the method will remain static.
    /// The actual max and min values will get cached once calculated as the point objects don't
    /// change over the program's lifetime.
    pub fn get_bounds<'p>(
        points: &'p Vec<&Point>,
        dimmensions: usize,
    ) -> Result<Bounds<MultivaluedFloat>, String> {
        if points.len() == 0 {
            return Err("Tried to find bounds in an empty point vector!".to_string());
        }

        // Find (min, max) of (x0,x1,...)
        let (min_vals, max_vals) = {
            let mut _min_vals = vec![std::f32::MAX; dimmensions];
            let mut _max_vals = vec![std::f32::MIN; dimmensions];

            points.iter().for_each(|p: &&Point| {
                let current = p.get_values().iter();

                current.enumerate().for_each(|(i, v): (usize, &f32)| {
                    if *v < _min_vals[i] {
                        _min_vals[i] = *v;
                    }
                    if *v > _max_vals[i] {
                        _max_vals[i] = *v;
                    }
                });
            });

            (_min_vals, _max_vals)
        };

        // Get order of (x0,x1,...)

        let (lower_order, upper_order) = {
            let _lower = min_vals.iter().map(|l| l.log10().floor() as i32);
            let _upper = max_vals.iter().map(|u| u.log10().ceil() as i32);
            (_lower, _upper)
        };

        // Get bounds for (x0,x1,...)
        let (lower, upper) = {
            let _lower = lower_order.map(|l| 10f32.powi(l)).collect::<Vec<f32>>();
            let _upper = upper_order.map(|u| 10f32.powi(u)).collect::<Vec<f32>>();

            (
                MultivaluedFloat::new(dimmensions, _lower),
                MultivaluedFloat::new(dimmensions, _upper),
            )
        };

        Ok(Bounds::new(lower, upper))
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn get_points(&self) -> &Vec<Point> {
        return &self.points;
    }

    pub fn into_ref_list<'s>(&'s mut self) -> Vec<&Point> {
        // If already tagged and clustered, flatten points
        match self.get_clusters() {
            Some(clusters) => {
                let flattened_clusters = {
                    let mut _flattened: Vec<&Point> = vec![];
                    clusters.iter().for_each(|c| {
                        c.get_points().iter().for_each(|p| {
                            _flattened.push(p);
                        });
                    });
                    _flattened
                };
            }

            None => {
                // Space not tagged yet.
                if self.is_tagged() {
                    panic!("Error in code!, this state should not be reachable");
                }

                return self.get_points().iter().collect::<Vec<&Point>>();
            }
        }
        unimplemented!();
    }

    pub fn is_tagged(&self) -> bool {
        return self.tagged;
    }

    pub fn set_tagged(&mut self, flag: bool) {
        self.tagged = flag;
    }

    pub fn get_clusters<'x>(&'x mut self) -> Option<&'x Vec<Cluster>> {
        match &self.clusters {
            Some(vec) => return Some(vec),
            None => return None,
        }
    }
}
