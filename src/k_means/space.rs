use super::point::Point;

pub type Center<'a> = Point<'a>;
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

pub struct Space<'b> {
    // clusters: Option<Vec<Cluster<'b>>>,
    dimmensions: usize,
    points: Vec<Point<'b>>,
}

impl<'b> Space<'b> {
    fn new(points: Vec<Point<'b>>, dimmensions: usize) -> Self {
        points.iter().for_each(|p| {
            if p.len() != dimmensions {
                panic!("All points must be of dimmension {}", dimmensions);
            }
        });

        Space {
            // clusters: None,
            dimmensions,
            points,
        }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn get_points(&self) -> &Vec<Point> {
        return &self.points;
    }
}
