use super::point::Point;

type Center = Point<f32>;

pub struct Cluster {
    center: Center,
    points: Vec<Point<f32>>,
}

pub struct Space {
    clusters: Option<Vec<Cluster>>,
    dimmensions: usize,
    k: Option<usize>,
}
