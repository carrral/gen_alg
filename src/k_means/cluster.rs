use super::point::Point;

pub enum PlotSettings {
    /// Stores (graph_char, hex)
    Pair(char, String),
}

/// Stores a list of Points,
// pub struct Cluster<'a> {
pub struct Cluster {
    points: Vec<Point>,
    id: usize,
    display_settings: Option<PlotSettings>,
    center: Point,
    current_point: usize,
}

impl Cluster {
    pub fn new(center: Point) -> Cluster {
        Cluster {
            points: vec![],
            id: center.get_id(),
            display_settings: None,
            center,
            current_point: 0,
        }
    }

    pub fn get_cluster_id(&self) -> usize {
        self.id
    }

    pub fn add_point(&mut self, point: Point) {
        self.points.push(point);
    }

    pub fn get_points<'b>(&'b self) -> &'b Vec<Point> {
        &self.points
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn set_plot_settings(&mut self, plot_settings: PlotSettings) {
        self.display_settings = Some(plot_settings);
    }

    pub fn get_plot_settings<'k>(&'k self) -> Option<&'k PlotSettings> {
        match &self.display_settings {
            Some(v) => Some(v),
            None => None,
        }
    }
}
