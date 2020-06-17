#[derive(Clone)]
// #[derive(Copy)]
pub struct Point<'a> {
    values: &'a [f32],
    id: Option<usize>,
}

impl<'a> Point<'a> {
    // impl Point {
    pub fn new(values: &'a [f32]) -> Self {
        Point { values, id: None }
    }

    pub fn get_values(&self) -> &[f32] {
        self.values
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn dist_euclidian(&self, point: &Point) -> Result<f32, &str> {
        if self.len() != point.len() {
            return Err("Error: Dimmension mismatch!");
        }

        let mut sum: f32 = 0.0;

        self.get_values()
            .iter()
            .zip(point.get_values().iter())
            .for_each(|tuple| {
                sum += (tuple.0 - tuple.1).powi(2);
            });

        Ok(sum.sqrt())
    }

    pub fn nth_value(&self, n: usize) -> f32 {
        self.values[n]
    }

    pub fn to_string(&self) -> String {
        format!("Point({:?})", self.get_values())
    }

    pub fn set_id(&mut self, id: usize) {
        self.id = Some(id);
    }

    pub fn get_id(&self) -> usize {
        self.id.unwrap()
    }

    // pub fn set_values(&mut self, values: &'a [f32]) {
    // self.values = values;
    // }
}

#[cfg(test)]
pub mod tests {
    use super::Point;

    #[test]
    fn test_euclidian_distance() {
        let mut v1 = vec![7.0, 11.0];
        let mut v2 = vec![40.0, -27.0];
        let p1: Point = Point::new(&v1[..]);
        let p2: Point = Point::new(&v2[..]);

        assert!((p1.dist_euclidian(&p2).unwrap() - 50.32).abs() <= 0.01);

        v1[0] = 0.0;
        v1[1] = -4.0;
        v2[0] = 8.0;
        v2[1] = 100.0;

        // assert!((p1.dist_euclidian(&p2).unwrap() - 104.3072).abs() <= 0.01);
    }
}
