pub struct Point<T> {
    values: Vec<T>,
}

impl<T> Point<T> {
    fn new(values: Vec<T>) -> Self {
        Point::<T> { values }
    }

    fn len(&self) -> usize {
        self.values.len()
    }
}

impl Point<f32> {
    pub fn dist_euclidian(&self, point: &Point<f32>) -> Result<f32, &str> {
        if self.len() != point.len() {
            return Err("Error: Dimmension mismatch!");
        }

        let mut sum: f32 = 0.0;

        self.values
            .iter()
            .zip(point.values.iter())
            .for_each(|tuple| {
                sum += (tuple.0 - tuple.1).powi(2);
            });

        Ok(sum.sqrt())
    }

    pub fn set_values(&mut self, values: Vec<f32>) {
        self.values = values;
    }
}

#[cfg(test)]
pub mod tests {
    use super::Point;

    #[test]
    fn test_euclidian_distance() {
        let p1: Point<f32> = Point::<f32>::new(vec![7.0, 11.0]);
        let p2: Point<f32> = Point::<f32>::new(vec![40.0, -27.0]);

        assert!((p1.dist_euclidian(&p2).unwrap() - 50.32).abs() <= 0.01);
    }
}
