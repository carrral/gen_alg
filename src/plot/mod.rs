const COLUMNS: usize = 90;
const ROWS: usize = 35;
const BLANK: char = ' ';

pub struct Plot2D {
    /// ROWS X COLUMNS matrix
    point_matrix: [[char; COLUMNS]; ROWS],
    displayable: bool,
    x_range: Option<(f32, f32)>,
    /// Equivalent displacement of x step in graph
    x_equ_step: usize,
    y_range: Option<(f32, f32)>,
    /// Equivalent displacement of y step in graph
    y_equ_step: usize,
}

impl Plot2D {
    const VERT_SEP: char = '|';
    const HOR_SEP: char = '_';

    pub fn new() -> Self {
        Plot2D {
            point_matrix: [[BLANK; COLUMNS]; ROWS],
            displayable: false,
            x_range: None,
            x_equ_step: 0,
            y_range: None,
            y_equ_step: 0,
        }
    }

    pub fn set_x_range<'k>(
        &'k mut self,
        min: f32,
        max: f32,
        step: f32,
    ) -> Result<&'k mut Plot2D, &str> {
        if max <= min || (max - min) < step {
            return Err("Invalid range");
        } else {
            self.x_range = Some((min, max));

            // Normalize x step
            self.x_equ_step = Self::map(
                step as isize,
                (min as isize, max as isize),
                (0, COLUMNS as isize),
            )
            .unwrap() as usize;

            if self.y_range.is_some() {
                self.displayable = true;
            }

            return Ok(self);
        }
    }

    pub fn set_y_range<'k>(
        &'k mut self,
        min: f32,
        max: f32,
        step: f32,
    ) -> Result<&'k mut Plot2D, &str> {
        if max <= min || (max - min) < step {
            return Err("Invalid range");
        } else {
            self.y_range = Some((min, max));

            // Normalize y step
            self.y_equ_step = Self::map(
                step as isize,
                (min as isize, max as isize),
                (0, ROWS as isize),
            )
            .unwrap() as usize;

            if self.x_range.is_some() {
                self.displayable = true;
            }

            return Ok(self);
        }
    }

    fn map_x<'k>(&'k self, x: f32) -> Result<isize, &str> {
        let (x_min, x_max) = self.x_range.unwrap();
        Self::map(
            x as isize,
            (x_min as isize, x_max as isize),
            (0, COLUMNS as isize),
        )
    }

    fn map_y<'k>(&'k self, y: f32) -> Result<isize, &str> {
        let (y_min, y_max) = self.y_range.unwrap();
        Self::map(
            y as isize,
            (y_min as isize, y_max as isize),
            (0, ROWS as isize),
        )
    }

    fn set_point<'k>(&'k mut self, (x, y): (f32, f32), c: char) -> Result<&'k mut Plot2D, &str> {
        let mut _x = 0;
        let mut _y = 0;

        let (x_min, x_max) = self.x_range.unwrap();
        let (y_min, y_max) = self.y_range.unwrap();

        if x < x_min || x > x_max {
            return Err("X value not in range");
        }

        if y < y_min || y > y_max {
            return Err("Y value not in range");
        }

        let x_mapped = self.map_x(x).unwrap();
        let y_mapped = self.map_y(y).unwrap();

        self.point_matrix[x_mapped as usize][y_mapped as usize] = c;

        return Ok(self);
    }

    pub fn draw<'w>(&'w self, file: &mut dyn std::io::Write) -> Result<(), &str> {
        if !self.displayable {
            return Err("2-D plot not displayable yet");
        }

        let axis_x_row = ROWS / 2 as usize;
        let axis_y_row = COLUMNS / 2 as usize;
        let mut vert_step_counter = 0;
        let mut hor_step_counter = 0;
        // let mut hor_axis_flag = false;
        // let mut vert_axis_flag = false;

        for i in 0..ROWS {
            for j in 0..COLUMNS {
                // if j == axis_y_row {
                // vert_axis_flag = true;
                // }

                if j == axis_y_row && i == axis_x_row {
                    write!(file, "{}", '.').unwrap();
                }
                if j == axis_y_row && i != axis_x_row {
                    write!(file, "{}", Self::VERT_SEP).unwrap();
                }
                if i == axis_x_row && j != axis_y_row {
                    write!(file, "{}", Self::HOR_SEP).unwrap();
                }
                if i != axis_x_row && j != axis_y_row {
                    write!(file, "{}", self.point_matrix[i][j]).unwrap();
                }

                vert_step_counter += 1;
            }
            write!(file, "\n").unwrap();
            hor_step_counter += 1;
        }

        Ok(())
    }

    /// Returns a value between 0 and k
    fn map<'w>(
        val: isize,
        (old_min, old_max): (isize, isize),
        (new_min, new_max): (isize, isize),
    ) -> Result<isize, &'w str> {
        if val < old_min || val > old_max {
            return Err("Value outside of range");
        }

        let old_range = old_max - old_min;

        if old_range == 0 {
            Ok(new_min)
        } else {
            let new_range = new_max - new_min;
            let new_val = ((val - old_min) * new_range) / old_range;
            Ok(new_val)
        }
    }
}
