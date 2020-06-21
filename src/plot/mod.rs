const COLUMNS: usize = 140;
const ROWS: usize = 40;
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
            self.x_equ_step = Self::map(step, (min, max), (0f32, COLUMNS as f32)).unwrap() as usize;

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
            self.y_equ_step = Self::map(step, (min, max), (0.0, ROWS as f32)).unwrap() as usize;

            if self.x_range.is_some() {
                self.displayable = true;
            }

            return Ok(self);
        }
    }

    /// Maps an x value to a valid [0,COLUMNS) value
    fn map_x<'k>(&'k self, x: f32) -> Result<f32, &str> {
        let (x_min, x_max) = self.x_range.unwrap();

        // Approximately where the y axis (x=0) is
        // let center = COLUMNS / 2 as usize;
        // println!("Center x: {}", center);

        // Maps center from screen domain to x domain
        // let mapped_center =
        // Self::map(center as f32, (0.0, COLUMNS as f32), (x_min, x_max)).unwrap();
        // println!("Mapped center x: {}", mapped_center);

        // if x < mapped_center {
        // Range x_min .. mapped_center
        // map to 0 .. center
        // Self::map(x, (mapped_center, x_min), (center as f32, 0.0))
        // } else {
        // Range  mapped_center .. x_max
        // map to center .. COLUMNS
        // Self::map(x, (mapped_center, x_max), (center as f32, COLUMNS as f32))
        //

        Self::map(x, (x_min, x_max), (0.0, COLUMNS as f32))
    }

    /// Maps a y value to a valid [0,ROWS) value
    fn map_y<'k>(&'k self, y: f32) -> Result<f32, &str> {
        let (y_min, y_max) = self.y_range.unwrap();

        let center = ROWS / 2 as usize;
        println!("Center y: {}", center);

        // Maps center from screen domain to y domain
        let mapped_center = Self::map(center as f32, (0.0, ROWS as f32), (y_max, y_min)).unwrap();
        println!("Mapped center y: {}", mapped_center);

        if y < mapped_center {
            // Range y_min .. mapped_center
            // map to 0 .. center
            Self::map(y, (y_min, mapped_center), (ROWS as f32, center as f32))
        } else {
            // Range  mapped_center .. y_may
            // map to center .. ROWS
            Self::map(y, (mapped_center, y_max), (center as f32, 0.0))
        }
    }

    pub fn set_point<'k>(
        &'k mut self,
        (x, y): (f32, f32),
        c: char,
    ) -> Result<&'k mut Plot2D, &str> {
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
        println!("({},{})", x_mapped, y_mapped);

        self.point_matrix[y_mapped as usize][x_mapped as usize] = c;

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
                    write!(file, "{}", '|').unwrap();
                }
                if j == axis_y_row && i != axis_x_row {
                    write!(file, "{}", Self::VERT_SEP).unwrap();
                }
                if i == axis_x_row && j != axis_y_row {
                    write!(file, "{}", Self::HOR_SEP).unwrap();
                }
                if i != axis_x_row && j != axis_y_row {
                    // println!("({},{})", i, j);
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
    pub fn map<'w>(val: f32, from_range: (f32, f32), to_range: (f32, f32)) -> Result<f32, &'w str> {
        let output = to_range.0
            + (val - from_range.0) * (to_range.1 - to_range.0) / (from_range.1 - from_range.0);

        Ok(output)
    }
}
