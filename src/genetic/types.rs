pub type FitnessReturn = f32;

#[derive(Clone)]
pub struct MultivaluedInteger {
    pub n_vars: usize,
    pub vars_value: Vec<isize>,
}

impl MultivaluedInteger {
    pub fn new(n_vars: usize, vars_value: Vec<isize>) -> Self {
        if vars_value.len() != n_vars {
            panic!("Declared length of vector doesn't match length of supplied values");
        }
        MultivaluedInteger { n_vars, vars_value }
    }

    pub fn to_string(&self) -> String {
        let mut s = String::new();
        for i in 0..(self.n_vars - 1) {
            let mut _s = format!("{}", self.vars_value[i]);
            _s.push_str(",");
            s.push_str(&*_s)
        }
        s.push_str(&*format!("{}", self.vars_value[self.n_vars - 1]));
        format!("({})", s)
    }

    pub fn update_vals(&mut self, values: &Vec<isize>) {
        //Raises error if values.len() != self.n_vars
        if values.len() != self.n_vars {
            panic!(format!(
                "Mismatch of variable length: expected {}, got {} ",
                self.n_vars,
                values.len()
            ));
        }

        for i in 0..self.n_vars {
            self.vars_value[i] = values[i];
        }
    }

    pub fn get_vals(&self) -> &Vec<isize> {
        &self.vars_value
    }
}

///Wrapper struct for (n0,n1,...) type of values, doesn't hold info about
///its candidate representation
#[derive(Clone)]
pub struct MultivaluedFloat {
    //n_vars is always equal to vars_value.len()
    pub n_vars: usize,
    pub vars_value: Vec<f32>,
}

impl MultivaluedFloat {
    pub fn new(n_vars: usize, vars_value: Vec<f32>) -> Self {
        if vars_value.len() != n_vars {
            panic!("Declared length of vector doesn't match length of supplied values");
        }
        MultivaluedFloat { n_vars, vars_value }
    }

    pub fn to_string(&self) -> String {
        let mut s = String::new();
        for i in 0..(self.n_vars - 1) {
            let mut _s = format!("{:05.3}", self.vars_value[i]);
            _s.push_str(",");
            s.push_str(&*_s)
        }
        s.push_str(&*format!("{:05.3}", self.vars_value[self.n_vars - 1]));
        format!("({})", s)
    }

    pub fn update_vals(&mut self, values: &Vec<f32>) {
        //Raises error if values.len() != self.n_vars
        if values.len() != self.n_vars {
            panic!(format!(
                "Mismatch of variable length: expected {}, got {} ",
                self.n_vars,
                values.len()
            ));
        }

        for i in 0..self.n_vars {
            self.vars_value[i] = values[i];
        }
    }

    pub fn get_vals(&self) -> &Vec<f32> {
        &self.vars_value
    }
}
