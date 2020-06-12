use super::genetic::traits::FitnessFunction;
use super::genetic::types::*;

pub fn squared(x: isize) -> FitnessReturn {
    isize::pow(x, 2) as f32
}

pub struct Rosenbrock {
    f: Box<dyn Fn(MultivaluedFloat) -> FitnessReturn>,
}

impl<'a> Rosenbrock {
    pub fn new() -> Self {
        Rosenbrock {
            f: Box::new(rosenbrock_banana),
        }
    }
}

impl<'a> FitnessFunction<'a, MultivaluedFloat> for Rosenbrock {
    fn get_closure<'b>(&self) -> &'a Box<dyn Fn(MultivaluedFloat) -> FitnessReturn> {
        // return &self.f;
        unimplemented!();
    }
    fn eval(&self, mvf: MultivaluedFloat) -> FitnessReturn {
        if mvf.n_vars != 2 {
            panic!(
                "Ésta función toma 2 parámetros, se recibieron {}",
                mvf.n_vars
            );
        }

        let x: f32 = mvf.vars_value[0];
        let y: f32 = mvf.vars_value[1];

        (1f32 - x).powi(2) + 100f32 * (y - x.powi(2)).powi(2)
    }
}

pub fn rosenbrock_banana(mvf: MultivaluedFloat) -> FitnessReturn {
    if mvf.n_vars != 2 {
        panic!(
            "Ésta función toma 2 parámetros, se recibieron {}",
            mvf.n_vars
        );
    }

    let x: f32 = mvf.vars_value[0];
    let y: f32 = mvf.vars_value[1];

    (1f32 - x).powi(2) + 100f32 * (y - x.powi(2)).powi(2)
}

pub fn multivalued_fn2(mvf: MultivaluedFloat) -> FitnessReturn {
    if mvf.n_vars != 2 {
        panic!(
            "Ésta función toma 2 parámetros, se recibieron {}",
            mvf.n_vars
        );
    }

    let x: f32 = mvf.vars_value[0];
    let y: f32 = mvf.vars_value[1];
    -(x - 5.0).powi(2) - (y - 7.0).powi(2) + 5.0
}

pub fn multivalued_fn_i_3(mvi: MultivaluedInteger) -> FitnessReturn {
    if mvi.n_vars != 3 {
        panic!(
            "Invalid number of variables: expected 3, got {}",
            mvi.n_vars
        );
    }
    let x = mvi.vars_value[0] as f32;
    let y = mvi.vars_value[1] as f32;
    let z = mvi.vars_value[2] as f32;

    let f = x + y + z;
    return f;
}
