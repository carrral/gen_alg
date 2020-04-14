// #![allow(dead_code)]
#![allow(non_camel_case_types)]
// #![allow(unused_variables)]

pub mod genetic_alg;

use genetic_alg::types::*;
use genetic_alg::{utils, Candidate, CandidateList, OptimizeType, StopCondition};
use gnuplot::{Caption, Color, Figure};
use rand::{distributions::WeightedIndex, prelude::*, Rng};

// ==-- STRUCTS --==

pub struct IntegerCandidateList {
    ind_size: usize,
    candidates: Vec<IntegerCandidate>,
}

#[derive(Clone)]
pub struct IntegerCandidate {
    pub value: String,
    pub fitness: Option<FitnessReturn>,
    pub mutated: bool,
    pub selected: bool,
}

struct MultivariedFloatCandidate {
    pub vars: MultivariedFloat,
    pub value: String,
    pub fitness: Option<FitnessReturn>,
    pub mutated: bool,
    pub selected: bool,
}

struct MVFCandidateList {
    n_vars: usize,
    /// Should be n_vars * 32
    ind_size: usize,
    candidates: Vec<MultivariedFloatCandidate>,
}

#[derive(Clone)]
struct MultivariedFloat {
    ///Wrapper struct for (n0,n1,...) type of values, doesn't hold info about
    ///its candidate representation
    //n_vars is always equal to vars_value.len()
    pub n_vars: usize,
    pub vars_value: Vec<f32>,
}

// ==-- STRUCT IMPLEMENTATIONS --==
impl IntegerCandidate {
    pub fn new(value: String) -> Self {
        IntegerCandidate {
            value,
            fitness: None,
            mutated: false,
            selected: false,
        }
    }

    fn get_integer_representation(&self) -> isize {
        let slice: &str = &*self.value;
        let self_int: isize = isize::from_str_radix(slice, 2).unwrap();
        return self_int;
    }

    fn len(&self) -> usize {
        self.value.chars().count()
    }
}

impl IntegerCandidateList {
    fn get_n_fittest(&mut self, n: usize, opt_type: &OptimizeType) -> &[IntegerCandidate] {
        self.sort(opt_type);

        let start_i: usize = self.len() - n;

        &self.candidates[start_i..]
    }
}

impl MultivariedFloat {
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

impl<'a> MultivariedFloatCandidate {
    const FLOAT_LEN: usize = 32;

    pub fn len(&self) -> usize {
        self.value.chars().count()
    }

    fn get_var_substrings(s: &String, n_vars: usize) -> Vec<&str> {
        // Returns a vector of f32 values entirely dependant on bit string

        // Sub divide single candidate string into individual sub strings
        let mut substrings: Vec<&str> = vec![""; n_vars];

        let r = MultivariedFloatCandidate::FLOAT_LEN;
        let l = s.chars().count();

        // Assumes l == n_vars*FLOAT_LEN
        for start_index in (0..l).step_by(r) {
            substrings[start_index / r] = &s[start_index..start_index + r];
        }

        return substrings;
    }

    fn get_vars_from_bit_string(&self) -> Vec<f32> {
        /// Returns a vector of the floats containted in self.value (bit string)
        /// Internal use meant only for updating MultivariedFloat's values, not intended for
        /// use by the user.
        let substrings =
            MultivariedFloatCandidate::get_var_substrings(&self.value, self.vars.n_vars);
        let values = substrings
            .iter()
            .map(|x| utils::parse_f32(&x.to_string()))
            .collect::<Vec<f32>>();

        return values;
    }

    pub fn update_values(&mut self) {
        let values = self.get_vars_from_bit_string();
        self.vars.update_vals(&values);
    }

    pub fn new(n_vars: usize, value: String) -> Self {
        //n_vars: Number of variables
        //value: bit string representation of parameters

        let len = value.chars().count();

        if !(len / MultivariedFloatCandidate::FLOAT_LEN == n_vars
            && len % MultivariedFloatCandidate::FLOAT_LEN == 0)
        {
            panic!("Length of value String and number of variables is incompatible");
        }

        let substrings = MultivariedFloatCandidate::get_var_substrings(&value, n_vars);

        let vars_value = substrings
            .iter()
            .map(|x| utils::parse_f32(&x.to_string()))
            .collect::<Vec<f32>>();

        MultivariedFloatCandidate {
            vars: MultivariedFloat { n_vars, vars_value },
            value,
            fitness: None,
            mutated: false,
            selected: false,
        }
    }
}

impl MVFCandidateList {
    pub fn get_n_fittest(
        &mut self,
        n: usize,
        opt_type: &OptimizeType,
    ) -> &[MultivariedFloatCandidate] {
        self.sort(opt_type);

        let start_i: usize = self.len() - n;

        &self.candidates[start_i..]
    }

    pub fn new(n_vars: usize) -> Self {
        MVFCandidateList {
            n_vars,
            ind_size: n_vars * MultivariedFloatCandidate::FLOAT_LEN,
            candidates: Default::default(),
        }
    }
}

//==-- TRAIT IMPLEMENTATIONS --==

impl CandidateList<isize, isize> for IntegerCandidateList {
    fn len(&self) -> usize {
        self.candidates.len()
    }

    // Generates an initial vector of Random Candidates and stores it in self
    fn generate_initial_candidates(&mut self, requested: usize) {
        for _ in 0..requested {
            let s: String = utils::generate_random_bitstring(self.ind_size);
            let c = IntegerCandidate::new(s);
            self.candidates.push(c);
        }
    }

    // Returns (max_fitness, avg_fitness)
    fn get_diagnostics(&self, opt_type: &OptimizeType) -> (FitnessReturn, FitnessReturn) {
        let mut total_fitness = 0.0;
        let mut max_fitness = match *opt_type {
            OptimizeType::MAX => std::f32::MIN,
            OptimizeType::MIN => std::f32::MAX,
        };

        if self.candidates[0].get_fitness().is_none() {
            panic!("Fitness hasn't been calculated yet");
        }

        for candidate in &self.candidates {
            let fitness = match candidate.get_fitness() {
                Some(v) => v,
                None => 0.0, // Unreachable
            };
            total_fitness += fitness;
            match *opt_type {
                OptimizeType::MAX => {
                    if max_fitness < fitness {
                        max_fitness = fitness;
                    }
                }
                OptimizeType::MIN => {
                    if max_fitness > fitness {
                        max_fitness = fitness;
                    }
                }
            };
        }
        (max_fitness, total_fitness / self.len() as f32)
    }

    // Updates internal CandidateList
    fn mate(&mut self, n_out: usize, n_selected: usize, prob_rep: f32, opt_type: &OptimizeType) {
        //FIXME: Regresa Result si n_selected > n_out o n_selected > self.len
        use genetic_alg::utils::*;

        // @prob_rep: Probability of reproduction
        // @n_out: # of Candidates selected each round for reproduction
        let mut new_candidates: Vec<IntegerCandidate> = Default::default();

        let size = self.ind_size;

        // Select @n_out best candidates
        let best_candidates: &[IntegerCandidate] = self.get_n_fittest(n_selected, opt_type);

        // if DEBUG {
        // println!();
        // debug_msg("Seleccionados");
        // debug_candidates(&best_candidates);
        // }

        //Probar de la siguiente manera:
        // Para cada Ciclo, hasta que no se junten los N requeridos:
        //      Se tienen dos listas idénticas de candidatos, con un desfasamiento
        //      Ciclo 1:    Candidato 1 -> Candidato 2
        //                  Candidato 2 -> Candidato 3
        //                  Candidato 3 -> Candidato 1
        //
        //      Ciclo 2:    Candidato 1 -> Candidato 3
        //                  Candidato 2 -> Candidato 1
        //                  Candidato 3 -> Candidato 2

        let mut offset: usize = 1;
        let k: usize = splitting_point(size, prob_rep);

        loop {
            let mut break_loop = false;
            for i in 0..n_selected {
                let offset_index = (i + offset) % n_selected;
                let current_offspring_vals = genetic_alg::utils::cross_strings(
                    &best_candidates[i].value,
                    &best_candidates[offset_index].value,
                    k,
                );
                let offspring = (
                    IntegerCandidate::new(String::from(current_offspring_vals.0)),
                    IntegerCandidate::new(String::from(current_offspring_vals.1)),
                );
                new_candidates.push(offspring.0);
                new_candidates.push(offspring.1);

                if new_candidates.len() >= n_out {
                    break_loop = true;
                    break;
                }

                // TODO: Reloj modulo para evitar out of bounds panic
            }

            if break_loop {
                break;
            }

            offset += 1;
        }

        self.candidates = new_candidates;
    }

    // Operates on the whole list with a given probability mut_pr
    fn mutate_list(&mut self, mut_pr: f32, opt: &OptimizeType) {
        let cointoss = [mut_pr, 1f32 - mut_pr];
        for candidate in &mut self.candidates {
            // result: 0 -> mutate, 1 -> do nothing
            let result = utils::roulette(&cointoss);
            match result {
                0 => candidate.mutate(opt),
                _ => continue,
            };
        }
    }

    //Evaluates fitness for the whole candidate list
    fn eval_fitness(&mut self, f: fn(isize) -> FitnessReturn) {
        for candidate in &mut self.candidates {
            candidate.eval_fitness(f);
        }
    }

    fn debug(&self) {
        for candidate in &self.candidates {
            candidate.debug();
        }
    }

    fn get_results(&mut self, opt_type: &OptimizeType) -> (isize, FitnessReturn) {
        let fittest = self.get_n_fittest(1, opt_type);
        let max_fitness = match fittest[0].get_fitness() {
            Some(v) => v,
            // TODO: Raise panic
            None => 0.0,
        };
        (fittest[0].get_integer_representation(), max_fitness)
    }

    fn sort(&mut self, opt_type: &OptimizeType) {
        match *opt_type {
            // Ordena el vector en forma ascendente ->  [0,1,...,N]
            OptimizeType::MAX => self
                .candidates
                .sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap()),
            // Ordena el vector en forma descendente -> [N,...,1,0]
            OptimizeType::MIN => self
                .candidates
                .sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap()),
        };
    }
}

impl Default for IntegerCandidateList {
    fn default() -> Self {
        IntegerCandidateList {
            ind_size: 8,
            candidates: Default::default(),
        }
    }
}

impl Candidate<isize> for IntegerCandidate {
    fn eval_fitness(&mut self, f: fn(isize) -> FitnessReturn) -> FitnessReturn {
        let self_int = self.get_integer_representation();
        self.fitness = Some(f(self_int));
        f(self_int)
    }

    fn get_fitness(&self) -> Option<FitnessReturn> {
        self.fitness
    }

    fn to_string(&self) -> String {
        let fitness = match self.get_fitness() {
            Some(v) => v.to_string(),
            None => String::from("UNDEFINED"),
        };
        let s = String::from(format!(
            "IntegerCandidate{{val: {}, int: {}, fit:{}, mut: {}}}",
            self.value,
            self.get_integer_representation(),
            fitness,
            self.mutated
        ));

        return s;
    }

    fn debug(&self) {
        println!("{}", self.to_string());
    }

    fn mutate(&mut self, opt_type: &OptimizeType) {
        let mutated = utils::generic_mutate(&self.value, opt_type);
        self.value = mutated;
        self.mutated = true;
    }
}

impl Default for IntegerCandidate {
    fn default() -> Self {
        IntegerCandidate::new(String::new())
    }
}

impl<'a> Candidate<MultivariedFloat> for MultivariedFloatCandidate {
    // Evalúa el valor de fitness de self.value, lo asigna en self.fitness
    // y lo regresa.
    fn eval_fitness(&mut self, f: fn(MultivariedFloat) -> FitnessReturn) -> FitnessReturn {
        //FIXME: Clone?
        // let v: MultivariedFloat = self.vars.clone();
        let f = f(self.vars.clone());

        self.fitness = Some(f);
        return f;
    }

    fn to_string(&self) -> String {
        let fit = match self.get_fitness() {
            Some(v) => v.to_string(),
            None => String::from("UNDEFINED"),
        };
        format!(
            "MVFCandidate{{vars:{}, fitness:{},selected:{}}}",
            self.vars.to_string(),
            fit,
            self.selected
        )
    }

    fn get_fitness(&self) -> Option<FitnessReturn> {
        self.fitness
    }
    fn debug(&self) {
        println!("{}", self.to_string());
    }
    fn mutate(&mut self, opt_type: &OptimizeType) {
        // TODO: es generic_mutate la función más apropiada?
        self.value = utils::generic_mutate(&self.value, opt_type);

        //Update vars_value
        let new_values = self.get_vars_from_bit_string();
        self.vars.update_vals(&new_values);
    }
}

impl CandidateList<MultivariedFloatCandidate, MultivariedFloat> for MVFCandidateList {
    // Generates an initial vector of Random Candidates
    fn generate_initial_candidates(&mut self, requested: usize) {
        for i in 0..requested {
            let s: String = utils::generate_random_bitstring(self.ind_size);
            let c = MultivariedFloatCandidate::new(self.n_vars, s);
            self.candidates.push(c);
        }
    }

    // Returns (max_fitness, avg_fitness)
    fn get_diagnostics(&self, opt_type: &OptimizeType) -> (FitnessReturn, FitnessReturn) {
        let mut total_fitness = 0.0;

        if self.candidates[0].get_fitness().is_none() {
            panic!("Fitness hasn't been calculated yet");
        }

        let mut max_fitness = match *opt_type {
            OptimizeType::MAX => std::f32::MIN,
            OptimizeType::MIN => std::f32::MAX,
        };

        for candidate in &self.candidates {
            let fitness = match candidate.get_fitness() {
                Some(v) => v,
                None => 0.0, // Unreachable
            };

            match *opt_type {
                OptimizeType::MAX => {
                    if fitness > max_fitness {
                        max_fitness = fitness
                    }
                }
                OptimizeType::MIN => {
                    if fitness < max_fitness {
                        max_fitness = fitness
                    }
                }
            }

            total_fitness += fitness;
        }

        (max_fitness, total_fitness / self.len() as f32)
    }

    // Updates internal CandidateList
    fn mate(&mut self, n_out: usize, n_selected: usize, prob_rep: f32, opt_type: &OptimizeType) {
        //FIXME: Regresa Result si n_selected > n_out o n_selected > self.len
        use genetic_alg::utils::*;

        // @prob_rep: Probability of reproduction
        // @n_out: # of Candidates selected each round for reproduction
        let mut new_candidates: Vec<MultivariedFloatCandidate> = Default::default();

        let size = self.ind_size;

        let n_vars = self.n_vars;

        // Select @n_out best candidates
        let best_candidates: &[MultivariedFloatCandidate] =
            self.get_n_fittest(n_selected, opt_type);

        // if DEBUG {
        // println!();
        // debug_msg("Seleccionados");
        // debug_candidates(&best_candidates);
        // }

        //Probar de la siguiente manera:
        // Para cada Ciclo, hasta que no se junten los N requeridos:
        //      Se tienen dos listas idénticas de candidatos, con un desfasamiento
        //      Ciclo 1:    Candidato 1 -> Candidato 2
        //                  Candidato 2 -> Candidato 3
        //                  Candidato 3 -> Candidato 1
        //
        //      Ciclo 2:    Candidato 1 -> Candidato 3
        //                  Candidato 2 -> Candidato 1
        //                  Candidato 3 -> Candidato 2

        let mut offset: usize = 1;
        let k: usize = splitting_point(size, prob_rep);

        loop {
            let mut break_loop = false;
            for i in 0..n_selected {
                let offset_index = (i + offset) % n_selected;
                let current_offspring_vals = genetic_alg::utils::cross_strings(
                    &best_candidates[i].value,
                    &best_candidates[offset_index].value,
                    k,
                );
                let offspring = (
                    MultivariedFloatCandidate::new(n_vars, String::from(current_offspring_vals.0)),
                    MultivariedFloatCandidate::new(n_vars, String::from(current_offspring_vals.1)),
                );
                new_candidates.push(offspring.0);
                new_candidates.push(offspring.1);

                if new_candidates.len() >= n_out {
                    break_loop = true;
                    break;
                }

                // TODO: Reloj modulo para evitar out of bounds panic
            }

            if break_loop {
                break;
            }

            offset += 1;
        }

        self.candidates = new_candidates;
    }

    // Operates on the whole list with a given probability mut_pr
    fn mutate_list(&mut self, mut_pr: f32, opt: &OptimizeType) {
        let cointoss = [mut_pr, 1f32 - mut_pr];
        for candidate in &mut self.candidates {
            // result: 0 -> mutate, 1 -> do nothing
            let result = utils::roulette(&cointoss);
            match result {
                0 => candidate.mutate(opt),
                _ => continue,
            };
        }
    }

    //Evaluates fitness for the whole candidate list
    fn eval_fitness(&mut self, f: fn(MultivariedFloat) -> FitnessReturn) {
        //FIXME: Agregar un genérico en la definición del trait, diferente a candidate
        for candidate in &mut self.candidates {
            candidate.eval_fitness(f);
        }
    }

    fn len(&self) -> usize {
        return self.candidates.len();
    }

    fn debug(&self) {
        for c in &self.candidates {
            c.debug();
        }
    }

    // Regresa el mejor resultado encontrado en una generación
    fn get_results(&mut self, opt_type: &OptimizeType) -> (MultivariedFloat, FitnessReturn) {
        self.sort(opt_type);
        let x = self.candidates.pop().unwrap();
        let (v, f): (MultivariedFloat, FitnessReturn) = (x.vars, x.fitness.unwrap());
        (v, f)
    }

    fn sort(&mut self, opt_type: &OptimizeType) {
        match *opt_type {
            // Ordena el vector en forma ascendente ->  [0,1,...,N]
            OptimizeType::MAX => self
                .candidates
                .sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap()),
            // Ordena el vector en forma descendente -> [N,...,1,0]
            OptimizeType::MIN => self
                .candidates
                .sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap()),
        };
    }
}

// TODO: Mover
trait ShowGraph {
    fn show_graph(
        x_axis: &[usize],
        y_axis: &[FitnessReturn],
        title: &str,
        color: &str,
    ) -> Result<bool, std::io::Error>;
}

fn squared(x: isize) -> FitnessReturn {
    isize::pow(x, 2) as f32
}

fn multivalued_fn2(mvf: MultivariedFloat) -> FitnessReturn {
    if mvf.n_vars != 2 {
        panic!(
            "Ésta función toma 2 parámetros, se recibieron {}",
            mvf.n_vars
        );
    }

    let x: f32 = mvf.vars_value[0];
    let y: f32 = mvf.vars_value[1];
    -x.powi(2) - y.powi(2) + 5.0
}

fn main() {
    let mut l = IntegerCandidateList::default();
    let mut mvfl = MVFCandidateList::new(2);

    let results = basic_genetic_algorithm(
        8,
        4,
        &mut mvfl,
        multivalued_fn2,
        0.5,
        0.1,
        &OptimizeType::MIN,
        &StopCondition::CYCLES(10),
    );

    let mvf = MultivariedFloat {
        n_vars: 3,
        vars_value: vec![1.0, 2.3, 3.4],
    };

    println!("{}", mvf.to_string());

    println!("Resultado: {}", results.unwrap().to_string());
}

fn basic_genetic_algorithm<T, U>(
    n: usize, // Tamaño de la población inicial
    selected_per_round: usize,
    candidates: &mut impl CandidateList<T, U>,
    fitness_fn: fn(U) -> FitnessReturn, // Función de adaptación
    mating_pr: f32,
    mut_pr: f32,
    opt: &OptimizeType, // MAX/MIN
    stop_cond: &StopCondition,
) -> Result<U, &'static str> {
    // @fitness: Fitness function
    // @opt: OptimizeType::MIN/OptimizeType::MAX

    // No sabe de una respuesta correcta, solo continúa hasta que stop_cond se cumple

    let mut results: (Result<U, &'static str>, FitnessReturn) = (Err("No calculations done"), 0f32);
    let mut fitness_over_time = Figure::new();
    let mut avg_fitness_over_time = Figure::new();

    let mut fitness_over_time_x: Vec<usize> = Default::default();
    let mut fitness_over_time_y: Vec<FitnessReturn> = Default::default();

    let mut avg_fitness_over_time_x: Vec<usize> = Default::default();
    let mut avg_fitness_over_time_y: Vec<FitnessReturn> = Default::default();
    let mut internal_state: genetic_alg::InternalState = Default::default();

    utils::debug_msg(&*format!("Tamaño de la población: {}", n));
    utils::debug_msg(&*format!("Optimización: {}", &*opt.to_string()));
    utils::debug_msg(&*format!("Probabilidad de reproducciónr: {}", mating_pr));
    utils::debug_msg(&*format!("Probabilidad de mutación: {}", mut_pr));

    // Generate initial round of candidates
    candidates.generate_initial_candidates(n);
    candidates.eval_fitness(fitness_fn);

    let tup = candidates.get_diagnostics(opt);
    // fitness_over_time_y.push(tup.0);
    // avg_fitness_over_time_y.push(tup.1);
    // fitness_over_time_x.push(0);
    // avg_fitness_over_time_x.push(0);

    internal_state.update_values(tup.0);

    loop {
        utils::debug_msg(&*format!("Generación {}:", internal_state.cycles));

        if internal_state.satisfies(stop_cond) {
            candidates.debug();
            utils::debug_msg("FIN\n\n");
            break;
        }

        candidates.debug();

        candidates.mate(n, selected_per_round, mating_pr, opt);
        candidates.mutate_list(mut_pr, opt);

        candidates.eval_fitness(fitness_fn);

        // Update stats
        let tup = candidates.get_diagnostics(opt);
        let current_gen = internal_state.cycles;

        // fitness_over_time_y.push(tup.0);
        // avg_fitness_over_time_y.push(tup.1);
        // fitness_over_time_x.push(current_gen);
        // avg_fitness_over_time_x.push(current_gen);

        // Update internal state
        results = match candidates.get_results(opt) {
            (val, fitness) => (Ok(val), fitness),
        };

        internal_state.update_values(results.1);
    }

    // fitness_over_time.axes2d().lines(
    // &fitness_over_time_x,
    // &fitness_over_time_y,
    // &[Caption("Fitness / Tiempo"), Color("black")],
    // );

    // avg_fitness_over_time.axes2d().lines(
    // &avg_fitness_over_time_x,
    // &avg_fitness_over_time_y,
    // &[Caption("Fitness promedio / Tiempo"), Color("red")],
    // );

    // let f = fitness_over_time.show();
    // let g = avg_fitness_over_time.show();

    return results.0;
}

#[cfg(test)]
mod tests {
    use super::*;
    use genetic_alg::InternalState;

    fn setup() -> IntegerCandidateList {
        let vals = ["0001", "0000", "0010", "0011"];
        let mut cl = IntegerCandidateList::default();
        for val in vals.iter() {
            let c = IntegerCandidate::new(String::from(*val));
            cl.candidates.push(c);
        }
        return cl;
    }

    #[test]
    fn test_impls() {
        let c = IntegerCandidate::new(String::from("0010"));
        assert_eq!(c.get_integer_representation(), 2);
    }

    #[test]
    fn test_diagnostics() {
        let mut cl = setup();
        cl.eval_fitness(squared);
        let d = cl.get_diagnostics(&OptimizeType::MAX);
        assert_eq!(d.0, 9.0);
        assert_eq!(d.1, 3.5);
    }

    #[test]
    fn test_sort() {
        let ascendente = [0.0, 1.0, 4.0, 9.0];
        let descendente = [9.0, 4.0, 1.0, 0.0];

        let mut cl = setup();
        cl.eval_fitness(squared);
        let s = cl.get_n_fittest(4, &OptimizeType::MAX);

        for i in 0..4 {
            let x = match s[i].get_fitness() {
                Some(v) => v,
                None => 0.0,
            };
            assert_eq!(ascendente[i], x);
        }

        let s = cl.get_n_fittest(4, &OptimizeType::MIN);
        for i in 0..4 {
            let x = match s[i].get_fitness() {
                Some(v) => v,
                None => 0.0,
            };
            assert_eq!(descendente[i], x);
        }
    }

    #[test]
    fn test_stop_condition() {
        let mut internal = InternalState::default();
        internal.max_achieved_fitness = 99.0;
        let stop = StopCondition::BOUND(100.0, 1.0);

        assert_eq!(internal.satisfies(&stop), true);

        internal.max_achieved_fitness = 98.9;
        assert_ne!(internal.satisfies(&stop), true);

        internal.max_achieved_fitness = 102.0;
        assert_eq!(internal.satisfies(&stop), true);
    }

    fn test_get_substrings() {}

    #[test]
    fn test_get_f32_values() {
        let f1 = "01000000000100110011001100110011"; // 2.3
        let f2 = "11000010111111110101011100001010"; // -127.67
        let f3 = "01000010000000111110000101001000"; // 32.97
        let mut v = String::default();
        v.push_str(f1);
        v.push_str(f2);
        v.push_str(f3);

        let mvfc = MultivariedFloatCandidate::new(3, v);

        let vals = mvfc.get_vars_from_bit_string();
        assert_eq!(vals, vec![2.3, -127.67, 32.97])
    }

    #[test]
    fn test_pow() {
        let s: f32 = 500.0;
        assert_eq!(s * 10f32.powi(-3), 0.5);
    }

    #[test]
    fn test_parse_f32() {
        assert_eq!(
            utils::parse_f32(&"00111110001000000000000000000000".to_string()),
            0.15625,
        );

        assert_eq!(
            utils::parse_f32(&"11000101101010010111101011101101".to_string()),
            -5423.36572265625
        );
    }
}
