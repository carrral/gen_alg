use super::*;

pub mod constants {
    // ==-- CONSTANTS --==
    pub const TEST_RANGE: isize = 10;
    pub const IND_SIZE: usize = 8; // Must be even
    pub const POP_SIZE: usize = 8;
    pub const GEN: usize = 10;
    pub const MUT_PR: f32 = 0.3;
    pub const REP_PR: f32 = 0.7;
    pub const SELECTED_EACH_ROUND: usize = 4;
    pub const DEBUG: bool = true;
}

// ==-- STRUCTS --==
#[derive(PartialEq, Eq)]
pub enum OptimizeType {
    MAX,
    MIN,
}

impl OptimizeType {
    pub fn to_string(&self) -> String {
        let s = match *self {
            OptimizeType::MAX => "MAX",
            OptimizeType::MIN => "MIN",
        };
        return format!("OptimizeType::{}", s);
    }
}

pub enum StopCondition {
    ///Stops until internal cycle count is met
    CYCLES(usize),

    ///Stops until error margin is uniform among candidates
    ERROR_MARGIN(f32),

    /// Stops until the internal max fitness value is within a certain radix of specified bound
    BOUND(types::FitnessReturn, f32),
}

pub struct Bounds<T> {
    lower: T,
    upper: T,
    strays: Option<Vec<T>>,
}

impl Bounds<MultivariedFloat> {
    pub fn new(lower: MultivariedFloat, upper: MultivariedFloat) -> Self {
        if lower.n_vars != upper.n_vars {
            panic!("Dimmension mismatch , {} != {}", lower.n_vars, upper.n_vars);
        }
        Bounds {
            lower,
            upper,
            strays: None,
        }
    }

    pub fn in_range(&self, mvf: &MultivariedFloat) -> bool {
        true
    }

    pub fn val_in_range(&self, v: f32, index: usize) -> bool {
        let upper = self.upper.get_vals()[index];
        let lower = self.lower.get_vals()[index];

        match &self.strays {
            Some(s) => {
                for var in s.iter() {
                    if v == var.get_vals()[index] {
                        return false;
                    }
                }
            }
            None => (),
        }

        return v >= lower && v <= upper;
    }
}

pub mod traits {

    use super::types::*;
    use super::{Bounds, InternalState, OptimizeType, StopCondition};

    pub trait Candidate<T> {
        // Evalúa el valor de fitness de self.value, lo asigna en self.fitness
        // y lo regresa.
        fn eval_fitness(&mut self, f: fn(T) -> FitnessReturn) -> FitnessReturn;
        fn to_string(&self) -> String;
        fn get_fitness(&self) -> Option<FitnessReturn>;
        fn debug(&self);
        fn mutate(&mut self, opt_type: &OptimizeType);
    }
    pub trait CandidateList<T, U> {
        /// @T: Type that will be held in internal candidate vector
        /// @U: Type that will be evaluated in the fitness function provided
        // Generates an initial vector of Random Candidates
        fn generate_initial_candidates(&mut self, requested: usize);

        // Returns (max_fitness, avg_fitness)
        fn get_diagnostics(&self, opt_type: &OptimizeType) -> (FitnessReturn, FitnessReturn);

        // Updates internal CandidateList
        fn mate(&mut self, n_out: usize, n_selected: usize, prob_rep: f32, opt_type: &OptimizeType);

        // Operates on the whole list with a given probability mut_pr
        fn mutate_list(&mut self, mut_pr: f32, opt: &OptimizeType);

        //Evaluates fitness for the whole candidate list
        fn eval_fitness(&mut self, f: fn(U) -> FitnessReturn);

        // fn get_fittest(&self, opt_type: &OptimizeType) -> &dyn Candidate<T>;

        fn len(&self) -> usize;

        fn debug(&self);

        // Regresa el mejor resultado encontrado en una generación
        fn get_results(&mut self, opt_type: &OptimizeType) -> (U, FitnessReturn);

        fn sort(&mut self, opt_type: &OptimizeType);

        /// Gets info from @stop_cond at the beginning of the algorithm
        fn track_stop_cond(&mut self, stop_cond: &StopCondition) -> Result<bool, String>;

        /// Updates internal values from @internal_state every generation
        /// without taking ownership.
        fn track_internal_state(&mut self, internal_state: &InternalState);
    }

    pub trait FitnessFunction<U> {
        // TODO: Should return Result<FitnessFunction>
        fn eval(t: U) -> FitnessReturn;
    }
}

pub mod impls {
    pub mod single_valued {
        use super::super::traits::*;
        use super::super::types::FitnessReturn;
        use super::super::utils::{
            cross_strings, generate_random_bitstring, generic_mutate, parse_f32, roulette,
            splitting_point,
        };
        use super::super::{InternalState, OptimizeType, StopCondition};
        #[derive(Clone)]
        pub struct IntegerCandidate {
            pub value: String,
            pub fitness: Option<FitnessReturn>,
            pub mutated: bool,
            pub selected: bool,
        }

        impl IntegerCandidate {
            pub fn new(value: String) -> Self {
                IntegerCandidate {
                    value,
                    fitness: None,
                    mutated: false,
                    selected: false,
                }
            }

            pub fn get_integer_representation(&self) -> isize {
                let slice: &str = &*self.value;
                let self_int: isize = isize::from_str_radix(slice, 2).unwrap();
                return self_int;
            }

            fn len(&self) -> usize {
                self.value.chars().count()
            }
        }
        impl Default for IntegerCandidate {
            fn default() -> Self {
                IntegerCandidate::new(String::new())
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
                let mutated = generic_mutate(&self.value, opt_type);
                self.value = mutated;
                self.mutated = true;
            }
        }
        pub struct IntegerCandidateList {
            pub ind_size: usize,
            pub candidates: Vec<IntegerCandidate>,
        }
        impl IntegerCandidateList {
            pub fn get_n_fittest(
                &mut self,
                n: usize,
                opt_type: &OptimizeType,
            ) -> &[IntegerCandidate] {
                self.sort(opt_type);

                let start_i: usize = self.len() - n;

                &self.candidates[start_i..]
            }
        }

        impl CandidateList<isize, isize> for IntegerCandidateList {
            fn len(&self) -> usize {
                self.candidates.len()
            }

            // Generates an initial vector of Random Candidates and stores it in self
            fn generate_initial_candidates(&mut self, requested: usize) {
                for _ in 0..requested {
                    let s: String = generate_random_bitstring(self.ind_size);
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
            fn mate(
                &mut self,
                n_out: usize,
                n_selected: usize,
                prob_rep: f32,
                opt_type: &OptimizeType,
            ) {
                //FIXME: Regresa Result si n_selected > n_out o n_selected > self.len

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
                        let current_offspring_vals = cross_strings(
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
                    let result = roulette(&cointoss);
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

            fn track_stop_cond(&mut self, stop_cond: &StopCondition) -> Result<bool, String> {
                Ok(true)
            }

            fn track_internal_state(&mut self, internal_state: &InternalState) {}
        }

        impl Default for IntegerCandidateList {
            fn default() -> Self {
                IntegerCandidateList {
                    ind_size: 8,
                    candidates: Default::default(),
                }
            }
        }
    }
    pub mod multi_valued {
        use super::super::traits::{Candidate, CandidateList};
        use super::super::types::{FitnessReturn, MultivariedFloat};
        use super::super::utils::*;
        use super::super::{Bounds, InternalState, OptimizeType, StopCondition};

        pub struct MultivariedFloatCandidate {
            pub vars: MultivariedFloat,
            pub value: String,
            pub fitness: Option<FitnessReturn>,
            pub mutated: bool,
            pub selected: bool,
        }

        #[derive(Clone)]
        pub struct RCCandidate {
            vars: MultivariedFloat,
            fitness: Option<FitnessReturn>,
        }

        impl RCCandidate {
            pub fn new(n_vars: usize, values: &Vec<f32>) -> Self {
                RCCandidate {
                    vars: MultivariedFloat {
                        n_vars,
                        vars_value: values.clone(),
                    },
                    fitness: None,
                }
            }
        }

        impl Candidate<MultivariedFloat> for RCCandidate {
            // Evalúa el valor de fitness de self.value, lo asigna en self.fitness
            // y lo regresa.
            fn eval_fitness(&mut self, f: fn(MultivariedFloat) -> FitnessReturn) -> FitnessReturn {
                let fit = f(self.vars.clone());
                self.fitness = Some(fit);
                fit
            }

            fn to_string(&self) -> String {
                self.vars.to_string()
            }

            fn get_fitness(&self) -> Option<FitnessReturn> {
                self.fitness
            }
            fn debug(&self) {
                println!("RCCandidate{{{}}}", self.to_string());
            }

            fn mutate(&mut self, opt_type: &OptimizeType) {
                ()
            }
        }

        pub struct RCCList {
            candidates: Vec<RCCandidate>,
            current_cycle: Option<usize>,
            max_cycles: Option<usize>,
            bounds: Option<Bounds<MultivariedFloat>>,

            ///Number of vars in each vector
            n_vars: usize,
        }

        impl RCCList {
            pub fn new(n_vars: usize) -> Self {
                RCCList {
                    candidates: Default::default(),
                    current_cycle: None,
                    max_cycles: None,
                    bounds: None,
                    n_vars,
                }
            }

            pub fn set_bounds(&mut self, bounds: Bounds<MultivariedFloat>) {
                self.bounds = Some(bounds);
            }

            pub fn get_n_fittest(&mut self, n: usize, opt_type: &OptimizeType) -> &[RCCandidate] {
                self.sort(opt_type);

                let start_index: usize = self.len() - n;
                &self.candidates[start_index..]
            }
        }

        /// This implementation of a Single Precision Floating Point GA requires a lower and upper bound for every dimmension.
        /// it also requires the stoping condition to be a hard cycle count.
        impl<'a> CandidateList<RCCandidate, MultivariedFloat> for RCCList {
            // Generates an initial vector of Random Candidates
            fn generate_initial_candidates(&mut self, requested: usize) {
                // Check if bounded

                let bounded = match &self.bounds {
                    Some(_b) => true,
                    None => false,
                };
                for _i in 0..requested {
                    let mut values: Vec<f32> = Vec::with_capacity(self.n_vars);
                    for j in 0..self.n_vars {
                        let val: f32;
                        if bounded {
                            // Get a random number in range of bounds
                            let mut lb = 0f32;
                            let mut ub = 0f32;
                            match &self.bounds {
                                Some(b) => {
                                    lb = b.lower.get_vals()[j];
                                    ub = b.upper.get_vals()[j];
                                }
                                None => (),
                            };
                            val = random_rangef32(lb, ub);
                        } else {
                            // Get a random between -1000.0 and 1000.0 and hope for the best
                            val = random_rangef32(-1000.0, 1000.0);
                        }

                        values.push(val);
                    }

                    let new_candidate = RCCandidate::new(self.n_vars, &values);
                    self.candidates.push(new_candidate);
                }
            }

            // Returns (max_fitness, avg_fitness)
            fn get_diagnostics(&self, opt_type: &OptimizeType) -> (FitnessReturn, FitnessReturn) {
                let mut total_fitness = 0.0;

                assert!(self.candidates.len() != 0);

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

            /// Arithmetic crossover method (AMXO)
            fn mate(
                &mut self,
                n_out: usize,
                n_selected: usize,
                prob_rep: f32,
                opt_type: &OptimizeType,
            ) {
                // In the first round, start mating  (c0, c1), (c1, c2)...
                // In the next round, it starts with (c0, c2), (c1, c3)...
                //                                   (c1, c2), (c2, c3)...
                // so that theres always a distance d such that 0 < d < list.len()
                // Until the requested number is met

                let selected: &[RCCandidate] = self.get_n_fittest(n_selected, opt_type);
                let list_len = selected.len();
                let mut new_candidate_list: Vec<RCCandidate> = Default::default();
                let mut count = 0;

                // Dependencia: s -> d -> i

                // Aumenta cada que d == list_len-1
                let mut s = 0;

                // Iterador principal
                let mut i = 0;

                // Distancia entre madre y padre. Aumenta cada que i == list_len
                let mut d = 1;

                loop {
                    if i == list_len {
                        i = 0;
                        d += 1;
                    }

                    if d == list_len - 1 {
                        d = 1;
                        s += 1;
                    }

                    let index_m = (i + s) % list_len;
                    let index_f = (i + s + d) % list_len;
                    let mother: RCCandidate = selected[index_m].clone();
                    let father: RCCandidate = selected[index_f].clone();
                    let n_vars = mother.vars.n_vars;
                    let mut son_a = RCCandidate::new(n_vars, &Vec::with_capacity(n_vars));
                    let mut son_b = RCCandidate::new(n_vars, &Vec::with_capacity(n_vars));

                    // Iterate over each xi
                    for z in 0..n_vars {
                        let alpha = get_alpha();
                        let xi_1 = mother.vars.vars_value[z];
                        let xi_2 = father.vars.vars_value[z];

                        let yi_1 = alpha * xi_1 + (1.0 - alpha) * xi_1;
                        let yi_2 = alpha * xi_2 + (1.0 - alpha) * xi_2;
                        son_a.vars.vars_value.push(yi_1);
                        son_b.vars.vars_value.push(yi_2);
                    }

                    new_candidate_list.push(son_a);
                    new_candidate_list.push(son_b);

                    count += 2;
                    i += 1;
                    if count >= n_out {
                        break;
                    }
                }

                self.candidates = new_candidate_list;
            }

            /// Michalewicz's non-uniform mutation
            /// Requires a bounded search space and a max cycle parameter
            fn mutate_list(&mut self, mut_pr: f32, _opt: &OptimizeType) {
                let cointoss = [mut_pr, 1f32 - mut_pr];
                let mut mutated_candidate_index = 0;
                let mut mutated_candidate_val_index = 0;
                let mut mutated_value: f32 = Default::default();
                let mut mutated_flag = false;
                // FIXME: Solve mutability HERE
                for candidate_index in 0..self.candidates.len() {
                    for i in 0..self.n_vars {
                        // result: 0 -> mutate, 1 -> do nothing
                        let result = roulette(&cointoss);

                        if result == 0 {
                            println!("Mutated!, index:{}", i);
                            mutated_candidate_index = candidate_index;
                            mutated_flag = true;
                            // Data needed for mutation:
                            // Max number of generations
                            // Current gen.
                            // r: either 0 or 1
                            // b: Degree of non-uniformity
                            // alpha: uniformly distributed random number [0,1)
                            // index: index ofmutated position
                            // lb: lower_bound
                            // ub: upper_bound

                            //Get a random index for mutation
                            mutated_candidate_val_index = i;
                            let alpha: f32 = get_alpha();
                            let current_gen;
                            let r = random_range(0, 2);
                            let b = 0;

                            let (mut lb, mut ub): (f32, f32) = (0f32, 0f32);

                            match &self.bounds {
                                Some(b) => {
                                    lb = b.lower.get_vals()[mutated_candidate_val_index];
                                    ub = b.upper.get_vals()[mutated_candidate_val_index];
                                }
                                None => (),
                            };

                            let max_gen = match self.max_cycles {
                                Some(c) => c as f32,
                                None => 100f32,
                            };

                            match self.current_cycle {
                                Some(v) => current_gen = v as f32,
                                None => panic!("Internal state not provided!"),
                            };

                            let delta = |y: f32| {
                                y * (1.0 - alpha.powf((1.0 - current_gen / max_gen).powi(b)))
                            };

                            let vk_mutated = |vk: f32| match r {
                                0 => vk + delta(ub - vk),
                                1 => vk - delta(vk - lb),
                                _ => 0f32,
                            };

                            // Change vector value

                            let old_value = self.candidates[mutated_candidate_index]
                                .vars
                                .vars_value[mutated_candidate_val_index];
                            mutated_value = vk_mutated(old_value);
                            self.candidates[mutated_candidate_index].debug();
                            println!(
                                "Old val: {}, New: {}",
                                self.candidates[mutated_candidate_index].vars.vars_value
                                    [mutated_candidate_val_index],
                                mutated_value
                            );
                            self.candidates[mutated_candidate_index].vars.vars_value
                                [mutated_candidate_val_index] = mutated_value;
                        } else {
                            continue;
                        }
                    }
                }
            }

            //Evaluates fitness for the whole candidate list
            fn eval_fitness(&mut self, f: fn(MultivariedFloat) -> FitnessReturn) {
                for c in &mut self.candidates {
                    c.eval_fitness(f);
                }
            }

            fn len(&self) -> usize {
                self.candidates.len()
            }

            fn debug(&self) {
                for c in &self.candidates {
                    c.debug();
                }
            }

            // Regresa el mejor resultado encontrado en una generación
            // TODO: should return Result
            fn get_results(
                &mut self,
                opt_type: &OptimizeType,
            ) -> (MultivariedFloat, FitnessReturn) {
                self.sort(opt_type);
                let x = self.candidates.pop().unwrap();
                let (v, f): (MultivariedFloat, FitnessReturn) = (x.vars, x.fitness.unwrap());
                (v, f)
            }

            // TODO: should return Result
            fn sort(&mut self, opt_type: &OptimizeType) {
                match *opt_type {
                    OptimizeType::MAX => self
                        .candidates
                        .sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap()),
                    OptimizeType::MIN => self
                        .candidates
                        .sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap()),
                }
            }

            /// Make sure @stop_cond is of type CYCLES and store the max_cycles value
            fn track_stop_cond(&mut self, stop_cond: &StopCondition) -> Result<bool, String> {
                match *stop_cond {
                    StopCondition::CYCLES(max_gen) => {
                        self.max_cycles = Some(max_gen);
                        Ok(true)
                    }
                    _ => Err(String::from(
                        "Stop condition doesn't match algorithm's requirements",
                    )),
                }
            }

            fn track_internal_state(&mut self, internal_state: &InternalState) {
                self.current_cycle = Some(internal_state.cycles);
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

            /// Returns a vector of the floats containted in self.value (bit string)
            /// Internal use meant only for updating MultivariedFloat's values, not intended for
            /// use by the user.
            pub fn get_vars_from_bit_string(&self) -> Vec<f32> {
                let substrings =
                    MultivariedFloatCandidate::get_var_substrings(&self.value, self.vars.n_vars);
                let values = substrings
                    .iter()
                    .map(|x| parse_f32(&x.to_string()))
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
                    .map(|x| parse_f32(&x.to_string()))
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
                    "MVFCandidate{{vars:{}, fitness:{}}}",
                    self.vars.to_string(),
                    fit,
                )
            }

            fn get_fitness(&self) -> Option<FitnessReturn> {
                self.fitness
            }
            fn debug(&self) {
                println!("{}", self.to_string());
            }
            fn mutate(&mut self, opt_type: &OptimizeType) {
                // Mutar en dígitos significativos
                // 0: sign (0)-> [0.5]
                // 1-8: exponente (6-8) -> [0.1,1,2]
                // 9-31: mantissa (9-12) -> [0.5,1,2,4]
                // Total = 8

                // Seleccionar al azar el valor del vector
                // sobre el cual actuará.

                let weights = [0.5, 0.1, 1.0, 2.0, 0.5, 1.0, 6.0, 10.0];
                let rel_start_i = match roulette(&weights) as usize {
                    0 => 0,
                    1 => 6,
                    2 => 7,
                    3 => 8,
                    4 => 9,
                    5 => 10,
                    6 => 11,
                    7 => 12,
                    _ => 12,
                };

                //Seleccionar la posición de la subcadena (0-12)
                //Mismo peso para todos
                let weights = [1.0; 13];
                let pos = roulette(&weights) as usize;

                let mut new_val = String::default();

                let abs_index = MultivariedFloatCandidate::FLOAT_LEN * rel_start_i + pos;

                for (i, c) in self.value.char_indices() {
                    if i == abs_index {
                        let new_char = match c {
                            '0' => '1',
                            '1' => '0',
                            _ => '0',
                        };
                        new_val.push(new_char);
                    } else {
                        new_val.push(c);
                    }
                }

                self.value = new_val;

                //Update vars_value
                let new_values = self.get_vars_from_bit_string();
                self.vars.update_vals(&new_values);
            }
        }
        pub struct MVFCandidateList {
            n_vars: usize,
            /// Should be n_vars * 32
            ind_size: usize,
            candidates: Vec<MultivariedFloatCandidate>,
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

        impl CandidateList<MultivariedFloatCandidate, MultivariedFloat> for MVFCandidateList {
            // Generates an initial vector of Random Candidates
            fn generate_initial_candidates(&mut self, requested: usize) {
                for i in 0..requested {
                    let s: String = generate_random_bitstring(self.ind_size);
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
            fn mate(
                &mut self,
                n_out: usize,
                n_selected: usize,
                prob_rep: f32,
                opt_type: &OptimizeType,
            ) {
                //FIXME: Regresa Result si n_selected > n_out o n_selected > self.len
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
                let k: usize =
                    splitting_point(n_vars, prob_rep) * MultivariedFloatCandidate::FLOAT_LEN;

                loop {
                    let mut break_loop = false;
                    for i in 0..n_selected {
                        let offset_index = (i + offset) % n_selected;
                        let current_offspring_vals = cross_strings(
                            &best_candidates[i].value,
                            &best_candidates[offset_index].value,
                            k,
                        );
                        let offspring = (
                            MultivariedFloatCandidate::new(
                                n_vars,
                                String::from(current_offspring_vals.0),
                            ),
                            MultivariedFloatCandidate::new(
                                n_vars,
                                String::from(current_offspring_vals.1),
                            ),
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
                    let result = roulette(&cointoss);
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
            fn get_results(
                &mut self,
                opt_type: &OptimizeType,
            ) -> (MultivariedFloat, FitnessReturn) {
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

            fn track_stop_cond(&mut self, stop_cond: &StopCondition) -> Result<bool, String> {
                Ok(true)
            }

            fn track_internal_state(&mut self, internal_state: &InternalState) {}
        }
    }
}

pub struct InternalState {
    pub cycles: usize,
    pub max_achieved_fitness: FitnessReturn,
    error_margin: f32,
}

impl InternalState {
    pub fn satisfies(&self, stop: &StopCondition) -> bool {
        // So if StopCondition is never met, it wont loop forever
        const HARD_STOP: usize = std::usize::MAX;
        let satisfied = match *stop {
            StopCondition::CYCLES(value) => value == self.cycles,
            StopCondition::ERROR_MARGIN(value) => value <= self.error_margin,

            StopCondition::BOUND(value, radix) => {
                (value - self.max_achieved_fitness).abs() <= radix
            }
        };

        return satisfied || self.cycles > HARD_STOP;
    }

    pub fn update_values(&mut self, max_fitness: FitnessReturn) {
        self.max_achieved_fitness = max_fitness;
        self.cycles += 1;
    }
}

impl Default for InternalState {
    fn default() -> Self {
        InternalState {
            cycles: 0,
            max_achieved_fitness: 0.0,
            error_margin: 0.0,
        }
    }
}

pub mod types {
    pub type FitnessReturn = f32;

    #[derive(Clone)]
    pub struct MultivariedFloat {
        ///Wrapper struct for (n0,n1,...) type of values, doesn't hold info about
        ///its candidate representation
        //n_vars is always equal to vars_value.len()
        pub n_vars: usize,
        pub vars_value: Vec<f32>,
    }

    impl MultivariedFloat {
        pub fn new(n_vars: usize, vars_value: Vec<f32>) -> Self {
            if vars_value.len() != n_vars {
                panic!("Declared length of vector doesn't match length of supplied values");
            }
            MultivariedFloat { n_vars, vars_value }
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
}

pub mod utils {

    use super::*;

    pub fn bin_to_int(bin: &String) -> isize {
        let r_int: isize = isize::from_str_radix(bin, 2).unwrap();
        r_int
    }

    pub fn get_alpha() -> f32 {
        let v: f32 = rand::thread_rng().gen();
        v
    }

    pub fn parse_f32(bin_str: &String) -> f32 {
        const FLOAT_LEN: usize = 32;
        const SIGN_LEN: usize = 1;
        const EXPONENT_LEN: usize = 8;
        const MANTISSA_LEN: usize = 23;

        let l = bin_str.chars().count();
        if l != FLOAT_LEN {
            panic!(format!(
                "Invalid string length: expected 32, instead got {}",
                l
            ));
        }

        let char_array: Vec<char> = bin_str.chars().collect();

        for c in &char_array {
            if *c != '0' && *c != '1' {
                panic!("Invalid values found while parsing bit string: {}", *c);
            }
        }

        let sign = match char_array[0] {
            '0' => 1,
            '1' => -1,
            _ => 1,
        };

        let exponent_str = &bin_str[SIGN_LEN..(SIGN_LEN + EXPONENT_LEN)];
        let mantissa_str =
            &bin_str[(SIGN_LEN + EXPONENT_LEN)..(SIGN_LEN + EXPONENT_LEN + MANTISSA_LEN)];

        // Exponent & mantissa extraction ok
        let exponent = i32::from_str_radix(exponent_str, 2).unwrap();

        let mantissa = {
            let mut s: f32 = 1.0;
            let mut i: i32 = -1;
            for c in mantissa_str.chars() {
                if c != '0' {
                    s += 2f32.powi(i);
                }
                i -= 1;
            }
            s
        };

        // return sign * mantissa.pow(exponent) as f32;
        (sign as f32) * (mantissa as f32) * 2f32.powi(exponent - 127i32)
    }

    pub fn cross_strings(father: &String, mother: &String, k: usize) -> (String, String) {
        // TODO: Volverlo un trait?
        // FIXME: Tomar como parámetro un IntegerCandidate
        //Regresa una tupla de hijos

        let gnomes_father = (&father[..k], &father[k..]);
        let gnomes_mother = (&mother[..k], &mother[k..]);
        let mut sons: (String, String) =
            (String::from(gnomes_father.0), String::from(gnomes_mother.0));

        sons.0.push_str(gnomes_mother.1);
        sons.1.push_str(gnomes_father.1);

        return sons;
    }

    pub fn generate_random_bitstring(n: usize) -> String {
        let mut s: String = String::new();

        for _j in 0..n {
            let r = utils::random_range(0, 2).to_string();
            s.push_str(&r);
        }
        return s;
    }

    pub fn generic_mutate(s: &String, opt_type: &OptimizeType) -> String {
        let mut mutated = String::new();

        let (unwanted_char, wanted) = match *opt_type {
            OptimizeType::MAX => ('0', '1'),
            OptimizeType::MIN => ('1', '0'),
        };
        let mut k: usize;
        let mut tries: usize = 0;
        loop {
            // TODO: Cambiar intento al azar por iterción izquierda->derecha, derecha -> izquierda
            k = utils::random_range(0, s.len() as isize) as usize;
            let char_array: Vec<char> = s.chars().collect();
            if char_array[k] == unwanted_char || tries > s.len() {
                break;
            }

            tries += 1;
        }

        let mut i: usize = 0;
        for c in s.chars() {
            let mutated_char = match i {
                a if a == k => wanted,
                _ => c,
            };
            mutated.push(mutated_char);
            i += 1;
        }

        return mutated;
    }

    pub fn debug_msg(msg: &str) {
        println!("  => {} ", msg);
    }

    pub fn random_range(start: isize, finish: isize) -> isize {
        let mut rng = thread_rng();
        return rng.gen_range(start, finish);
    }

    pub fn random_rangef32(start: f32, finish: f32) -> f32 {
        let mut rng = thread_rng();
        rng.gen_range(start, finish)
    }

    pub fn splitting_point(n: usize, pr: f32) -> usize {
        let spf: f32 = pr * (n as f32);
        return spf as usize;
    }

    pub fn roulette(weights: &[f32]) -> isize {
        //TODO: Implementar generics
        // Regresa 0 <= valor < weights.len()

        let mut rng = thread_rng();
        let weighted_dist = WeightedIndex::new(weights).unwrap();

        return weighted_dist.sample(&mut rng) as isize;
    }
}
