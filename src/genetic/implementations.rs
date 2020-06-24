pub mod single_valued {
    use super::super::traits::*;
    use super::super::types::FitnessReturn;
    use super::super::utils::{
        cross_strings, generate_random_bitstring, generic_mutate, roulette, splitting_point,
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

    impl<'a> Candidate<'a, isize> for IntegerCandidate {
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

        fn eval_fitness(&mut self, f: &dyn FitnessFunction<'a, isize>) -> FitnessReturn {
            let self_int = self.get_integer_representation();
            let fit = f.eval(self_int);
            self.fitness = Some(fit);
            return fit;
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
        pub fn get_n_fittest(&mut self, n: usize, opt_type: &OptimizeType) -> &[IntegerCandidate] {
            self.sort(opt_type);

            let start_i: usize = self.len() - n;

            &self.candidates[start_i..]
        }
    }

    impl<'a> CandidateList<'a, isize, isize> for IntegerCandidateList {
        fn len(&self) -> usize {
            self.candidates.len()
        }

        // Generates an initial vector of Random Candidates and stores it in self
        fn generate_initial_candidates(&mut self, requested: usize) {
            for _ in 0..=requested {
                let s: String = generate_random_bitstring(self.ind_size);
                let c = IntegerCandidate::new(s);
                self.candidates.push(c);
            }
        }

        fn mark_for_selection(&mut self, opt_type: &OptimizeType, n_selected: usize) {}
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
        fn eval_fitness(&mut self, f: &dyn FitnessFunction<'a, isize>) {
            for candidate in &mut self.candidates {
                candidate.eval_fitness(f);
            }
        }

        fn debug(&self, debug_value: bool) {
            if debug_value {
                for candidate in &self.candidates {
                    candidate.debug();
                }
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
    use super::super::traits::{Candidate, CandidateList, FitnessFunction};
    use super::super::types::{FitnessReturn, MultivaluedFloat, MultivaluedInteger};
    use super::super::utils::*;
    use super::super::{Bounds, InternalState, OptimizeType, StopCondition};

    #[derive(Clone)]
    pub struct RCCandidate {
        vars: MultivaluedFloat,
        fitness: Option<FitnessReturn>,
        selected_for_mating: bool,
    }

    impl RCCandidate {
        pub fn new(n_vars: usize, values: &Vec<f32>) -> Self {
            RCCandidate {
                vars: MultivaluedFloat {
                    n_vars,
                    vars_value: values.clone(),
                },
                fitness: None,
                selected_for_mating: false,
            }
        }
    }

    impl<'a> Candidate<'a, MultivaluedFloat> for RCCandidate {
        // Evalúa el valor de fitness de self.value, lo asigna en self.fitness
        // y lo regresa.

        fn to_string(&self) -> String {
            let fit = match self.fitness {
                Some(v) => v.to_string(),
                None => String::from("UNDEFINED"),
            };
            format!(
                "{},fitness:{}, selected: {}",
                self.vars.to_string(),
                fit,
                self.selected_for_mating,
            )
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
        fn eval_fitness(&mut self, f: &dyn FitnessFunction<'a, MultivaluedFloat>) -> FitnessReturn {
            let fit = f.eval(self.vars.clone());
            self.fitness = Some(fit);
            return fit;
        }
    }

    pub struct RCCList {
        pub candidates: Vec<RCCandidate>,
        current_cycle: Option<usize>,
        max_cycles: Option<usize>,
        bounds: Option<Bounds<MultivaluedFloat>>,

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

        pub fn set_bounds(&mut self, bounds: Bounds<MultivaluedFloat>) {
            self.bounds = Some(bounds);
        }

        pub fn get_n_fittest(&mut self, n: usize, opt_type: &OptimizeType) -> &[RCCandidate] {
            self.sort(opt_type);

            let start_index: usize = self.len() - n;

            &self.candidates[start_index..]
        }

        pub fn set_debug(debug: bool) {}
    }

    /// This implementation of a Single Precision Floating Point GA requires a lower and upper bound for every dimmension.
    /// it also requires the stoping condition to be a hard cycle count.
    impl<'a> CandidateList<'a, RCCandidate, MultivaluedFloat> for RCCList {
        // Generates an initial vector of Random Candidates
        fn generate_initial_candidates(&mut self, requested: usize) {
            // Check if bounded

            let bounded = match &self.bounds {
                Some(_b) => true,
                None => false,
            };
            for _i in 0..=requested {
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

        fn mark_for_selection(&mut self, opt_type: &OptimizeType, n_selected: usize) {
            //Maybe always call this and eliminate sorting from main mating function
            self.sort(opt_type);
            let start_index = self.len() - n_selected;
            for i in start_index..self.len() {
                self.candidates[i].selected_for_mating = true;
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

            // println!("Selected");
            // for c in selected {
            // c.debug();
            // }
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
                assert!(index_m != index_f);
                let mut mother: RCCandidate = selected[index_m].clone();
                mother.selected_for_mating = true;

                let mut father: RCCandidate = selected[index_f].clone();
                father.selected_for_mating = true;

                let n_vars = mother.vars.n_vars;
                let mut son_a = RCCandidate::new(n_vars, &Vec::with_capacity(n_vars));
                let mut son_b = RCCandidate::new(n_vars, &Vec::with_capacity(n_vars));

                // Iterate over each xi
                for z in 0..n_vars {
                    let alpha = get_alpha();
                    let xi_1 = mother.vars.vars_value[z];
                    let xi_2 = father.vars.vars_value[z];

                    let yi_1 = alpha * xi_1 + (1.0 - alpha) * xi_2;
                    let yi_2 = alpha * xi_2 + (1.0 - alpha) * xi_1;
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
                        // println!("Mutated!, index:{}", i);
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

                        let delta =
                            |y: f32| y * (1.0 - alpha.powf((1.0 - current_gen / max_gen).powi(b)));

                        let vk_mutated = |vk: f32| match r {
                            0 => vk + delta(ub - vk),
                            1 => vk - delta(vk - lb),
                            _ => 0f32,
                        };

                        // Change vector value

                        let old_value = self.candidates[mutated_candidate_index].vars.vars_value
                            [mutated_candidate_val_index];
                        mutated_value = vk_mutated(old_value);
                        // self.candidates[mutated_candidate_index].debug();
                        // println!(
                        // "Old val: {}, New: {}",
                        // self.candidates[mutated_candidate_index].vars.vars_value
                        // [mutated_candidate_val_index],
                        // mutated_value
                        // );
                        self.candidates[mutated_candidate_index].vars.vars_value
                            [mutated_candidate_val_index] = mutated_value;
                    } else {
                        continue;
                    }
                }
            }
        }

        //Evaluates fitness for the whole candidate list
        fn eval_fitness(&mut self, f: &dyn FitnessFunction<'a, MultivaluedFloat>) {
            for c in &mut self.candidates {
                c.eval_fitness(f);
            }
        }

        fn len(&self) -> usize {
            self.candidates.len()
        }

        fn debug(&self, debug_value: bool) {
            if debug_value {
                for c in &self.candidates {
                    c.debug();
                }
            }
        }

        // Regresa el mejor resultado encontrado en una generación
        // TODO: should return Result
        fn get_results(&mut self, opt_type: &OptimizeType) -> (MultivaluedFloat, FitnessReturn) {
            self.sort(opt_type);
            let x = self.candidates.pop().unwrap();
            let (v, f): (MultivaluedFloat, FitnessReturn) = (x.vars, x.fitness.unwrap());
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

    pub struct MultivaluedIntCandidate {
        pub vars: MultivaluedInteger,
        pub value: String,
        pub fitness: Option<FitnessReturn>,
        pub mutated: bool,
        pub selected: bool,
    }

    impl<'a> MultivaluedIntCandidate {
        const INT_LEN: usize = 32;

        pub fn len(&self) -> usize {
            self.value.chars().count()
        }

        fn get_var_substrings(s: &String, n_vars: usize) -> Vec<&str> {
            // Returns a vector of f32 values entirely dependant on bit string

            // Sub divide single candidate string into individual sub strings
            let mut substrings: Vec<&str> = vec![""; n_vars];

            let r = MultivaluedIntCandidate::INT_LEN;
            let l = s.chars().count();

            // Assumes l == n_vars*INT_LEN
            for start_index in (0..l).step_by(r) {
                substrings[start_index / r] = &s[start_index..start_index + r];
            }

            return substrings;
        }

        /// Returns a vector of the ints containted in self.value (bit string)
        /// Internal use meant only for updating MultivaluedInts's values, not intended for
        /// use by the user.
        pub fn get_vars_from_bit_string(&self) -> Vec<isize> {
            let substrings =
                MultivaluedIntCandidate::get_var_substrings(&self.value, self.vars.n_vars);
            let values = substrings
                .iter()
                .map(|x| bin_to_int(&x.to_string()))
                .collect::<Vec<isize>>();

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

            if !(len / MultivaluedIntCandidate::INT_LEN == n_vars
                && len % MultivaluedIntCandidate::INT_LEN == 0)
            {
                panic!("Length of value String and number of variables is incompatible: expected {}, got {}", MultivaluedIntCandidate::INT_LEN*n_vars, len);
            }

            let substrings = MultivaluedIntCandidate::get_var_substrings(&value, n_vars);

            let vars_value = substrings
                .iter()
                .map(|x| bin_to_int(&x.to_string()))
                .collect::<Vec<isize>>();

            MultivaluedIntCandidate {
                vars: MultivaluedInteger { n_vars, vars_value },
                value,
                fitness: None,
                mutated: false,
                selected: false,
            }
        }
    }

    impl<'a> Candidate<'a, MultivaluedInteger> for MultivaluedIntCandidate {
        // Evalúa el valor de fitness de self.value, lo asigna en self.fitness
        // y lo regresa.

        fn to_string(&self) -> String {
            let fit = match self.get_fitness() {
                Some(v) => v.to_string(),
                None => String::from("UNDEFINED"),
            };
            format!(
                "MVICandidate{{vars:{}, fitness:{}}}",
                self.vars.to_string(),
                fit,
            )
        }

        fn eval_fitness(
            &mut self,
            f: &dyn FitnessFunction<'a, MultivaluedInteger>,
        ) -> FitnessReturn {
            //FIXME: Clone?
            // let v: MultivaluedInteger = self.vars.clone();
            let f = f.eval(self.vars.clone());

            self.fitness = Some(f);
            return f;
        }

        fn get_fitness(&self) -> Option<FitnessReturn> {
            self.fitness
        }
        fn debug(&self) {
            println!("{}", self.to_string());
        }
        fn mutate(&mut self, opt_type: &OptimizeType) {
            // At this point, the vector should be brand new and not related to the Points in Space
            // in any way.
            let weights = [1.0; 32];
            let pos = roulette(&weights) as usize;
            let mut new_val = String::new();

            for (i, c) in self.value.char_indices() {
                if i == pos {
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
    pub struct MVICandidateList {
        n_vars: usize,
        /// Should be n_vars * 32
        ind_size: usize,
        candidates: Vec<MultivaluedIntCandidate>,
    }

    impl MVICandidateList {
        pub fn get_n_fittest(
            &mut self,
            n: usize,
            opt_type: &OptimizeType,
        ) -> &[MultivaluedIntCandidate] {
            self.sort(opt_type);

            let start_i: usize = self.len() - n;

            &self.candidates[start_i..]
        }

        pub fn new(n_vars: usize) -> Self {
            MVICandidateList {
                n_vars,
                ind_size: n_vars * MultivaluedIntCandidate::INT_LEN,
                candidates: Default::default(),
            }
        }
    }

    impl<'a> CandidateList<'a, MultivaluedIntCandidate, MultivaluedInteger> for MVICandidateList {
        // Generates an initial vector of Random Candidates
        fn generate_initial_candidates(&mut self, requested: usize) {
            for i in 0..=requested {
                let s: String = generate_random_bitstring(self.ind_size);
                let c = MultivaluedIntCandidate::new(self.n_vars, s);
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

        fn mark_for_selection(&mut self, opt_type: &OptimizeType, n_selected: usize) {}
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
            let mut new_candidates: Vec<MultivaluedIntCandidate> = Default::default();

            let size = self.ind_size;

            let n_vars = self.n_vars;

            // Select @n_out best candidates
            let best_candidates: &[MultivaluedIntCandidate] =
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
            let k: usize = splitting_point(n_vars, prob_rep) * MultivaluedIntCandidate::INT_LEN;

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
                        MultivaluedIntCandidate::new(
                            n_vars,
                            String::from(current_offspring_vals.0),
                        ),
                        MultivaluedIntCandidate::new(
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
        fn eval_fitness(&mut self, f: &dyn FitnessFunction<'a, MultivaluedInteger>) {
            //FIXME: Agregar un genérico en la definición del trait, diferente a candidate
            for candidate in &mut self.candidates {
                candidate.eval_fitness(f);
            }
        }

        fn len(&self) -> usize {
            return self.candidates.len();
        }

        fn debug(&self, debug_value: bool) {
            if debug_value {
                for c in &self.candidates {
                    c.debug();
                }
            }
        }

        // Regresa el mejor resultado encontrado en una generación
        fn get_results(&mut self, opt_type: &OptimizeType) -> (MultivaluedInteger, FitnessReturn) {
            self.sort(opt_type);
            let x = self.candidates.pop().unwrap();
            let (v, f): (MultivaluedInteger, FitnessReturn) = (x.vars, x.fitness.unwrap());
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
