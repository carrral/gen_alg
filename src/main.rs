#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(unused_variables)]

use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::Rng;
use gnuplot::{Figure, Caption, Color};

/* Constants  */
const TEST_RANGE: isize = 10;
const IND_SIZE: usize = 8;

// Must be even
const POP_SIZE: usize = 8;
const GEN: usize = 30;

type ParamType = String;
type FitnessReturn = f64;
type t_int = isize;

/* Structs */

struct Candidate {
    value: ParamType,
    fitness: FitnessReturn,
    prob: f64,
}

fn main() {
    algoritmo_genetico_simple(fitness, "MAX");
}

fn random_range(start: isize, finish: isize) -> isize {
    let mut rng = thread_rng();
    return rng.gen_range(start, finish);
}

fn roulette(weights: &[FitnessReturn; POP_SIZE]) -> t_int {
    //TODO Implementar generics
    // Regresa 0 <= valor < weights.len()

    let mut rng = thread_rng();
    let values = 0..weights.len();
    let weighted_dist = WeightedIndex::new(weights).unwrap();

    return weighted_dist.sample(&mut rng) as t_int;
}

fn fitness(x: &ParamType) -> FitnessReturn {
    let x_int: isize = isize::from_str_radix(x, 2).unwrap();
    return x_int.pow(2) as FitnessReturn;
}

fn mate(father: &ParamType, mother: &ParamType, k: usize) -> (ParamType, ParamType) {
    // FIXME: Tomar como parámetro un Candidate
    //Regresa una tupla de hijos

    let gnomes_father = (&father[0..k], &father[k..IND_SIZE]);
    let gnomes_mother = (&mother[0..k], &mother[k..IND_SIZE]);
    let mut sons: (ParamType, ParamType) =
        (String::from(gnomes_father.0), String::from(gnomes_mother.0));

    sons.0.push_str(gnomes_mother.1);
    sons.1.push_str(gnomes_father.1);

    return sons;
}

fn generate_initial_candidates() -> Vec<Candidate> {
    let mut candidates: Vec<Candidate> = Vec::new();

    for _i in 0..POP_SIZE {
        let mut current_cand_str = String::new();
        for _j in 0..IND_SIZE {
            let val = random_range(0, 2);
            current_cand_str.push_str(&val.to_string());
        }

        let current_candidate = Candidate {
            value: current_cand_str,
            fitness: 0.0,
            prob: 0.0,
        };
        candidates.push(current_candidate);
    }

    return candidates;
}

fn bin_to_int(bin: &String ) -> t_int {
    let r_int: t_int = isize::from_str_radix(bin, 2).unwrap();
    r_int
}

fn algoritmo_genetico_simple(funcion: fn(&ParamType) -> FitnessReturn, opt: &str) {
    /* funcion: Función a optimizar
     * opt: "MAX"/"MIN" */

    // No sabe de una respuesta correcta, solo continúa hasta que alcanza GEN
    let mut fitness_over_time = Figure::new();
    let mut avg_fitness_over_time = Figure::new();

    let mut fitness_over_time_x : [usize;GEN] = [0; GEN];
    let mut fitness_over_time_y: [FitnessReturn;GEN] = Default::default();

    let mut avg_fitness_over_time_x : [usize;GEN] = [0; GEN];
    let mut avg_fitness_over_time_y : [FitnessReturn;GEN] = Default::default();

    for i in 0..GEN{
        fitness_over_time_x[i] = i;
        avg_fitness_over_time_x[i] = i;
    }

    let mut total_fitness: FitnessReturn = 0.0;

    // Generar ronda inicial de candidatos
    let mut candidatos: Vec<Candidate> = generate_initial_candidates();

    for _gen in 0..GEN {
        let mut max_fitness: FitnessReturn = {
            if opt == "MAX" {
                -1f64
            } else {
                1000f64
            }
        };

        // Obtener fitness de cada candidato
        for candidato in &mut candidatos {
            candidato.fitness = funcion(&candidato.value);
            total_fitness += candidato.fitness;

            // Obtener máximo de cada generación
            max_fitness = {
                if opt == "MAX" {
                    if max_fitness < candidato.fitness {
                        candidato.fitness
                    } else {
                        max_fitness
                    }
                } else {
                    if max_fitness > candidato.fitness {
                        candidato.fitness
                    } else {
                        max_fitness
                    }
                }
            }
        }

        fitness_over_time_y[_gen] = max_fitness;
        avg_fitness_over_time_y[_gen] = total_fitness/(POP_SIZE as f64);

        // println!("Fitness máximo de la generación {}°: {}", _gen+1, max_fitness); 
        // println!("Fitness promedio: {} ",total_fitness/(POP_SIZE as f64));
        // println!("    Candidatos:");
        // for candidato in &mut candidatos{
            // println!("     * ({}, {})",candidato.value, bin_to_int(&candidato.value));
        // }

        // Obtener vector de pesos
        let mut weights: [FitnessReturn; POP_SIZE] = [0f64; POP_SIZE];

        // FIXME: Nos podemos ahorrar este ciclo?
        for i in 0..POP_SIZE {
            weights[i] = candidatos[i].fitness;
        }

        // Obtener POP_SIZE/2 madres y padres para reproducción
        let mut madres: [ParamType; POP_SIZE / 2] = Default::default();
        let mut padres: [ParamType; POP_SIZE / 2] = Default::default();

        for i in 0..POP_SIZE / 2 {
            // Obtener padres de acuerdo a pesos de candidatos
            // Seleccionar un índice al azar con peso
            let index_p = roulette(&weights) as usize;
            let index_m = roulette(&weights) as usize;

            // Copiar valor de candidato
            padres[i] = candidatos[index_p].value.clone();
            madres[i] = candidatos[index_m].value.clone();
        }

        // Reproducir
        let mut new_candidates : Vec<Candidate> = Vec::new();

        for i in 0..POP_SIZE / 2 {
            let current_mother: &ParamType = &madres[i];
            let current_father: &ParamType = &padres[i];

            //Elegir punto de corte al azar
            let k: usize = random_range(1, IND_SIZE as isize) as usize;

            let sons_vals: (ParamType, ParamType) = mate(current_father, current_mother, k);

            new_candidates.push(Candidate{
                value: sons_vals.0,
                fitness: 0.0,
                prob: 0.0
            });
            new_candidates.push(Candidate{
                value: sons_vals.1,
                fitness: 0.0,
                prob: 0.0
            });
        }

        // Para este punto ya tenemos POP_SIZE nuevos candidatos
        candidatos = new_candidates;
    }

    fitness_over_time.axes2d()
        .lines(&fitness_over_time_x,
               &fitness_over_time_y,
               &[Caption("Fitness / Tiempo"), Color("black")]);

    avg_fitness_over_time.axes2d()
        .lines(&avg_fitness_over_time_x,
               &avg_fitness_over_time_y,
               &[Caption("Fitness promedio / Tiempo"), Color("red")]);

    fitness_over_time.show();
    avg_fitness_over_time.show();
}
