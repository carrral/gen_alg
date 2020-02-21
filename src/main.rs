#![allow(dead_code)]
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::Rng;

const TEST_RANGE: u32 = 10;
const IND_SIZE: usize = 4;
const POP_SIZE: u32 = 4;
const GEN: u32 = 100;

type ParamType = String;

fn main() {
    let d = [2, 0, 3, 5];

    // Test random_ranage()
    for _i in 0..TEST_RANGE {
        // println!("Valor: {}",random_ranage(0,5));
        println!("Elegido (0,1): {} ", roulette(&d));
    }
}

fn random_ranage(start: i64, finish: i64) -> i64 {
    let mut rng = thread_rng();
    return rng.gen_range(start, finish);
}

fn roulette(weights: &[u32]) -> u32 {
    // Regresa 0 <= valor < weights.len()

    let mut rng = thread_rng();
    let values = 0..weights.len();
    let weighted_dist = WeightedIndex::new(weights).unwrap();

    return weighted_dist.sample(&mut rng) as u32;
}

fn fitness(x: u32) -> u32 {
    return x.pow(2);
}

fn mate(father: &ParamType,mother: &ParamType, k: usize) -> (ParamType, ParamType) {
    //Regresa una tupla de hijos
    
    let gnomes_father = (&father[0..k], &father[k..IND_SIZE]);
    let gnomes_mother = (&mother[0..k], &mother[k..IND_SIZE]);
    let mut sons : (ParamType, ParamType) = (String::from(gnomes_father.0),String::from(gnomes_mother.0));

    sons.0.push_str(gnomes_mother.1);
    sons.1.push_str(gnomes_father.1);

    return sons;
}
