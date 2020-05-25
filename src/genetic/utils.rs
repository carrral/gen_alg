use super::*;

/// Two's complement repr. of signed integer representation
pub fn bin_to_int(bin: &String) -> isize {
    // TODO: Should return Result

    let mut bin_array = bin.chars();
    let mut negated = String::new();

    for e in bin.chars() {
        if e != '0' && e != '1' {
            panic!("Invalid element in bit string: {}", e);
        }
    }

    let negative = match bin_array.nth(0).unwrap() {
        '1' => true,
        _ => false,
    };

    if negative {
        // Negate string
        bin_array.for_each(|c| {
            let pushed = match c {
                '1' => '0',
                '0' => '1',
                _ => ' ',
            };
            negated.push(pushed);
        });
    }

    if negative {
        return (-1) * isize::from_str_radix(&negated, 2).unwrap() - 1;
    } else {
        return isize::from_str_radix(bin, 2).unwrap();
    }
}

pub fn bin_to_i32(bin: &String) -> i32 {
    i32::from_str_radix(bin, 2).unwrap()
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
    let mut sons: (String, String) = (String::from(gnomes_father.0), String::from(gnomes_mother.0));

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

pub fn debug_msg(msg: String, debug: bool) {
    if debug {
        println!("  => {} ", msg);
    }
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
