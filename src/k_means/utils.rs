use super::super::genetic::utils::random_rangef32;
use super::cluster::PlotSettings;
use super::point::Point;

/// Returns a list of non-repeating tags
pub fn gen_tags_list<'a>(k: usize) -> Result<Vec<char>, &'a str> {
    const SYMBOLS: &str = ".+x*sSoOtTdDrR";

    if k >= SYMBOLS.chars().count() {
        return Err("GNU Plot is unable to display, too many clusters!");
    }

    let symbols = SYMBOLS.chars().collect::<Vec<char>>();
    let mut tags: Vec<char> = Default::default();
    for i in 0..k {
        tags.push(tags[i]);
    }

    return Ok(tags);
}

pub fn gen_id_list(k: usize) -> Vec<usize> {
    (0..k).collect::<Vec<usize>>()
}

/// Gets a list of k unique plot settings
pub fn gen_plot_settings<'a>(k: usize) -> Result<Vec<PlotSettings>, &'a str> {
    const COLORS: [&str; 56] = [
        "FF0000", "00FF00", "0000FF", "FFFF00", "FF00FF", "00FFFF", "000000", "800000", "008000",
        "000080", "808000", "800080", "008080", "808080", "C00000", "00C000", "0000C0", "C0C000",
        "C000C0", "00C0C0", "C0C0C0", "400000", "004000", "000040", "404000", "400040", "004040",
        "404040", "200000", "002000", "000020", "202000", "200020", "002020", "202020", "600000",
        "006000", "000060", "606000", "600060", "006060", "606060", "A00000", "00A000", "0000A0",
        "A0A000", "A000A0", "00A0A0", "A0A0A0", "E00000", "00E000", "0000E0", "E0E000", "E000E0",
        "00E0E0", "E0E0E0",
    ];

    const CHARS: [char; 14] = [
        '.', '+', 'x', '*', 's', 'S', 'o', 'O', 't', 'T', 'd', 'D', 'r', 'R',
    ];

    if k > COLORS.len() {
        return Err("Too many unique identifiers requested");
    }

    let mut settings: Vec<PlotSettings> = vec![];
    if k < CHARS.len() {
        // Return an unique (char,hex) pair
        COLORS.iter().zip(CHARS.iter()).for_each(|(color, chr)| {
            let plot_setting = PlotSettings::Pair(*chr, color.to_string());
            settings.push(plot_setting);
        });
    } else {
        // Return settings distinguished only by color
        (0..k).for_each(|i| {
            let plot_setting = PlotSettings::Pair(CHARS[i], COLORS[i].to_string());
        });
    }

    Ok(settings)
}

pub fn gen_random_points<'a>(dim: usize, n: usize, range: (f32, f32)) -> Vec<Point> {
    let mut points: Vec<Point> = Default::default();
    for i in 0..n {
        let values = (0..dim)
            .map(|_| random_rangef32(range.0, range.1))
            .collect::<Vec<f32>>();

        points.push(Point::new(values));
    }

    return points;
}
