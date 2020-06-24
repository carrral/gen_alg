use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct OilField {
    company: String,
    pub x1: f32,
    pub x2: f32,
    pub x3: f32,
    pub x4: f32,
    pub x5: f32,
    pub x6: f32,
    pub x7: f32,
    pub x8: f32,
}
