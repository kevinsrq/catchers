use crate::utils::stats::basic::{mean, std_dev};

/// Returns the absolute energy of the time series which is the sum over the squared values.
pub fn abs_energy(x: &[f64]) -> f64 {
    x.iter().map(|&v| v * v).sum()
}

/// Complexity-invariant distance estimate.
pub fn cid_ce(x: &[f64], normalize: bool) -> f64 {
    if x.len() < 2 { return 0.0; }
    
    let working_x: Vec<f64>;
    let x_ref = if normalize {
        let s = std_dev(x);
        if s == 0.0 { return 0.0; }
        let m = mean(x);
        working_x = x.iter().map(|&v| (v - m) / s).collect();
        &working_x
    } else {
        x
    };

    let sum_sq_diff: f64 = x_ref.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
    sum_sq_diff.sqrt()
}
