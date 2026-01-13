use std::f64;

use crate::utils::stats::basic::{max_, mean, median, min_, std_dev};

/// Boolean variable denoting if the distribution of x looks symmetric.
/// This is the case if |mean(x) - median(x)| < r * (max(x) - min(x)).
pub fn symmetry_looking(x: &[f64], r: f64) -> bool {
    let mean_val = mean(x);
    let median_val = median(x);
    let max_val = max_(x);
    let min_val = min_(x);

    (mean_val - median_val).abs() < r * (max_val - min_val)
}

/// Check if the standard deviation of x is higher than r * (max(x) - min(x)).
pub fn large_standard_deviation(x: &[f64], r: f64) -> bool {
    let std = std_dev(x);
    let max_val = max_(x);
    let min_val = min_(x);

    std > r * (max_val - min_val)
}

/// Calculates the q-th quantile of the time series x.
///
/// :param x: the time series to calculate the feature of
/// :param q: the quantile to calculate (0.0 to 1.0)
pub fn quantile(x: &[f64], q: f64) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }

    let mut sorted = x.to_vec();
    // Sort handling NaNs (push to end or similar).
    // Here we treat them equal to themselves for simple sort
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }

    // numpy percentile style interpolation (linear)?
    // Simplified:
    let pos = (n as f64 - 1.0) * q;
    let base = pos.floor() as usize;
    let rest = pos - base as f64;

    if base + 1 < n {
        sorted[base] + rest * (sorted[base + 1] - sorted[base])
    } else {
        sorted[base]
    }
}

/// Returns the ratio of values that are more than r * std(x) away from the mean of x.
pub fn ratio_beyond_r_sigma(x: &[f64], r: f64) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let m = mean(x);
    let s = std_dev(x);

    if s == 0.0 {
        return 0.0;
    }

    let count = x.iter().filter(|&&v| (v - m).abs() > r * s).count();
    count as f64 / x.len() as f64
}
