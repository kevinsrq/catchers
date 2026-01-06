use crate::utils::stats::basic::{mean, std_dev, median, max_, min_};
use polars::prelude::*;

/// Is variance higher than the standard deviation?
pub fn variance_larger_than_standard_deviation(x: &[f64]) -> bool {
    let s = std_dev(x);
    let var = s.powi(2);
    var > s
}

/// Ratio of values that are more than r * std(x) away from the mean of x.
pub fn ratio_beyond_r_sigma(x: &[f64], r: f64) -> f64 {
    if x.is_empty() { return 0.0; }
    let m = mean(x);
    let s = std_dev(x);
    if s == 0.0 { return 0.0; }
    
    let count = x.iter().filter(|&&v| (v - m).abs() > r * s).count();
    count as f64 / x.len() as f64
}

/// Boolean variable denoting if the standard dev of x is higher than 'r' times the range.
pub fn large_standard_deviation(x: &[f64], r: f64) -> bool {
    if x.is_empty() { return false; }
    let s = std_dev(x);
    let range = max_(x) - min_(x);
    if range == 0.0 { return false; }
    s > r * range
}

/// Boolean variable denoting if the distribution of x looks symmetric.
pub fn symmetry_looking(x: &[f64], r: f64) -> bool {
    if x.is_empty() { return false; }
    let m = mean(x);
    let med = median(x);
    let range = max_(x) - min_(x);
    if range == 0.0 { return true; }
    (m - med).abs() < r * range
}

/// Checks if the maximum value of x is observed more than once.
pub fn has_duplicate_max(x: &[f64]) -> bool {
    if x.is_empty() { return false; }
    let m = max_(x);
    x.iter().filter(|&&v| v == m).count() >= 2
}

/// Checks if the minimal value of x is observed more than once.
pub fn has_duplicate_min(x: &[f64]) -> bool {
    if x.is_empty() { return false; }
    let m = min_(x);
    x.iter().filter(|&&v| v == m).count() >= 2
}

/// Checks if any value in x occurs more than once.
pub fn has_duplicate(x: &[f64]) -> bool {
    // Note: for f64, exact equality is tricky but tsfresh uses it.
    let mut sorted = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted.windows(2).any(|w| w[0] == w[1])
}

/// Returns the root mean square (rms) of the time series.
pub fn root_mean_square(x: &[f64]) -> f64 {
    if x.is_empty() { return 0.0; }
    let sum_sq: f64 = x.iter().map(|v| v * v).sum();
    (sum_sq / x.len() as f64).sqrt()
}

/// Returns the variation coefficient (standard deviation / mean).
pub fn variation_coefficient(x: &[f64]) -> f64 {
    let m = mean(x);
    if m == 0.0 { return f64::NAN; }
    std_dev(x) / m
}

/// Returns the sample skewness of x (Fisher-Pearson coefficient of skewness).
pub fn skewness(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 3 { return f64::NAN; }
    
    let m = mean(x);
    let s = std_dev(x);
    
    if s == 0.0 { return 0.0; }
    
    let sum_cubed_diff: f64 = x.iter().map(|&v| {
        let diff = (v - m) / s;
        diff.powi(3)
    }).sum();
    
    sum_cubed_diff / n as f64
}

/// Returns the kurtosis of x (Fisher kurtosis, 3.0 for normal distribution).
/// tsfresh uses scipy.stats.kurtosis which is Fisher (normal = 0.0) by default? 
/// Or Pearson (normal = 3.0)? 
/// fresh.py docs say "Calculates the kurtosis as the fourth standardized moment... In the limit of a dirac delta, kurtosis should be 3".
/// So it seems they calculate Pearson kurtosis but maybe check if they subtract 3.
/// fresh.py `get_kurtosis`: `(moment(4) ... ) / var^2`. If they don't subtract 3, it's Pearson.
/// Wait, fresh.py: `get_kurtosis` implementation: `... / get_variance(y)**2`. It doesn't look like it subtracts 3.
/// But scipy.stats.kurtosis defaults to fisher=True (excess kurtosis, subtracts 3).
/// Let's assume standard Pearson moment-based kurtosis for now, or check fresh.py `kurtosis` function specifically.
/// I will check `fresh.py` again.
pub fn kurtosis(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 4 { return f64::NAN; }
    
    let m = mean(x);
    let s = std_dev(x);
    
    if s == 0.0 { return 0.0; } // flat line
    
    let sum_fourth_diff: f64 = x.iter().map(|&v| {
        let diff = (v - m) / s;
        diff.powi(4)
    }).sum();
    
    // Pearson kurtosis = E[((x-mu)/sigma)^4]. Normal dist is 3.
    // Excess kurtosis = Pearson - 3.
    // tsfresh `kurtosis` function uses `scipy.stats.kurtosis(x)`. 
    // scipy defaults to fisher=True (excess).
    // So we should subtract 3.
    
    (sum_fourth_diff / n as f64) - 3.0
}
