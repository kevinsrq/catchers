// Removed unused mean import

/// Average over first differences absolute values.
pub fn mean_abs_change(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return f64::NAN;
    }
    let sum_abs_diff: f64 = x.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
    sum_abs_diff / (x.len() - 1) as f64
}

/// Average over first differences.
pub fn mean_change(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return f64::NAN;
    }
    (x[x.len() - 1] - x[0]) / (x.len() - 1) as f64
}

/// Mean value of a central approximation of the second derivative.
pub fn mean_second_derivative_central(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 3 {
        return f64::NAN;
    }
    // Sum_{i=0 to n-3} 0.5 * (x[i+2] - 2*x[i+1] + x[i])
    // The sum telescopes to: 0.5 * (x[n-1] - x[n-2] - x[1] + x[0])
    (x[n - 1] - x[n - 2] - x[1] + x[0]) / (2.0 * (n as f64 - 2.0))
}

/// Returns the sum over the absolute value of consecutive changes.
pub fn absolute_sum_of_changes(x: &[f64]) -> f64 {
    x.windows(2).map(|w| (w[1] - w[0]).abs()).sum()
}
