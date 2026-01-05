//! Basic statistical helper functions.
//!
//! This module provides fundamental statistical operations required by the catchers library.

/// Calculates the arithmetic mean of a slice.
pub fn mean(a: &[f64]) -> f64 {
    let sum: f64 = a.iter().sum();
    if a.is_empty() { 0.0 } else { sum / a.len() as f64 }
}

/// Calculates the median of a slice.
///
/// Note: This function clones and sorts the data.
pub fn median(a: &[f64]) -> f64 {
    let mut c = a.to_vec();
    c.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = c.len() / 2;
    if c.len() % 2 == 0 { (c[mid - 1] + c[mid]) / 2.0 } else { c[mid] }
}

/// Returns the maximum value in a slice.
pub fn max_(a: &[f64]) -> f64 {
    a.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
}

/// Returns the minimum value in a slice.
pub fn min_(a: &[f64]) -> f64 {
    a.iter().fold(f64::INFINITY, |a, &b| a.min(b))
}

/// Calculates the standard deviation of a slice.
pub fn std_dev(a: &[f64]) -> f64 {
    let m = mean(a);
    let variance = a.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (a.len().saturating_sub(1) as f64).max(1.0);
    variance.sqrt()
}

/// Checks if all elements in the slice are effectively constant (within tolerance).
pub fn is_constant(a: &[f64]) -> bool {
    if a.is_empty() { return true; }
    let first = a[0];
    a.iter().all(|&x| (x - first).abs() < 1e-10)
}

/// Computes the first difference of the slice.
pub fn diff(a: &[f64]) -> Vec<f64> {
    if a.len() < 2 { return vec![]; }
    a.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Calculates the Euclidean norm (L2 norm) of the slice.
pub fn norm(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Coarse-grains a time series into `num_groups` quantiles.
///
/// Returns a vector of group indices (1 to `num_groups`).
pub fn coarsegrain(a: &[f64], num_groups: usize) -> Vec<usize> {
    let mut c = a.to_vec();
    c.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut out = vec![0; a.len()];
    for (i, &val) in a.iter().enumerate() {
        let rank = c.binary_search_by(|p| p.partial_cmp(&val).unwrap()).unwrap_or(0);
        let group = (rank as f64 * num_groups as f64 / a.len() as f64).floor() as usize + 1;
        out[i] = group.min(num_groups);
    }
    out
}

/// Calculates the entropy of a probability distribution.
pub fn f_entropy(probs: &[f64]) -> f64 {
    probs.iter().map(|&p| if p > 0.0 { -p * p.ln() } else { 0.0 }).sum()
}

/// Computes the covariance matrix of a dataset (rows are observations, columns are variables).
pub fn covariance_matrix(matrix: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let rows = matrix.len();
    if rows == 0 { return vec![]; }
    let cols = matrix[0].len();
    let means: Vec<f64> = matrix.iter().map(|row| mean(row)).collect();
    let mut cov = vec![vec![0.0; rows]; rows];
    for i in 0..rows {
        for j in 0..rows {
             let mut sum = 0.0;
             for k in 0..cols {
                 sum += (matrix[i][k] - means[i]) * (matrix[j][k] - means[j]);
             }
             cov[i][j] = sum / (cols.saturating_sub(1) as f64).max(1.0);
        }
    }
    cov
}
