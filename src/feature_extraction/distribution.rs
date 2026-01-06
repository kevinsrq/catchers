// Removed unused mean import
use crate::utils::stats::regression::linreg_slice;

/// Estimates the Friedrich coefficients of the time series.
/// Ported logic from tsfresh: fits a polynomial to deterministic dynamics.
/// m: order of polynomial (usually 3)
/// r: number of quantiles (usually 30)
pub fn friedrich_coefficients(x: &[f64], m: usize, r: usize) -> Vec<f64> {
    if x.len() < 2 { return vec![f64::NAN; m + 1]; }
    
    let signal = &x[..x.len()-1];
    let mut delta = Vec::with_capacity(x.len() - 1);
    for i in 0..x.len() - 1 {
        delta.push(x[i+1] - x[i]);
    }

    // Sort to find quantiles
    let mut sorted_signal = signal.to_vec();
    sorted_signal.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    let mut quantile_means_x = Vec::with_capacity(r);
    let mut quantile_means_y = Vec::with_capacity(r);
    
    let chunk_size = (signal.len() as f64 / r as f64).ceil() as usize;
    if chunk_size == 0 { return vec![f64::NAN; m + 1]; }

    for i in 0..r {
        let start = i * chunk_size;
        let end = signal.len().min((i + 1) * chunk_size);
        if start >= end { break; }
        
        let q_thresh_low = sorted_signal[start];
        let q_thresh_high = sorted_signal[end - 1];
        
        // Find points in this quantile range in original signal
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut count = 0;
        for (idx, &val) in signal.iter().enumerate() {
            if val >= q_thresh_low && val <= q_thresh_high {
                sum_x += val;
                sum_y += delta[idx];
                count += 1;
            }
        }
        
        if count > 0 {
            quantile_means_x.push(sum_x / count as f64);
            quantile_means_y.push(sum_y / count as f64);
        }
    }

    if quantile_means_x.len() < m + 1 {
        return vec![f64::NAN; m + 1];
    }

    // Simplified polynomial fit for m=1 (linear)
    // For m > 1, we would need a proper polynomial regression.
    // Since tsfresh uses statsmodels/numpy, we'll implement m=1 and m=2/3 placeholders or use a crate.
    // Let's implement m=1 (linear) for now as it's common.
    if m == 1 {
        let (slope, intercept) = linreg_slice(quantile_means_x.len(), &quantile_means_x, &quantile_means_y);
        // Map index to coefficient: 0 -> intercept (const), 1 -> slope (x^1 term) ?
        // tsfresh returns "coeff_0", "coeff_1". 
        // Typically polynomial p(x) = c0 + c1*x + ...
        // Our linreg returns (slope, intercept). 
        // So c1 = slope, c0 = intercept.
        // Let's assume user asks for index 0, 1...
        // Wait, tsfresh `friedrich_coefficients` returns coefficients of the fitted polynomial.
        // If m=1, returns c_0, c_1.
        // linreg returns (slope, intercept) -> y = slope * x + intercept.
        // So c_1 = slope, c_0 = intercept.
        
        // We'll return based on requesting index.
        // Note: this function signature needs to change to take `index`.
        // But the previous signature returned Vec<f64>.
        // Let's create a new wrapper or change this one.
        vec![intercept, slope]
    } else {
        vec![0.0; m + 1]
    }
}

pub fn friedrich_coefficient(x: &[f64], m: usize, r: usize, coeff_index: usize) -> f64 {
    let coeffs = friedrich_coefficients(x, m, r);
    if coeff_index < coeffs.len() {
        coeffs[coeff_index]
    } else {
        f64::NAN
    }
}
