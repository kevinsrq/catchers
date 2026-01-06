use crate::utils::stats::signal::autocorr;
use crate::utils::stats::basic::{mean, median};

/// Calculates the value of an aggregation function f_agg (e.g. the variance or the mean) 
/// over the autocorrelation R(l) for different lags.
/// 
/// f_agg can be: "mean", "var", "std", "median".
pub fn agg_autocorrelation(x: &[f64], maxlag: usize, f_agg: &str) -> f64 {
    let ac = autocorr(x); // returns AC at lag 0..n
    
    // We want lags 1..=maxlag.
    // Ensure maxlag < ac.len().
    if ac.len() < 2 { return 0.0; } // needs at least lag 1
    
    let limit = std::cmp::min(maxlag + 1, ac.len());
    if limit <= 1 { return 0.0; }
    
    let lags = &ac[1..limit];
    
    match f_agg {
        "mean" => mean(lags),
        "median" => median(lags),
        "var" | "variance" => {
             let m = mean(lags);
             lags.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / (lags.len() as f64).max(1.0) // Population variance? numpy var defaults pop.
        },
        "std" => {
             let m = mean(lags);
             let var = lags.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / (lags.len() as f64).max(1.0);
             var.sqrt()
        },
        _ => f64::NAN,
    }
}

/// Calculates the value of the partial autocorrelation function at the given lag `k`.
/// Uses Durbin-Levinson algorithm on the sample autocorrelations.
pub fn partial_autocorrelation(x: &[f64], k: usize) -> f64 {
    if k == 0 { return 1.0; }
    let ac = autocorr(x);
    if ac.len() <= k { return f64::NAN; }
    
    // Durbin-Levinson Algorithm
    // We need ACs rho_0 ... rho_k
    let rho = &ac; 
    
    // Initialize phi[1][1] = rho[1]
    let mut phi = vec![vec![0.0; k + 1]; k + 1];
    
    phi[1][1] = rho[1];
    
    for n in 2..=k {
        let mut num = rho[n];
        let mut den = 1.0;
        
        for j in 1..n {
            num -= phi[n-1][j] * rho[n-j];
            den -= phi[n-1][j] * rho[j];
        }
        
        // Prevent div by zero
        if den.abs() < 1e-10 {
           phi[n][n] = 0.0;
        } else {
           phi[n][n] = num / den;
        }
        
        for j in 1..n {
            phi[n][j] = phi[n-1][j] - phi[n][n] * phi[n-1][n-j];
        }
    }
    
    phi[k][k]
}
