use polars::prelude::*;
use crate::utils::stats::basic::mean;

// ==============================================================================
// 2. Estatística e Regressão
// ==============================================================================

/// Computes a simple linear regression on a Polars ChunkedArray.
///
/// Returns (slope, intercept).
/// Note: Indexes are assumed to be 0..n-1.
#[allow(dead_code)]
pub fn linreg<T>(ca: &ChunkedArray<T>) -> PolarsResult<(f64, f64)>
where
    T: PolarsNumericType,
{
    let y = ca.cast(&DataType::Float64)?;
    let y = y.f64()?;
    let n = y.len();
    if n < 2 {
        return Ok((0.0, 0.0));
    }

    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y_vec: Vec<f64> = y.into_no_null_iter().collect();

    let x_mean = mean(&x);
    let y_mean = mean(&y_vec);

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..n {
        numerator += (x[i] - x_mean) * (y_vec[i] - y_mean);
        denominator += (x[i] - x_mean).powi(2);
    }

    if denominator == 0.0 {
        return Ok((0.0, y_mean));
    }

    let slope = numerator / denominator;
    let intercept = y_mean - slope * x_mean;

    Ok((slope, intercept))
}

/// Calculates the slope of a linear regression on a Polars ChunkedArray.
#[allow(dead_code)]
pub fn slope<T>(ca: &ChunkedArray<T>) -> PolarsResult<f64>
where
    T: PolarsNumericType,
{
    let (slope, _) = linreg(ca)?;
    Ok(slope)
}

/// Computes a linear regression on `&[f64]` slices.
///
/// Returns (slope, intercept).
pub fn linreg_slice(n: usize, x: &[f64], y: &[f64]) -> (f64, f64) {
    let eff_n = n.min(x.len()).min(y.len());
    
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_xx = 0.0;
    
    for i in 0..eff_n {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_xx += x[i] * x[i];
    }
    
    let dn = eff_n as f64;
    let denom = dn * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-10 { return (0.0, 0.0); }
    
    let slope = (dn * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y * sum_xx - sum_x * sum_xy) / denom;
    (slope, intercept)
}
