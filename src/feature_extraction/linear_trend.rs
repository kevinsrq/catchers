use crate::utils::stats::basic::{mean, std_dev};

/// Calculate a linear least-squares regression for the values of the time series versus the sequence from 0 to
/// length of the time series minus one.
///
/// Returns: (slope, intercept, r_value, p_value, std_err)
/// Note: p_value and std_err calculation might be simplified or omitted if not critical,
/// but tsfresh includes them. We'll implement slope, intercept, r_value, and rss/stderr.
///
/// Uses simple OLS formulas:
/// slope = Cov(x, y) / Var(x)
/// intercept = mean(y) - slope * mean(x)
/// r_value = Cov(x, y) / (std(x) * std(y))
pub fn linear_trend(x: &[f64]) -> (f64, f64, f64, f64, f64) {
    // slope, intercept, r_value, p_value, stderr
    let n = x.len() as f64;
    if n < 2.0 {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }

    // x values are 0, 1, ..., n-1
    let idx: Vec<f64> = (0..x.len()).map(|i| i as f64).collect();

    let mean_x = mean(&idx);
    let mean_y = mean(x);
    let std_x = std_dev(&idx);
    let std_y = std_dev(x);

    if std_x == 0.0 {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }

    // Covariance(x, y)
    // = sum((x_i - mean_x) * (y_i - mean_y)) / (n-1)
    let cov: f64 = idx
        .iter()
        .zip(x.iter())
        .map(|(&i, &v)| (i - mean_x) * (v - mean_y))
        .sum::<f64>()
        / (n - 1.0);

    let slope = cov / (std_x * std_x);
    let intercept = mean_y - slope * mean_x;

    let r_value = if std_y == 0.0 {
        0.0
    } else {
        cov / (std_x * std_y)
    };

    // Standard error of the estimate (residual standard error)
    // RSS = sum((y_i - (intercept + slope * x_i))^2)
    let rss: f64 = idx
        .iter()
        .zip(x.iter())
        .map(|(&i, &v)| {
            let pred = intercept + slope * i;
            (v - pred).powi(2)
        })
        .sum();

    let stderr = if n > 2.0 {
        (rss / (n - 2.0)).sqrt() / (std_x * (n - 1.0).sqrt())
    } else {
        0.0
    };

    // P-value requires t-distribution cdf, typically involves `statrs` or similar.
    // For now, returning NaN for p-value to avoid heavy deps, as it's less commonly used in ML feats than slope/r2.
    // Or we can approximate if strictly needed. tsfresh uses scipy.stats.linregress.
    let p_value = f64::NAN;

    (slope, intercept, r_value, p_value, stderr)
}

/// Helper to extract specific attribute from linear trend
pub fn linear_trend_attr(x: &[f64], attr: &str) -> f64 {
    let (slope, intercept, r_value, p_value, stderr) = linear_trend(x);
    match attr {
        "slope" => slope,
        "intercept" => intercept,
        "rvalue" => r_value,
        "pvalue" => p_value,
        "stderr" => stderr,
        _ => f64::NAN,
    }
}
