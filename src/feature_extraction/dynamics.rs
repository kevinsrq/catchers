

/// Returns the time reversal asymmetry statistic.
///
/// This function calculates the value of:
/// mean(x(i + 2*lag)^2 * x(i + lag) - x(i + lag) * x(i)^2)
///
/// :param x: the time series to calculate the feature of
/// :param lag: the lag that should be used in the calculation of the feature
pub fn time_reversal_asymmetry_statistic(x: &[f64], lag: usize) -> f64 {
    let n = x.len();
    if 2 * lag >= n {
        return 0.0;
    }

    // We iterate from 0 to n - 2*lag
    let limit = n - 2 * lag;
    let mut sum = 0.0;
    
    for i in 0..limit {
        let x_i = x[i];
        let x_lag = x[i + lag];
        let x_2lag = x[i + 2 * lag];
        
        // (x_{i+2lag}^2 * x_{i+lag}) - (x_{i+lag} * x_{i}^2)
        sum += (x_2lag * x_2lag * x_lag) - (x_lag * x_i * x_i);
    }
    
    sum / limit as f64
}

/// Uses c3 statistics to measure non linearity in the time series.
///
/// This function calculates the value of:
/// mean(x(i + 2*lag) * x(i + lag) * x(i))
///
/// :param x: the time series to calculate the feature of
/// :param lag: the lag that should be used in the calculation of the feature
pub fn c3(x: &[f64], lag: usize) -> f64 {
    let n = x.len();
    if 2 * lag >= n {
        return 0.0;
    }
    
    let limit = n - 2 * lag;
    let mut sum = 0.0;
    
    for i in 0..limit {
        sum += x[i + 2 * lag] * x[i + lag] * x[i];
    }
    
    sum / limit as f64
}
