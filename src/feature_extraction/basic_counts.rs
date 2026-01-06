use crate::utils::stats::basic::mean;

/// Returns the number of values in x that are higher than the mean of x.
pub fn count_above_mean(x: &[f64]) -> f64 {
    let m = mean(x);
    x.iter().filter(|&&v| v > m).count() as f64
}

/// Returns the number of values in x that are lower than the mean of x.
pub fn count_below_mean(x: &[f64]) -> f64 {
    let m = mean(x);
    x.iter().filter(|&&v| v < m).count() as f64
}

/// Returns the length of the longest consecutive subsequence in x that is greater than the mean of x.
pub fn longest_strike_above_mean(x: &[f64]) -> f64 {
    let m = mean(x);
    let mut max_strike = 0;
    let mut current_strike = 0;
    
    for &v in x {
        if v > m {
            current_strike += 1;
        } else {
            max_strike = max_strike.max(current_strike);
            current_strike = 0;
        }
    }
    max_strike.max(current_strike) as f64
}

/// Returns the length of the longest consecutive subsequence in x that is smaller than the mean of x.
pub fn longest_strike_below_mean(x: &[f64]) -> f64 {
    let m = mean(x);
    let mut max_strike = 0;
    let mut current_strike = 0;
    
    for &v in x {
        if v < m {
            current_strike += 1;
        } else {
            max_strike = max_strike.max(current_strike);
            current_strike = 0;
        }
    }
    max_strike.max(current_strike) as f64
}

/// Helper to get unique counts via sorting to avoid float hashing issues.
/// Returns (value, count) pairs.
fn get_unique_counts(x: &[f64]) -> Vec<(f64, usize)> {
    let mut sorted = x.to_vec();
    // sort_by handles NaNs by putting them at the end or panic, let's filter or handle carefully.
    // tsfresh usually ignores NaNs or handles them. Here we'll handle standard floats.
    // We treat NaNs as equal to each other for grouping if present, or push to end.
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    let mut counts = Vec::new();
    if sorted.is_empty() { return counts; }
    
    let mut current_val = sorted[0];
    let mut current_count = 1;
    
    for &v in sorted.iter().skip(1) {
        if (v - current_val).abs() < f64::EPSILON { // float equality
            current_count += 1;
        } else {
            counts.push((current_val, current_count));
            current_val = v;
            current_count = 1;
        }
    }
    counts.push((current_val, current_count));
    counts
}

/// Returns the percentage of unique values that are present in the time series more than once.
/// len(different values occurring more than once) / len(different values)
pub fn percentage_of_reoccurring_values_to_all_values(x: &[f64]) -> f64 {
    let counts = get_unique_counts(x);
    if counts.is_empty() { return 0.0; }
    
    let num_unique = counts.len() as f64;
    let num_reoccurring = counts.iter().filter(|&&(_, c)| c > 1).count() as f64;
    
    num_reoccurring / num_unique
}

/// Returns the percentage of non-unique data points.
/// # of data points occurring more than once / # of all data points
pub fn percentage_of_reoccurring_datapoints_to_all_datapoints(x: &[f64]) -> f64 {
    let counts = get_unique_counts(x);
    if x.is_empty() { return 0.0; }
    
    let sum_reoccurring: usize = counts.iter()
        .filter(|&&(_, c)| c > 1)
        .map(|&(_, c)| c)
        .sum();
    
    sum_reoccurring as f64 / x.len() as f64
}

/// Returns the sum of all values, that are present in the time series more than once.
pub fn sum_of_reoccurring_values(x: &[f64]) -> f64 {
    let counts = get_unique_counts(x);
    counts.iter()
        .filter(|&&(_, c)| c > 1)
        .map(|&(v, _)| v)
        .sum()
}

/// Returns the sum of all data points, that are present in the time series more than once.
pub fn sum_of_reoccurring_data_points(x: &[f64]) -> f64 {
    let counts = get_unique_counts(x);
    counts.iter()
        .filter(|&&(_, c)| c > 1)
        .map(|&(v, c)| v * c as f64)
        .sum()
}

/// Returns a factor which is 1 if all values in the time series occur only once,
/// and below one if this is not the case. len(unique) / len(x)
pub fn ratio_value_number_to_time_series_length(x: &[f64]) -> f64 {
    if x.is_empty() { return 0.0; }
    let counts = get_unique_counts(x);
    counts.len() as f64 / x.len() as f64
}
