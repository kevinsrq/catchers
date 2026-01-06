use std::f64;

/// Returns the calculated relative first location of the maximum value of x.
/// The position is calculated relatively to the length of x.
pub fn first_location_of_maximum(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }

    let (idx, _) =
        x.iter()
            .enumerate()
            .fold((0, f64::NEG_INFINITY), |(max_idx, max_val), (i, &val)| {
                if val > max_val {
                    (i, val)
                } else {
                    (max_idx, max_val)
                }
            });

    idx as f64 / x.len() as f64
}

/// Returns the relative last location of the maximum value of x.
/// The position is calculated relatively to the length of x.
pub fn last_location_of_maximum(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }

    // Iterate backwards to find the last occurrence of the maximum
    let (rev_idx, _) =
        x.iter()
            .rev()
            .enumerate()
            .fold((0, f64::NEG_INFINITY), |(max_idx, max_val), (i, &val)| {
                if val > max_val {
                    (i, val)
                } else {
                    (max_idx, max_val)
                }
            });

    // Matches tsfresh: 1.0 - argmax(x[::-1]) / len(x)
    1.0 - (rev_idx as f64 / x.len() as f64)
}

/// Returns the first location of the minimal value of x.
/// The position is calculated relatively to the length of x.
pub fn first_location_of_minimum(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }

    let (idx, _) =
        x.iter()
            .enumerate()
            .fold((0, f64::INFINITY), |(min_idx, min_val), (i, &val)| {
                if val < min_val {
                    (i, val)
                } else {
                    (min_idx, min_val)
                }
            });

    idx as f64 / x.len() as f64
}

/// Returns the last location of the minimal value of x.
/// The position is calculated relatively to the length of x.
pub fn last_location_of_minimum(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }

    let (rev_idx, _) =
        x.iter()
            .rev()
            .enumerate()
            .fold((0, f64::INFINITY), |(min_idx, min_val), (i, &val)| {
                if val < min_val {
                    (i, val)
                } else {
                    (min_idx, min_val)
                }
            });

    1.0 - (rev_idx as f64 / x.len() as f64)
}

/// Calculates the number of peaks of at least support n in the time series x.
/// A peak of support n is defined as a subsequence of x where a value occurs,
/// which is bigger than its n neighbours to the left and to the right.
pub fn number_peaks(x: &[f64], n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    } // Support 0 undefined or no peaks? tsfresh loop range(1, n+1), so n=0 loops 1..1 empty.
    if x.len() < 2 * n + 1 {
        return 0.0;
    }

    let mut count = 0;

    for i in n..(x.len() - n) {
        let val = x[i];
        let mut is_peak = true;

        // Check left neighbors
        for j in 1..=n {
            if val <= x[i - j] {
                is_peak = false;
                break;
            }
        }

        if !is_peak {
            continue;
        }

        // Check right neighbors
        for j in 1..=n {
            if val <= x[i + j] {
                is_peak = false;
                break;
            }
        }

        if is_peak {
            count += 1;
        }
    }

    count as f64
}

/// Calculates the relative index i of time series x where q% of the mass of x lies left of i.
pub fn index_mass_quantile(x: &[f64], q: f64) -> f64 {
    let abs_sum: f64 = x.iter().map(|v| v.abs()).sum();
    if abs_sum == 0.0 {
        return f64::NAN;
    }

    let mut current_sum = 0.0;
    for (i, &val) in x.iter().enumerate() {
        current_sum += val.abs();
        if current_sum / abs_sum >= q {
            return (i + 1) as f64 / x.len() as f64;
        }
    }

    f64::NAN // Should technically be reached inside loop unless q > 1.0 or issues
}
