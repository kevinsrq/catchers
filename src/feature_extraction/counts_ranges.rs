use crate::utils::stats::basic::{max_, min_};

/// Calculates the number of crossings of x with m.
/// A crossing is defined as two sequential values where the first value is lower than m and the next is greater,
/// or vice-versa. If one of the values is equal to m, it is count as a crossing.
pub fn number_crossing_m(x: &[f64], m: f64) -> f64 {
    if x.len() < 2 { return 0.0; }
    
    let mut count = 0;
    // We iterate through pairs
    for w in x.windows(2) {
        let a = w[0];
        let b = w[1];
        
        // Check if one > m and other < m, or equal.
        // Formula: (x[i] > m and x[i+1] <= m) or (x[i] < m and x[i+1] >= m)
        // Or simplified: (x[i] - m) * (x[i+1] - m) <= 0
        if (a - m) * (b - m) <= 0.0 {
            // But wait, if they are BOTH equal to m? (0*0 <= 0). Is that a crossing?
            // tsfresh: "If one of the values is equal to m, it is count as a crossing."
            // But if x = [m, m], is that a crossing?
            // fresh.py: ((x > m) & (x_shift <= m)) | ((x < m) & (x_shift >= m))
            // if x=m, x>m is False. x<m is False.
            // So if x[i] == m:
            // Term 1: False & ... = False
            // Term 2: False & ... = False
            // So x[i] == m is NOT a crossing start unless formula handles it logic wise.
            // Let's check fresh.py code logic precisely if possible or stick to simple crossing def.
            // fresh.py: `res = ((x > m) & (_roll(x, -1) <= m)) | ((x < m) & (_roll(x, -1) >= m))`
            // If x[i] = m: (False & ...) | (False & ...) -> False.
            // If x[i+1] = m: (True & True) | (False & True) -> True (if x[i] > m).
            // So it counts strictly passing through m, or landing on m from distinct side.
            // My condition (a-m)*(b-m) <= 0 includes a=m, b=m case which might be wrong.
            // Let's replicate fresh.py logic exactly.
            
            let a_gt = a > m;
            let a_lt = a < m;
            let b_le = b <= m;
            let b_ge = b >= m;
            
            if (a_gt && b_le) || (a_lt && b_ge) {
                count += 1;
            }
        }
    }
    count as f64
}

/// Count of values within interval [min, max].
pub fn range_count(x: &[f64], min: f64, max: f64) -> f64 {
    x.iter().filter(|&&v| v >= min && v <= max).count() as f64
}

/// Count occurrences of a specific value.
pub fn value_count(x: &[f64], value: f64) -> f64 {
    // Exact float equality check? tsfresh probably does exact check or with epsilon?
    // fresh.py `value_count` uses `np.sum(x == value)`.
    // Standard float equality in Rust might fail. But if input is from discrete data it's fine.
    // We'll use strict equality or epsilon. Let's start with proper epsilon for safer 'value count'.
    // Or just strictly if the user sends specific floats.
    // Let's use a small epsilon for robustness.
    
    x.iter().filter(|&&v| (v - value).abs() < f64::EPSILON).count() as f64
}
