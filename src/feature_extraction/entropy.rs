use crate::utils::stats::basic::std_dev;

/// Calculates and returns sample entropy of x.
/// m: length of sequences to compare (usually 2)
/// r: tolerance (usually 0.2 * std_dev)
pub fn sample_entropy(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 3 { return f64::NAN; }
    
    let m = 2;
    let r = 0.2 * std_dev(x);
    if r == 0.0 { return f64::NAN; }

    fn count_matches(data: &[f64], m: usize, r: f64) -> f64 {
        let n = data.len();
        let mut count = 0;
        for i in 0..n - m {
            for j in i + 1..n - m {
                let mut match_found = true;
                for k in 0..m {
                    if (data[i + k] - data[j + k]).abs() > r {
                        match_found = false;
                        break;
                    }
                }
                if match_found {
                    count += 1;
                }
            }
        }
        count as f64
    }

    let a = count_matches(x, m + 1, r);
    let b = count_matches(x, m, r);

    if a == 0.0 || b == 0.0 {
        return f64::NAN;
    }

    -(a / b).ln()
}

/// Calculate the Binned Entropy.
pub fn binned_entropy(x: &[f64], max_bins: usize) -> f64 {
    if x.is_empty() { return 0.0; }
    
    // Simple histogram-based entropy
    use crate::utils::stats::histogram::histcounts;
    let (counts, _) = histcounts(x, max_bins);
    let total = x.len() as f64;
    
    counts.iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total;
            -p * p.ln()
        })
        .sum()
}
