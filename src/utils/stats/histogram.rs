//! Histogram utility functions.

use crate::utils::stats::basic::{max_, min_};

/// Automatically calculates the number of bins using the Freedman-Diaconis rule.
pub fn num_bins_auto(a: &[f64]) -> usize {
    if a.len() < 2 { return 1; }
    let iqr = {
        let mut c = a.to_vec();
        c.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q1 = c[c.len() / 4];
        let q3 = c[c.len() * 3 / 4];
        q3 - q1
    };
    if iqr == 0.0 { return 1; }
    let h = 2.0 * iqr * (a.len() as f64).powf(-1.0 / 3.0);
    let range = max_(a) - min_(a);
    (range / h).ceil() as usize
}

/// Calculates histogram counts and edges.
pub fn histcounts(a: &[f64], n_bins: usize) -> (Vec<usize>, Vec<f64>) {
    let min_v = min_(a);
    let max_v = max_(a);
    let mut counts = vec![0; n_bins];
    let mut edges = vec![0.0; n_bins + 1];
    if n_bins == 0 { return (counts, edges); }
    let step = (max_v - min_v) / n_bins as f64;
    for i in 0..=n_bins { edges[i] = min_v + i as f64 * step; }
    edges[n_bins] = max_v + 1e-10; 
    for &x in a {
        for i in 0..n_bins {
            if x >= edges[i] && x < edges[i+1] {
                counts[i] += 1;
                break;
            }
        }
    }
    (counts, edges)
}

/// Assigns each value in the slice to a histogram bin defined by `edges`.
///
/// Returns a vector of bin indices.
pub fn histbinassign(a: &[f64], edges: &[f64]) -> Vec<usize> {
    a.iter().map(|&x| {
        for i in 0..edges.len()-1 {
            if x >= edges[i] && x < edges[i+1] { return i + 1; }
        }
        if x >= edges[edges.len()-1] - 1e-9 { return edges.len() - 1; }
        0 
    }).collect()
}

/// Counts frequencies of pre-binned values using edges.
/// note: this function seems to treat `bins` as raw values to be counted against `edges`?
/// Based on usage in catchers.rs (joint_hist), it seems `bins` input here are actually values (like `bins12`).
pub fn histcount_edges(bins: &[f64], edges: &[f64]) -> Vec<usize> {
    let mut counts = vec![0; edges.len() - 1];
    for &val in bins {
        let v = val; 
         for i in 0..edges.len()-1 {
            if v >= edges[i] && v < edges[i+1] {
                counts[i] += 1;
                break;
            }
        }
    }
    counts
}
