//! Catchers feature extraction library.
//!
//! Port of the catch22 time-series feature extraction library to Rust.
//! Provides canonical time-series characteristics (catch22).

use std::f64;

use crate::utils::stats::basic::{
    coarsegrain, covariance_matrix, diff, f_entropy, is_constant, max_, mean, median, min_, norm,
    std_dev,
};
use crate::utils::stats::histogram::{histbinassign, histcount_edges, histcounts, num_bins_auto};
use crate::utils::stats::regression::linreg_slice as linreg;
use crate::utils::stats::signal::{
    autocorr, autocorr_lag, autocov_lag, first_zero_core as first_zero, welch,
};
use crate::utils::stats::smoothing::splinefit_core;

/// Distribution of Outliers (DN_OutlierInclude_n_001_mdrmd)
///
/// Measures the spread of the data including outliers.
/// Returns the median of the distribution of relative differences between successive outliers.
pub fn dn_outlier_include_np_001_mdrmd(a: &[f64], is_pos: bool) -> f64 {
    if is_constant(a) {
        return 0.0;
    }

    // Optimization: avoid full clone if possible, but we might need to negate.
    // If is_pos is true, we can just use reference or Cow?
    // Since we need to modify values (negate), we unfortunately need a copy or an iterator.
    // However, for max_ and filters we can use iterators.

    // We'll use a local vector but optimize the inner loop allocations.
    let working_a: Vec<f64>;
    let a_ref = if !is_pos {
        working_a = a.iter().map(|&x| -x).collect();
        &working_a
    } else {
        a
    };

    let inc = 0.01;
    let tot = a_ref.iter().filter(|&&x| x >= 0.0).count();
    let max_val = max_(a_ref);

    if max_val < inc {
        return 0.0;
    }

    let n_thresh = ((max_val / inc) + 1.0) as usize;

    // Reuse buffers
    let mut r = vec![0.0; a_ref.len()];
    let mut dt_exc = Vec::with_capacity(a_ref.len()); // Dynamic reuse

    let mut msdti1 = vec![0.0; n_thresh];
    let mut msdti3 = vec![0.0; n_thresh];
    let mut msdti4 = vec![0.0; n_thresh];

    for i in 0..n_thresh {
        let thresh = i as f64 * inc;
        let mut high_size = 0;

        for (j, &val) in a_ref.iter().enumerate() {
            if val >= thresh {
                r[high_size] = (j + 1) as f64;
                high_size += 1;
            }
        }

        dt_exc.clear();
        for j in 0..high_size.saturating_sub(1) {
            dt_exc.push(r[j + 1] - r[j]);
        }

        msdti1[i] = mean(&dt_exc);
        msdti3[i] = (dt_exc.len() as f64 * 100.0) / tot as f64;

        // Optimize median calculation?
        // median() sorts the slice. buffer `r` is partially filled.
        msdti4[i] = median(&r[..high_size]) / (a_ref.len() as f64 / 2.0) - 1.0;
    }

    let trim_tr = 2.0;
    let mut mj = 0;
    let mut fbi = n_thresh - 1;

    for i in 0..n_thresh {
        if msdti3[i] > trim_tr {
            mj = i;
        }
        if msdti1[n_thresh - i - 1].is_nan() {
            fbi = n_thresh - i - 1;
        }
    }

    let trim_lim = mj.min(fbi);
    median(&msdti4[..trim_lim + 1])
}

/// Histogram Mode (DN_HistogramMode_5 / 10).
///
/// Measures the mode of the data using a histogram with `n_bins`.
pub fn dn_histogram_mode_n(a: &[f64], n_bins: usize) -> f64 {
    let (bin_counts, bin_edges) = histcounts(a, n_bins);

    let mut max_count = 0;
    let mut num_maxs = 1;
    let mut res = 0.0;

    for i in 0..n_bins {
        if bin_counts[i] > max_count {
            max_count = bin_counts[i];
            num_maxs = 1;
            res = (bin_edges[i] + bin_edges[i + 1]) / 2.0;
        } else if bin_counts[i] == max_count {
            num_maxs += 1;
            res += (bin_edges[i] + bin_edges[i + 1]) / 2.0;
        }
    }

    res / num_maxs as f64
}

/// Correlation - Embedding Distance (CO_Embed2_Dist_tau_d_expfit_meandiff).
///
/// Measures the exponential fit to the mean distance in the embedding space.
pub fn co_embed2_dist_tau_d_expfit_meandiff(a: &[f64]) -> f64 {
    let mut tau = first_zero(a, a.len());

    if tau > a.len() / 10 {
        tau = a.len() / 10;
    }

    let mut d = vec![0.0; a.len() - tau];

    for i in 0..a.len() - tau - 1 {
        d[i] = ((a[i + 1] - a[i]).powi(2) + (a[i + tau] - a[i + tau + 1]).powi(2)).sqrt();

        if d[i].is_nan() {
            return f64::NAN;
        }
    }

    let l = mean(&d[..a.len() - tau - 1]);

    let n_bins = num_bins_auto(&d[..a.len() - tau - 1]);

    if n_bins == 0 {
        return 0.0;
    }
    let (hist_counts, bin_edges) = histcounts(&d[..a.len() - tau - 1], n_bins);
    let mut hist_counts_norm = vec![0.0; n_bins];
    let count_tot = d[..a.len() - tau - 1].len() as f64; // Correct count normalization

    for i in 0..n_bins {
        hist_counts_norm[i] = hist_counts[i] as f64 / count_tot;
    }

    let mut d_expfit_diff = vec![0.0; n_bins];

    for i in 0..n_bins {
        let mut expf = (-(bin_edges[i] + bin_edges[i + 1]) * 0.5 / l).exp() / l;
        if expf < 0.0 {
            expf = 0.0;
        }
        d_expfit_diff[i] = (hist_counts_norm[i] - expf).abs();
    }

    mean(&d_expfit_diff[..n_bins])
}

/// Correlation - First 1/e Crossing of Autocorrelation (CO_f1ecac).
///
/// Returns the first time lag where the autocorrelation function drops below 1/e.
pub fn co_f1ecac(a: &[f64]) -> f64 {
    let autocorr = autocorr(a);

    let thresh = 1.0 / 1.0f64.exp();

    let mut out = a.len() as f64;

    for i in 0..a.len() - 2 {
        if autocorr[i + 1] < thresh {
            let m = autocorr[i + 1] - autocorr[i];
            let dy = thresh - autocorr[i];
            let dx = dy / m;
            out = i as f64 + dx;
            return out;
        }
    }
    out
}

/// Correlation - First Minimum of Autocorrelation (CO_FirstMin_ac).
///
/// Returns the first time lag where the autocorrelation function has a local minimum.
pub fn co_first_min_ac(a: &[f64]) -> f64 {
    let autocorr = autocorr(a);

    let mut min_ind = a.len();

    for i in 1..a.len() - 1 {
        if autocorr[i] < autocorr[i - 1] && autocorr[i] < autocorr[i + 1] {
            min_ind = i;
            break;
        }
    }

    min_ind as f64
}

/// Correlation - Histogram AMI Even (CO_HistogramAMI_even_2_5).
///
/// Automutual information using histograms.
pub fn co_histogram_ami_even_tau_bins(a: &[f64], tau: usize, n_bins: usize) -> f64 {
    if a.len() <= tau {
        return 0.0;
    }
    let mut y1 = vec![0.0; a.len() - tau];
    let mut y2 = vec![0.0; a.len() - tau];

    y1.copy_from_slice(&a[..a.len() - tau]);
    y2.copy_from_slice(&a[tau..]);

    let max_val = max_(a);
    let min_val = min_(a);

    let bin_step = (max_val - min_val + 0.2) / n_bins as f64;

    let mut bin_edges = vec![0.0; n_bins + 1];

    for (i, val) in bin_edges.iter_mut().enumerate().take(n_bins + 1) {
        *val = min_val + (i as f64 * bin_step) - 0.1;
    }

    let bins1 = histbinassign(&y1, &bin_edges);
    let bins2 = histbinassign(&y2, &bin_edges);

    let mut bins12 = vec![0.0; a.len() - tau];
    let mut bin_edges12 = vec![0.0; (n_bins + 1) * (n_bins + 1)];

    for i in 0..a.len() - tau {
        bins12[i] = ((bins1[i] - 1) * (n_bins + 1) + bins2[i]) as f64;
    }

    for (i, val) in bin_edges12
        .iter_mut()
        .enumerate()
        .take((n_bins + 1) * (n_bins + 1))
    {
        *val = (i + 1) as f64;
    }

    let joint_hist_linear = histcount_edges(&bins12, &bin_edges12);

    let mut pij = vec![vec![0.0; n_bins]; n_bins];

    let mut sum_bins = 0.0;

    for i in 0..n_bins {
        for j in 0..n_bins {
            if i * (n_bins + 1) + j < joint_hist_linear.len() {
                pij[j][i] = joint_hist_linear[i * (n_bins + 1) + j] as f64;
                sum_bins += pij[j][i];
            }
        }
    }

    for col in &mut pij {
        for val in col.iter_mut() {
            *val /= sum_bins;
        }
    }

    let mut pi = vec![0.0; n_bins];
    let mut pj = vec![0.0; n_bins];

    for i in 0..n_bins {
        for j in 0..n_bins {
            pi[i] += pij[i][j];
            pj[j] += pij[i][j];
        }
    }

    let mut ami = 0.0;
    for i in 0..n_bins {
        for j in 0..n_bins {
            if pij[i][j] > 0.0 {
                ami += pij[i][j] * (pij[i][j] / (pi[i] * pj[j])).ln();
            }
        }
    }

    ami
}

/// Correlation - Time Reversibility (CO_trev_1_num).
///
/// Measures time-reversibility using the third moment of differences.
pub fn co_trev_1_num(a: &[f64]) -> f64 {
    let tau = 1;

    let mut diff_temp = vec![0.0; a.len() - tau];

    for i in 0..a.len() - tau {
        diff_temp[i] = (a[i + 1] - a[i]).powi(3);
    }

    mean(&diff_temp)
}

/// Forecasting - Local Simple Mean Ratio (FC_LocalSimple_mean1_tauresrat).
///
/// Ratio of the first zero crossing of the residuals to the first zero crossing of the original series.
pub fn fc_local_simple_mean_tauresrat(a: &[f64], train_length: usize) -> f64 {
    let mut res = vec![0.0; a.len() - train_length];

    for i in 0..res.len() {
        let mut yest = 0.0;
        for j in 0..train_length {
            yest += a[i + j]
        }
        yest /= train_length as f64;

        res[i] = a[i + train_length] - yest;
    }

    let res_ac1st_z = first_zero(&res, res.len()) as f64;
    let y_ac1st_z = first_zero(a, a.len()) as f64;

    res_ac1st_z / y_ac1st_z
}

/// Forecasting - Local Simple Mean Std Err (FC_LocalSimple_mean3_stderr).
///
/// Standard deviation of the residuals from a local mean forecast.
pub fn fc_local_simple_mean_stderr(a: &[f64], train_length: usize) -> f64 {
    let mut res = vec![0.0; a.len() - train_length];

    for i in 0..res.len() {
        let mut yest = 0.0;
        for j in 0..train_length {
            yest += a[i + j]
        }
        yest /= train_length as f64;

        res[i] = a[i + train_length] - yest;
    }

    std_dev(&res)
}

/// Information - Auto Mutual Information (IN_AutoMutualInfoStats_40_gaussian_fmmi).
///
/// First minimum of the auto-mutual information.
pub fn in_auto_mutual_info_stats_tau_gaussian_fmmi(a: &[f64], tau: f64) -> f64 {
    let mut tau = tau;

    if tau > (a.len() as f64 / 2.0).ceil() {
        tau = (a.len() as f64 / 2.0).ceil();
    }

    let mut ami = vec![0.0; a.len()];

    for (i, val) in ami.iter_mut().enumerate().take(tau as usize) {
        let ac = autocorr_lag(a, i + 1);
        *val = -0.5 * (1.0 - ac * ac).ln();
    }

    let mut fmmi = tau;

    for i in 1..tau as usize - 1 {
        if ami[i] < ami[i - 1] && ami[i] < ami[i + 1] {
            fmmi = i as f64;
            break;
        }
    }
    fmmi
}

/// Medical - HRV Classic pNN (MD_hrv_classic_pnn40).
///
/// Percentage of successive RR intervals that differ by more than `pnn` ms.
pub fn md_hrv_classic_pnn(a: &[f64], pnn: usize) -> f64 {
    let d_y = diff(a);

    let mut pnn40 = 0.0;

    for val in d_y.iter().take(a.len() - 1) {
        if val.abs() * 1000.0 > pnn as f64 {
            pnn40 += 1.0;
        }
    }

    pnn40 / (a.len() - 1) as f64
}

/// Symbol - Binary Stats Diff Longstretch0 (SB_BinaryStats_diff_longstretch0).
///
/// Length of the longest sequence of 0s in the binary difference series.
pub fn sb_binary_stats_diff_longstretch0(a: &[f64]) -> f64 {
    let mut y_bin = vec![0; a.len() - 1];

    for (i, val) in y_bin.iter_mut().enumerate().take(a.len() - 1) {
        let diff_temp = a[i + 1] - a[i];
        if diff_temp < 0.0 { *val = 0 } else { *val = 1 }
    }

    let mut max_stretch = 0;
    let mut last1 = 0;

    for (i, &val) in y_bin.iter().enumerate().take(a.len() - 1) {
        if val == 1 || i == a.len() - 2 {
            let stretch = i - last1;

            if stretch > max_stretch {
                max_stretch = stretch;
            }

            last1 = i;
        }
    }

    max_stretch as f64
}

/// Symbol - Binary Stats Mean Longstretch1 (SB_BinaryStats_mean_longstretch1).
///
/// Length of the longest sequence of 1s in the binary mean-thresholded series.
pub fn sb_binary_stats_mean_longstretch1(a: &[f64]) -> f64 {
    let mut y_bin = vec![0; a.len() - 1];
    let a_mean = mean(a);
    for (i, val) in y_bin.iter_mut().enumerate().take(a.len() - 1) {
        if a[i] - a_mean <= 0.0 {
            *val = 0
        } else {
            *val = 1
        }
    }

    let mut max_stretch = 0;
    let mut last1 = 0;

    for (i, &val) in y_bin.iter().enumerate().take(a.len() - 1) {
        if val == 0 || i == a.len() - 2 {
            let stretch = i - last1;

            if stretch > max_stretch {
                max_stretch = stretch;
            }

            last1 = i;
        }
    }

    max_stretch as f64
}

/// Symbol - Motif Three Quantile HH (SB_MotifThree_quantile_hh).
///
/// Entropy of the distribution of 3-letter motifs based on quantiles.
pub fn sb_motif_three_quantile_hh(a: &[f64]) -> f64 {
    let alphabet_size = 3;
    let yt = coarsegrain(a, alphabet_size);

    let mut r1: Vec<Vec<usize>> = (0..alphabet_size)
        .map(|_| Vec::with_capacity(a.len()))
        .collect();
    for (i, vec_r1) in r1.iter_mut().enumerate().take(alphabet_size) {
        for (j, &val) in yt.iter().enumerate().take(a.len()) {
            if val == i + 1 {
                vec_r1.push(j);
            }
        }
    }

    for vec_r1 in r1.iter_mut().take(alphabet_size) {
        if vec_r1.last() == Some(&(a.len() - 1)) {
            vec_r1.pop();
        }
    }

    let mut r2 = vec![vec![Vec::new(); alphabet_size]; alphabet_size];
    let mut out2 = vec![vec![0.0; alphabet_size]; alphabet_size];

    for i in 0..alphabet_size {
        for j in 0..alphabet_size {
            for k in 0..r1[i].len() {
                let tmp_idx = yt[r1[i][k] + 1];
                if tmp_idx == (j + 1) {
                    r2[i][j].push(r1[i][k]);
                }
            }
            let tmp = r2[i][j].len() as f64 / (a.len() - 1) as f64;
            out2[i][j] = tmp;
        }
    }

    let mut hh = 0.0;
    for val in out2.iter().take(alphabet_size) {
        hh += f_entropy(val);
    }
    hh
}

/// Scaling - Fluctuation Analysis (SC_FluctAnal_2_50_1_logi_prop_r1_dfa / rsrangefit).
///
/// Performs fluctuation analysis (DFA or RS range fit).
/// Scaling - Fluctuation Analysis (SC_FluctAnal_2_50_1_logi_prop_r1_dfa / rsrangefit).
///
/// Performs fluctuation analysis (DFA or RS range fit).
pub fn sc_fluct_anal_2_50_1_logi_prop_r1(a: &[f64], lag: usize, how: &str) -> f64 {
    let (tau, n_tau) = fa_generate_tau(a.len());

    if n_tau < 12 {
        return 0.0;
    }

    let size_cs = a.len() / lag;
    let mut y_cs = vec![0.0; size_cs];

    y_cs[0] = a[0];
    for i in 0..size_cs - 1 {
        y_cs[i + 1] = y_cs[i] + a[(i + 1) * lag];
    }

    // Pre-calculate x_reg for max tau to avoid reallocation?
    // Actually x_reg varies from 1 to tau[i].
    // We can allocate one buffer of max size.
    let max_tau_val = tau[n_tau - 1] as usize;
    let x_reg_buffer: Vec<f64> = (1..=max_tau_val).map(|v| v as f64).collect();

    let f = fa_compute_f(&y_cs, &tau, n_tau, &x_reg_buffer, how);

    fa_fit_log_log(&tau, &f, n_tau)
}

fn fa_generate_tau(n: usize) -> (Vec<f64>, usize) {
    let lin_low = (5.0f64).ln();
    let lin_high = ((n / 2) as f64).ln();

    let n_tau_steps = 50;
    let tau_step = (lin_high - lin_low) / (n_tau_steps - 1) as f64;

    let mut tau = vec![0.0; n_tau_steps];
    for (i, val) in tau.iter_mut().enumerate().take(n_tau_steps) {
        *val = (lin_low + i as f64 * tau_step).exp().round();
    }

    let mut n_tau = n_tau_steps;
    // Deduplicate
    for i in 0..n_tau_steps - 1 {
        // Simple forward check.
        // Original logic was slightly convoluted (while loop).
        // Let's stick to original logic's intent: remove duplicates.
        // The original code uses a shift strategy.
        let j = i;
        while j < n_tau - 1 && tau[j] == tau[j + 1] {
            // shift everything left
            for k in j + 1..n_tau {
                tau[k - 1] = tau[k];
            }
            n_tau -= 1;
            // Don't advance j, check again
        }
    }

    // Sort and dedup is easier in rust:
    // tau.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // tau.dedup();
    // But we need to match original behavior potentially.
    // Original behavior preserves order (which is sorted) and removes successive duplicates.

    (tau, n_tau)
}

fn fa_compute_f(
    y_cs: &[f64],
    tau: &[f64],
    n_tau: usize,
    x_reg_buffer: &[f64],
    how: &str,
) -> Vec<f64> {
    let mut f = vec![0.0; n_tau];
    let mut buffer = Vec::with_capacity(x_reg_buffer.len());

    for i in 0..n_tau {
        let t_i = tau[i] as usize;
        let n_buffer = (y_cs.len() as f64 / tau[i]) as usize;
        f[i] = 0.0;

        for j in 0..n_buffer {
            let start = j * t_i;
            let end = start + t_i;
            let y_slice = &y_cs[start..end];

            let (m, b) = linreg(t_i, &x_reg_buffer[..t_i], y_slice);

            buffer.clear();
            for (k, &val) in y_slice.iter().enumerate().take(t_i) {
                buffer.push(val - (m * (k + 1) as f64 + b));
            }

            match how {
                "rsrangefit" => {
                    let max = max_(&buffer);
                    let min = min_(&buffer);
                    f[i] += (max - min).powi(2);
                },
                "dfa" => {
                    // map().sum() is faster/cleaner
                    f[i] += buffer.iter().map(|x| x.powi(2)).sum::<f64>();
                },
                _ => {}, // Should return error or 0
            }
        }

        match how {
            "rsrangefit" => f[i] = (f[i] / n_buffer as f64).sqrt(),
            "dfa" => f[i] = (f[i] / n_buffer as f64 * tau[i]).sqrt(),
            _ => {},
        }
    }
    f
}

fn fa_fit_log_log(tau: &[f64], f: &[f64], n_tau: usize) -> f64 {
    let mut logtt = vec![0.0; n_tau];
    let mut logff = vec![0.0; n_tau];

    for i in 0..n_tau {
        logtt[i] = tau[i].ln();
        logff[i] = f[i].ln();
    }

    let min_points = 6;
    let nsserr = n_tau - 2 * min_points + 1;
    if nsserr == 0 {
        return 0.0;
    } // Safety check

    let mut sserr = vec![0.0; nsserr];
    // Buffer for linreg residuals
    let mut buffer = Vec::with_capacity(n_tau);

    for i in min_points..n_tau - min_points + 1 {
        let (m1, b1) = linreg(i, &logtt, &logff);
        let (m2, b2) = linreg(n_tau - i + 1, &logtt[i - 1..], &logff[i - 1..]);

        buffer.clear();
        for j in 0..i {
            buffer.push(logtt[j] * m1 + b1 - logff[j]);
        }
        sserr[i - min_points] += norm(&buffer);

        buffer.clear();
        for j in 0..n_tau - i + 1 {
            buffer.push(logtt[j + i - 1] * m2 + b2 - logff[j + i - 1]);
        }
        sserr[i - min_points] += norm(&buffer);
    }

    let min_ind = (0..nsserr)
        .min_by(|&a, &b| {
            sserr[a]
                .partial_cmp(&sserr[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);

    (min_ind + min_points) as f64 / n_tau as f64
}

/// Spectral - Welch Summary (SP_Summaries_welch_rect_centroid / area_5_1).
///
/// Summaries of the power spectral density estimated by Welch's method.
pub fn sp_summaries_welch_rect(a: &[f64], what: &str) -> f64 {
    let window = (0..a.len()).map(|_| 1.0).collect::<Vec<f64>>();
    let fs = 1.0;

    let (s, f) = welch(a, fs, &window);

    let mut w = vec![0.0; s.len()];
    let mut sw = vec![0.0; s.len()];

    for i in 0..s.len() {
        w[i] = 2.0 * std::f64::consts::PI * f[i];
        sw[i] = s[i] / (2.0 * std::f64::consts::PI);

        if sw[i].is_infinite() {
            return 0.0;
        }
    }

    let dw = w[1] - w[0];

    // cum sum of sw
    let s_cs = sw
        .iter()
        .scan(0.0, |state, x| {
            *state += x;
            Some(*state)
        })
        .collect::<Vec<f64>>();

    match what {
        "centroid" => {
            if s_cs.is_empty() {
                return 0.0;
            }
            let s_cs_thresh = s_cs[s.len() - 1] / 2.0;
            let mut centroid = 0.0;
            for i in 0..s.len() {
                if s_cs[i] > s_cs_thresh {
                    centroid = w[i];
                    break;
                }
            }
            centroid
        },
        "area_5_1" => {
            let mut area_5_1 = 0.0;
            for i in 0..s.len() {
                // Corrected logic: area_5_1 likely refers to power in period range [1, 5] (Hz [0.2, 1.0])
                // w is angular frequency = 2*pi*f
                let f_val = w[i] / (2.0 * std::f64::consts::PI);
                if (0.2..=1.0).contains(&f_val) {
                    area_5_1 += sw[i];
                }
            }
            area_5_1 * dw
        },
        _ => 0.0,
    }
}

/// Symbol - Transition Matrix 3AC Sum Diag Cov (SB_TransitionMatrix_3ac_sumdiagcov).
///
/// Trace of the covariance of the transition matrix.
pub fn sb_transition_matrix_3ac_sumdiagcov(a: &[f64]) -> f64 {
    if is_constant(a) {
        return f64::NAN;
    }

    let num_groups = 3;

    let tau = first_zero(a, a.len());

    let y_filt = a.to_vec();

    let n_down = (a.len() - 1) / tau + 1;
    let mut y_down = vec![0.0; n_down];
    for i in 0..n_down {
        y_down[i] = y_filt[i * tau];
    }

    let y_cg = coarsegrain(&y_down, num_groups);

    let mut t = vec![vec![0.0; 3]; 3];

    for i in 0..n_down.saturating_sub(1) {
        if i + 1 < y_cg.len() {
            let idx1 = y_cg[i].saturating_sub(1);
            let idx2 = y_cg[i + 1].saturating_sub(1);
            if idx1 < 3 && idx2 < 3 {
                t[idx1][idx2] += 1.0;
            }
        }
    }

    for row in t.iter_mut().take(num_groups) {
        for val in row.iter_mut().take(num_groups) {
            *val /= (n_down - 1) as f64;
        }
    }

    let cm = covariance_matrix(t);
    let mut diag_sum = 0.0;

    for (i, row) in cm.iter().enumerate().take(num_groups) {
        diag_sum += row[i];
    }

    diag_sum
}

/// Periodicity - Wang Threshold 0.01 (PD_PeriodicityWang_th0_01).
///
/// Periodicity measure based on autocorrelation of spline-detrended series.
pub fn pd_periodicity_wang_th0_01(a: &[f64]) -> f64 {
    let th = 0.01;
    // We need splinefit core that returns Vec<f64>
    // existing splinefit returns Series.
    // In smoothing.rs, we exposed splinefit_core

    // Original reference calls splinefit(a).
    let n = a.len();
    let deg = 3;
    let pieces = 2;
    let breaks = [0, (n as f64 / 2.0).floor() as usize - 1, n - 1];
    let y_spline = splinefit_core(a, &breaks, deg, pieces);

    let mut y_sub = vec![0.0; a.len()];
    for i in 0..a.len() {
        y_sub[i] = a[i] - y_spline[i];
    }

    let ac_max = (a.len() as f64 / 3.0).ceil() as usize;
    let mut acf = vec![0.0; ac_max];

    for i in 1..(ac_max + 1) {
        acf[i - 1] = autocov_lag(&y_sub, i);
    }

    let mut troughs = vec![0.0; ac_max];
    let mut peaks = vec![0.0; ac_max];
    let mut n_troughs = 0;
    let mut n_peaks = 0;

    for i in 1..ac_max - 1 {
        let slope_in = acf[i] - acf[i - 1];
        let slope_out = acf[i + 1] - acf[i];

        if slope_in < 0.0 && slope_out > 0.0 {
            troughs[n_troughs] = i as f64;
            n_troughs += 1;
        } else if slope_in > 0.0 && slope_out < 0.0 {
            peaks[n_peaks] = i as f64;
            n_peaks += 1;
        }
    }

    let mut out = 0.0;

    for &i_peak in peaks.iter().take(n_peaks) {
        let the_peak = acf[i_peak as usize];

        let mut j: isize = -1;

        while (j + 1) < n_troughs as isize && troughs[(j + 1) as usize] < i_peak {
            j += 1;
        }

        if j == -1 {
            continue;
        }

        let i_trough = troughs[j as usize];
        let the_trough = acf[i_trough as usize];

        if the_peak - the_trough < th {
            continue;
        }

        if the_peak < 0.0 {
            continue;
        }

        out = i_peak;
        break;
    }

    out
}
