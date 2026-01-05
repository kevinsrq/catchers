#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use crate::catchers;

// --------------------------------------------------------------------------------
// Helper to extract data
// --------------------------------------------------------------------------------
fn series_to_f64_vec(series: &Series) -> PolarsResult<Vec<f64>> {
    let ca = series.cast(&DataType::Float64)?;
    let ca = ca.f64()?;
    // Treating nulls as 0.0 to match previous behavior and ensure continuity for stats
    Ok(ca.into_iter().map(|v| v.unwrap_or(0.0)).collect())
}

// --------------------------------------------------------------------------------
// Kwargs Definitions
// --------------------------------------------------------------------------------

#[derive(Deserialize)]
struct IsPosKwargs {
    is_pos: bool,
}

#[derive(Deserialize)]
struct NBinsKwargs {
    n_bins: usize,
}

#[derive(Deserialize)]
struct TauNBinsKwargs {
    tau: usize,
    n_bins: usize,
}

#[derive(Deserialize)]
struct TrainLengthKwargs {
    train_length: usize,
}

#[derive(Deserialize)]
struct TauF64Kwargs {
    tau: f64,
}

#[derive(Deserialize)]
struct PnnKwargs {
    pnn: usize,
}

#[derive(Deserialize)]
struct FluctAnalKwargs {
    lag: usize,
    how: String,
}

#[derive(Deserialize)]
struct WelchKwargs {
    what: String,
}

// --------------------------------------------------------------------------------
// Expressions
// --------------------------------------------------------------------------------

/// Distribution of Outliers (DN_OutlierInclude_n_001_mdrmd)
/// 
/// measures the spread of the data including outliers.
/// 
/// # Arguments
/// * `is_pos` - Whether to consider positive (true) or negative (false) deviations.
#[polars_expr(output_type=Float64)]
fn dn_outlier_include_np_001_mdrmd(inputs: &[Series], kwargs: IsPosKwargs) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = catchers::dn_outlier_include_np_001_mdrmd(&data, kwargs.is_pos);
    Ok(Series::new("out".into(), [out]))
}

/// Histogram Mode (DN_HistogramMode_5 / 10)
/// 
/// Measures the mode of the data using a histogram with `n_bins`.
/// 
/// # Arguments
/// * `n_bins` - Number of bins for the histogram.
#[polars_expr(output_type=Float64)]
fn dn_histogram_mode(inputs: &[Series], kwargs: NBinsKwargs) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = catchers::dn_histogram_mode_n(&data, kwargs.n_bins);
    Ok(Series::new("out".into(), [out]))
}

/// Correlation - Embedding Distance (CO_Embed2_Dist_tau_d_expfit_meandiff)
/// 
/// Measures the exponential fit to the mean distance in the embedding space.
#[polars_expr(output_type=Float64)]
fn co_embed2_dist_tau_d_expfit_meandiff(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = catchers::co_embed2_dist_tau_d_expfit_meandiff(&data);
    Ok(Series::new("out".into(), [out]))
}

/// Correlation - First 1/e Crossing of Autocorrelation (CO_f1ecac)
#[polars_expr(output_type=Float64)]
fn co_f1ecac(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = catchers::co_f1ecac(&data);
    Ok(Series::new("out".into(), [out]))
}

/// Correlation - First Minimum of Autocorrelation (CO_FirstMin_ac)
#[polars_expr(output_type=Float64)]
fn co_first_min_ac(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = catchers::co_first_min_ac(&data);
    Ok(Series::new("out".into(), [out]))
}

/// Correlation - Histogram AMI Even (CO_HistogramAMI_even_2_5)
/// 
/// Automutual information using histograms.
/// 
/// # Arguments
/// * `tau` - The time lag.
/// * `n_bins` - Number of bins.
#[polars_expr(output_type=Float64)]
fn co_histogram_ami_even_tau_bins(inputs: &[Series], kwargs: TauNBinsKwargs) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = catchers::co_histogram_ami_even_tau_bins(&data, kwargs.tau, kwargs.n_bins);
    Ok(Series::new("out".into(), [out]))
}

/// Correlation - Time Reversibility (CO_trev_1_num)
#[polars_expr(output_type=Float64)]
fn co_trev_1_num(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = catchers::co_trev_1_num(&data);
    Ok(Series::new("out".into(), [out]))
}

/// Forecasting - Local Simple Mean Ratio (FC_LocalSimple_mean1_tauresrat)
/// 
/// # Arguments
/// * `train_length` - Length of the training period for the local mean.
#[polars_expr(output_type=Float64)]
fn fc_local_simple_mean_tauresrat(inputs: &[Series], kwargs: TrainLengthKwargs) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = catchers::fc_local_simple_mean_tauresrat(&data, kwargs.train_length);
    Ok(Series::new("out".into(), [out]))
}

/// Forecasting - Local Simple Mean Std Err (FC_LocalSimple_mean3_stderr)
/// 
/// # Arguments
/// * `train_length` - Length of the training period.
#[polars_expr(output_type=Float64)]
fn fc_local_simple_mean_stderr(inputs: &[Series], kwargs: TrainLengthKwargs) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = catchers::fc_local_simple_mean_stderr(&data, kwargs.train_length);
    Ok(Series::new("out".into(), [out]))
}

/// Information - Auto Mutual Information (IN_AutoMutualInfoStats_40_gaussian_fmmi)
/// 
/// # Arguments
/// * `tau` - The time lag (as float/time unit, but used as value here).
#[polars_expr(output_type=Float64)]
fn in_auto_mutual_info_stats_tau_gaussian_fmmi(inputs: &[Series], kwargs: TauF64Kwargs) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = catchers::in_auto_mutual_info_stats_tau_gaussian_fmmi(&data, kwargs.tau);
    Ok(Series::new("out".into(), [out]))
}

/// Medical - HRV Classic pNN (MD_hrv_classic_pnn40)
/// 
/// Percentage of successive RR intervals that differ by more than `pnn` ms.
/// 
/// # Arguments
/// * `pnn` - The threshold (e.g. 40 in pnn40).
#[polars_expr(output_type=Float64)]
fn md_hrv_classic_pnn(inputs: &[Series], kwargs: PnnKwargs) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = catchers::md_hrv_classic_pnn(&data, kwargs.pnn);
    Ok(Series::new("out".into(), [out]))
}

/// Symbol - Binary Stats Diff Longstretch0 (SB_BinaryStats_diff_longstretch0)
#[polars_expr(output_type=Float64)]
fn sb_binary_stats_diff_longstretch0(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = catchers::sb_binary_stats_diff_longstretch0(&data);
    Ok(Series::new("out".into(), [out]))
}

/// Symbol - Binary Stats Mean Longstretch1 (SB_BinaryStats_mean_longstretch1)
#[polars_expr(output_type=Float64)]
fn sb_binary_stats_mean_longstretch1(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = catchers::sb_binary_stats_mean_longstretch1(&data);
    Ok(Series::new("out".into(), [out]))
}

/// Symbol - Motif Three Quantile HH (SB_MotifThree_quantile_hh)
#[polars_expr(output_type=Float64)]
fn sb_motif_three_quantile_hh(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = catchers::sb_motif_three_quantile_hh(&data);
    Ok(Series::new("out".into(), [out]))
}

/// Scaling - Fluctuation Analysis (SC_FluctAnal_2_50_1_logi_prop_r1_dfa / rsrangefit)
/// 
/// # Arguments
/// * `lag` - The lag parameter.
/// * `how` - The method ("dfa" or "rsrangefit").
#[polars_expr(output_type=Float64)]
fn sc_fluct_anal_2_50_1_logi_prop_r1(inputs: &[Series], kwargs: FluctAnalKwargs) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = catchers::sc_fluct_anal_2_50_1_logi_prop_r1(&data, kwargs.lag, &kwargs.how);
    Ok(Series::new("out".into(), [out]))
}

/// Spectral - Welch Summary (SP_Summaries_welch_rect_centroid / area_5_1)
/// 
/// # Arguments
/// * `what` - The summary statistic to calculate ("centroid" or "area_5_1").
#[polars_expr(output_type=Float64)]
fn sp_summaries_welch_rect(inputs: &[Series], kwargs: WelchKwargs) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = catchers::sp_summaries_welch_rect(&data, &kwargs.what);
    Ok(Series::new("out".into(), [out]))
}

/// Symbol - Transition Matrix 3AC Sum Diag Cov (SB_TransitionMatrix_3ac_sumdiagcov)
#[polars_expr(output_type=Float64)]
fn sb_transition_matrix_3ac_sumdiagcov(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = catchers::sb_transition_matrix_3ac_sumdiagcov(&data);
    Ok(Series::new("out".into(), [out]))
}

/// Periodicity - Wang Threshold 0.01 (PD_PeriodicityWang_th0_01)
#[polars_expr(output_type=Float64)]
fn pd_periodicity_wang_th0_01(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = catchers::pd_periodicity_wang_th0_01(&data);
    Ok(Series::new("out".into(), [out]))
}
