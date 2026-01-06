#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use crate::catchers;
use crate::feature_extraction::{statistics, energy_and_complexity, change, fourier, wavelets, entropy, distribution, basic_counts, locations, correlation, dynamics, linear_trend, counts_ranges, stats_tests};

// --------------------------------------------------------------------------------
// Helper to extract data
// --------------------------------------------------------------------------------
fn series_to_f64_vec(series: &Series) -> PolarsResult<Vec<f64>> {
    let ca = series.cast(&DataType::Float64)?;
    let ca = ca.f64()?;
    // Treating nulls as 0.0 to match previous behavior and ensure continuity for stats
    Ok(ca.into_iter().map(|v| v.unwrap_or(0.0)).collect())
}

// Removed unused list_f64_output helper
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

#[derive(Deserialize)]
struct RKwargs {
    r: f64,
}

#[derive(Deserialize)]
struct CidCeKwargs {
    normalize: bool,
}

#[derive(Deserialize)]
struct FftCoeffKwargs {
    k: usize,
    attr: String,
}

#[derive(Deserialize)]
struct FftAggKwargs {
    agg_type: String,
}

#[derive(Deserialize)]
struct CwtCoeffKwargs {
    width: f64,
    index: usize,
}

#[derive(Deserialize)]
struct MaxBinsKwargs {
    max_bins: usize,
}

#[derive(Deserialize)]
struct FriedrichKwargs {
    m: usize,
    r: usize,
    coeff_index: usize,
}

#[derive(Deserialize)]
struct NumberPeaksKwargs {
    n: usize,
}

#[derive(Deserialize)]
struct MassQuantileKwargs {
    q: f64,
}

#[derive(Deserialize)]
struct AggAutocorrKwargs {
    match_agg: String, // "mean", "std", etc.
    maxlag: usize,
}

#[derive(Deserialize)]
struct PartialAutocorrKwargs {
    lag: usize,
}

#[derive(Deserialize)]
struct LagKwargs {
    lag: usize,
}

#[derive(Deserialize)]
struct LinearTrendKwargs {
    attr: String,
}

#[derive(Deserialize)]
struct ThresholdKwargs {
    m: f64,
}

#[derive(Deserialize)]
struct RangeKwargs {
    min: f64,
    max: f64,
}

#[derive(Deserialize)]
struct ValueKwargs {
    value: f64,
}


#[derive(Deserialize)]
struct QuantileKwargs {
    q: f64,
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

// --------------------------------------------------------------------------------
// tsfresh Expressions
// --------------------------------------------------------------------------------

#[polars_expr(output_type=Boolean)]
fn fresh_variance_larger_than_standard_deviation(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = statistics::variance_larger_than_standard_deviation(&data);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Float64)]
fn fresh_ratio_beyond_r_sigma(inputs: &[Series], kwargs: RKwargs) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = statistics::ratio_beyond_r_sigma(&data, kwargs.r);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Boolean)]
fn fresh_large_standard_deviation(inputs: &[Series], kwargs: RKwargs) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = statistics::large_standard_deviation(&data, kwargs.r);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Boolean)]
fn fresh_symmetry_looking(inputs: &[Series], kwargs: RKwargs) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = statistics::symmetry_looking(&data, kwargs.r);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Boolean)]
fn fresh_has_duplicate_max(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = statistics::has_duplicate_max(&data);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Boolean)]
fn fresh_has_duplicate_min(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = statistics::has_duplicate_min(&data);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Boolean)]
fn fresh_has_duplicate(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = statistics::has_duplicate(&data);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Float64)]
fn fresh_root_mean_square(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = statistics::root_mean_square(&data);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Float64)]
fn fresh_skewness(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = statistics::skewness(&data);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Float64)]
fn fresh_kurtosis(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = statistics::kurtosis(&data);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Float64)]
fn fresh_variation_coefficient(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = statistics::variation_coefficient(&data);
    Ok(Series::new("out".into(), [out]))
}


#[polars_expr(output_type=Float64)]
fn fresh_abs_energy(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = energy_and_complexity::abs_energy(&data);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Float64)]
fn fresh_cid_ce(inputs: &[Series], kwargs: CidCeKwargs) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = energy_and_complexity::cid_ce(&data, kwargs.normalize);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Float64)]
fn fresh_mean_abs_change(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = change::mean_abs_change(&data);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Float64)]
fn fresh_mean_change(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = change::mean_change(&data);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Float64)]
fn fresh_mean_second_derivative_central(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = change::mean_second_derivative_central(&data);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Float64)]
fn fresh_absolute_sum_of_changes(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = change::absolute_sum_of_changes(&data);
    Ok(Series::new("out".into(), [out]))
}

// --------------------------------------------------------------------------------
// Fourier Expressions
// --------------------------------------------------------------------------------

#[polars_expr(output_type=Float64)]
fn fresh_fft_coefficient(inputs: &[Series], kwargs: FftCoeffKwargs) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = fourier::fft_coefficient(&data, kwargs.k, &kwargs.attr);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Float64)]
fn fresh_fft_aggregated(inputs: &[Series], kwargs: FftAggKwargs) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = fourier::fft_aggregated(&data, &kwargs.agg_type);
    Ok(Series::new("out".into(), [out]))
}

// --------------------------------------------------------------------------------
// Wavelet Expressions
// --------------------------------------------------------------------------------

#[polars_expr(output_type=Float64)]
fn fresh_cwt_coefficient(inputs: &[Series], kwargs: CwtCoeffKwargs) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = wavelets::cwt_coefficient(&data, kwargs.width, kwargs.index);
    Ok(Series::new("out".into(), [out]))
}

// --------------------------------------------------------------------------------
// Entropy Expressions
// --------------------------------------------------------------------------------

#[polars_expr(output_type=Float64)]
fn fresh_sample_entropy(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = entropy::sample_entropy(&data);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Float64)]
fn fresh_binned_entropy(inputs: &[Series], kwargs: MaxBinsKwargs) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = entropy::binned_entropy(&data, kwargs.max_bins);
    Ok(Series::new("out".into(), [out]))
}

// --------------------------------------------------------------------------------
// Distribution Expressions
// --------------------------------------------------------------------------------

#[polars_expr(output_type=Float64)]
fn fresh_friedrich_coefficient(inputs: &[Series], kwargs: FriedrichKwargs) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = distribution::friedrich_coefficient(&data, kwargs.m, kwargs.r, kwargs.coeff_index);
    Ok(Series::new("out".into(), [out]))
}

// --------------------------------------------------------------------------------
// Basic Counts & Streaks Expressions
// --------------------------------------------------------------------------------

#[polars_expr(output_type=Float64)]
fn fresh_count_above_mean(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = basic_counts::count_above_mean(&data);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Float64)]
fn fresh_count_below_mean(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = basic_counts::count_below_mean(&data);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Float64)]
fn fresh_longest_strike_above_mean(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = basic_counts::longest_strike_above_mean(&data);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Float64)]
fn fresh_longest_strike_below_mean(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = basic_counts::longest_strike_below_mean(&data);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Float64)]
fn fresh_percentage_of_reoccurring_values_to_all_values(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = basic_counts::percentage_of_reoccurring_values_to_all_values(&data);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Float64)]
fn fresh_percentage_of_reoccurring_datapoints_to_all_datapoints(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = basic_counts::percentage_of_reoccurring_datapoints_to_all_datapoints(&data);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Float64)]
fn fresh_sum_of_reoccurring_values(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = basic_counts::sum_of_reoccurring_values(&data);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Float64)]
fn fresh_sum_of_reoccurring_data_points(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = basic_counts::sum_of_reoccurring_data_points(&data);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Float64)]
fn fresh_ratio_value_number_to_time_series_length(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = basic_counts::ratio_value_number_to_time_series_length(&data);
    Ok(Series::new("out".into(), [out]))
}

// --------------------------------------------------------------------------------
// Locations & Peaks Expressions
// --------------------------------------------------------------------------------

#[polars_expr(output_type=Float64)]
fn fresh_first_location_of_maximum(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = locations::first_location_of_maximum(&data);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Float64)]
fn fresh_last_location_of_maximum(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = locations::last_location_of_maximum(&data);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Float64)]
fn fresh_first_location_of_minimum(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = locations::first_location_of_minimum(&data);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Float64)]
fn fresh_last_location_of_minimum(inputs: &[Series]) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = locations::last_location_of_minimum(&data);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Float64)]
fn fresh_number_peaks(inputs: &[Series], kwargs: NumberPeaksKwargs) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = locations::number_peaks(&data, kwargs.n);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Float64)]
fn fresh_index_mass_quantile(inputs: &[Series], kwargs: MassQuantileKwargs) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = locations::index_mass_quantile(&data, kwargs.q);
    Ok(Series::new("out".into(), [out]))
}

// --------------------------------------------------------------------------------
// Correlation Expressions
// --------------------------------------------------------------------------------

#[polars_expr(output_type=Float64)]
fn fresh_agg_autocorrelation(inputs: &[Series], kwargs: AggAutocorrKwargs) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = correlation::agg_autocorrelation(&data, kwargs.maxlag, &kwargs.match_agg);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Float64)]
fn fresh_partial_autocorrelation(inputs: &[Series], kwargs: PartialAutocorrKwargs) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = correlation::partial_autocorrelation(&data, kwargs.lag);
    Ok(Series::new("out".into(), [out]))
}

// --------------------------------------------------------------------------------
// Dynamics Expressions
// --------------------------------------------------------------------------------

#[polars_expr(output_type=Float64)]
fn fresh_time_reversal_asymmetry_statistic(inputs: &[Series], kwargs: LagKwargs) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = dynamics::time_reversal_asymmetry_statistic(&data, kwargs.lag);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Float64)]
fn fresh_c3(inputs: &[Series], kwargs: LagKwargs) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = dynamics::c3(&data, kwargs.lag);
    Ok(Series::new("out".into(), [out]))
}

// --------------------------------------------------------------------------------
// Linear Trend Expressions
// --------------------------------------------------------------------------------

#[polars_expr(output_type=Float64)]
fn fresh_linear_trend(inputs: &[Series], kwargs: LinearTrendKwargs) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = linear_trend::linear_trend_attr(&data, &kwargs.attr);
    Ok(Series::new("out".into(), [out]))
}

// --------------------------------------------------------------------------------
// Counts & Ranges Expressions
// --------------------------------------------------------------------------------

#[polars_expr(output_type=Float64)]
fn fresh_number_crossing_m(inputs: &[Series], kwargs: ThresholdKwargs) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = counts_ranges::number_crossing_m(&data, kwargs.m);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Float64)]
fn fresh_range_count(inputs: &[Series], kwargs: RangeKwargs) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = counts_ranges::range_count(&data, kwargs.min, kwargs.max);
    Ok(Series::new("out".into(), [out]))
}

#[polars_expr(output_type=Float64)]
fn fresh_value_count(inputs: &[Series], kwargs: ValueKwargs) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = counts_ranges::value_count(&data, kwargs.value);
    Ok(Series::new("out".into(), [out]))
}

// --------------------------------------------------------------------------------
// Distribution Tests Expressions
// --------------------------------------------------------------------------------

#[polars_expr(output_type=Float64)]
fn fresh_quantile(inputs: &[Series], kwargs: QuantileKwargs) -> PolarsResult<Series> {
    let data = series_to_f64_vec(&inputs[0])?;
    let out = stats_tests::quantile(&data, kwargs.q);
    Ok(Series::new("out".into(), [out]))
}









