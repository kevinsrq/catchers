from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from catchers._internal import __version__ as __version__

if TYPE_CHECKING:
    from catchers.typing import IntoExprColumn

LIB = Path(__file__).parent


@pl.api.register_expr_namespace("catchers")
class CatchersNamespace:
    """Polars expression namespace for catchers (catch22) features."""

    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def _register(self, name: str, kwargs: dict | None = None) -> pl.Expr:
        return register_plugin_function(
            args=[self._expr],
            plugin_path=LIB,
            function_name=name,
            kwargs=kwargs,
            is_elementwise=False,
            returns_scalar=True,
        )

    # --- Short Names (Official Catch22 aliases) ---

    def mode_5(self) -> pl.Expr:
        """5-bin histogram mode (DN_HistogramMode_5)."""
        return self.dn_histogram_mode_n(n_bins=5)

    def mode_10(self) -> pl.Expr:
        """10-bin histogram mode (DN_HistogramMode_10)."""
        return self.dn_histogram_mode_n(n_bins=10)

    def outlier_timing_pos(self) -> pl.Expr:
        """Positive outlier timing (DN_OutlierInclude_p_001_mdrmd)."""
        return self.dn_outlier_include_np_001_mdrmd(is_pos=True)

    def outlier_timing_neg(self) -> pl.Expr:
        """Negative outlier timing (DN_OutlierInclude_n_001_mdrmd)."""
        return self.dn_outlier_include_np_001_mdrmd(is_pos=False)

    def acf_timescale(self) -> pl.Expr:
        """First 1/e crossing of the ACF (first1e_acf_tau)."""
        return self.co_f1ecac()

    def acf_first_min(self) -> pl.Expr:
        """First minimum of the ACF (firstMin_acf)."""
        return self.co_first_min_ac()

    def low_freq_power(self) -> pl.Expr:
        """Power in lowest 20% frequencies (SP_Summaries_welch_rect_area_5_1)."""
        return self.sp_summaries_welch_rect(what="area_5_1")

    def centroid_freq(self) -> pl.Expr:
        """Centroid frequency (SP_Summaries_welch_rect_centroid)."""
        return self.sp_summaries_welch_rect(what="centroid")

    def forecast_error(self, train_length: int = 3) -> pl.Expr:
        """Error of rolling mean forecast (FC_LocalSimple_mean3_stderr)."""
        return self.fc_local_simple_mean_stderr(train_length=train_length)

    def whiten_timescale(self, train_length: int = 1) -> pl.Expr:
        """Change in autocorrelation timescale after differencing (FC_LocalSimple_mean1_tauresrat)."""
        return self.fc_local_simple_mean_tauresrat(train_length=train_length)

    def high_fluctuation(self, pnn: int = 40) -> pl.Expr:
        """Proportion of high incremental changes (MD_hrv_classic_pnn40)."""
        return self.md_hrv_classic_pnn(pnn=pnn)

    def stretch_high(self) -> pl.Expr:
        """Longest stretch of above-mean values (SB_BinaryStats_mean_longstretch1)."""
        return self.sb_binary_stats_mean_longstretch1()

    def stretch_decreasing(self) -> pl.Expr:
        """Longest stretch of decreasing values (SB_BinaryStats_diff_longstretch0)."""
        return self.sb_binary_stats_diff_longstretch0()

    def entropy_pairs(self) -> pl.Expr:
        """Entropy of successive pairs in symbolized series (SB_MotifThree_quantile_hh)."""
        return self.sb_motif_three_quantile_hh()

    def ami2(self, tau: int = 2, n_bins: int = 5) -> pl.Expr:
        """Histogram-based automutual information (lag 2, 5 bins) (CO_HistogramAMI_even_2_5)."""
        return self.co_histogram_ami_even_tau_bins(tau=tau, n_bins=n_bins)

    def trev(self) -> pl.Expr:
        """Time reversibility (CO_trev_1_num)."""
        return self.co_trev_1_num()

    def ami_timescale(self, tau: float = 40.0) -> pl.Expr:
        """First minimum of the AMI function (IN_AutoMutualInfoStats_40_gaussian_fmmi)."""
        return self.in_auto_mutual_info_stats_tau_gaussian_fmmi(tau=tau)

    def transition_variance(self) -> pl.Expr:
        """Transition matrix column variance (SB_TransitionMatrix_3ac_sumdiagcov)."""
        return self.sb_transition_matrix_3ac_sumdiagcov()

    def periodicity(self) -> pl.Expr:
        """Wang's periodicity metric (PD_PeriodicityWang_th001)."""
        return self.pd_periodicity_wang_th0_01()

    def embedding_dist(self) -> pl.Expr:
        """Goodness of exponential fit to embedding distance distribution (CO_Embed2_Dist_tau_d_expfit_meandiff)."""
        return self.co_embed2_dist_tau_d_expfit_meandiff()

    def rs_range(self, lag: int = 2) -> pl.Expr:
        """Rescaled range fluctuation analysis (SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1)."""
        return self.sc_fluct_anal_2_50_1_logi_prop_r1(lag=lag, how="rsrangefit")

    def dfa(self, lag: int = 2) -> pl.Expr:
        """Detrended fluctuation analysis (SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1)."""
        return self.sc_fluct_anal_2_50_1_logi_prop_r1(lag=lag, how="dfa")

    # --- Catch All ---

    def catch_all(self, include_mean_std: bool = False) -> pl.Expr:
        """
        Calculates all 22 (or 24) features in a single call.
        
        Returns a Struct containing all features.
        """
        feats = {
            "mode_5": self.mode_5(),
            "mode_10": self.mode_10(),
            "outlier_timing_pos": self.outlier_timing_pos(),
            "outlier_timing_neg": self.outlier_timing_neg(),
            "acf_timescale": self.acf_timescale(),
            "acf_first_min": self.acf_first_min(),
            "low_freq_power": self.low_freq_power(),
            "centroid_freq": self.centroid_freq(),
            "forecast_error": self.forecast_error(),
            "whiten_timescale": self.whiten_timescale(),
            "high_fluctuation": self.high_fluctuation(),
            "stretch_high": self.stretch_high(),
            "stretch_decreasing": self.stretch_decreasing(),
            "entropy_pairs": self.entropy_pairs(),
            "ami2": self.ami2(),
            "trev": self.trev(),
            "ami_timescale": self.ami_timescale(),
            "transition_variance": self.transition_variance(),
            "periodicity": self.periodicity(),
            "embedding_dist": self.embedding_dist(),
            "rs_range": self.rs_range(),
            "dfa": self.dfa(),
        }
        if include_mean_std:
            feats["mean"] = self._expr.mean()
            feats["std"] = self._expr.std()
            
        return pl.struct(**feats)

    # --- Generic Names (Mapping to Rust Expressions) ---

    def dn_outlier_include_np_001_mdrmd(self, is_pos: bool = True) -> pl.Expr:
        return self._register("dn_outlier_include_np_001_mdrmd", {"is_pos": is_pos})

    def dn_histogram_mode_n(self, n_bins: int = 5) -> pl.Expr:
        return self._register("dn_histogram_mode", {"n_bins": n_bins})

    def co_embed2_dist_tau_d_expfit_meandiff(self) -> pl.Expr:
        return self._register("co_embed2_dist_tau_d_expfit_meandiff")

    def co_f1ecac(self) -> pl.Expr:
        return self._register("co_f1ecac")

    def co_first_min_ac(self) -> pl.Expr:
        return self._register("co_first_min_ac")

    def co_histogram_ami_even_tau_bins(self, tau: int = 1, n_bins: int = 5) -> pl.Expr:
        return self._register("co_histogram_ami_even_tau_bins", {"tau": tau, "n_bins": n_bins})

    def co_trev_1_num(self) -> pl.Expr:
        return self._register("co_trev_1_num")

    def fc_local_simple_mean_tauresrat(self, train_length: int = 10) -> pl.Expr:
        return self._register("fc_local_simple_mean_tauresrat", {"train_length": train_length})

    def fc_local_simple_mean_stderr(self, train_length: int = 10) -> pl.Expr:
        return self._register("fc_local_simple_mean_stderr", {"train_length": train_length})

    def in_auto_mutual_info_stats_tau_gaussian_fmmi(self, tau: float = 1.0) -> pl.Expr:
        return self._register("in_auto_mutual_info_stats_tau_gaussian_fmmi", {"tau": tau})

    def md_hrv_classic_pnn(self, pnn: int = 40) -> pl.Expr:
        return self._register("md_hrv_classic_pnn", {"pnn": pnn})

    def sb_binary_stats_diff_longstretch0(self) -> pl.Expr:
        return self._register("sb_binary_stats_diff_longstretch0")

    def sb_binary_stats_mean_longstretch1(self) -> pl.Expr:
        return self._register("sb_binary_stats_mean_longstretch1")

    def sb_motif_three_quantile_hh(self) -> pl.Expr:
        return self._register("sb_motif_three_quantile_hh")

    def sc_fluct_anal_2_50_1_logi_prop_r1(self, lag: int = 1, how: str = "dfa") -> pl.Expr:
        return self._register("sc_fluct_anal_2_50_1_logi_prop_r1", {"lag": lag, "how": how})

    def sp_summaries_welch_rect(self, what: str = "centroid") -> pl.Expr:
        return self._register("sp_summaries_welch_rect", {"what": what})

    def sb_transition_matrix_3ac_sumdiagcov(self) -> pl.Expr:
        return self._register("sb_transition_matrix_3ac_sumdiagcov")

    def pd_periodicity_wang_th0_01(self) -> pl.Expr:
        return self._register("pd_periodicity_wang_th0_01")


@pl.api.register_expr_namespace("fresh")
class FreshNamespace:
    """Polars expression namespace for tsfresh-inspired features."""

    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def _register(self, name: str, kwargs: dict | None = None) -> pl.Expr:
        return register_plugin_function(
            args=[self._expr],
            plugin_path=LIB,
            function_name=name,
            kwargs=kwargs,
            is_elementwise=False,
            returns_scalar=True,
        )

    def variance_larger_than_standard_deviation(self) -> pl.Expr:
        return self._register("fresh_variance_larger_than_standard_deviation")

    def ratio_beyond_r_sigma(self, r: float = 1.0) -> pl.Expr:
        return self._register("fresh_ratio_beyond_r_sigma", {"r": r})

    def large_standard_deviation(self, r: float = 0.25) -> pl.Expr:
        return self._register("fresh_large_standard_deviation", {"r": r})

    def symmetry_looking(self, r: float = 0.1) -> pl.Expr:
        return self._register("fresh_symmetry_looking", {"r": r})

    def has_duplicate_max(self) -> pl.Expr:
        return self._register("fresh_has_duplicate_max")

    def has_duplicate_min(self) -> pl.Expr:
        return self._register("fresh_has_duplicate_min")

    def has_duplicate(self) -> pl.Expr:
        return self._register("fresh_has_duplicate")

    def root_mean_square(self) -> pl.Expr:
        return self._register("fresh_root_mean_square")

    # --- Batch 3: Additional Statistics ---

    def skewness(self) -> pl.Expr:
        return self._register("fresh_skewness")

    def kurtosis(self) -> pl.Expr:
        return self._register("fresh_kurtosis")

    def variation_coefficient(self) -> pl.Expr:
        return self._register("fresh_variation_coefficient")

    def abs_energy(self) -> pl.Expr:
        return self._register("fresh_abs_energy")

    def cid_ce(self, normalize: bool = True) -> pl.Expr:
        return self._register("fresh_cid_ce", {"normalize": normalize})

    def mean_abs_change(self) -> pl.Expr:
        return self._register("fresh_mean_abs_change")

    def mean_change(self) -> pl.Expr:
        return self._register("fresh_mean_change")

    def mean_second_derivative_central(self) -> pl.Expr:
        return self._register("fresh_mean_second_derivative_central")

    def absolute_sum_of_changes(self) -> pl.Expr:
        return self._register("fresh_absolute_sum_of_changes")

    def fft_coefficient(self, k: int, attr: str = "abs") -> pl.Expr:
        return self._register("fresh_fft_coefficient", {"k": k, "attr": attr})

    def fft_aggregated(self, agg_type: str = "centroid") -> pl.Expr:
        return self._register("fresh_fft_aggregated", {"agg_type": agg_type})

    def cwt_coefficient(self, width: float, index: int) -> pl.Expr:
        return self._register("fresh_cwt_coefficient", {"width": width, "index": index})

    def sample_entropy(self) -> pl.Expr:
        return self._register("fresh_sample_entropy")

    def binned_entropy(self, max_bins: int = 10) -> pl.Expr:
        return self._register("fresh_binned_entropy", {"max_bins": max_bins})

    def friedrich_coefficient(self, m: int = 1, r: int = 30, coeff_index: int = 0) -> pl.Expr:
        return self._register("fresh_friedrich_coefficient", {"m": m, "r": r, "coeff_index": coeff_index})

    # --- Batch 1: Basic Counts & Streaks ---

    def count_above_mean(self) -> pl.Expr:
        return self._register("fresh_count_above_mean")

    def count_below_mean(self) -> pl.Expr:
        return self._register("fresh_count_below_mean")

    def longest_strike_above_mean(self) -> pl.Expr:
        return self._register("fresh_longest_strike_above_mean")

    def longest_strike_below_mean(self) -> pl.Expr:
        return self._register("fresh_longest_strike_below_mean")

    def percentage_of_reoccurring_values_to_all_values(self) -> pl.Expr:
        return self._register("fresh_percentage_of_reoccurring_values_to_all_values")

    def percentage_of_reoccurring_datapoints_to_all_datapoints(self) -> pl.Expr:
        return self._register("fresh_percentage_of_reoccurring_datapoints_to_all_datapoints")

    def sum_of_reoccurring_values(self) -> pl.Expr:
        return self._register("fresh_sum_of_reoccurring_values")

    def sum_of_reoccurring_data_points(self) -> pl.Expr:
        return self._register("fresh_sum_of_reoccurring_data_points")

    def ratio_value_number_to_time_series_length(self) -> pl.Expr:
        return self._register("fresh_ratio_value_number_to_time_series_length")

    # --- Batch 2: Locations & Peaks ---

    def first_location_of_maximum(self) -> pl.Expr:
        return self._register("fresh_first_location_of_maximum")

    def last_location_of_maximum(self) -> pl.Expr:
        return self._register("fresh_last_location_of_maximum")

    def first_location_of_minimum(self) -> pl.Expr:
        return self._register("fresh_first_location_of_minimum")

    def last_location_of_minimum(self) -> pl.Expr:
        return self._register("fresh_last_location_of_minimum")

    def number_peaks(self, n: int = 1) -> pl.Expr:
        return self._register("fresh_number_peaks", {"n": n})

    
    def index_mass_quantile(self, q: float = 0.5) -> pl.Expr:
        return self._register("fresh_index_mass_quantile", {"q": q})

    # --- Batch 4: Correlation ---

    def agg_autocorrelation(self, f_agg: str = "mean", maxlag: int = 10) -> pl.Expr:
        # Map f_agg to match_agg due to param names? No, kwargs uses struct field name.
        return self._register("fresh_agg_autocorrelation", {"match_agg": f_agg, "maxlag": maxlag})

    def partial_autocorrelation(self, lag: int = 1) -> pl.Expr:
        return self._register("fresh_partial_autocorrelation", {"lag": lag})

    # --- Batch 5: Dynamics ---

    def time_reversal_asymmetry_statistic(self, lag: int = 1) -> pl.Expr:
        return self._register("fresh_time_reversal_asymmetry_statistic", {"lag": lag})

    def c3(self, lag: int = 1) -> pl.Expr:
        return self._register("fresh_c3", {"lag": lag})

    # --- Batch 6: Linear Trend ---

    def linear_trend(self, attr: str = "slope") -> pl.Expr:
        # attr: "slope", "intercept", "rvalue", "pvalue", "stderr"
        return self._register("fresh_linear_trend", {"attr": attr})

    # --- Batch 7: Counts & Ranges ---

    def number_crossing_m(self, m: float = 0.0) -> pl.Expr:
        return self._register("fresh_number_crossing_m", {"m": m})

    def range_count(self, min: float = -1.0, max: float = 1.0) -> pl.Expr:
        return self._register("fresh_range_count", {"min": min, "max": max})

    def value_count(self, value: float = 0.0) -> pl.Expr:
        return self._register("fresh_value_count", {"value": value})

    # --- Batch 8: Distribution Tests ---

    def symmetry_looking(self, r: float = 0.05) -> pl.Expr:
        return self._register("fresh_symmetry_looking", {"r": r})

    def large_standard_deviation(self, r: float = 0.25) -> pl.Expr:
        return self._register("fresh_large_standard_deviation", {"r": r})

    def quantile(self, q: float = 0.5) -> pl.Expr:
        return self._register("fresh_quantile", {"q": q})

    def ratio_beyond_r_sigma(self, r: float = 2.0) -> pl.Expr:
        return self._register("fresh_ratio_beyond_r_sigma", {"r": r})

    def catch_all(self) -> pl.Expr:
        """Calculates a set of essential tsfresh features. 
        Compatible with df.rolling(w).agg(pl.col("x").fresh.catch_all()) if unnested."""
        feats = {
            "abs_energy": self.abs_energy(),
            "root_mean_square": self.root_mean_square(),
            "mean_abs_change": self.mean_abs_change(),
            "mean_change": self.mean_change(),
            "cid_ce": self.cid_ce(),
            "sample_entropy": self.sample_entropy(),
            "has_duplicate": self.has_duplicate(),
            "variance_larger_std": self.variance_larger_than_standard_deviation(),
            "fft_centroid": self.fft_aggregated(agg_type="centroid"),
            "fft_variance": self.fft_aggregated(agg_type="variance"),
            "friedrich_coeff_0": self.friedrich_coefficient(coeff_index=0),
            "friedrich_coeff_1": self.friedrich_coefficient(coeff_index=1),
            # Batch 1
            "count_above_mean": self.count_above_mean(),
            "count_below_mean": self.count_below_mean(),
            "longest_strike_above_mean": self.longest_strike_above_mean(),
            "percentage_reoccurring": self.percentage_of_reoccurring_values_to_all_values(),
            # Batch 2
            "first_loc_max": self.first_location_of_maximum(),
            "number_peaks_n1": self.number_peaks(n=1),
            "mass_quantile_50": self.index_mass_quantile(q=0.5),
            # Batch 3
            "skewness": self.skewness(),
            "kurtosis": self.kurtosis(),
            "variation_coefficient": self.variation_coefficient(),
            # Batch 4
            "autocorr_mean_lag10": self.agg_autocorrelation(f_agg="mean", maxlag=10),
            "partial_autocorr_lag1": self.partial_autocorrelation(lag=1),
            # Batch 5
            "time_rev_asym_lag1": self.time_reversal_asymmetry_statistic(lag=1),
            "c3_lag1": self.c3(lag=1),
            # Batch 6
            "linear_trend_slope": self.linear_trend(attr="slope"),
            # Batch 7
            "number_crossing_0": self.number_crossing_m(m=0.0),
            # Batch 8
            "quantile_05": self.quantile(q=0.05),
            "quantile_95": self.quantile(q=0.95),
            "check_large_std": self.large_standard_deviation(r=0.25).cast(pl.Float64),
            "check_symmetry": self.symmetry_looking(r=0.05).cast(pl.Float64),
        }
        return pl.struct(**feats)