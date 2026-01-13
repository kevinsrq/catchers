# polars-catchers

`polars-catchers` is a high-performance **Polars extension** that implements:
1.  The **catch22** (CAnonical Time-series CHaracteristics) feature set.
2.  A comprehensive subset of **tsfresh** (Time Series FeatuRe Extraction on Basis of Scalable Hypothesis tests) features.

Designed for speed and ease of use, it allows you to extract essential time-series features directly within your Polars pipelines using optimized Rust implementations.

## Features

- **Native Performance**: Implemented in Rust for maximum efficiency.
- **Polars Integration**: Works seamlessly as a Polars Expression Namespace.
- **Dual Namespaces**:
    - `.catchers`: Access the 22 canonical catch22 features.
    - `.fresh`: Access a wide range of tsfresh features (stats, entropy, dynamics, correlation, etc.).
- **catch_all()**: Calculate feature sets in a single call for either namespace.
- **Customizable**: Flexible parameters for individual features.

## Usage

Access the features through the `.catchers` or `.fresh` namespaces on any Polars expression.

### Catch22 Examples

```python
import polars as pl
import polars_catchers

df = pl.DataFrame({
    "id": ["A", "A", "B", "B"],
    "val": [1.0, 2.0, 5.0, 4.0]
})

# Individual features
df.select(
    pl.col("val").catchers.mode_5().alias("mode"),
    pl.col("val").catchers.dfa().alias("dfa")
)

# All catch22 features
df.group_by("id").agg(
    pl.col("val").catchers.catch_all()
).unnest("val")
```

### TSFresh Examples

```python
import polars as pl
import polars_catchers

df = pl.DataFrame({
    "x": [10.0, 12.0, 15.0, 14.0, 10.0, 12.0]
})

# Individual features
df.select(
    pl.col("x").fresh.skewness(),
    pl.col("x").fresh.number_peaks(n=1),
    pl.col("x").fresh.c3(lag=1)
)

# Comprehensive feature set
df.select(
    pl.col("x").fresh.catch_all()
).unnest("x")
```

## Available Features

### Catch22 (`.catchers`)

| Short Name | Description |
| :--- | :--- |
| `mode_5` | 5-bin histogram mode |
| `mode_10` | 10-bin histogram mode |
| `outlier_timing_pos` | Positive outlier timing |
| `outlier_timing_neg` | Negative outlier timing |
| `acf_timescale` | First 1/e crossing of the ACF |
| `acf_first_min` | First minimum of the ACF |
| `low_freq_power` | Power in lowest 20% frequencies |
| `centroid_freq` | Centroid frequency |
| `forecast_error` | Error of 3-point rolling mean forecast |
| `whiten_timescale` | Change in AC timescale after differencing |
| `high_fluctuation` | Proportion of high incremental changes |
| `stretch_high` | Longest stretch of above-mean values |
| `stretch_decreasing` | Longest stretch of decreasing values |
| `entropy_pairs` | Entropy of successive pairs |
| `ami2` | Automutual information (lag 2, 5 bins) |
| `trev` | Time reversibility |
| `ami_timescale` | First minimum of the AMI function |
| `transition_variance` | Transition matrix column variance |
| `periodicity` | Wang's periodicity metric |
| `embedding_dist` | Exponential fit to embedding distance |
| `rs_range` | Rescaled range fluctuation analysis |
| `dfa` | Detrended fluctuation analysis |

### TSFresh (`.fresh`)

The `.fresh` namespace covers a wide range of feature categories:

*   **Basic Counts**: `count_above_mean`, `count_below_mean`, `longest_strike_above_mean`, `percentage_reoccurring`, etc.
*   **Locations & Peaks**: `first_location_of_maximum`, `number_peaks`, `mass_quantile`.
*   **Statistics**: `skewness`, `kurtosis`, `variation_coefficient`, `mean`, `variance`.
*   **Correlation**: `agg_autocorrelation`, `partial_autocorrelation`.
*   **Dynamics**: `time_reversal_asymmetry_statistic`, `c3` (non-linearity), `mean_abs_change`.
*   **Linear Trend**: `linear_trend` (slope, intercept, r-value).
*   **Counts & Ranges**: `number_crossing_m`, `range_count`, `value_count`.
*   **Distribution Tests**: `symmetry_looking`, `large_standard_deviation`, `quantile`, `ratio_beyond_r_sigma`.
*   **Entropy & Complexity**: `sample_entropy`, `binned_entropy`, `approximate_entropy`, `cid_ce`, `abs_energy`.
*   **Spectral**: `fft_aggregated` (centroid, variance, skew, kurtosis), `fft_coefficient`.

## License

This project is licensed under the MIT License.
