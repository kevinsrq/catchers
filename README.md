# catchers 🦀📊

`catchers` is a high-performance **Polars extension** that implements the [catch22](https://time-series-features.gitbook.io/catch22/) (CAnonical Time-series CHaracteristics) feature set.

Designed for speed and ease of use, it allows you to extract 22 essential time-series features directly within your Polars pipelines.

## Features

- **🚀 Native Performance**: Implemented in Rust for maximum efficiency.
- **⚡ Polars Integration**: Works seamlessly as a Polars Expression Namespace.
- **🎁 catch_all()**: Calculate all 22 canonical features in a single call.
- **🏷️ Official Short Names**: Access features using the standard Catch22 short names (`mode_5`, `dfa`, `acf_timescale`, etc.).
- **🔧 Customizable**: Flexible parameters for individual features if you need to go beyond the defaults.

## Inspired By

This project is inspired by and builds upon the excellent work of:

- [pycatch22](https://github.com/DynamicsAndNeuralSystems/pycatch22): The official Python implementation.
- [catch22_rs](https://github.com/irazza/catch22_rs): A Rust implementation of Catch22 features.

## Installation

This project is optimized for use with [uv](https://github.com/astral-sh/uv).

```bash
# Clone the repository
git clone https://github.com/kevin/catchers
cd catchers

# Build and install for development
uv run maturin develop
```

## Usage

Access the features through the `.catchers` namespace on any Polars expression.

### Basic Examples

```python
import polars as pl
import catchers

# Sample data
df = pl.DataFrame({
    "id": ["A", "A", "B", "B"],
    "val": [1.0, 2.0, 5.0, 4.0]
})

# Calculate individual features
result = df.select(
    pl.col("val").catchers.mode_5().alias("mode"),
    pl.col("val").catchers.dfa().alias("dfa")
)

# Calculate ALL 22 features grouped by ID
full_results = (
    df.group_by("id")
    .agg(pl.col("val").catchers.catch_all())
    .unnest("val") # catch_all returns a Struct
)

print(full_results)
```

### Tips
>
> [!NOTE]
> Catch22 features are designed to be calculated on **z-scored** time-series data to focus on the time-ordering properties rather than raw values.

```python
# Canonical way to use catchers
df.select(
    ((pl.col("val") - pl.col("val").mean()) / pl.col("val").std())
    .catchers.catch_all()
)
```

## Available Features (Short Names)

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

## License

This project is licensed under the MIT License.
