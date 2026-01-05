//! Catchers: A Rust port of the Catch22 time-series feature extraction library.
//!
//! This library provides efficient implementations of 22 canonical time-series characteristics
//! designed for feature extraction in machine learning pipelines.
//!
//! It is designed to be used as a Python extension via Polars or directly in Rust.

mod expressions;
mod utils;
pub mod catchers;

use pyo3::prelude::*;
use pyo3_polars::PolarsAllocator;

#[pymodule]
fn _internal(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();
