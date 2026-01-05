//! Statistical utility functions and internal modules.
pub mod signal;
pub mod regression;
pub mod smoothing;

// Re-export specific items to match original stats.rs interface
pub use signal::{first_zero, welch_psd};
pub use regression::{linreg, slope};
pub use smoothing::splinefit;




pub mod basic;
pub mod histogram;
