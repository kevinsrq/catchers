use num_traits::Zero;
use polars::prelude::*;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

use crate::utils::stats::basic::mean;

// ==============================================================================
// 1. Signal Processing
// ==============================================================================

/// Calculates the first index where the autocorrelation crosses zero.
/// Useful for determining the natural "lag" of a time series.
/// Optimized to reuse the core autocorrelation logic.
#[allow(dead_code)]
pub fn first_zero<T>(ca: &ChunkedArray<T>, max_tau: usize) -> PolarsResult<usize>
where
    T: PolarsNumericType,
{
    let s = autocorr_polars(ca)?;
    let ca_ac = s.f64()?;

    let limit = std::cmp::min(max_tau, ca_ac.len());
    let mut zero_cross_ind = 0;

    for i in 0..limit {
        // Safe because we know the length
        let val = unsafe { ca_ac.get_unchecked(i) }.unwrap_or(0.0);
        if val <= 0.0 {
            break;
        }
        zero_cross_ind += 1;
    }

    Ok(zero_cross_ind)
}

/// Core implementation of first_zero for slices.
/// Uses the unified autocorr function to avoid code duplication.
pub fn first_zero_core(a: &[f64], max_tau: usize) -> usize {
    let ac = autocorr(a); // Uses FFT-based autocorr
    let limit = std::cmp::min(max_tau, ac.len());
    let mut zero_cross_ind = 0;
    for &val in ac.iter().take(limit) {
        if val <= 0.0 {
            break;
        }
        zero_cross_ind += 1;
    }
    zero_cross_ind
}

/// Power Spectral Density Estimation (Welch's Method).
/// Returns a DataFrame with "frequency" and "power" columns.
#[allow(dead_code)]
pub fn welch_psd<T>(ca: &ChunkedArray<T>, fs: f64) -> PolarsResult<DataFrame>
where
    T: PolarsNumericType,
{
    if ca.is_empty() {
        return Err(PolarsError::ComputeError(
            "Empty array for Welch Method".into(),
        ));
    }

    let s_f64: Series = ca.cast(&DataType::Float64)?;
    let ca_f64 = s_f64.f64()?;

    // Efficiently collect data, filling nulls with mean
    let mean_val = ca_f64.mean().unwrap_or(0.0);
    let data: Vec<f64> = ca_f64
        .into_iter()
        .map(|opt_v| opt_v.unwrap_or(mean_val))
        .collect();

    // Rectangular window (default catch22 behavior)
    // Avoid allocating a full vector of 1.0s if we can just pass a slice or handle it inside welch
    // For API compatibility with existing `welch`, we create it here.
    let window = vec![1.0; data.len()];

    let (pxx, freqs) = welch(&data, fs, &window);

    let s_freq = Series::new("frequency".into(), freqs);
    let s_power = Series::new("power".into(), pxx);

    DataFrame::new(vec![s_freq.into(), s_power.into()])
}

/// Computes the power spectral density using Welch's method.
pub fn welch(signal: &[f64], fs: f64, window: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = signal.len();
    let nperseg = window.len();
    let noverlap = nperseg / 2;
    let step = nperseg - noverlap;

    if step == 0 {
        // avoid division by zero if nperseg is huge or something wrong
        return (vec![], vec![]);
    }

    let n_segments = (n - noverlap) / step;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(nperseg);

    // Reusing the buffer for FFT to avoid allocations per segment
    // However, rustfft requires a specific buffer size.
    let mut buffer: Vec<Complex<f64>> = vec![Complex::zero(); nperseg];
    let mut psd_acc = vec![0.0; nperseg / 2 + 1];

    let window_energy: f64 = window.iter().map(|w| w.powi(2)).sum();
    let scale_factor = if window_energy > 0.0 {
        2.0 / (fs * window_energy)
    } else {
        0.0
    };

    for i in 0..n_segments {
        let start = i * step;
        let end = start + nperseg;
        if end > n {
            break;
        }

        // Fill buffer
        for (j, (&s, &w)) in signal[start..end].iter().zip(window.iter()).enumerate() {
            buffer[j] = Complex { re: s * w, im: 0.0 };
        }

        fft.process(&mut buffer);

        // Accumulate PSD
        for j in 0..=nperseg / 2 {
            let amp = buffer[j].norm_sqr();
            let mut scale = scale_factor;
            if j == 0 || (nperseg.is_multiple_of(2) && j == nperseg / 2) {
                scale /= 2.0;
            }
            psd_acc[j] += amp * scale;
        }
    }

    if n_segments > 0 {
        for v in psd_acc.iter_mut() {
            *v /= n_segments as f64;
        }
    }

    let mut freqs = vec![0.0; nperseg / 2 + 1];
    let df = fs / nperseg as f64;
    for (i, val) in freqs.iter_mut().enumerate().take(nperseg / 2 + 1) {
        *val = i as f64 * df;
    }

    (psd_acc, freqs)
}

/// Computes autocorrelation using FFT.
/// Optimized to perform calculation once.
pub fn autocorr(a: &[f64]) -> Vec<f64> {
    let n = a.len();
    if n == 0 {
        return vec![];
    }

    let mean_val = mean(a);

    // Zero-padding is essential for linear convolution via FFT to avoid aliasing (circular convolution)
    // Standard approach: 2*N length
    let padded_len = n.next_power_of_two() * 2; // Optimization: Power of 2 usually faster for FFT

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(padded_len);
    let ifft = planner.plan_fft_inverse(padded_len);

    let mut buffer: Vec<Complex<f64>> = vec![Complex::zero(); padded_len];

    // Centering and copying to buffer
    for i in 0..n {
        buffer[i] = Complex {
            re: a[i] - mean_val,
            im: 0.0,
        };
    }

    fft.process(&mut buffer);

    // Compute PSD (Conj multiplication)
    for c in buffer.iter_mut() {
        *c *= c.conj();
    }

    ifft.process(&mut buffer);

    // Normalize
    let mut ac = Vec::with_capacity(n);
    // The first element of IFFT result corresponds to lag 0
    // Scale for Inverse FFT typically required if library doesn't do it?
    // RustFFT does NOT normalize inverse FFT.
    let scale = 1.0 / padded_len as f64;

    // Variance is at index 0 (lag 0) * scale / N (biased estimator)
    let var = (buffer[0].re * scale) / n as f64;

    if var.abs() < 1e-12 {
        return vec![0.0; n];
    }

    for buffer_item in buffer.iter().take(n) {
        let val = (buffer_item.re * scale) / n as f64; // Biased autocorrelation
        ac.push(val / var);
    }

    ac
}

/// Computes autocorrelation for a single lag `tau`.
/// Prefer `autocorr` for multiple lags. Kept for single-lag efficiency.
pub fn autocorr_lag(a: &[f64], tau: usize) -> f64 {
    if tau >= a.len() {
        return 0.0;
    }
    let mean_val = mean(a);
    let mut num = 0.0;
    let mut den = 0.0;

    // Single pass for denominator could be cached if calling multiple times,
    // but here we strictly optimize the isolated function call.
    for x in a {
        den += (x - mean_val).powi(2);
    }

    for i in 0..a.len() - tau {
        num += (a[i] - mean_val) * (a[i + tau] - mean_val);
    }

    if den == 0.0 { 0.0 } else { num / den }
}

/// Computes autocovariance for a single lag `tau`.
pub fn autocov_lag(a: &[f64], tau: usize) -> f64 {
    if tau >= a.len() {
        return 0.0;
    }
    let mean_val = mean(a);
    let mut num = 0.0;
    for i in 0..a.len() - tau {
        num += (a[i] - mean_val) * (a[i + tau] - mean_val);
    }
    num / a.len() as f64
}

#[allow(dead_code)]
fn autocorr_polars<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsNumericType,
{
    let ca = ca.cast(&DataType::Float64)?;
    let ca_f64 = ca.f64()?;

    // Efficient collection
    let signal: Vec<f64> = ca_f64.into_iter().map(|v| v.unwrap_or(0.0)).collect();

    let ac = autocorr(&signal);
    Ok(Series::new("autocorrelation".into(), ac))
}
