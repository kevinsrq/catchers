use std::sync::Arc;
use polars::prelude::*;

use rustfft::{num_complex::Complex, FftPlanner};
use num_traits::{ToPrimitive, Zero};
use crate::utils::stats::basic::mean;

// ==============================================================================
// 1. Processamento de Sinais (Signal Processing)
// ==============================================================================

/// Calcula o primeiro índice onde a autocorrelação cruza zero.
/// Útil para determinar a "lag" natural de uma série temporal.
pub fn first_zero<T>(ca: &ChunkedArray<T>, max_tau: usize) -> PolarsResult<usize>
where
    T: PolarsNumericType,
{
    // Reutiliza a lógica de autocorrelação (assumindo que já temos a função autocorr disponível ou implementada aqui)
    // Para ser self-contained, implementarei uma versão simplificada ou chamaria a autocorr definida anteriormente.
    // Aqui, faremos o cálculo direto para performance se não precisarmos da série inteira,
    // mas o padrão catch22 usa a autocorrelação completa.
    
    let s = autocorr_polars(ca)?; // Chama a função autocorr definida no seu arquivo original (ou importada)
    let ca_ac = s.f64()?;
    
    // To use core logic if needed, but here simple enough.
    let mut zero_cross_ind = 0;
    let limit = std::cmp::min(max_tau, ca_ac.len());

    for i in 0..limit {
        let val = ca_ac.get(i).unwrap_or(0.0);
        if val <= 0.0 {
            break;
        }
        zero_cross_ind += 1;
    }

    Ok(zero_cross_ind)
}

pub fn first_zero_core(a: &[f64], max_tau: usize) -> usize {
    // Requires autocorr of 'a'
    // This is circular if first_zero requires autocorr which returns signal.
    // reference.rs implementation:
    // let mut tau = first_zero(a, a.len()); ...
    // -> it assumes first_zero computes autocorrelation internally or is passed it.
    // In reference code, first_zero(a, ...) seems to do autocorr inside.
    
    // Simple logic implementation for reference usage:
    // We can interpret `first_zero` in reference logic as: compute autocorr, find first zero crossing.
    // use crate::utils::stats::basic::autocorr; // Removed
    // let ac = autocorr(a);
    let ac = autocorr(a);
    let limit = std::cmp::min(max_tau, ac.len());
    let mut zero_cross_ind = 0;
    for i in 0..limit {
        if ac[i] <= 0.0 { break; }
        zero_cross_ind += 1;
    }
    zero_cross_ind
}

/// Estimativa de Densidade Espectral de Potência (Welch's Method).
/// Retorna um DataFrame com colunas "frequency" e "power".
/// O padrão Catch22 usa janela retangular, mas deixei a estrutura pronta para janelas.
pub fn welch_psd<T>(ca: &ChunkedArray<T>, fs: f64) -> PolarsResult<DataFrame>
where
    T: PolarsNumericType,
{
    if ca.is_empty() {
        return Err(PolarsError::ComputeError("Empty array for Welch Method".into()));
    }

    // ... logic inside welch_psd using chunked array
    // Simplify: Call core welch function
    let fs = fs;
    let s_f64: Series = ca.cast(&DataType::Float64)?;
    let mean_val = s_f64.mean().unwrap_or(0.0);

    // Prepara dados de entrada (fill nulls com média)
    let data: Vec<f64> = ca.into_iter()
        .map(|opt_v| opt_v.map(|v| v.to_f64().unwrap_or(mean_val)).unwrap_or(mean_val))
        .collect();
    
    let window = vec![1.0; data.len()]; 
    let (pxx, freqs) = welch(&data, fs, &window);

    let s_freq = Series::new("frequency".into(), freqs);
    let s_power = Series::new("power".into(), pxx);

    DataFrame::new(vec![s_freq.into(), s_power.into()])
}

/// Computes the power spectral density using Welch's method.
///
/// * `signal` - Input signal array.
/// * `fs` - Sampling frequency.
/// * `window` - Window function to apply.
///
/// Returns a tuple `(S, f)` where `S` is the power spectrum and `f` are frequencies.
pub fn welch(signal: &[f64], fs: f64, window: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = signal.len();
    let nperseg = window.len();
    let noverlap = nperseg / 2;
    let step = nperseg - noverlap;
    
    // Number of segments
    let n_segments = (n - noverlap) / step;
    
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(nperseg);
    
    let mut psd_acc = vec![0.0; nperseg / 2 + 1];
    
    // Window energy normalization
    let window_energy: f64 = window.iter().map(|w| w.powi(2)).sum();

    for i in 0..n_segments {
        let start = i * step;
        let end = start + nperseg;
        if end > n { break; }
        
        let segment = &signal[start..end];
        let mut buffer: Vec<Complex<f64>> = segment.iter().zip(window.iter()).map(|(&s, &w)| Complex { re: s * w, im: 0.0 }).collect();
        
        fft.process(&mut buffer);
        
        // Compute periodogram for this segment
        for j in 0..=nperseg / 2 {
             let amp = buffer[j].norm_sqr(); // |X[k]|^2
             // Scale: 2 for one-sided (except DC and Nyquist), / (fs * window_energy)
             let mut scale = 2.0 / (fs * window_energy);
             if j == 0 || (nperseg % 2 == 0 && j == nperseg / 2) {
                 scale /= 2.0;
             }
             psd_acc[j] += amp * scale;
        }
    }
    
    // Average
    for v in psd_acc.iter_mut() {
        *v /= n_segments as f64;
    }
    
    // Frequencies
    let mut freqs = vec![0.0; nperseg / 2 + 1];
    for i in 0..=nperseg / 2 {
        freqs[i] = i as f64 * fs / nperseg as f64;
    }
    
    (psd_acc, freqs)
}

/// Computes autocorrelation for a specific lag using FFT.
pub fn autocorr(a: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mean_val = mean(a);
    let mut normalized = vec![0.0; n];
    for i in 0..n {
        normalized[i] = a[i] - mean_val;
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(2 * n);
    let ifft = planner.plan_fft_inverse(2 * n);

    let mut buffer: Vec<Complex<f64>> = vec![Complex::zero(); 2 * n];
    for i in 0..n {
        buffer[i] = Complex { re: normalized[i], im: 0.0 };
    }

    fft.process(&mut buffer);

    for i in 0..2 * n {
        buffer[i] = buffer[i] * buffer[i].conj();
    }

    ifft.process(&mut buffer);

    let mut ac = vec![0.0; n];
    let var = buffer[0].re / (2.0 * n as f64); 

    for i in 0..n {
        ac[i] = (buffer[i].re / (2.0 * n as f64)) / var;
    }
    ac
}

/// Computes autocorrelation for a single lag `tau`.
pub fn autocorr_lag(a: &[f64], tau: usize) -> f64 {
    if tau >= a.len() { return 0.0; }
    let mean_val = mean(a);
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..a.len() {
        den += (a[i] - mean_val).powi(2);
    }
    for i in 0..a.len()-tau {
        num += (a[i] - mean_val) * (a[i+tau] - mean_val);
    }
    if den == 0.0 { 0.0 } else { num / den }
}

/// Computes autocovariance for a single lag `tau`.
pub fn autocov_lag(a: &[f64], tau: usize) -> f64 {
    if tau >= a.len() { return 0.0; }
    // autocov is unnormalized autocorrelation (covariance of signal with lagged self)
    let mean_val = mean(a);
    let mut num = 0.0;
    for i in 0..a.len() - tau {
        num += (a[i] - mean_val) * (a[i+tau] - mean_val);
    }
    num / a.len() as f64
}

fn autocorr_polars<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsNumericType,
{
    let ca = ca.cast(&DataType::Float64)?;
    let ca = ca.f64()?;
    let signal: Vec<f64> = ca.into_iter().map(|v| v.unwrap_or(0.0)).collect();
    let ac = autocorr(&signal);
    Ok(Series::new("autocorrelation".into(), ac))
}
