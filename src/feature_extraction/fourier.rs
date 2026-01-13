use realfft::RealFftPlanner;
use rustfft::num_complex::Complex;

/// Compute the Real FFT of a signal.
pub fn compute_rfft(x: &[f64]) -> Vec<Complex<f64>> {
    let n = x.len();
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);

    let mut input = x.to_vec();
    let mut output = fft.make_output_vec();
    fft.process(&mut input, &mut output).unwrap();
    output
}

/// Returns the real, imag, abs or angle of the k-th FFT coefficient.
pub fn fft_coefficient(x: &[f64], k: usize, attr: &str) -> f64 {
    let fft = compute_rfft(x);
    if k >= fft.len() {
        return f64::NAN;
    }

    let val = fft[k];
    match attr {
        "real" => val.re,
        "imag" => val.im,
        "abs" => (val.re.powi(2) + val.im.powi(2)).sqrt(),
        "angle" => val.im.atan2(val.re).to_degrees(),
        _ => f64::NAN,
    }
}

/// Helper to calculate spectral moments.
fn get_spectral_moment(magnitudes: &[f64], moment: i32) -> f64 {
    let sum_mag: f64 = magnitudes.iter().sum();
    if sum_mag == 0.0 {
        return 0.0;
    }

    let weighted_sum: f64 = magnitudes
        .iter()
        .enumerate()
        .map(|(i, &m)| (i as f64).powi(moment) * m)
        .sum();

    weighted_sum / sum_mag
}

/// Returns spectral centroid, variance, skew, or kurtosis of the absolute FFT spectrum.
pub fn fft_aggregated(x: &[f64], agg_type: &str) -> f64 {
    let fft = compute_rfft(x);
    let magnitudes: Vec<f64> = fft
        .iter()
        .map(|c| (c.re.powi(2) + c.im.powi(2)).sqrt())
        .collect();

    let m1 = get_spectral_moment(&magnitudes, 1); // Centroid
    let m2 = get_spectral_moment(&magnitudes, 2);
    let m3 = get_spectral_moment(&magnitudes, 3);
    let m4 = get_spectral_moment(&magnitudes, 4);

    let variance = m2 - m1.powi(2);

    match agg_type {
        "centroid" => m1,
        "variance" => variance,
        "skew" => {
            if variance < 0.5 {
                return f64::NAN;
            }
            (m3 - 3.0 * m1 * variance - m1.powi(3)) / variance.powf(1.5)
        },
        "kurtosis" => {
            if variance < 0.5 {
                return f64::NAN;
            }
            (m4 - 4.0 * m1 * m3 + 6.0 * m2 * m1.powi(2) - 3.0 * m1.powi(4)) / variance.powi(2)
        },
        _ => f64::NAN,
    }
}
