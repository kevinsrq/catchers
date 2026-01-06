use std::f64::consts::PI;

/// Ricker wavelet (Mexican Hat wavelet)
/// scale: width parameter 'a'
pub fn ricker_wavelet(t: f64, scale: f64) -> f64 {
    let t2 = t * t;
    let s2 = scale * scale;
    let factor = 2.0 / ((3.0 * scale).sqrt() * PI.powf(0.25));
    factor * (1.0 - t2 / s2) * (-(t2 / (2.0 * s2))).exp()
}

/// Continuous Wavelet Transform at a specific scale and position.
/// This is a simplified version of CWT for the Ricker wavelet.
/// In tsfresh, it's often used with specific widths.
pub fn cwt_coefficient(x: &[f64], width: f64, index: usize) -> f64 {
    if index >= x.len() {
        return f64::NAN;
    }

    // The wavelet is usually sampled over a finite range.
    // pywt uses a default length proportional to the scale.
    // For Ricker, +/- 10*scale is usually enough to capture the energy.
    let window_half = (10.0 * width).ceil() as isize;
    let mut sum = 0.0;

    for offset in -window_half..=window_half {
        let x_idx = (index as isize + offset) as usize;
        if x_idx < x.len() {
            // center the wavelet at 'index'
            // s = offset/width
            let val = x[x_idx];
            let w = ricker_wavelet(offset as f64, width);
            sum += val * w;
        }
    }

    // Normalization: sum /= sqrt(width) ?
    // pywt's normalization depends on the implementation.
    // Ricker wavelet defined above already has a 1/sqrt(scale) component.
    sum
}
