//! Weight initialization schemes for neural network layers.

use rand::Rng;

/// Orthogonal weight initialization (simplified QR via Gram-Schmidt).
///
/// Generates an (out_dim × in_dim) weight matrix where rows are orthonormal
/// (scaled by `gain`). This is the standard initialization for PPO networks.
pub fn orthogonal_init(in_dim: usize, out_dim: usize, gain: f32, rng: &mut impl Rng) -> Vec<f32> {
    let rows = out_dim;
    let cols = in_dim;
    let n = rows.max(cols);

    // Generate random matrix.
    let mut a: Vec<f32> = (0..n * cols).map(|_| rng.gen_range(-1.0..1.0f32)).collect();

    // Gram-Schmidt orthogonalization on the first `rows` row vectors.
    for i in 0..rows.min(n) {
        // Subtract projections onto previous rows.
        for j in 0..i {
            let mut dot = 0.0f32;
            for k in 0..cols {
                dot += a[i * cols + k] * a[j * cols + k];
            }
            for k in 0..cols {
                a[i * cols + k] -= dot * a[j * cols + k];
            }
        }
        // Normalize.
        let norm: f32 = (0..cols).map(|k| a[i * cols + k] * a[i * cols + k]).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for k in 0..cols {
                a[i * cols + k] /= norm;
            }
        }
    }

    // Extract the first `rows` rows and scale by gain.
    let mut weights = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for k in 0..cols {
            weights[i * cols + k] = a[i * cols + k] * gain;
        }
    }
    weights
}