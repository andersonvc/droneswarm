//! Weight initialization schemes for neural network layers.

use rand::Rng;

/// Orthogonal weight initialization (simplified QR via Gram-Schmidt).
///
/// Generates an (out_dim x in_dim) weight matrix where rows are orthonormal
/// (scaled by `gain`). This is the standard initialization for PPO networks.
///
/// When out_dim > in_dim, we generate an (in_dim x out_dim) random matrix,
/// orthogonalize its in_dim rows (each living in out_dim-dimensional space,
/// so there is always room), scale by gain, then transpose to get the
/// (out_dim x in_dim) weight matrix. This avoids dead rows that occur when
/// trying to orthogonalize more vectors than the dimension allows.
pub fn orthogonal_init(in_dim: usize, out_dim: usize, gain: f32, rng: &mut impl Rng) -> Vec<f32> {
    if out_dim <= in_dim {
        // Standard case: out_dim rows in in_dim-dimensional space.
        // We can orthogonalize up to in_dim rows, so out_dim <= in_dim is fine.
        let rows = out_dim;
        let cols = in_dim;

        let mut a: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(-1.0..1.0f32)).collect();

        gram_schmidt(&mut a, rows, cols);

        // Scale by gain.
        for i in 0..rows {
            for k in 0..cols {
                a[i * cols + k] *= gain;
            }
        }
        a
    } else {
        // Tall case: out_dim > in_dim.
        // Generate (in_dim x out_dim) matrix — in_dim rows of out_dim elements.
        // Orthogonalize in_dim rows in out_dim-dimensional space (always room).
        // Then transpose to get (out_dim x in_dim).
        let rows = in_dim;
        let cols = out_dim;

        let mut a: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(-1.0..1.0f32)).collect();

        gram_schmidt(&mut a, rows, cols);

        // Scale by gain.
        for v in a.iter_mut() {
            *v *= gain;
        }

        // Transpose (in_dim x out_dim) -> (out_dim x in_dim).
        let mut weights = vec![0.0f32; out_dim * in_dim];
        for i in 0..in_dim {
            for j in 0..out_dim {
                weights[j * in_dim + i] = a[i * out_dim + j];
            }
        }
        weights
    }
}

/// Gram-Schmidt orthonormalization on `num_rows` row vectors, each of length
/// `num_cols`, stored row-major in `a`.
fn gram_schmidt(a: &mut [f32], num_rows: usize, num_cols: usize) {
    for i in 0..num_rows {
        // Subtract projections onto previous rows.
        for j in 0..i {
            let mut dot = 0.0f32;
            for k in 0..num_cols {
                dot += a[i * num_cols + k] * a[j * num_cols + k];
            }
            for k in 0..num_cols {
                a[i * num_cols + k] -= dot * a[j * num_cols + k];
            }
        }
        // Normalize.
        let norm: f32 = (0..num_cols)
            .map(|k| a[i * num_cols + k] * a[i * num_cols + k])
            .sum::<f32>()
            .sqrt();
        if norm > 1e-8 {
            for k in 0..num_cols {
                a[i * num_cols + k] /= norm;
            }
        }
    }
}