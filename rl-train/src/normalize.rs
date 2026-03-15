use serde::{Deserialize, Serialize};

/// Running mean and variance tracker using Welford's online algorithm.
/// Used to normalize observations before feeding to the network.
#[derive(Clone, Serialize, Deserialize)]
pub struct RunningMeanStd {
    pub mean: Vec<f32>,
    pub var: Vec<f32>,
    pub count: f64,
    dim: usize,
}

impl RunningMeanStd {
    /// Initialize with zero mean, unit variance, count=0.
    pub fn new(dim: usize) -> Self {
        Self {
            mean: vec![0.0; dim],
            var: vec![1.0; dim],
            count: 0.0,
            dim,
        }
    }

    /// Update running stats from a batch of observations using Welford's algorithm.
    ///
    /// `batch` is `[batch_size, dim]` row-major (length = batch_size * dim).
    pub fn update_batch(&mut self, batch: &[f32], batch_size: usize) {
        debug_assert_eq!(
            batch.len(),
            batch_size * self.dim,
            "batch length {} != batch_size {} * dim {}",
            batch.len(),
            batch_size,
            self.dim,
        );

        // M2 tracks the sum of squared deviations from the current mean.
        // Reconstruct from existing var * count so we can continue accumulating.
        let mut m2: Vec<f64> = self
            .var
            .iter()
            .map(|&v| v as f64 * self.count)
            .collect();

        let mut mean: Vec<f64> = self.mean.iter().map(|&m| m as f64).collect();
        let mut count = self.count;

        for row in 0..batch_size {
            count += 1.0;
            let offset = row * self.dim;
            for d in 0..self.dim {
                let x = batch[offset + d] as f64;
                let delta = x - mean[d];
                mean[d] += delta / count;
                let delta2 = x - mean[d];
                m2[d] += delta * delta2;
            }
        }

        self.count = count;
        for d in 0..self.dim {
            self.mean[d] = mean[d] as f32;
            self.var[d] = if count > 0.0 {
                (m2[d] / count) as f32
            } else {
                1.0
            };
        }
    }

    /// Normalize a single observation: `(x - mean) / sqrt(var + 1e-8)`, clipped to [-10, 10].
    pub fn normalize(&self, input: &[f32]) -> Vec<f32> {
        debug_assert_eq!(input.len(), self.dim);
        input
            .iter()
            .zip(self.mean.iter().zip(self.var.iter()))
            .map(|(&x, (&m, &v))| ((x - m) / (v + 1e-8).sqrt()).clamp(-10.0, 10.0))
            .collect()
    }

    /// Normalize a batch of observations. Returns a new vec of the same layout.
    ///
    /// `batch` is `[batch_size, dim]` row-major.
    pub fn normalize_batch(&self, batch: &[f32], batch_size: usize) -> Vec<f32> {
        debug_assert_eq!(batch.len(), batch_size * self.dim);
        let mut out = Vec::with_capacity(batch.len());
        for row in 0..batch_size {
            let offset = row * self.dim;
            for d in 0..self.dim {
                let x = batch[offset + d];
                let normed =
                    ((x - self.mean[d]) / (self.var[d] + 1e-8).sqrt()).clamp(-10.0, 10.0);
                out.push(normed);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_initializes_correctly() {
        let rms = RunningMeanStd::new(3);
        assert_eq!(rms.mean, vec![0.0; 3]);
        assert_eq!(rms.var, vec![1.0; 3]);
        assert_eq!(rms.count, 0.0);
    }

    #[test]
    fn test_update_single_sample() {
        let mut rms = RunningMeanStd::new(2);
        rms.update_batch(&[4.0, 6.0], 1);
        assert_eq!(rms.count, 1.0);
        assert_eq!(rms.mean, vec![4.0, 6.0]);
        // Variance after 1 sample: M2/count = 0/1 = 0
        assert_eq!(rms.var, vec![0.0, 0.0]);
    }

    #[test]
    fn test_update_batch_mean_and_variance() {
        let mut rms = RunningMeanStd::new(1);
        // Feed [2, 4, 6] as 3 samples of dim=1
        rms.update_batch(&[2.0, 4.0, 6.0], 3);
        assert_eq!(rms.count, 3.0);
        // mean = (2+4+6)/3 = 4.0
        assert!((rms.mean[0] - 4.0).abs() < 1e-6);
        // population variance = ((2-4)^2 + (4-4)^2 + (6-4)^2)/3 = 8/3
        assert!((rms.var[0] - 8.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_incremental_matches_batch() {
        // Incremental (one at a time) should match batch update
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 samples, dim=2

        let mut batch_rms = RunningMeanStd::new(2);
        batch_rms.update_batch(&data, 3);

        let mut incr_rms = RunningMeanStd::new(2);
        incr_rms.update_batch(&data[0..2], 1);
        incr_rms.update_batch(&data[2..4], 1);
        incr_rms.update_batch(&data[4..6], 1);

        for d in 0..2 {
            assert!(
                (batch_rms.mean[d] - incr_rms.mean[d]).abs() < 1e-5,
                "mean mismatch at dim {d}"
            );
            assert!(
                (batch_rms.var[d] - incr_rms.var[d]).abs() < 1e-5,
                "var mismatch at dim {d}"
            );
        }
    }

    #[test]
    fn test_normalize_clips() {
        let mut rms = RunningMeanStd::new(1);
        // After a single sample of 0.0, mean=0, var=0
        rms.update_batch(&[0.0], 1);
        // Input far from mean: (1000 - 0) / sqrt(0 + 1e-8) is huge
        let normed = rms.normalize(&[1000.0]);
        assert_eq!(normed[0], 10.0); // clipped to 10
        let normed_neg = rms.normalize(&[-1000.0]);
        assert_eq!(normed_neg[0], -10.0); // clipped to -10
    }

    #[test]
    fn test_normalize_batch_matches_single() {
        let mut rms = RunningMeanStd::new(2);
        rms.update_batch(&[1.0, 3.0, 5.0, 7.0], 2);

        let batch_out = rms.normalize_batch(&[2.0, 4.0, 6.0, 8.0], 2);
        let single_0 = rms.normalize(&[2.0, 4.0]);
        let single_1 = rms.normalize(&[6.0, 8.0]);

        assert_eq!(batch_out[0], single_0[0]);
        assert_eq!(batch_out[1], single_0[1]);
        assert_eq!(batch_out[2], single_1[0]);
        assert_eq!(batch_out[3], single_1[1]);
    }
}
