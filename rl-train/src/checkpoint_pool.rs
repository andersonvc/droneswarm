use std::collections::VecDeque;

/// Pool of serialized model checkpoints for diverse self-play opponents.
///
/// Maintains a fixed-size ring buffer of serialized model JSONs.
/// Each environment samples its opponent from the pool on episode reset.
pub struct CheckpointPool {
    /// Serialized model JSONs (most recent at back).
    checkpoints: VecDeque<String>,
    /// Maximum number of checkpoints to retain.
    max_size: usize,
}

impl CheckpointPool {
    /// Create an empty pool with the given capacity.
    pub fn new(max_size: usize) -> Self {
        Self {
            checkpoints: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    /// Add a checkpoint. If at capacity, remove the oldest (front).
    pub fn push(&mut self, model_json: String) {
        if self.checkpoints.len() == self.max_size {
            self.checkpoints.pop_front();
        }
        self.checkpoints.push_back(model_json);
    }

    /// Sample a random checkpoint from the pool.
    ///
    /// Uses simple `seed % len` for deterministic per-env selection.
    /// Returns `None` if the pool is empty.
    pub fn sample(&self, rng_seed: u64) -> Option<&str> {
        if self.checkpoints.is_empty() {
            return None;
        }
        let idx = ((rng_seed.wrapping_mul(2654435769) >> 16) % self.checkpoints.len() as u64) as usize;
        Some(&self.checkpoints[idx])
    }

    /// Number of checkpoints in the pool.
    pub fn len(&self) -> usize {
        self.checkpoints.len()
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.checkpoints.is_empty()
    }

    /// Get the most recently added checkpoint (back of deque).
    pub fn latest(&self) -> Option<&str> {
        self.checkpoints.back().map(|s| s.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_pool_returns_none() {
        let pool = CheckpointPool::new(5);
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);
        assert!(pool.sample(42).is_none());
        assert!(pool.latest().is_none());
    }

    #[test]
    fn push_and_retrieve() {
        let mut pool = CheckpointPool::new(3);
        pool.push("ckpt_0".into());
        pool.push("ckpt_1".into());

        assert_eq!(pool.len(), 2);
        assert_eq!(pool.latest(), Some("ckpt_1"));
        assert!(pool.sample(0).is_some());
    }

    #[test]
    fn evicts_oldest_at_capacity() {
        let mut pool = CheckpointPool::new(2);
        pool.push("a".into());
        pool.push("b".into());
        pool.push("c".into());

        assert_eq!(pool.len(), 2);
        // "a" was evicted; only "b" and "c" remain
        assert_eq!(pool.sample(0), Some("b"));
        assert_eq!(pool.sample(1), Some("c"));
        assert_eq!(pool.latest(), Some("c"));
    }

    #[test]
    fn deterministic_sampling() {
        let mut pool = CheckpointPool::new(5);
        for i in 0..5 {
            pool.push(format!("ckpt_{i}"));
        }
        // Same seed always returns the same checkpoint
        let a = pool.sample(77);
        let b = pool.sample(77);
        assert_eq!(a, b);
    }
}
