//! Functional-equivalence cache: per-task scores keyed by (fingerprint, task id).
//!
//! Owner: WP1. Contract section 5.5. Pure: no I/O.
//!
//! Layout: a hash map from key to slot, plus an intrusive doubly linked list over a slot
//! vector in LRU order (head = least recently used). Get, touch, insert and evict are O(1).
//! The map is never iterated, so its hash seed cannot influence any result.
#![forbid(unsafe_code)]

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Empty link.
const NIL: u32 = u32::MAX;

/// Largest usable capacity (slot indices are u32 and NIL is reserved).
const MAX_CAPACITY: usize = (u32::MAX - 1) as usize;

/// One cached score in the LRU list.
#[derive(Clone, Copy, Debug)]
struct Node {
    fp: u64,
    task: u64,
    aulc: f32,
    prev: u32,
    next: u32,
}

/// A change waiting for `commit`.
#[derive(Clone, Copy, Debug)]
enum Staged {
    Touch(u64, u64),
    Insert(u64, u64, f32),
}

/// Per-task score cache keyed by (fingerprint, task id) -> AULC.
/// Reads during a batch are immutable; writes and LRU touches are staged and applied by
/// `commit` in slot order, so results do not depend on thread scheduling.
pub struct ScoreCache {
    capacity: usize,
    map: HashMap<(u64, u64), u32>,
    nodes: Vec<Node>,
    free: Vec<u32>,
    head: u32,
    tail: u32,
    staged: Vec<Staged>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CacheSnapshot {
    pub capacity: usize,
    /// Entries in LRU order, oldest first: (fingerprint, task id, aulc).
    pub entries: Vec<(u64, u64, f32)>,
}

impl ScoreCache {
    /// An empty cache. Capacity 0 caches nothing.
    pub fn new(capacity: usize) -> Self {
        ScoreCache {
            capacity: capacity.min(MAX_CAPACITY),
            map: HashMap::new(),
            nodes: Vec::new(),
            free: Vec::new(),
            head: NIL,
            tail: NIL,
            staged: Vec::new(),
        }
    }

    /// Committed score of (fp, task). Does not change the LRU order (use `stage_touch`).
    pub fn get(&self, fp: u64, task: u64) -> Option<f32> {
        let slot = *self.map.get(&(fp, task))?;

        self.nodes.get(slot as usize).map(|n| n.aulc)
    }

    /// Mark (fp, task) as used; applied at the next `commit`.
    pub fn stage_touch(&mut self, fp: u64, task: u64) {
        self.staged.push(Staged::Touch(fp, task));
    }

    /// Store a score; applied at the next `commit`. Non-finite scores are never cached (I8).
    pub fn stage_insert(&mut self, fp: u64, task: u64, aulc: f32) {
        if aulc.is_finite() {
            self.staged.push(Staged::Insert(fp, task, aulc));
        }
    }

    /// Apply staged touches/inserts in staging order; evict least recently used entries.
    pub fn commit(&mut self) {
        let staged = std::mem::take(&mut self.staged);

        for change in &staged {
            match *change {
                Staged::Touch(fp, task) => self.touch(fp, task),
                Staged::Insert(fp, task, aulc) => self.insert(fp, task, aulc),
            }
        }

        // Keep the allocation for the next batch.
        self.staged = staged;
        self.staged.clear();
    }

    /// Committed entries.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// True when the cache holds no committed entries (errata E5: clippy len_without_is_empty).
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Committed entries in LRU order, oldest first. Staged changes are not included.
    pub fn snapshot(&self) -> CacheSnapshot {
        let mut entries = Vec::with_capacity(self.len());
        let mut at = self.head;

        while let Some(node) = self.nodes.get(at as usize) {
            entries.push((node.fp, node.task, node.aulc));
            at = node.next;
        }

        CacheSnapshot {
            capacity: self.capacity,
            entries,
        }
    }

    /// Rebuild a cache by inserting the entries in order: duplicates keep the last score,
    /// non-finite scores are skipped, and only the newest `capacity` entries survive.
    pub fn from_snapshot(s: CacheSnapshot) -> Self {
        let mut cache = ScoreCache::new(s.capacity);

        for (fp, task, aulc) in s.entries {
            if aulc.is_finite() {
                cache.insert(fp, task, aulc);
            }
        }

        cache
    }

    fn touch(&mut self, fp: u64, task: u64) {
        if let Some(&slot) = self.map.get(&(fp, task)) {
            self.unlink(slot);
            self.push_back(slot);
        }
    }

    fn insert(&mut self, fp: u64, task: u64, aulc: f32) {
        if self.capacity == 0 {
            return;
        }

        if let Some(&slot) = self.map.get(&(fp, task)) {
            if let Some(node) = self.nodes.get_mut(slot as usize) {
                node.aulc = aulc;
            }

            self.unlink(slot);
            self.push_back(slot);

            return;
        }

        if self.map.len() >= self.capacity {
            self.evict_oldest();
        }

        let node = Node {
            fp,
            task,
            aulc,
            prev: NIL,
            next: NIL,
        };

        let slot = match self.free.pop() {
            Some(slot) => {
                self.nodes[slot as usize] = node;

                slot
            }

            None => {
                self.nodes.push(node);

                (self.nodes.len() - 1) as u32
            }
        };

        self.map.insert((fp, task), slot);
        self.push_back(slot);
    }

    fn evict_oldest(&mut self) {
        let slot = self.head;
        let Some(node) = self.nodes.get(slot as usize).copied() else {
            return;
        };

        self.unlink(slot);
        self.map.remove(&(node.fp, node.task));
        self.free.push(slot);
    }

    /// Remove `slot` from the list (it stays allocated).
    fn unlink(&mut self, slot: u32) {
        let Some(node) = self.nodes.get(slot as usize).copied() else {
            return;
        };

        match self.nodes.get_mut(node.prev as usize) {
            Some(prev) => prev.next = node.next,
            None => self.head = node.next,
        }

        match self.nodes.get_mut(node.next as usize) {
            Some(next) => next.prev = node.prev,
            None => self.tail = node.prev,
        }

        if let Some(n) = self.nodes.get_mut(slot as usize) {
            n.prev = NIL;
            n.next = NIL;
        }
    }

    /// Append `slot` as the most recently used entry.
    fn push_back(&mut self, slot: u32) {
        let old_tail = self.tail;

        if let Some(n) = self.nodes.get_mut(slot as usize) {
            n.prev = old_tail;
            n.next = NIL;
        }

        match self.nodes.get_mut(old_tail as usize) {
            Some(t) => t.next = slot,
            None => self.head = slot,
        }

        self.tail = slot;
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use super::*;

    fn keys(c: &ScoreCache) -> Vec<(u64, u64)> {
        c.snapshot().entries.iter().map(|e| (e.0, e.1)).collect()
    }

    /// T-FEC-1: commit order determinism, LRU eviction at capacity, snapshot round trip.
    #[test]
    fn t_fec_1_order_eviction_snapshot() {
        let mut c = ScoreCache::new(3);

        assert!(c.is_empty());

        // Staged changes are invisible until commit.
        c.stage_insert(1, 10, 0.1);
        c.stage_insert(2, 20, 0.2);
        c.stage_insert(3, 30, 0.3);

        assert_eq!(c.get(1, 10), None);

        c.commit();

        assert_eq!(c.len(), 3);
        assert_eq!(c.get(2, 20), Some(0.2));
        assert_eq!(keys(&c), vec![(1, 10), (2, 20), (3, 30)]);

        // A touch makes (1, 10) the newest; the next insert evicts (2, 20).
        c.stage_touch(1, 10);
        c.stage_insert(4, 40, 0.4);
        c.commit();

        assert_eq!(keys(&c), vec![(3, 30), (1, 10), (4, 40)]);
        assert_eq!(c.get(2, 20), None);

        // Re-inserting an existing key updates it and makes it newest; no eviction.
        c.stage_insert(3, 30, 0.35);
        c.stage_touch(9, 99);
        c.commit();

        assert_eq!(keys(&c), vec![(1, 10), (4, 40), (3, 30)]);
        assert_eq!(c.get(3, 30), Some(0.35));

        // Same key, different task id: a different entry.
        c.stage_insert(1, 11, 0.5);
        c.commit();

        assert_eq!(keys(&c), vec![(4, 40), (3, 30), (1, 11)]);

        // Snapshot round trip, also through JSON.
        let snap = c.snapshot();
        let text = serde_json::to_string(&snap).expect("serialise");
        let back: CacheSnapshot = serde_json::from_str(&text).expect("parse");
        let restored = ScoreCache::from_snapshot(back.clone());

        assert_eq!(back, snap);
        assert_eq!(restored.snapshot(), snap);

        // Restoring into a smaller capacity keeps the newest entries.
        let small = ScoreCache::from_snapshot(CacheSnapshot {
            capacity: 2,
            entries: snap.entries.clone(),
        });

        assert_eq!(keys(&small), vec![(3, 30), (1, 11)]);
    }

    /// Two caches fed the same staged changes in the same order end identical, no matter
    /// how the reads were interleaved (reads never change the cache).
    #[test]
    fn commit_is_deterministic_and_reads_are_pure() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut a = ScoreCache::new(500);
        let mut b = ScoreCache::new(500);

        for _ in 0..50 {
            let batch: Vec<(bool, u64, u64, f32)> = (0..200)
                .map(|_| {
                    (
                        rng.random_bool(0.3),
                        rng.random_range(0..40),
                        rng.random_range(0..40),
                        rng.random(),
                    )
                })
                .collect();

            // Cache b is read a lot during the batch; a is not read at all.
            for &(_, fp, task, _) in &batch {
                let _ = b.get(fp, task);
            }

            for cache in [&mut a, &mut b] {
                for &(touch, fp, task, aulc) in &batch {
                    if touch {
                        cache.stage_touch(fp, task);
                    } else {
                        cache.stage_insert(fp, task, aulc);
                    }
                }

                cache.commit();
            }

            assert_eq!(a.snapshot(), b.snapshot());
            assert!(a.len() <= 500);
        }

        assert_eq!(a.len(), 500);
    }

    /// The linked list agrees with a naive model after many random operations.
    #[test]
    fn matches_a_naive_lru_model() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut cache = ScoreCache::new(17);
        let mut model: Vec<(u64, u64, f32)> = Vec::new();

        for _ in 0..20_000 {
            let key = (rng.random_range(0..30u64), rng.random_range(0..3u64));

            if rng.random_bool(0.4) {
                cache.stage_touch(key.0, key.1);

                if let Some(pos) = model.iter().position(|e| (e.0, e.1) == key) {
                    let e = model.remove(pos);

                    model.push(e);
                }
            } else {
                let aulc: f32 = rng.random();

                cache.stage_insert(key.0, key.1, aulc);

                if let Some(pos) = model.iter().position(|e| (e.0, e.1) == key) {
                    model.remove(pos);
                } else if model.len() == 17 {
                    model.remove(0);
                }

                model.push((key.0, key.1, aulc));
            }

            if rng.random_bool(0.1) {
                cache.commit();

                assert_eq!(cache.snapshot().entries, model);
            }
        }
    }

    #[test]
    fn edge_cases() {
        let mut zero = ScoreCache::new(0);

        zero.stage_insert(1, 1, 0.5);
        zero.commit();

        assert!(zero.is_empty());
        assert_eq!(zero.get(1, 1), None);

        // Non-finite scores are never cached (and never reach a snapshot).
        let mut c = ScoreCache::new(4);

        c.stage_insert(1, 1, f32::NAN);
        c.stage_insert(2, 2, f32::INFINITY);
        c.commit();

        assert!(c.is_empty());

        let restored = ScoreCache::from_snapshot(CacheSnapshot {
            capacity: 4,
            entries: vec![(1, 1, f32::NAN), (2, 2, 0.5), (2, 2, 0.75)],
        });

        assert_eq!(restored.snapshot().entries, vec![(2, 2, 0.75)]);

        // Touching a missing key does nothing.
        c.stage_touch(5, 5);
        c.commit();

        assert!(c.is_empty());

        // Huge capacities are clamped instead of overflowing slot indices.
        assert_eq!(
            ScoreCache::new(usize::MAX).snapshot().capacity,
            MAX_CAPACITY
        );
    }
}
