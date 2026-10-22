//! ARC grids as per-cell learning tasks: phi32 features, pass@2 solving and scoring.
//!
//! Owner: WP3. Contract section 5.12, errata E3 and E9. Pure: no I/O.
//!
//! A cell task maps the 32 hand-made features of one input cell to the colour of the output
//! cell at the same place. phi32 is hand-made prior knowledge and is reported as such.
#![forbid(unsafe_code)]

use std::collections::HashMap;

use sha2::{Digest, Sha256};

use crate::learner::protocol::run_learner_with_outputs;
use crate::learner::types::{
    ArcPair, ArcTask, EvalSettings, FamilyId, Grid, Group, Labels, LearnKind, LearnerGenome,
    TaskInstance,
};
use crate::util::derive_seed;

/// Feature width of a cell task (one of SUPPORTED_F).
const ARC_F: usize = 32;

/// ARC colours 0..=9; also the number of classes of a cell task.
const N_COLOURS: usize = 10;

/// Largest grid side.
const MAX_SIDE: usize = 30;

/// Stream length bounds: steps = clamp(3 * n_train_cells, MIN_STEPS, MAX_STEPS).
const MIN_STEPS: usize = 384;
const MAX_STEPS: usize = 8192;

/// Neighbour offsets in phi32 order: (dr, dc) over (-1, 0, 1)^2 without (0, 0).
const NEIGHBOURS: [(isize, isize); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

/// First bit of the "neighbour equals centre" block.
const SAME_BIT: usize = 10;

/// First bit of the "neighbour non-zero" block.
const NONZERO_BIT: usize = 18;

/// First bit of the position block (r%2, c%2, r==0, r==h-1, c==0, c==w-1).
const POS_BIT: usize = 26;

/// phi32 features of cell (r, c): one-hot centre colour (10); 8 "neighbour equals centre"
/// bits; 8 "neighbour non-zero" bits (neighbours in order (dr,dc) over (-1,0,1)^2 without
/// (0,0); outside the grid counts as 0); then r%2, c%2, r==0, r==h-1, c==0, c==w-1.
///
/// "Outside counts as 0" means both neighbour bits are 0 there, as in the prototype
/// `arc_cell_ceiling.py`. A cell outside the grid, or a malformed grid, gives all zeros.
pub fn phi32(g: &Grid, r: usize, c: usize) -> [f32; 32] {
    let mut out = [0.0f32; 32];

    expand_key(phi32_key(g, r, c), &mut out);

    out
}

/// phi32 as a bit set: bit i is feature i (every feature is 0 or 1).
fn phi32_key(g: &Grid, r: usize, c: usize) -> u32 {
    let h = g.h as usize;
    let w = g.w as usize;

    if r >= h || c >= w || g.cells.len() != h * w {
        return 0;
    }

    let centre = g.cells[r * w + c];
    let mut key = 0u32;

    if (centre as usize) < N_COLOURS {
        key |= 1 << centre;
    }

    for (n, &(dr, dc)) in NEIGHBOURS.iter().enumerate() {
        let (Some(rr), Some(cc)) = (r.checked_add_signed(dr), c.checked_add_signed(dc)) else {
            continue;
        };

        if rr >= h || cc >= w {
            continue;
        }

        let colour = g.cells[rr * w + cc];

        if colour == centre {
            key |= 1 << (SAME_BIT + n);
        }

        if colour != 0 {
            key |= 1 << (NONZERO_BIT + n);
        }
    }

    let flags = [
        r % 2 == 1,
        c % 2 == 1,
        r == 0,
        r == h - 1,
        c == 0,
        c == w - 1,
    ];

    for (n, &on) in flags.iter().enumerate() {
        key |= (on as u32) << (POS_BIT + n);
    }

    key
}

/// Write the 32 bits of `key` as 0.0 / 1.0 into `out` (length 32).
fn expand_key(key: u32, out: &mut [f32]) {
    for (i, x) in out.iter_mut().enumerate() {
        *x = ((key >> i) & 1) as f32;
    }
}

/// True when every train pair keeps its shape (the solver predicts output shape = input).
/// A task without train pairs gives false: there is nothing to learn from.
pub fn train_keeps_shape(t: &ArcTask) -> bool {
    !t.train.is_empty() && t.train.iter().all(same_shape)
}

/// Input and output have the same height and width.
fn same_shape(p: &ArcPair) -> bool {
    p.input.h == p.output.h && p.input.w == p.output.w
}

/// 1..=30 on both sides, h * w cells, colours 0..=9.
fn grid_ok(g: &Grid) -> bool {
    let h = g.h as usize;
    let w = g.w as usize;

    (1..=MAX_SIDE).contains(&h)
        && (1..=MAX_SIDE).contains(&w)
        && g.cells.len() == h * w
        && g.cells.iter().all(|&v| (v as usize) < N_COLOURS)
}

/// At least one train and one test pair, and every grid well-formed.
fn task_ok(t: &ArcTask) -> bool {
    !t.train.is_empty()
        && !t.test.is_empty()
        && t.train
            .iter()
            .chain(&t.test)
            .all(|p| grid_ok(&p.input) && grid_ok(&p.output))
}

/// Training cells of a task: train pairs in order, cells row-major.
struct TrainCells {
    /// Row-major n x ARC_F phi32 rows of the input cells.
    x: Vec<f32>,
    /// Output colour of each cell.
    y: Vec<u8>,
    n: usize,
}

/// Build the training cells. The caller has checked `task_ok` and `train_keeps_shape`.
fn train_cells(t: &ArcTask) -> TrainCells {
    let n: usize = t.train.iter().map(|p| p.input.cells.len()).sum();
    let mut x = vec![0.0f32; n * ARC_F];
    let mut y = Vec::with_capacity(n);
    let mut offset = 0;

    for p in &t.train {
        let cells = p.input.cells.len();

        write_rows(&p.input, &mut x[offset * ARC_F..(offset + cells) * ARC_F]);
        y.extend_from_slice(&p.output.cells);

        offset += cells;
    }

    TrainCells { x, y, n }
}

/// phi32 rows of every cell of `g`, row-major, into `out` (length h * w * ARC_F).
fn write_rows(g: &Grid, out: &mut [f32]) {
    let w = (g.w as usize).max(1);

    for (i, row) in out.chunks_exact_mut(ARC_F).enumerate() {
        expand_key(phi32_key(g, i / w, i % w), row);
    }
}

/// First 8 bytes (LE) of SHA-256 over "family|seed|f|n_train|n_valid|level" (section 4).
fn instance_id(
    family: &FamilyId,
    seed: u64,
    f: usize,
    n_train: usize,
    n_valid: usize,
    level: u8,
) -> u64 {
    let text = format!("{}|{seed}|{f}|{n_train}|{n_valid}|{level}", family.0);
    let digest = Sha256::digest(text.as_bytes());
    let mut first = [0u8; 8];

    first.copy_from_slice(&digest[..8]);

    u64::from_le_bytes(first)
}

/// The cell task of test pair `j`, or None when that pair changes shape.
/// The caller has checked `task_ok` and `train_keeps_shape`.
fn pair_task(t: &ArcTask, j: usize, train: &TrainCells) -> Option<TaskInstance> {
    let pair = t.test.get(j)?;

    if !same_shape(pair) {
        return None;
    }

    let n_valid = pair.input.cells.len();
    let mut x_valid = vec![0.0f32; n_valid * ARC_F];

    write_rows(&pair.input, &mut x_valid);

    let changed: Vec<bool> = pair
        .input
        .cells
        .iter()
        .zip(&pair.output.cells)
        .map(|(a, b)| a != b)
        .collect();

    // Errata E9: seed tag "arc-cell/<id>/<j>", family "arc-cell/<id>", level 0.
    let family = FamilyId(format!("arc-cell/{}", t.id));
    let seed = derive_seed(0, &format!("arc-cell/{}/{j}", t.id));
    let steps = train.n.saturating_mul(3).clamp(MIN_STEPS, MAX_STEPS);
    let id = instance_id(&family, seed, ARC_F, train.n, n_valid, 0);

    Some(TaskInstance {
        id,
        family,
        group: Group::ArcCell,
        f: ARC_F,
        kind: LearnKind::Multiclass { k: N_COLOURS as u8 },
        x_train: train.x.clone(),
        y_train: Labels::Classes(train.y.clone()),
        y_train_late: None,
        drift_at: None,
        x_valid,
        y_valid: Labels::Classes(pair.output.cells.clone()),
        n_train: train.n,
        n_valid,
        steps,
        seed,
        changed_mask: Some(changed),
    })
}

/// One TaskInstance per test pair: F = 32, Multiclass{k: 10}, train = all cells of all
/// train pairs (input phi32 -> output colour), valid = cells of that test input with the
/// test output as labels, steps = clamp(3 * n_train_cells, 384, 8192),
/// changed_mask = test output != test input. None when !train_keeps_shape(t).
///
/// Also None when a test pair changes shape (its labels cannot line up with its cells;
/// this never happens in the pinned evaluation sets) or when a grid is malformed.
pub fn cell_tasks(t: &ArcTask) -> Option<Vec<TaskInstance>> {
    if !task_ok(t) || !train_keeps_shape(t) {
        return None;
    }

    let train = train_cells(t);

    (0..t.test.len()).map(|j| pair_task(t, j, &train)).collect()
}

#[derive(Clone, Debug, PartialEq)]
pub struct ArcResult {
    pub task_id: String,
    /// Per test pair: whether attempt 0 or attempt 1 matched exactly.
    pub correct: Vec<bool>,
    /// Balanced cell accuracy per test pair with both changed and unchanged cells.
    pub bca: Vec<f64>,
}

/// Two attempts (interpreter attempts 0 and 1), argmax colour per cell, exact grid match.
/// Tasks that change shape yield all-false.
///
/// Attempt 1 runs only when attempt 0 missed. BCA uses attempt 0; an aborted attempt
/// predicts nothing and scores 0 there.
pub fn solve_pass2(
    g: &LearnerGenome,
    t: &ArcTask,
    settings: &EvalSettings,
    stop: &dyn Fn() -> bool,
) -> ArcResult {
    solve_with(t, |task, attempt| {
        if stop() {
            return None;
        }

        let (result, outputs) = run_learner_with_outputs(g, task, attempt, settings, stop);

        match result.abort {
            Some(_) => None,
            None => Some(outputs),
        }
    })
}

/// The pass@2 driver behind `solve_pass2`. `attempt(task, a)` returns the linked
/// validation outputs of attempt `a` (n_valid rows of 10 values), or None when it aborted.
pub(crate) fn solve_with<A>(t: &ArcTask, mut attempt: A) -> ArcResult
where
    A: FnMut(&TaskInstance, u32) -> Option<Vec<f32>>,
{
    let mut result = ArcResult {
        task_id: t.id.clone(),
        correct: vec![false; t.test.len()],
        bca: Vec::new(),
    };

    if !task_ok(t) || !train_keeps_shape(t) {
        return result;
    }

    let train = train_cells(t);

    for (j, pair) in t.test.iter().enumerate() {
        let Some(task) = pair_task(t, j, &train) else {
            continue;
        };

        let target = pair.output.cells.as_slice();
        let first = attempt(&task, 0).and_then(|out| predicted_grid(&out, task.n_valid));
        let mut solved = first.as_deref() == Some(target);

        if !solved {
            let second = attempt(&task, 1).and_then(|out| predicted_grid(&out, task.n_valid));

            solved = second.as_deref() == Some(target);
        }

        result.correct[j] = solved;

        if let Some(bca) = balanced_accuracy(first.as_deref(), pair) {
            result.bca.push(bca);
        }
    }

    result
}

/// Argmax colour per cell (lowest index on ties; NaN never wins).
/// None when `outputs` is not n_cells rows of 10 values.
fn predicted_grid(outputs: &[f32], n_cells: usize) -> Option<Vec<u8>> {
    if outputs.len() != n_cells * N_COLOURS {
        return None;
    }

    let grid = outputs
        .chunks_exact(N_COLOURS)
        .map(|row| {
            let mut best = 0;

            for (k, &v) in row.iter().enumerate().skip(1) {
                if v > row[best] || (row[best].is_nan() && !v.is_nan()) {
                    best = k;
                }
            }

            best as u8
        })
        .collect();

    Some(grid)
}

/// 0.5 * accuracy on unchanged cells + 0.5 * accuracy on changed cells.
/// None unless the pair has both kinds; a missing prediction scores 0.
fn balanced_accuracy(pred: Option<&[u8]>, pair: &ArcPair) -> Option<f64> {
    let mut hits = [0usize; 2];
    let mut totals = [0usize; 2];

    for (i, (&a, &b)) in pair.input.cells.iter().zip(&pair.output.cells).enumerate() {
        let changed = (a != b) as usize;

        totals[changed] += 1;

        if pred.and_then(|p| p.get(i)) == Some(&b) {
            hits[changed] += 1;
        }
    }

    if totals[0] == 0 || totals[1] == 0 {
        return None;
    }

    let unchanged = hits[0] as f64 / totals[0] as f64;
    let changed = hits[1] as f64 / totals[1] as f64;

    Some(0.5 * unchanged + 0.5 * changed)
}

/// Mean over ALL n_tasks of (correct outputs / test outputs); missing tasks count 0.
/// The denominator is at least `results.len()`, so the score stays in [0, 1].
pub fn score_official(results: &[ArcResult], n_tasks: usize) -> f64 {
    let denom = n_tasks.max(results.len());

    if denom == 0 {
        return 0.0;
    }

    let total: f64 = results
        .iter()
        .filter(|r| !r.correct.is_empty())
        .map(|r| solved_outputs(r) as f64 / r.correct.len() as f64)
        .sum();

    total / denom as f64
}

/// Correct outputs / all test outputs.
/// The denominator is at least the number of outputs in `results`.
pub fn score_kaggle(results: &[ArcResult], n_test_outputs: usize) -> f64 {
    let present: usize = results.iter().map(|r| r.correct.len()).sum();
    let denom = n_test_outputs.max(present);

    if denom == 0 {
        return 0.0;
    }

    let solved: usize = results.iter().map(solved_outputs).sum();

    solved as f64 / denom as f64
}

/// Number of exactly matched test outputs of one task.
fn solved_outputs(r: &ArcResult) -> usize {
    r.correct.iter().filter(|&&c| c).count()
}

/// Mean balanced cell accuracy over every scored test pair; None when there is none.
/// Non-finite entries (impossible from `solve_pass2`) are skipped.
pub fn mean_bca(results: &[ArcResult]) -> Option<f64> {
    let mut sum = 0.0;
    let mut n = 0usize;

    for &b in results.iter().flat_map(|r| &r.bca) {
        if b.is_finite() {
            sum += b;
            n += 1;
        }
    }

    if n == 0 {
        return None;
    }

    Some(sum / n as f64)
}

/// SHA-256 of the lexicographically smallest serialisation over the 8 D4 transforms
/// (applied to input and output together) with colours relabelled by first appearance
/// (0 kept as 0) across input then output.
///
/// Serialisation: h, w and the cells of the input, then h, w and the cells of the output,
/// one byte each.
pub fn canonical_pair_hash(p: &ArcPair) -> [u8; 32] {
    let len = 4 + p.input.cells.len() + p.output.cells.len();
    let mut best: Vec<u8> = Vec::with_capacity(len);
    let mut current: Vec<u8> = Vec::with_capacity(len);

    for t in 0..8u8 {
        let mut labels = Relabel::new();

        current.clear();
        push_transformed(&p.input, t, &mut labels, &mut current);
        push_transformed(&p.output, t, &mut labels, &mut current);

        if t == 0 || current < best {
            std::mem::swap(&mut best, &mut current);
        }
    }

    Sha256::digest(&best).into()
}

/// Colour relabelling by first appearance; 0 stays 0.
struct Relabel {
    map: [u16; 256],
    next: u16,
}

impl Relabel {
    fn new() -> Self {
        let mut map = [u16::MAX; 256];

        map[0] = 0;

        Relabel { map, next: 1 }
    }

    fn label(&mut self, colour: u8) -> u8 {
        let slot = &mut self.map[colour as usize];

        if *slot == u16::MAX {
            *slot = self.next;
            self.next = self.next.saturating_add(1);
        }

        (*slot).min(u8::MAX as u16) as u8
    }
}

/// Append `g` under D4 transform `t`: dims first, then the relabelled cells row-major.
fn push_transformed(g: &Grid, t: u8, labels: &mut Relabel, out: &mut Vec<u8>) {
    let h = g.h as usize;
    let w = g.w as usize;
    let (th, tw) = d4_dims(t, h, w);

    out.push(th as u8);
    out.push(tw as u8);

    for i in 0..th {
        for j in 0..tw {
            let (r, c) = d4_source(t, i, j, h, w);
            let colour = g.cells.get(r * w + c).copied().unwrap_or(0);

            out.push(labels.label(colour));
        }
    }
}

/// Shape of an h x w grid after D4 transform `t`.
fn d4_dims(t: u8, h: usize, w: usize) -> (usize, usize) {
    match t {
        1 | 3 | 6 | 7 => (w, h),
        _ => (h, w),
    }
}

/// Source cell (row, col) of target cell (i, j) under D4 transform `t` of an h x w grid.
/// Only called with (i, j) inside the target shape, so h, w >= 1.
fn d4_source(t: u8, i: usize, j: usize, h: usize, w: usize) -> (usize, usize) {
    match t {
        // Identity.
        0 => (i, j),
        // Rotate 90 degrees clockwise.
        1 => (h - 1 - j, i),
        // Rotate 180 degrees.
        2 => (h - 1 - i, w - 1 - j),
        // Rotate 270 degrees clockwise.
        3 => (j, w - 1 - i),
        // Mirror left-right.
        4 => (i, w - 1 - j),
        // Mirror up-down.
        5 => (h - 1 - i, j),
        // Transpose.
        6 => (j, i),
        // Anti-transpose.
        _ => (h - 1 - j, w - 1 - i),
    }
}

/// Port of arc_cell_ceiling.py "phi32 ceiling": all pairs (train and test) keep shape and
/// the map phi32-key -> output colour is consistent over train+test. Test helper.
pub fn phi32_ceiling(t: &ArcTask) -> bool {
    if !task_ok(t) || !t.train.iter().chain(&t.test).all(same_shape) {
        return false;
    }

    let mut table: HashMap<u32, u8> = HashMap::new();

    for p in t.train.iter().chain(&t.test) {
        let w = p.input.w as usize;

        for (i, &colour) in p.output.cells.iter().enumerate() {
            let key = phi32_key(&p.input, i / w, i % w);

            if *table.entry(key).or_insert(colour) != colour {
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::BTreeSet;

    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn grid(rows: &[&[u8]]) -> Grid {
        Grid {
            h: rows.len() as u8,
            w: rows[0].len() as u8,
            cells: rows.iter().flat_map(|r| r.iter().copied()).collect(),
        }
    }

    fn pair(input: &[&[u8]], output: &[&[u8]]) -> ArcPair {
        ArcPair {
            input: grid(input),
            output: grid(output),
        }
    }

    fn task(id: &str, train: Vec<ArcPair>, test: Vec<ArcPair>) -> ArcTask {
        ArcTask {
            id: id.to_string(),
            train,
            test,
        }
    }

    /// Indices of the features that are 1; asserts every feature is 0 or 1.
    fn on_bits(v: &[f32; 32]) -> Vec<usize> {
        assert!(v.iter().all(|&x| x == 0.0 || x == 1.0), "{v:?}");

        (0..32).filter(|&i| v[i] == 1.0).collect()
    }

    /// The 3 x 3 grid of the golden vectors.
    fn golden_grid() -> Grid {
        grid(&[&[1, 1, 0], &[0, 1, 2], &[0, 0, 2]])
    }

    // T-ARC-1: phi32 golden vectors (worked out by hand).
    #[test]
    fn phi32_golden_vectors() {
        let g = golden_grid();

        // Corner: centre 1; right (n=4) and down-right (n=7) are 1, down (n=6) is 0.
        assert_eq!(on_bits(&phi32(&g, 0, 0)), vec![1, 14, 17, 22, 25, 28, 30]);

        // Middle: up-left and up equal the centre; right and down-right are non-zero 2s.
        assert_eq!(
            on_bits(&phi32(&g, 1, 1)),
            vec![1, 10, 11, 18, 19, 22, 25, 26, 27]
        );

        // Bottom-right corner: centre 2.
        assert_eq!(on_bits(&phi32(&g, 2, 2)), vec![2, 11, 18, 19, 29, 31]);

        // Centre 0 at the left edge: outside neighbours set neither "same" nor "non-zero".
        assert_eq!(
            on_bits(&phi32(&g, 1, 0)),
            vec![0, 16, 17, 19, 20, 22, 26, 30]
        );

        // A 1 x 1 grid touches all four borders.
        assert_eq!(
            on_bits(&phi32(&grid(&[&[5]]), 0, 0)),
            vec![5, 28, 29, 30, 31]
        );
    }

    #[test]
    fn phi32_never_panics_on_bad_input() {
        let g = golden_grid();

        assert_eq!(phi32(&g, 3, 0), [0.0; 32]);
        assert_eq!(phi32(&g, 0, 99), [0.0; 32]);

        let short = Grid {
            h: 2,
            w: 2,
            cells: vec![1],
        };

        assert_eq!(phi32(&short, 0, 0), [0.0; 32]);

        // Colour 12 has no one-hot slot; the other bits still work.
        let odd = grid(&[&[12, 12]]);

        assert_eq!(on_bits(&phi32(&odd, 0, 0)), vec![14, 22, 28, 29, 30]);
    }

    /// Two train pairs (9 + 4 cells), two test pairs (9 + 2 cells); colour 1 becomes 2.
    fn recolour_task() -> ArcTask {
        task(
            "t1",
            vec![
                pair(
                    &[&[1, 0, 0], &[0, 1, 0], &[3, 0, 1]],
                    &[&[2, 0, 0], &[0, 2, 0], &[3, 0, 2]],
                ),
                pair(&[&[1, 3], &[0, 0]], &[&[2, 3], &[0, 0]]),
            ],
            vec![
                pair(
                    &[&[0, 1, 0], &[1, 1, 1], &[0, 3, 0]],
                    &[&[0, 2, 0], &[2, 2, 2], &[0, 3, 0]],
                ),
                pair(&[&[3, 1]], &[&[3, 2]]),
            ],
        )
    }

    // T-ARC-1: cell_tasks sizes, labels and seeds.
    #[test]
    fn cell_tasks_have_the_contract_shape() {
        let t = recolour_task();
        let tasks = cell_tasks(&t).expect("shape-keeping task");

        assert_eq!(tasks.len(), 2);

        let train_labels: Vec<u8> = vec![2, 0, 0, 0, 2, 0, 3, 0, 2, 2, 3, 0, 0];

        for (j, ti) in tasks.iter().enumerate() {
            let test = &t.test[j];

            assert_eq!(ti.f, 32);
            assert_eq!(ti.kind, LearnKind::Multiclass { k: 10 });
            assert_eq!(ti.group, Group::ArcCell);
            assert_eq!(ti.family, FamilyId("arc-cell/t1".into()));
            assert_eq!(ti.n_train, 13);
            assert_eq!(ti.x_train.len(), 13 * 32);
            assert_eq!(ti.y_train, Labels::Classes(train_labels.clone()));
            assert_eq!(ti.steps, 384);
            assert_eq!(ti.seed, derive_seed(0, &format!("arc-cell/t1/{j}")));
            assert_eq!(ti.n_valid, test.input.cells.len());
            assert_eq!(ti.x_valid.len(), ti.n_valid * 32);
            assert_eq!(ti.y_valid, Labels::Classes(test.output.cells.clone()));
            assert!(ti.y_train_late.is_none() && ti.drift_at.is_none());
            assert!(ti.kind.k_out() <= ti.f);

            let mask: Vec<bool> = test
                .input
                .cells
                .iter()
                .zip(&test.output.cells)
                .map(|(a, b)| a != b)
                .collect();

            assert_eq!(ti.changed_mask.as_ref(), Some(&mask));
        }

        // Rows are phi32 of the input cells: pair 0 from row 0, pair 1 from row 9.
        let first = &tasks[0];

        assert_eq!(first.x_train[..32], phi32(&t.train[0].input, 0, 0));
        assert_eq!(
            first.x_train[9 * 32..10 * 32],
            phi32(&t.train[1].input, 0, 0)
        );
        assert_eq!(first.x_train[12 * 32..], phi32(&t.train[1].input, 1, 1));
        assert_eq!(first.x_valid[4 * 32..5 * 32], phi32(&t.test[0].input, 1, 1));
        assert_eq!(tasks[1].x_valid[32..], phi32(&t.test[1].input, 0, 1));

        // Distinct ids per test pair; the same ids on a rebuild.
        assert_ne!(tasks[0].id, tasks[1].id);
        assert_eq!(cell_tasks(&t).expect("again")[1].id, tasks[1].id);
    }

    /// A task whose train pairs hold `cells` cells in all (1 x 30 rows of zeros).
    fn task_with_cells(cells: usize) -> ArcTask {
        let mut train = Vec::new();
        let mut left = cells;

        while left > 0 {
            let n = left.min(30);
            let g = Grid {
                h: 1,
                w: n as u8,
                cells: vec![0; n],
            };

            train.push(ArcPair {
                input: g.clone(),
                output: g,
            });

            left -= n;
        }

        task("big", train, vec![pair(&[&[0]], &[&[0]])])
    }

    // T-ARC-1: steps = clamp(3 * n_train_cells, 384, 8192).
    #[test]
    fn steps_are_clamped() {
        let cases = [
            (13, 384),
            (128, 384),
            (200, 600),
            (2730, 8190),
            (2731, 8192),
            (9000, 8192),
        ];

        for (cells, steps) in cases {
            let tasks = cell_tasks(&task_with_cells(cells)).expect("same shape");

            assert_eq!(tasks[0].n_train, cells);
            assert_eq!(tasks[0].steps, steps, "cells {cells}");
        }
    }

    #[test]
    fn cell_tasks_refuse_shape_changes_and_bad_grids() {
        let grows = task(
            "g",
            vec![pair(&[&[1]], &[&[1, 1]])],
            vec![pair(&[&[1]], &[&[1]])],
        );

        assert!(!train_keeps_shape(&grows));
        assert!(cell_tasks(&grows).is_none());

        let mut test_grows = recolour_task();

        test_grows.test[1] = pair(&[&[3, 1]], &[&[3], &[2]]);

        assert!(train_keeps_shape(&test_grows));
        assert!(cell_tasks(&test_grows).is_none());

        let mut bad_colour = recolour_task();

        bad_colour.train[0].output.cells[0] = 10;

        assert!(cell_tasks(&bad_colour).is_none());

        let no_train = task("n", vec![], vec![pair(&[&[1]], &[&[1]])]);

        assert!(!train_keeps_shape(&no_train));
        assert!(cell_tasks(&no_train).is_none());
    }

    /// Linked outputs that put probability 1 on `colours`.
    fn one_hot(colours: &[u8]) -> Vec<f32> {
        let mut out = vec![0.0; colours.len() * 10];

        for (i, &c) in colours.iter().enumerate() {
            out[i * 10 + c as usize] = 1.0;
        }

        out
    }

    /// A perfect learner: reads the labels it should not see.
    fn oracle(task: &TaskInstance) -> Vec<f32> {
        match &task.y_valid {
            Labels::Classes(c) => one_hot(c),
            Labels::Values(_) => Vec::new(),
        }
    }

    /// Copies the centre colour (features 0..10) of every validation cell.
    fn copy_input(task: &TaskInstance) -> Vec<f32> {
        let colours: Vec<u8> = task
            .x_valid
            .chunks_exact(32)
            .map(|row| row[..10].iter().position(|&x| x == 1.0).unwrap_or(0) as u8)
            .collect();

        one_hot(&colours)
    }

    #[test]
    fn solve_with_scores_exact_matches_and_bca() {
        let t = recolour_task();
        let mut calls = 0;
        let r = solve_with(&t, |task, _| {
            calls += 1;

            Some(oracle(task))
        });

        assert_eq!(r.task_id, "t1");
        assert_eq!(r.correct, vec![true, true]);
        assert_eq!(r.bca, vec![1.0, 1.0]);
        assert_eq!(calls, 2, "attempt 1 only runs after a miss");

        // Copying the input never matches a recolouring; BCA is exactly 0.5.
        let r = solve_with(&t, |task, _| Some(copy_input(task)));

        assert_eq!(r.correct, vec![false, false]);
        assert_eq!(r.bca, vec![0.5, 0.5]);
    }

    #[test]
    fn solve_with_counts_the_second_attempt() {
        let t = recolour_task();
        let mut seen = Vec::new();
        let r = solve_with(&t, |task, attempt| {
            seen.push(attempt);

            if attempt == 0 {
                Some(copy_input(task))
            } else {
                Some(oracle(task))
            }
        });

        assert_eq!(r.correct, vec![true, true]);
        assert_eq!(seen, vec![0, 1, 0, 1]);

        // BCA reports attempt 0, not the better attempt 1.
        assert_eq!(r.bca, vec![0.5, 0.5]);
    }

    #[test]
    fn solve_with_treats_aborts_and_bad_outputs_as_misses() {
        let t = recolour_task();
        let r = solve_with(&t, |_, _| None);

        assert_eq!(r.correct, vec![false, false]);
        assert_eq!(r.bca, vec![0.0, 0.0]);

        let r = solve_with(&t, |task, _| {
            let mut out = oracle(task);

            out.pop();

            Some(out)
        });

        assert_eq!(r.correct, vec![false, false]);

        let r = solve_with(&t, |task, _| Some(vec![f32::NAN; task.n_valid * 10]));

        assert_eq!(r.correct, vec![false, false]);
    }

    #[test]
    fn argmax_breaks_ties_low_and_ignores_nan() {
        let mut row = [0.0f32; 10];

        row[3] = 0.5;
        row[7] = 0.5;

        assert_eq!(predicted_grid(&row, 1), Some(vec![3]));

        row[0] = f32::NAN;

        assert_eq!(predicted_grid(&row, 1), Some(vec![3]));
        assert_eq!(predicted_grid(&[f32::NAN; 10], 1), Some(vec![0]));
        assert_eq!(predicted_grid(&row, 2), None);
    }

    #[test]
    fn solve_with_skips_shape_changes() {
        let grows = task(
            "g",
            vec![pair(&[&[1]], &[&[1, 1]])],
            vec![pair(&[&[1]], &[&[1, 1]]), pair(&[&[2]], &[&[2, 2]])],
        );
        let mut calls = 0;
        let r = solve_with(&grows, |task, _| {
            calls += 1;

            Some(oracle(task))
        });

        assert_eq!(r.correct, vec![false, false]);
        assert!(r.bca.is_empty());
        assert_eq!(calls, 0);

        // Train keeps shape but test pair 1 does not: pair 0 still counts.
        let mut t = recolour_task();

        t.test[1] = pair(&[&[3, 1]], &[&[3], &[2]]);

        let r = solve_with(&t, |task, _| Some(oracle(task)));

        assert_eq!(r.correct, vec![true, false]);
        assert_eq!(r.bca, vec![1.0]);
    }

    #[test]
    fn bca_needs_changed_and_unchanged_cells() {
        // Every cell changes: no BCA for this pair.
        let t = task(
            "all",
            vec![pair(&[&[1, 1]], &[&[2, 2]])],
            vec![pair(&[&[1, 1]], &[&[2, 2]])],
        );
        let r = solve_with(&t, |task, _| Some(oracle(task)));

        assert_eq!(r.correct, vec![true]);
        assert!(r.bca.is_empty());

        // 3 unchanged cells (2 right), 1 changed cell (right): 0.5 * 2/3 + 0.5 * 1.
        let p = pair(&[&[1, 0, 0, 0]], &[&[2, 0, 0, 0]]);
        let bca = balanced_accuracy(Some(&[2, 0, 0, 5]), &p).expect("both kinds");

        assert!((bca - (0.5 * 2.0 / 3.0 + 0.5)).abs() < 1e-12);
    }

    // T-ARC-3: scoring on hand fixtures, with a multi-output task and missing tasks.
    #[test]
    fn scores_match_hand_computations() {
        let results = vec![
            ArcResult {
                task_id: "a".into(),
                correct: vec![true],
                bca: vec![1.0],
            },
            ArcResult {
                task_id: "b".into(),
                correct: vec![true, false],
                bca: vec![0.5],
            },
            ArcResult {
                task_id: "c".into(),
                correct: vec![false, false, false],
                bca: vec![],
            },
        ];

        // 5 tasks, 2 of them missing (1 test output each): 8 outputs in all.
        assert!((score_official(&results, 5) - (1.0 + 0.5) / 5.0).abs() < 1e-12);
        assert!((score_kaggle(&results, 8) - 2.0 / 8.0).abs() < 1e-12);
        assert!((mean_bca(&results).expect("two pairs") - 0.75).abs() < 1e-12);

        // Nothing to score: zero, never NaN.
        assert_eq!(score_official(&[], 0), 0.0);
        assert_eq!(score_kaggle(&[], 0), 0.0);
        assert_eq!(mean_bca(&[]), None);

        // A denominator smaller than the results cannot push a score above 1.
        assert!(score_official(&results, 1) <= 1.0);
        assert!(score_kaggle(&results, 1) <= 1.0);

        // A task without test outputs counts 0; a NaN BCA is skipped.
        let empty = vec![ArcResult {
            task_id: "e".into(),
            correct: vec![],
            bca: vec![f64::NAN],
        }];

        assert_eq!(score_official(&empty, 1), 0.0);
        assert_eq!(mean_bca(&empty), None);
    }

    /// Rotate a grid 90 degrees clockwise (independent of d4_source).
    fn rotate(g: &Grid) -> Grid {
        let (h, w) = (g.h as usize, g.w as usize);
        let mut cells = Vec::with_capacity(h * w);

        for c in 0..w {
            for r in (0..h).rev() {
                cells.push(g.cells[r * w + c]);
            }
        }

        Grid {
            h: g.w,
            w: g.h,
            cells,
        }
    }

    /// Mirror a grid left-right.
    fn mirror(g: &Grid) -> Grid {
        let w = g.w as usize;
        let cells = g
            .cells
            .chunks_exact(w)
            .flat_map(|row| row.iter().rev().copied())
            .collect();

        Grid { cells, ..g.clone() }
    }

    fn map_pair(p: &ArcPair, f: impl Fn(&Grid) -> Grid) -> ArcPair {
        ArcPair {
            input: f(&p.input),
            output: f(&p.output),
        }
    }

    fn random_grid(rng: &mut ChaCha8Rng) -> Grid {
        let h = rng.random_range(1..=6usize);
        let w = rng.random_range(1..=6usize);

        Grid {
            h: h as u8,
            w: w as u8,
            cells: (0..h * w).map(|_| rng.random_range(0..10u8)).collect(),
        }
    }

    // T-ARC-2: invariant under the 8 D4 transforms times colour permutations keeping 0.
    #[test]
    fn canonical_hash_is_invariant_under_d4_and_recolouring() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let mut distinct = BTreeSet::new();

        for _ in 0..50 {
            let p = ArcPair {
                input: random_grid(&mut rng),
                output: random_grid(&mut rng),
            };
            let h = canonical_pair_hash(&p);

            distinct.insert(h);

            for mirrored in [false, true] {
                let mut q = if mirrored {
                    map_pair(&p, mirror)
                } else {
                    p.clone()
                };

                for _ in 0..4 {
                    // A random permutation of 1..=9; 0 stays 0.
                    let mut perm: Vec<u8> = (1..10).collect();

                    for i in (1..perm.len()).rev() {
                        perm.swap(i, rng.random_range(0..=i));
                    }

                    let recolour = |g: &Grid| Grid {
                        cells: g
                            .cells
                            .iter()
                            .map(|&c| if c == 0 { 0 } else { perm[c as usize - 1] })
                            .collect(),
                        ..g.clone()
                    };

                    assert_eq!(canonical_pair_hash(&map_pair(&q, recolour)), h);

                    q = map_pair(&q, rotate);
                }
            }
        }

        assert_eq!(distinct.len(), 50, "random pairs should not collide");
    }

    #[test]
    fn canonical_hash_separates_real_differences() {
        let base = pair(&[&[1, 2]], &[&[1, 2]]);
        let h = canonical_pair_hash(&base);

        // Output colours permuted apart from the input.
        assert_ne!(canonical_pair_hash(&pair(&[&[1, 2]], &[&[2, 1]])), h);

        // One cell differs.
        assert_ne!(canonical_pair_hash(&pair(&[&[1, 2]], &[&[1, 1]])), h);

        // A recolouring that moves 0 is a real difference.
        assert_ne!(canonical_pair_hash(&pair(&[&[0, 2]], &[&[0, 2]])), h);

        // Malformed grids hash without panicking.
        let bad = ArcPair {
            input: Grid {
                h: 3,
                w: 3,
                cells: vec![255],
            },
            output: Grid {
                h: 0,
                w: 5,
                cells: vec![],
            },
        };

        let _ = canonical_pair_hash(&bad);
    }

    #[test]
    fn d4_transforms_match_rotate_and_mirror() {
        let g = grid(&[&[1, 2, 3], &[4, 5, 6]]);
        let r1 = rotate(&g);
        let r2 = rotate(&r1);
        let r3 = rotate(&r2);
        let m = mirror(&g);
        let expected = [
            g.clone(),
            r1,
            r2,
            r3,
            m.clone(),
            rotate(&rotate(&m)),
            rotate(&rotate(&rotate(&m))),
            rotate(&m),
        ];

        for (t, want) in expected.iter().enumerate() {
            let t = t as u8;
            let (th, tw) = d4_dims(t, 2, 3);
            let mut cells = Vec::new();

            for i in 0..th {
                for j in 0..tw {
                    let (r, c) = d4_source(t, i, j, 2, 3);

                    cells.push(g.cells[r * 3 + c]);
                }
            }

            assert_eq!((th as u8, tw as u8), (want.h, want.w), "transform {t}");
            assert_eq!(cells, want.cells, "transform {t}");
        }
    }

    #[test]
    fn phi32_ceiling_on_fixtures() {
        // Recolouring 1 -> 2 is a function of the centre colour.
        assert!(phi32_ceiling(&recolour_task()));

        // The same 1 x 1 key with two different output colours.
        let conflict = task(
            "c",
            vec![pair(&[&[5]], &[&[5]])],
            vec![pair(&[&[5]], &[&[6]])],
        );

        assert!(!phi32_ceiling(&conflict));

        // Any shape change, train or test, rules a task out.
        let mut test_grows = recolour_task();

        test_grows.test[1] = pair(&[&[3, 1]], &[&[3], &[2]]);

        assert!(!phi32_ceiling(&test_grows));
        assert!(!phi32_ceiling(&task("e", vec![], vec![])));
    }
}
