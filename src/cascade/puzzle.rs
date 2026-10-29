//! ARC-style grid puzzles for Cascade breeding (Stage 2c, ADR 0005): a seeded generator
//! of puzzles with hidden rules, and the plain-text protocol between the orchestrator and
//! a Worker model.
//!
//! This is the evaluator side. A `Puzzle` carries its hidden rule and its ground truth,
//! but the prompt renderers read only the worked examples and the test input, so no
//! prompt can carry the answer. The parsers take arbitrary model output: anything that
//! does not follow the protocol gives `None`, never a panic.
//!
//! The protocol is line based. A grid is one line per row, its cells digits 0..=9
//! separated by spaces. A solve prompt is `TASK`, the worked examples, `TEST INPUT` and
//! the answers of earlier stages (`PREVIOUS ANSWER k`), closed by `END`; a judge prompt
//! is `JUDGE`, the examples, the test input and two candidates `ANSWER A` and
//! `ANSWER B`. A Worker replies with the line `ANSWER` and a grid, a judge with the line
//! `VERDICT A` or `VERDICT B`.
#![forbid(unsafe_code)]

use std::collections::HashSet;

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::util::{derive_seed, sha256_hex};

/// A grid of colours, row by row. Colour 0 is the background.
pub type Grid = Vec<Vec<u8>>;

/// Largest number of rows, and of columns, a grid may have (as in ARC).
pub const MAX_SIDE: usize = 30;

/// Worked examples per generated puzzle.
pub const TRAIN_PAIRS: usize = 3;

/// Text appended to a Worker's system prompt by the orchestrator when its `self_check`
/// Trait is on.
pub const SELF_CHECK_TEXT: &str =
    "Before you answer, test your rule on every example and drop any rule that fails one.";

/// The instruction line of a solve prompt.
const SOLVE_INTRO: &str = "Each example maps an input grid to an output grid by one hidden rule. Apply the rule to the test input.";

/// The instruction line of a judge prompt.
const JUDGE_INTRO: &str = "Two answers were given for the test input. Decide which one follows the rule of the examples. Reply with the line VERDICT A or VERDICT B.";

/// The colour map that changes nothing.
const SAME_COLOURS: [u8; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

/// The geometric steps a generated rule may use: every one but Identity.
const MOVES: [Geo; 7] = [
    Geo::FlipH,
    Geo::FlipV,
    Geo::Rot90,
    Geo::Rot180,
    Geo::Rot270,
    Geo::Transpose,
    Geo::GravityDown,
];

/// Grid draws per rule, and rule draws per puzzle, before the generator falls back to a
/// fixed puzzle. Neither bound is reached in practice; they only make the loop finite.
const GRID_ATTEMPTS: usize = 64;
const RULE_ATTEMPTS: usize = 64;

/// True for a well-formed grid: 1..=30 rows, every row the same length 1..=30, every
/// cell a colour 0..=9.
pub fn is_valid_grid(g: &Grid) -> bool {
    let Some(first) = g.first() else {
        return false;
    };
    let width = first.len();

    g.len() <= MAX_SIDE
        && (1..=MAX_SIDE).contains(&width)
        && g.iter()
            .all(|row| row.len() == width && row.iter().all(|&c| c <= 9))
}

/// One worked example: an input grid and the output the hidden rule makes of it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pair {
    pub input: Grid,
    pub output: Grid,
}

/// The geometric step of a rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Geo {
    /// No change.
    Identity,
    /// Mirror left-right.
    FlipH,
    /// Mirror top-bottom.
    FlipV,
    /// Quarter turn clockwise.
    Rot90,
    /// Half turn.
    Rot180,
    /// Quarter turn anticlockwise.
    Rot270,
    /// Rows become columns (mirror on the main diagonal).
    Transpose,
    /// Non-zero cells fall to the bottom of their column, keeping their order.
    GravityDown,
}

/// A hidden rule: geometric step first, then an optional colour map (index = old colour).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rule {
    pub geo: Geo,
    pub colours: Option<[u8; 10]>,
}

/// One puzzle: worked examples, a test input and its hidden answer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Puzzle {
    /// First 16 hex of SHA-256 over the canonical JSON of (train, test_input, test_output).
    pub id: String,
    /// The hidden rule, for tests and reports only; never rendered into any prompt.
    pub rule: Rule,
    /// The worked examples.
    pub train: Vec<Pair>,
    pub test_input: Grid,
    /// The ground truth; never rendered into any prompt.
    pub test_output: Grid,
}

/// Colour at row `r`, column `c`; 0 outside the grid, so no step panics on a ragged grid.
fn cell(g: &Grid, r: usize, c: usize) -> u8 {
    g.get(r).and_then(|row| row.get(c)).copied().unwrap_or(0)
}

/// A `rows` x `cols` grid whose cells are `f(row, column)`.
fn build(rows: usize, cols: usize, f: impl Fn(usize, usize) -> u8) -> Grid {
    (0..rows)
        .map(|r| (0..cols).map(|c| f(r, c)).collect())
        .collect()
}

/// Apply one geometric step. The width is that of the first row; a ragged grid is padded
/// with 0 or cut to it rather than causing a panic.
pub fn apply_geo(geo: Geo, g: &Grid) -> Grid {
    let rows = g.len();
    let cols = g.first().map_or(0, Vec::len);
    let at = |r: usize, c: usize| cell(g, r, c);

    match geo {
        Geo::Identity => build(rows, cols, at),
        Geo::FlipH => build(rows, cols, |r, c| at(r, cols - 1 - c)),
        Geo::FlipV => build(rows, cols, |r, c| at(rows - 1 - r, c)),
        Geo::Rot90 => build(cols, rows, |r, c| at(rows - 1 - c, r)),
        Geo::Rot180 => build(rows, cols, |r, c| at(rows - 1 - r, cols - 1 - c)),
        Geo::Rot270 => build(cols, rows, |r, c| at(c, cols - 1 - r)),
        Geo::Transpose => build(cols, rows, |r, c| at(c, r)),
        Geo::GravityDown => {
            // Each column bottom-aligned: background on top, then its colours in order.
            let columns: Grid = (0..cols)
                .map(|c| {
                    let colours: Vec<u8> =
                        (0..rows).map(|r| at(r, c)).filter(|&v| v != 0).collect();
                    let mut column = vec![0; rows - colours.len()];

                    column.extend(colours);
                    column
                })
                .collect();

            build(rows, cols, |r, c| cell(&columns, c, r))
        }
    }
}

/// Apply a whole rule: its geometric step, then its colour map.
pub fn apply(rule: &Rule, g: &Grid) -> Grid {
    let moved = apply_geo(rule.geo, g);

    match &rule.colours {
        Some(map) => moved
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&v| map.get(usize::from(v)).copied().unwrap_or(v))
                    .collect()
            })
            .collect(),
        None => moved,
    }
}

/// Fit one consistent per-colour map from (x, y) pairs of equal shape; colours never seen
/// map to themselves. None if inconsistent, if shapes differ, or if a cell is not 0..=9.
pub fn fit_colour_map(pairs: &[(&Grid, &Grid)]) -> Option<[u8; 10]> {
    let mut fitted: [Option<u8>; 10] = [None; 10];

    for (x, y) in pairs {
        if x.len() != y.len() {
            return None;
        }

        for (row_x, row_y) in x.iter().zip(y.iter()) {
            if row_x.len() != row_y.len() {
                return None;
            }

            for (&from, &to) in row_x.iter().zip(row_y) {
                let slot = fitted.get_mut(usize::from(from))?;

                if to > 9 || slot.is_some_and(|earlier| earlier != to) {
                    return None;
                }

                *slot = Some(to);
            }
        }
    }

    let mut map = SAME_COLOURS;

    for (target, found) in map.iter_mut().zip(fitted) {
        if let Some(to) = found {
            *target = to;
        }
    }

    Some(map)
}

/// Deterministic panel of `n` puzzles from `seed` (puzzle i uses
/// `derive_seed(seed, &format!("puzzle/{i}"))`, so a longer panel extends a shorter one).
///
/// Every puzzle has three worked examples and is well posed: every colour of the test
/// input occurs in some example input, the test output differs from the test input, at
/// least two examples change their input, and a gravity rule moves cells of the test
/// input and of at least two example inputs. Ids are unique within the panel.
pub fn generate_panel(seed: u64, n: usize) -> Vec<Puzzle> {
    let mut panel = Vec::new();
    let mut ids = HashSet::new();

    for i in 0..n {
        let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(seed, &format!("puzzle/{i}")));
        let mut puzzle = generate_puzzle(&mut rng);

        if !ids.insert(puzzle.id.clone()) {
            // Same content as an earlier puzzle: salt the hashed material with the index.
            puzzle.id = puzzle_id(
                &puzzle.train,
                &puzzle.test_input,
                &puzzle.test_output,
                Some(i),
            );
            ids.insert(puzzle.id.clone());
        }

        panel.push(puzzle);
    }

    panel
}

/// One well-posed puzzle: redraw grids, then the rule, within fixed bounds.
fn generate_puzzle(rng: &mut ChaCha8Rng) -> Puzzle {
    for _ in 0..RULE_ATTEMPTS {
        let palette = draw_palette(rng);
        let rule = draw_rule(rng, &palette);

        for _ in 0..GRID_ATTEMPTS {
            let inputs = (0..TRAIN_PAIRS).map(|_| draw_grid(rng, &palette)).collect();
            let test_input = draw_grid(rng, &palette);
            let puzzle = build_puzzle(rule.clone(), inputs, test_input);

            if is_well_posed(&puzzle) {
                return puzzle;
            }
        }
    }

    fallback_puzzle()
}

/// A uniform index below `n`, drawn as u32 so the stream is the same on every platform;
/// 0 when `n` is 0.
fn pick(rng: &mut ChaCha8Rng, n: usize) -> usize {
    match u32::try_from(n) {
        Ok(bound) if bound > 0 => rng.random_range(0..bound) as usize,
        _ => 0,
    }
}

/// Three or four distinct colours from 1..=9.
fn draw_palette(rng: &mut ChaCha8Rng) -> Vec<u8> {
    let mut colours: Vec<u8> = (1..=9).collect();

    colours.shuffle(rng);
    colours.truncate(3 + pick(rng, 2));
    colours
}

/// A rule from the design mix: about 45 % one geometric step, 25 % a colour map alone,
/// 30 % a geometric step followed by a colour map.
fn draw_rule(rng: &mut ChaCha8Rng, palette: &[u8]) -> Rule {
    let roll: f64 = rng.random();
    let geo = if (0.45..0.70).contains(&roll) {
        Geo::Identity
    } else {
        draw_move(rng)
    };
    let colours = (roll >= 0.45).then(|| draw_permutation(rng, palette));

    Rule { geo, colours }
}

/// One of the non-identity geometric steps, uniformly.
fn draw_move(rng: &mut ChaCha8Rng) -> Geo {
    MOVES
        .get(pick(rng, MOVES.len()))
        .copied()
        .unwrap_or(Geo::FlipH)
}

/// A colour map that permutes the palette (never as the identity) and fixes every other
/// colour, the background included.
fn draw_permutation(rng: &mut ChaCha8Rng, palette: &[u8]) -> [u8; 10] {
    let mut targets = palette.to_vec();

    targets.shuffle(rng);

    if targets == palette && !targets.is_empty() {
        targets.rotate_left(1);
    }

    let mut map = SAME_COLOURS;

    for (&from, &to) in palette.iter().zip(&targets) {
        if let Some(slot) = map.get_mut(usize::from(from)) {
            *slot = to;
        }
    }

    map
}

/// A grid of 3..=6 rows and 3..=6 columns; each cell is background with probability one
/// half, else a palette colour.
fn draw_grid(rng: &mut ChaCha8Rng, palette: &[u8]) -> Grid {
    let rows = 3 + pick(rng, 4);
    let cols = 3 + pick(rng, 4);
    let mut draw_cell = || {
        if rng.random_bool(0.5) {
            0
        } else {
            palette.get(pick(rng, palette.len())).copied().unwrap_or(0)
        }
    };

    (0..rows)
        .map(|_| (0..cols).map(|_| draw_cell()).collect())
        .collect()
}

/// A puzzle from a rule and its inputs; every output comes from the rule.
fn build_puzzle(rule: Rule, inputs: Vec<Grid>, test_input: Grid) -> Puzzle {
    let train: Vec<Pair> = inputs
        .into_iter()
        .map(|input| Pair {
            output: apply(&rule, &input),
            input,
        })
        .collect();
    let test_output = apply(&rule, &test_input);
    let id = puzzle_id(&train, &test_input, &test_output, None);

    Puzzle {
        id,
        rule,
        train,
        test_input,
        test_output,
    }
}

/// The constraints of `generate_panel`.
fn is_well_posed(p: &Puzzle) -> bool {
    let mut seen = [false; 10];

    for &c in p.train.iter().flat_map(|pair| pair.input.iter().flatten()) {
        if let Some(slot) = seen.get_mut(usize::from(c)) {
            *slot = true;
        }
    }

    let colours_shown = p
        .test_input
        .iter()
        .flatten()
        .all(|&c| seen.get(usize::from(c)).copied().unwrap_or(false));
    let changed = p
        .train
        .iter()
        .filter(|pair| pair.output != pair.input)
        .count();
    let moves = |g: &Grid| apply_geo(p.rule.geo, g) != *g;
    let gravity_shown = p.rule.geo != Geo::GravityDown
        || (moves(&p.test_input) && p.train.iter().filter(|pair| moves(&pair.input)).count() >= 2);

    colours_shown && p.test_output != p.test_input && changed >= 2 && gravity_shown
}

/// A fixed, well-posed puzzle (mirror left-right), returned only if every bounded random
/// draw failed, so that the generator always terminates.
fn fallback_puzzle() -> Puzzle {
    let inputs = vec![
        vec![vec![1, 2, 0], vec![0, 3, 0], vec![0, 0, 0]],
        vec![vec![3, 0, 0], vec![1, 1, 0], vec![0, 0, 2]],
        vec![vec![0, 0, 0], vec![2, 0, 0], vec![3, 3, 1]],
    ];
    let test_input = vec![vec![1, 0, 0], vec![0, 2, 0], vec![3, 0, 0]];
    let rule = Rule {
        geo: Geo::FlipH,
        colours: None,
    };

    build_puzzle(rule, inputs, test_input)
}

/// First 16 hex of SHA-256 over the canonical JSON of (train, test_input, test_output),
/// with `salt` appended to the material when given.
fn puzzle_id(train: &[Pair], test_input: &Grid, test_output: &Grid, salt: Option<usize>) -> String {
    let json = match salt {
        None => serde_json::to_string(&(train, test_input, test_output)),
        Some(i) => serde_json::to_string(&(train, test_input, test_output, i)),
    };

    sha256_hex(json.unwrap_or_default().as_bytes())[..16].to_string()
}

/// A grid as text: rows joined by '\n', cells by ' ', no trailing newline.
pub fn render_grid(g: &Grid) -> String {
    g.iter()
        .map(|row| row.iter().map(u8::to_string).collect::<Vec<_>>().join(" "))
        .collect::<Vec<_>>()
        .join("\n")
}

/// A Worker's reply in protocol form: the line ANSWER, then the grid.
pub fn render_answer(g: &Grid) -> String {
    if g.is_empty() {
        "ANSWER".to_string()
    } else {
        format!("ANSWER\n{}", render_grid(g))
    }
}

/// A judge's reply in protocol form.
pub fn render_verdict(v: Verdict) -> &'static str {
    match v {
        Verdict::A => "VERDICT A",
        Verdict::B => "VERDICT B",
    }
}

/// Append a block: its title line, then its grid.
fn push_block(out: &mut String, title: &str, g: &Grid) {
    out.push_str(title);
    out.push('\n');
    out.push_str(&render_grid(g));
    out.push('\n');
}

/// The worked examples and the test input, shared by both prompts. This is the only place
/// a prompt reads a Puzzle, and it reads neither the rule nor the test output.
fn push_task(out: &mut String, p: &Puzzle) {
    for (k, pair) in p.train.iter().enumerate() {
        out.push_str(&format!("EXAMPLE {}\n", k + 1));
        push_block(out, "INPUT", &pair.input);
        push_block(out, "OUTPUT", &pair.output);
    }

    push_block(out, "TEST INPUT", &p.test_input);
}

/// The solve prompt: the worked examples, the test input and the answers of earlier
/// stages (the residual path of a Cascade), oldest first.
pub fn render_solve_prompt(p: &Puzzle, previous: &[Grid]) -> String {
    let mut out = format!("TASK\n{SOLVE_INTRO}\n");

    push_task(&mut out, p);

    for (k, g) in previous.iter().enumerate() {
        push_block(&mut out, &format!("PREVIOUS ANSWER {}", k + 1), g);
    }

    out.push_str("END");
    out
}

/// The judge prompt: the worked examples, the test input and the candidates A and B.
pub fn render_judge_prompt(p: &Puzzle, a: &Grid, b: &Grid) -> String {
    let mut out = format!("JUDGE\n{JUDGE_INTRO}\n");

    push_task(&mut out, p);
    push_block(&mut out, "ANSWER A", a);
    push_block(&mut out, "ANSWER B", b);
    out.push_str("END");
    out
}

/// A prompt as read back from its text. A simulated Worker sees only this, never a Puzzle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsedPrompt {
    /// True for a judge prompt, false for a solve prompt.
    pub judge: bool,
    pub examples: Vec<Pair>,
    pub test_input: Grid,
    /// The answers of earlier stages, oldest first (solve prompts only).
    pub previous: Vec<Grid>,
    /// The candidates A and B (judge prompts only).
    pub candidates: Option<(Grid, Grid)>,
}

/// How one line reads as a grid row.
enum Row {
    /// A row of colours.
    Cells(Vec<u8>),
    /// Numbers only, but not a row: a number above 9, or more than 30 cells.
    Malformed,
    /// Anything else, a blank line included. A grid ends before such a line.
    Other,
}

/// Classify one line.
fn read_row(line: &str) -> Row {
    let numeric = |token: &str| token.bytes().all(|b| b.is_ascii_digit());

    if line.trim().is_empty() || !line.split_whitespace().all(numeric) {
        return Row::Other;
    }

    let mut cells = Vec::new();

    for token in line.split_whitespace() {
        match token.as_bytes() {
            [digit] if cells.len() < MAX_SIDE => cells.push(digit - b'0'),
            _ => return Row::Malformed,
        }
    }

    Row::Cells(cells)
}

/// A cursor over the trimmed lines of a text.
struct Cursor<'a> {
    lines: Vec<&'a str>,
    at: usize,
}

impl<'a> Cursor<'a> {
    fn new(text: &'a str) -> Self {
        Cursor {
            lines: text.lines().map(str::trim).collect(),
            at: 0,
        }
    }

    fn peek(&self) -> Option<&'a str> {
        self.lines.get(self.at).copied()
    }

    fn skip_blank(&mut self) {
        while self.peek() == Some("") {
            self.at += 1;
        }
    }

    /// Consume the next non-blank line if it is exactly `word`.
    fn accept(&mut self, word: &str) -> bool {
        self.skip_blank();

        let found = self.peek() == Some(word);

        if found {
            self.at += 1;
        }

        found
    }

    /// Like `accept`, as an Option for `?`.
    fn keyword(&mut self, word: &str) -> Option<()> {
        self.accept(word).then_some(())
    }

    /// Consume a grid: rows from the next non-blank line up to the first line that is not
    /// a row. None unless they form a valid grid.
    fn grid(&mut self) -> Option<Grid> {
        self.skip_blank();

        let mut g = Grid::new();

        while let Some(line) = self.peek() {
            match read_row(line) {
                Row::Cells(cells) if g.len() < MAX_SIDE => g.push(cells),
                Row::Cells(_) | Row::Malformed => return None,
                Row::Other => break,
            }

            self.at += 1;
        }

        is_valid_grid(&g).then_some(g)
    }
}

/// Read a solve or judge prompt back. Lines are trimmed, blank lines between blocks are
/// skipped, and the instruction text up to `EXAMPLE 1` is free; any other deviation from
/// the protocol gives None.
pub fn parse_prompt(text: &str) -> Option<ParsedPrompt> {
    let mut cur = Cursor::new(text);

    cur.skip_blank();

    let judge = match cur.peek()? {
        "TASK" => false,
        "JUDGE" => true,
        _ => return None,
    };

    while cur.peek()? != "EXAMPLE 1" {
        cur.at += 1;
    }

    let mut examples = Vec::new();

    while cur.accept(&format!("EXAMPLE {}", examples.len() + 1)) {
        cur.keyword("INPUT")?;
        let input = cur.grid()?;
        cur.keyword("OUTPUT")?;
        let output = cur.grid()?;

        examples.push(Pair { input, output });
    }

    cur.keyword("TEST INPUT")?;

    let test_input = cur.grid()?;
    let mut previous = Vec::new();
    let mut candidates = None;

    if judge {
        cur.keyword("ANSWER A")?;
        let a = cur.grid()?;
        cur.keyword("ANSWER B")?;
        let b = cur.grid()?;

        candidates = Some((a, b));
    } else {
        while cur.accept(&format!("PREVIOUS ANSWER {}", previous.len() + 1)) {
            previous.push(cur.grid()?);
        }
    }

    cur.keyword("END")?;

    Some(ParsedPrompt {
        judge,
        examples,
        test_input,
        previous,
        candidates,
    })
}

/// Grid lines after the LAST line that trims to exactly "ANSWER"; stops at the first blank
/// or non-grid line (blank lines directly after ANSWER are skipped); must be a valid grid.
/// A row of numbers that is not a grid row (a number above 9) gives None, not a cut grid.
pub fn parse_answer(text: &str) -> Option<Grid> {
    let mut cur = Cursor::new(text);
    let last = cur.lines.iter().rposition(|line| *line == "ANSWER")?;

    cur.at = last + 1;
    cur.grid()
}

/// A judge's choice between the candidates A and B.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verdict {
    A,
    B,
}

/// The last line of the form "VERDICT A" / "VERDICT B" (trimmed).
pub fn parse_verdict(text: &str) -> Option<Verdict> {
    text.lines().rev().find_map(|line| match line.trim() {
        "VERDICT A" => Some(Verdict::A),
        "VERDICT B" => Some(Verdict::B),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(rows: &[&[u8]]) -> Grid {
        rows.iter().map(|row| row.to_vec()).collect()
    }

    /// The colours that occur in any of `grids`.
    fn colours<'a>(grids: impl IntoIterator<Item = &'a Grid>) -> HashSet<u8> {
        grids.into_iter().flatten().flatten().copied().collect()
    }

    /// A one-example puzzle (mirror left-right), for exact-format tests.
    fn tiny() -> Puzzle {
        let rule = Rule {
            geo: Geo::FlipH,
            colours: None,
        };

        build_puzzle(rule, vec![grid(&[&[1, 2]])], grid(&[&[3, 0]]))
    }

    #[test]
    fn valid_grids_are_rectangular_small_and_coloured() {
        assert!(is_valid_grid(&grid(&[&[0]])));
        assert!(is_valid_grid(&vec![vec![9; MAX_SIDE]; MAX_SIDE]));
        assert!(!is_valid_grid(&Grid::new()), "no rows");
        assert!(!is_valid_grid(&vec![vec![]]), "empty row");
        assert!(!is_valid_grid(&grid(&[&[1, 2], &[3]])), "ragged");
        assert!(!is_valid_grid(&grid(&[&[10]])), "colour above 9");
        assert!(!is_valid_grid(&vec![vec![0; MAX_SIDE + 1]]), "too wide");
        assert!(!is_valid_grid(&vec![vec![0]; MAX_SIDE + 1]), "too tall");
    }

    #[test]
    fn geometric_steps_move_cells_as_named() {
        let g = grid(&[&[1, 2, 3], &[4, 5, 6]]);
        let step = |geo: Geo| apply_geo(geo, &g);

        assert_eq!(step(Geo::Identity), g);
        assert_eq!(step(Geo::FlipH), grid(&[&[3, 2, 1], &[6, 5, 4]]));
        assert_eq!(step(Geo::FlipV), grid(&[&[4, 5, 6], &[1, 2, 3]]));
        assert_eq!(step(Geo::Rot90), grid(&[&[4, 1], &[5, 2], &[6, 3]]));
        assert_eq!(step(Geo::Rot180), grid(&[&[6, 5, 4], &[3, 2, 1]]));
        assert_eq!(step(Geo::Rot270), grid(&[&[3, 6], &[2, 5], &[1, 4]]));
        assert_eq!(step(Geo::Transpose), grid(&[&[1, 4], &[2, 5], &[3, 6]]));
        assert_eq!(apply_geo(Geo::Rot270, &step(Geo::Rot90)), g);

        // Background stays on top; the colours of a column keep their order.
        let fall = |g: &Grid| apply_geo(Geo::GravityDown, g);
        let settled = grid(&[&[0, 0, 0], &[1, 0, 0], &[4, 3, 2]]);

        assert_eq!(fall(&grid(&[&[1, 0, 2], &[0, 3, 0], &[4, 0, 0]])), settled);
        assert_eq!(fall(&settled), settled);
    }

    #[test]
    fn steps_never_panic_on_odd_grids() {
        let ragged = grid(&[&[1, 2], &[3]]);

        for g in [
            Grid::new(),
            vec![vec![]],
            ragged.clone(),
            grid(&[&[1], &[2, 3]]),
        ] {
            for geo in MOVES {
                let _ = apply_geo(geo, &g);
            }
        }

        // A ragged grid is read at the width of its first row, short rows padded with 0.
        assert_eq!(apply_geo(Geo::Identity, &ragged), grid(&[&[1, 2], &[3, 0]]));
    }

    #[test]
    fn colour_map_fits_only_consistent_pairs() {
        let fit = |pairs: &[(Grid, Grid)]| {
            let refs: Vec<(&Grid, &Grid)> = pairs.iter().map(|(x, y)| (x, y)).collect();

            fit_colour_map(&refs)
        };
        let (x, y) = (grid(&[&[1, 2], &[0, 1]]), grid(&[&[3, 2], &[0, 3]]));
        let map = fit(&[(x.clone(), y.clone())]).expect("consistent");
        let recolour = Rule {
            geo: Geo::Identity,
            colours: Some(map),
        };
        let more = [(x.clone(), y.clone()), (grid(&[&[5, 1]]), grid(&[&[6, 3]]))];

        assert_eq!(map, [0, 3, 2, 3, 4, 5, 6, 7, 8, 9], "unseen colours stay");
        assert_eq!(apply(&recolour, &x), y);
        assert_eq!(fit(&more).map(|m| (m[1], m[5])), Some((3, 6)));
        assert_eq!(fit(&[]), Some(SAME_COLOURS));

        let refused = [
            (
                "1 -> 3, then 1 -> 1",
                vec![(x.clone(), y), (grid(&[&[1]]), grid(&[&[1]]))],
            ),
            (
                "1 -> 2 and 1 -> 3",
                vec![(grid(&[&[1, 1]]), grid(&[&[2, 3]]))],
            ),
            ("shapes differ", vec![(x.clone(), grid(&[&[3, 2]]))]),
            ("row lengths differ", vec![(x, grid(&[&[3, 2], &[0]]))]),
            ("not a colour", vec![(grid(&[&[1]]), grid(&[&[12]]))]),
        ];

        for (what, pairs) in &refused {
            assert_eq!(fit(pairs), None, "{what}");
        }
    }

    #[test]
    fn panel_is_deterministic_and_seed_sensitive() {
        let a = generate_panel(7, 25);

        assert_eq!(a.len(), 25);
        assert_eq!(a, generate_panel(7, 25));
        assert_ne!(a, generate_panel(8, 25));
        assert_eq!(generate_panel(7, 30)[..25], a[..], "extends");
        assert!(generate_panel(7, 0).is_empty());
    }

    #[test]
    fn panel_puzzles_are_well_formed_and_well_posed() {
        let mut ids = HashSet::new();

        for p in generate_panel(3, 300) {
            let examples = p.train.iter().map(|t| (&t.input, &t.output));
            let changed = p.train.iter().filter(|t| t.output != t.input).count();
            let shown = colours(p.train.iter().map(|t| &t.input));

            assert_eq!(p.train.len(), TRAIN_PAIRS);

            for (input, output) in examples.chain([(&p.test_input, &p.test_output)]) {
                assert!(is_valid_grid(input) && is_valid_grid(output), "{}", p.id);
                assert!((3..=6).contains(&input.len()) && (3..=6).contains(&input[0].len()));
                assert_eq!(*output, apply(&p.rule, input), "{}", p.id);
            }

            assert!(colours([&p.test_input]).is_subset(&shown), "{}", p.id);
            assert!(p.test_output != p.test_input && changed >= 2, "{}", p.id);
            assert!(is_well_posed(&p), "{}", p.id);
            assert_eq!(
                p.id,
                puzzle_id(&p.train, &p.test_input, &p.test_output, None)
            );
            assert!(ids.insert(p.id.clone()), "duplicate id {}", p.id);
        }
    }

    #[test]
    fn rule_mix_follows_the_design() {
        let panel = generate_panel(11, 600);
        let share = |f: &dyn Fn(&Rule) -> bool| {
            panel.iter().filter(|p| f(&p.rule)).count() as f64 / panel.len() as f64
        };
        let mix = [
            ("step only", share(&|r| r.colours.is_none()), 0.37..=0.53),
            (
                "colours only",
                share(&|r| r.geo == Geo::Identity),
                0.18..=0.32,
            ),
            (
                "both",
                share(&|r| r.geo != Geo::Identity && r.colours.is_some()),
                0.23..=0.37,
            ),
        ];

        for (what, measured, expected) in mix {
            assert!(expected.contains(&measured), "{what}: {measured}");
        }

        for geo in MOVES {
            assert!(
                panel.iter().any(|p| p.rule.geo == geo),
                "{geo:?} never drawn"
            );
        }

        let no_op = Rule {
            geo: Geo::Identity,
            colours: None,
        };

        assert!(panel.iter().all(|p| p.rule != no_op));

        for map in panel.iter().filter_map(|p| p.rule.colours) {
            let mut sorted = map;

            sorted.sort();
            assert_eq!(sorted, SAME_COLOURS, "{map:?} is not a permutation");
            assert!(map != SAME_COLOURS && map[0] == 0, "{map:?}");
        }
    }

    #[test]
    fn fallback_puzzle_is_well_posed() {
        let p = fallback_puzzle();

        assert!(is_well_posed(&p));
        assert_eq!(p.train.len(), TRAIN_PAIRS);
    }

    #[test]
    fn prompts_have_the_exact_format() {
        let p = tiny();
        let (a, b) = (grid(&[&[0, 3]]), grid(&[&[3, 0]]));
        let task = "EXAMPLE 1\nINPUT\n1 2\nOUTPUT\n2 1\nTEST INPUT\n3 0";

        assert_eq!(
            render_solve_prompt(&p, std::slice::from_ref(&a)),
            format!("TASK\n{SOLVE_INTRO}\n{task}\nPREVIOUS ANSWER 1\n0 3\nEND")
        );
        assert_eq!(
            render_judge_prompt(&p, &a, &b),
            format!("JUDGE\n{JUDGE_INTRO}\n{task}\nANSWER A\n0 3\nANSWER B\n3 0\nEND")
        );
    }

    #[test]
    fn prompts_round_trip() {
        for p in generate_panel(5, 10) {
            let two = vec![p.train[0].output.clone(), p.test_input.clone()];

            for previous in [Vec::new(), two] {
                let text = render_solve_prompt(&p, &previous);
                let expected = ParsedPrompt {
                    judge: false,
                    examples: p.train.clone(),
                    test_input: p.test_input.clone(),
                    previous,
                    candidates: None,
                };

                assert_eq!(parse_prompt(&text).as_ref(), Some(&expected));
                assert_eq!(parse_prompt(&text.replace('\n', "\r\n")), Some(expected));
            }

            let (a, b) = (p.train[1].output.clone(), p.test_input.clone());
            let judged = parse_prompt(&render_judge_prompt(&p, &a, &b)).expect("judge prompt");

            assert!(judged.judge && judged.previous.is_empty());
            assert_eq!(
                (judged.examples, judged.test_input),
                (p.train, p.test_input)
            );
            assert_eq!(judged.candidates, Some((a, b)));
        }
    }

    #[test]
    fn prompts_never_carry_the_ground_truth() {
        let p = generate_panel(9, 1).remove(0);
        let mut q = p.clone();

        q.id = "another".to_string();
        q.rule.geo = Geo::Identity;
        q.test_output = vec![vec![7; 5]; 4];

        let previous = [p.train[0].output.clone()];
        let (a, b) = (&p.test_input, &p.train[2].output);

        assert_eq!(render_solve_prompt(&p, &[]), render_solve_prompt(&q, &[]));
        assert_eq!(
            render_solve_prompt(&p, &previous),
            render_solve_prompt(&q, &previous)
        );
        assert_eq!(render_judge_prompt(&p, a, b), render_judge_prompt(&q, a, b));
        assert!(!render_solve_prompt(&p, &[]).contains(&p.id));
    }

    #[test]
    fn malformed_prompts_are_refused() {
        let p = tiny();
        let solve = render_solve_prompt(&p, &[]);
        let judge = render_judge_prompt(&p, &p.test_input, &p.test_input);
        let broken = [
            ("unknown header", solve.replace("TASK", "QUEST")),
            ("numbering", solve.replace("EXAMPLE 1", "EXAMPLE 2")),
            ("no END", solve.replace("\nEND", "")),
            ("colour above 9", solve.replace("3 0", "3 10")),
            ("no candidate B", judge.replace("ANSWER B", "ANSWER C")),
            (
                "candidate in a solve prompt",
                solve.replace("END", "ANSWER A\n1\nEND"),
            ),
        ];

        assert!(parse_prompt(&solve).is_some() && parse_prompt(&judge).is_some());

        for (what, text) in &broken {
            assert_eq!(parse_prompt(text), None, "{what}");
        }

        // Every strict prefix of a prompt is cut somewhere and must be refused.
        for (i, _) in solve.char_indices() {
            assert_eq!(parse_prompt(&solve[..i]), None, "prefix of {i} bytes");
        }
    }

    #[test]
    fn parse_answer_reads_the_last_answer_block() {
        let two_by_two = Some(grid(&[&[1, 2], &[3, 4]]));
        let cases = [
            (
                "Let me think.\nThe rule flips.\nANSWER\n1 2\n3 4",
                two_by_two.clone(),
            ),
            (
                "ANSWER\n1 1\n\nNo, wait:\nANSWER\n1 2\n3 4\nDone.",
                two_by_two,
            ),
            ("  ANSWER  \r\n\r\n0 1 2\r\n", Some(grid(&[&[0, 1, 2]]))),
            ("ANSWER\n1 2\n\n3 4", Some(grid(&[&[1, 2]]))),
            ("ANSWER\n1 2\n3", None),
            ("ANSWER\n1 2\n3 12", None),
            ("ANSWER\n1 2\n3 4\nANSWER", None),
            ("1 2\n3 4", None),
            ("ANSWER: 1 2", None),
        ];

        for (text, expected) in cases {
            assert_eq!(parse_answer(text), expected, "{text:?}");
        }

        assert_eq!(
            parse_answer(&render_answer(&grid(&[&[5, 0]]))),
            Some(grid(&[&[5, 0]]))
        );
    }

    #[test]
    fn parse_verdict_takes_the_last_verdict_line() {
        let cases = [
            ("VERDICT A", Some(Verdict::A)),
            ("thinking...\n  VERDICT B  \n", Some(Verdict::B)),
            ("VERDICT A\nno, wait\nVERDICT B", Some(Verdict::B)),
            ("VERDICT B\nVERDICT C", Some(Verdict::B)),
            ("My VERDICT A is final", None),
            ("verdict a", None),
            ("", None),
            (render_verdict(Verdict::A), Some(Verdict::A)),
        ];

        for (text, expected) in cases {
            assert_eq!(parse_verdict(text), expected, "{text:?}");
        }
    }

    #[test]
    fn parsers_survive_garbage() {
        let p = generate_panel(2, 1).remove(0);
        let wide = format!("ANSWER\n{}", "1 ".repeat(10_000));
        let tall = format!("ANSWER\n{}", "1\n".repeat(10_000));
        let huge = "ANSWER\n99999999999999999999999999999999 1".to_string();
        let mut texts: Vec<String> = [
            "",
            "\n\n\n",
            "TASK",
            "JUDGE\nEND",
            "TASK\nEXAMPLE 1\nINPUT\nOUTPUT\nTEST INPUT\nEND",
            "ANSWER\n-1 0\n1.5 2",
            "Grüße 🌍\nANSWER\n１ ２\n٣",
            "\u{0}\u{1}\u{7f}\u{feff}ANSWER\u{2028}1",
            "VERDICT\nVERDICT AB\nVERDICT",
        ]
        .map(String::from)
        .to_vec();

        texts.push(format!("TASK\nEXAMPLE 1\nINPUT\n{}", "2 2\n".repeat(5_000)));

        for text in [
            render_solve_prompt(&p, std::slice::from_ref(&p.test_input)),
            render_judge_prompt(&p, &p.test_input, &p.train[0].output),
        ] {
            texts.extend(text.char_indices().map(|(i, _)| text[..i].to_string()));
        }

        for text in texts.iter().chain([&wide, &tall, &huge]) {
            let _ = (parse_prompt(text), parse_answer(text), parse_verdict(text));
        }

        for text in [&wide, &tall, &huge] {
            assert_eq!(parse_answer(text), None);
        }
    }
}
