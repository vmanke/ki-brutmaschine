//! The plain-text protocol of circuit tasks (step 1, ticket 03): solve and judge prompts
//! that show a target unitary, and the parsers that read prompts and answers back.
//!
//! This is the evaluator side. A prompt shows only the target, the gate set, the gate cap
//! and earlier answers; a task's name, id or reference cost is never rendered. The parsers
//! take arbitrary model output: anything that does not follow the protocol gives None,
//! never a panic and never a cut circuit.
//!
//! A solve prompt is `CIRCUIT TASK`, a free instruction line, `QUBITS n`, the gate set
//! line, `MAX GATES m`, `TARGET` with one row per input basis state, the answers of earlier
//! stages (`PREVIOUS ANSWER k`, one gate per line) and `END`. A judge prompt is
//! `CIRCUIT JUDGE`, its instruction, the same target block, `ANSWER A` and `ANSWER B` with
//! their gates, and `END`.
//!
//! A row such as `|10> -> +|10> -|11>` maps one input to the normalised equal-weight sum of
//! its terms. A term is a sign, an optional phase symbol and a ket: + is w^0, +w is w^1, +i
//! is w^2, +iw is w^3, and - negates (w^4), with w = e^(i pi/4). Bits list qubit 0 first.
//! Terms are separated by single spaces, and the rows list every input in order.
//!
//! A Worker replies with the line `ANSWER` and one gate per line, such as `H 0` or
//! `CNOT 0 1` (control first). A judge replies with `VERDICT A` or `VERDICT B`, which
//! `puzzle::parse_verdict` reads.
#![forbid(unsafe_code)]

use crate::cascade::qubit::{Circuit, Column, Gate, MAX_GATES, MAX_UNITARY_QUBITS, Target};

/// Text appended to a Worker's system prompt by the orchestrator when its `self_check`
/// Trait is on, for circuit tasks.
pub const CIRCUIT_SELF_CHECK_TEXT: &str = "Before you answer, run your circuit on every input basis state and compare it with the target.";

/// The instruction line of a solve prompt.
const SOLVE_INTRO: &str = "Build a circuit from the gates below that maps every input basis state as shown, up to one global phase. Use as few T and TDG gates as you can, then as few CNOT gates. Bits list qubit 0 first. w = e^(i pi/4); each line maps one input to the normalised equal-weight sum of its terms.";

/// The instruction line of a judge prompt.
const JUDGE_INTRO: &str = "A circuit that implements the target exactly, up to one global phase, beats one that does not; between two that do, fewer T and TDG gates win, then fewer CNOT gates. Reply with the line VERDICT A or VERDICT B.";

/// The gate set line, the same in every prompt.
const GATES_LINE: &str = "GATES H S SDG T TDG X Z CNOT";

/// The sign and phase symbol of a term, indexed by k for w^k.
const PHASES: [&str; 8] = ["+", "+w", "+i", "+iw", "-", "-w", "-i", "-iw"];

/// A basis index as a ket of n bits, qubit 0 first: 6 on three qubits is |110>.
fn ket(index: u32, qubits: usize) -> String {
    let bits: String = (0..qubits)
        .map(|q| {
            if (index >> (qubits - 1 - q)) & 1 == 1 {
                '1'
            } else {
                '0'
            }
        })
        .collect();

    format!("|{bits}>")
}

/// The target block of both prompts: qubit count, gate set, gate cap (at most MAX_GATES)
/// and one row per input. This is the only place a prompt reads a target.
fn push_target(out: &mut String, target: &Target, max_gates: usize) {
    let n = target.qubits();

    out.push_str(&format!(
        "QUBITS {n}\n{GATES_LINE}\nMAX GATES {}\nTARGET\n",
        max_gates.min(MAX_GATES)
    ));

    for (j, column) in (0u32..).zip(target.columns()) {
        out.push_str(&ket(j, n));
        out.push_str(" ->");

        for &(b, p) in column.terms() {
            out.push(' ');
            out.push_str(PHASES[usize::from(p % 8)]);
            out.push_str(&ket(b, n));
        }

        out.push('\n');
    }
}

/// Append a block: its title line, then one gate per line.
fn push_circuit(out: &mut String, title: &str, c: &Circuit) {
    out.push_str(title);
    out.push('\n');

    for g in c.gates() {
        out.push_str(&g.render());
        out.push('\n');
    }
}

/// The solve prompt: the target block and the answers of earlier stages (the residual path
/// of a Cascade), oldest first. The gate cap is written as at most MAX_GATES. Circuits are
/// written as they are; the parser refuses one that leaves the target's qubits.
pub fn render_circuit_solve_prompt(
    target: &Target,
    max_gates: usize,
    previous: &[Circuit],
) -> String {
    let mut out = format!("CIRCUIT TASK\n{SOLVE_INTRO}\n");

    push_target(&mut out, target, max_gates);

    for (k, c) in previous.iter().enumerate() {
        push_circuit(&mut out, &format!("PREVIOUS ANSWER {}", k + 1), c);
    }

    out.push_str("END");
    out
}

/// The judge prompt: the target block and the candidates A and B.
pub fn render_circuit_judge_prompt(
    target: &Target,
    max_gates: usize,
    a: &Circuit,
    b: &Circuit,
) -> String {
    let mut out = format!("CIRCUIT JUDGE\n{JUDGE_INTRO}\n");

    push_target(&mut out, target, max_gates);
    push_circuit(&mut out, "ANSWER A", a);
    push_circuit(&mut out, "ANSWER B", b);
    out.push_str("END");
    out
}

/// A circuit prompt as read back from its text. A simulated designer sees only this.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsedCircuitPrompt {
    /// True for a judge prompt, false for a solve prompt.
    pub judge: bool,
    pub target: Target,
    /// The gate cap the prompt states.
    pub max_gates: usize,
    /// The answers of earlier stages, oldest first (solve prompts only).
    pub previous: Vec<Circuit>,
    /// The candidates A and B (judge prompts only).
    pub candidates: Option<(Circuit, Circuit)>,
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

    /// The next non-blank line, consumed.
    fn next_line(&mut self) -> Option<&'a str> {
        self.skip_blank();

        let line = self.peek()?;

        self.at += 1;

        Some(line)
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

    /// The number after `prefix` on the next non-blank line: "QUBITS 3" gives 3. Digits only.
    fn number(&mut self, prefix: &str) -> Option<usize> {
        let digits = self.next_line()?.strip_prefix(prefix)?;

        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }

        digits.parse().ok()
    }

    /// Gate lines up to the first line that is not a gate, which stays unread (it must be
    /// the next keyword). None if a gate leaves the register or the block breaks a cap.
    fn circuit(&mut self, qubits: usize) -> Option<Circuit> {
        self.skip_blank();

        let mut gates = Vec::new();

        while let Some(g) = self.peek().and_then(Gate::parse) {
            if !fits(g, qubits) || gates.len() == MAX_GATES {
                return None;
            }

            gates.push(g);
            self.at += 1;
        }

        Circuit::new(gates).ok()
    }
}

/// True when every qubit of `g` is below `qubits`.
fn fits(g: Gate, qubits: usize) -> bool {
    let (a, b) = g.qubits();

    usize::from(a) < qubits && b.is_none_or(|b| usize::from(b) < qubits)
}

/// A ket of exactly n bits, as its basis index.
fn parse_ket(word: &str, qubits: usize) -> Option<u32> {
    let bits = word.strip_prefix('|')?.strip_suffix('>')?;

    if bits.len() != qubits || !bits.bytes().all(|b| b == b'0' || b == b'1') {
        return None;
    }

    u32::from_str_radix(bits, 2).ok()
}

/// One term: sign and phase symbol, then a ket, with nothing between them.
fn parse_term(word: &str, qubits: usize) -> Option<(u32, u8)> {
    let (symbol, ket) = word.split_at(word.find('|')?);
    let phase = PHASES.iter().position(|&s| s == symbol)?;

    Some((parse_ket(ket, qubits)?, u8::try_from(phase).ok()?))
}

/// Row j of the target: its input ket, "->", then the terms, all separated by single
/// spaces.
fn parse_row(line: &str, j: u32, qubits: usize) -> Option<Column> {
    let mut words = line.split(' ');

    if parse_ket(words.next()?, qubits)? != j || words.next()? != "->" {
        return None;
    }

    let terms = words
        .map(|w| parse_term(w, qubits))
        .collect::<Option<Vec<(u32, u8)>>>()?;

    Column::new(terms).ok()
}

/// Read a solve or judge prompt back. Lines are trimmed. Blank lines are skipped before the
/// header, between header lines, between target rows and around a gate block; a blank line
/// inside a gate block ends that block, so gates after it make the prompt None. The
/// instruction text up to `QUBITS` is free, and text after `END` is ignored, as in the grid
/// protocol. Any other deviation gives None: a qubit count outside
/// 1..=MAX_UNITARY_QUBITS, a gate cap above MAX_GATES, a row that is out of order or not
/// well formed, a target that is not exactly unitary, or a block with a gate outside the
/// register.
pub fn parse_circuit_prompt(text: &str) -> Option<ParsedCircuitPrompt> {
    let mut cur = Cursor::new(text);
    let judge = match cur.next_line()? {
        "CIRCUIT TASK" => false,
        "CIRCUIT JUDGE" => true,
        _ => return None,
    };

    while !cur.peek()?.starts_with("QUBITS") {
        cur.at += 1;
    }

    let qubits = cur.number("QUBITS ")?;

    if !(1..=MAX_UNITARY_QUBITS).contains(&qubits) {
        return None;
    }

    cur.keyword(GATES_LINE)?;

    let max_gates = cur.number("MAX GATES ")?;

    if max_gates > MAX_GATES {
        return None;
    }

    cur.keyword("TARGET")?;

    let mut columns = Vec::new();

    for j in 0..(1u32 << qubits) {
        columns.push(parse_row(cur.next_line()?, j, qubits)?);
    }

    let target = Target::new(qubits, columns).ok()?;
    let mut previous = Vec::new();
    let mut candidates = None;

    if judge {
        cur.keyword("ANSWER A")?;
        let a = cur.circuit(qubits)?;
        cur.keyword("ANSWER B")?;
        let b = cur.circuit(qubits)?;

        candidates = Some((a, b));
    } else {
        while cur.accept(&format!("PREVIOUS ANSWER {}", previous.len() + 1)) {
            previous.push(cur.circuit(qubits)?);
        }
    }

    cur.keyword("END")?;

    Some(ParsedCircuitPrompt {
        judge,
        target,
        max_gates,
        previous,
        candidates,
    })
}

/// How one line of an answer reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateLine {
    /// Exactly one gate.
    Gate(Gate),
    /// Not gate-like: a blank line or prose. It ends a block of gates.
    NotAGate,
    /// Gate-like, but not one valid gate. It spoils the whole answer.
    Malformed,
}

/// Classify one line. A line that `Gate::parse` reads is a gate. Otherwise the line is
/// split into words at every character that is not a letter or digit, and it is
/// gate-like, and so malformed, when its first word begins with a gate name or alias that
/// no other letter follows (H0, S† 1, CNOT(0,1)), or when any word holds a digit (Y(0),
/// CZ 0 1, `X 0`, 2. CNOT 0 1, This uses 1 CNOT.). A blank line or a line of words
/// without digits is not a gate. Prose after the gates must therefore follow a blank line
/// or hold no digit; otherwise the answer is refused rather than cut.
pub fn parse_gate_line(line: &str) -> GateLine {
    if let Some(g) = Gate::parse(line) {
        return GateLine::Gate(g);
    }

    let mut words = line
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty());
    let Some(first) = words.next() else {
        return GateLine::NotAGate;
    };
    let lead: String = first.chars().take_while(|c| c.is_alphabetic()).collect();
    let numeric = |w: &str| w.chars().any(char::is_numeric);

    if Gate::is_name(&lead) || numeric(first) || words.any(numeric) {
        GateLine::Malformed
    } else {
        GateLine::NotAGate
    }
}

/// The circuit after the LAST line that trims to exactly "ANSWER". Blank lines right after
/// it are skipped; then gate lines are read up to the first blank or non-gate line. Any
/// malformed line gives None, never a cut circuit. So do a qubit index of `qubits` or more,
/// no gates at all, more than `max_gates` gates (never more than MAX_GATES), and a circuit
/// that breaks another cap, such as more than MAX_H H gates.
pub fn parse_circuit_answer(text: &str, qubits: usize, max_gates: usize) -> Option<Circuit> {
    let lines: Vec<&str> = text.lines().map(str::trim).collect();
    let last = lines.iter().rposition(|line| *line == "ANSWER")?;
    let limit = max_gates.min(MAX_GATES);
    let mut gates = Vec::new();

    for line in lines
        .iter()
        .skip(last + 1)
        .skip_while(|line| line.is_empty())
    {
        match parse_gate_line(line) {
            GateLine::Gate(g) if fits(g, qubits) && gates.len() < limit => gates.push(g),
            GateLine::Gate(_) | GateLine::Malformed => return None,
            GateLine::NotAGate => break,
        }
    }

    if gates.is_empty() {
        return None;
    }

    Circuit::new(gates).ok()
}

/// A Worker's reply in protocol form: the line ANSWER, then one gate per line.
pub fn render_circuit_answer(c: &Circuit) -> String {
    let mut out = String::from("ANSWER");

    for g in c.gates() {
        out.push('\n');
        out.push_str(&g.render());
    }

    out
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::cascade::qubit::{Dressing, MAX_H, MAX_QUBITS};

    fn circuit(text: &str) -> Circuit {
        text.parse().expect("circuit text")
    }

    fn column(terms: Vec<(u32, u8)>) -> Column {
        Column::new(terms).expect("column")
    }

    /// Controlled-H with control qubit 0: the example of the plan.
    fn ch() -> Target {
        let columns = vec![
            Column::single(0, 0),
            Column::single(1, 0),
            column(vec![(2, 0), (3, 0)]),
            column(vec![(2, 0), (3, 4)]),
        ];

        Target::new(2, columns).expect("unitary")
    }

    /// The two-qubit Fourier transform: entry (b, j) is i^(b j) / 2.
    fn qft2() -> Target {
        let columns = (0..4u32)
            .map(|j| {
                column(
                    (0..4u32)
                        .map(|b| (b, u8::try_from(2 * j * b % 8).expect("phase")))
                        .collect(),
                )
            })
            .collect();

        Target::new(2, columns).expect("unitary")
    }

    fn toffoli() -> Target {
        Target::monomial(3, |j| (if j >= 6 { j ^ 1 } else { j }, 0)).expect("toffoli")
    }

    /// diag(w^0, ..., w^7) on three qubits: every phase symbol once.
    fn phases() -> Target {
        Target::monomial(3, |j| (j, u8::try_from(j).expect("small"))).expect("diagonal")
    }

    /// A random circuit of `len` gates on n qubits (CNOTs only when n > 1).
    fn random_circuit(rng: &mut ChaCha8Rng, n: usize, len: usize) -> Circuit {
        let kinds = if n > 1 { 8 } else { 7 };
        let gates = (0..len)
            .map(|_| {
                let q = rng.random_range(0..n);
                let qb = u8::try_from(q).expect("small index");

                match rng.random_range(0..kinds) {
                    0 => Gate::H(qb),
                    1 => Gate::S(qb),
                    2 => Gate::Sdg(qb),
                    3 => Gate::T(qb),
                    4 => Gate::Tdg(qb),
                    5 => Gate::X(qb),
                    6 => Gate::Z(qb),
                    _ => Gate::Cnot {
                        control: qb,
                        target: u8::try_from((q + rng.random_range(1..n)) % n)
                            .expect("small index"),
                    },
                }
            })
            .collect();

        Circuit::new(gates).expect("within the caps")
    }

    /// The prompt a parsed prompt renders to.
    fn rendered(p: &ParsedCircuitPrompt) -> String {
        match &p.candidates {
            Some((a, b)) if p.judge => render_circuit_judge_prompt(&p.target, p.max_gates, a, b),
            _ => render_circuit_solve_prompt(&p.target, p.max_gates, &p.previous),
        }
    }

    #[test]
    fn prompts_have_the_exact_format() {
        let block = "QUBITS 2\nGATES H S SDG T TDG X Z CNOT\nMAX GATES 64\nTARGET\n\
                     |00> -> +|00>\n|01> -> +|01>\n|10> -> +|10> +|11>\n|11> -> +|10> -|11>";
        let solve = format!(
            "CIRCUIT TASK\nBuild a circuit from the gates below that maps every input basis \
             state as shown, up to one global phase. Use as few T and TDG gates as you can, \
             then as few CNOT gates. Bits list qubit 0 first. w = e^(i pi/4); each line maps \
             one input to the normalised equal-weight sum of its terms.\n{block}\n\
             PREVIOUS ANSWER 1\nH 1\nEND"
        );
        let judge = format!(
            "CIRCUIT JUDGE\nA circuit that implements the target exactly, up to one global \
             phase, beats one that does not; between two that do, fewer T and TDG gates win, \
             then fewer CNOT gates. Reply with the line VERDICT A or VERDICT B.\n{block}\n\
             ANSWER A\nH 1\nANSWER B\nS 1\nH 1\nT 1\nCNOT 0 1\nTDG 1\nH 1\nSDG 1\nEND"
        );
        let witness = circuit("S 1; H 1; T 1; CNOT 0 1; TDG 1; H 1; SDG 1");

        assert_eq!(
            render_circuit_solve_prompt(&ch(), 64, &[circuit("H 1")]),
            solve
        );
        assert_eq!(
            render_circuit_judge_prompt(&ch(), 64, &circuit("H 1"), &witness),
            judge
        );
        assert!(render_circuit_solve_prompt(&ch(), 64, &[]).ends_with("+|10> -|11>\nEND"));
        assert!(render_circuit_solve_prompt(&ch(), 1000, &[]).contains("\nMAX GATES 256\n"));
        assert_eq!(
            render_circuit_answer(&circuit("H 0; CNOT 0 1")),
            "ANSWER\nH 0\nCNOT 0 1"
        );
        assert_eq!(render_circuit_answer(&Circuit::default()), "ANSWER");
        assert_eq!(
            CIRCUIT_SELF_CHECK_TEXT,
            "Before you answer, run your circuit on every input basis state and compare it with the target."
        );
    }

    #[test]
    fn rows_spell_every_phase_symbol() {
        let rows = "|000> -> +|000>\n|001> -> +w|001>\n|010> -> +i|010>\n|011> -> +iw|011>\n\
                    |100> -> -|100>\n|101> -> -w|101>\n|110> -> -i|110>\n|111> -> -iw|111>";
        let fourier = "|00> -> +|00> +|01> +|10> +|11>\n|01> -> +|00> +i|01> -|10> -i|11>\n\
                       |10> -> +|00> -|01> +|10> -|11>\n|11> -> +|00> -i|01> -|10> +i|11>";

        assert!(
            render_circuit_solve_prompt(&phases(), 64, &[])
                .ends_with(&format!("TARGET\n{rows}\nEND"))
        );
        assert!(
            render_circuit_solve_prompt(&qft2(), 64, &[])
                .ends_with(&format!("TARGET\n{fourier}\nEND"))
        );
        assert!(
            render_circuit_solve_prompt(&toffoli(), 8, &[])
                .contains("|110> -> +|111>\n|111> -> +|110>\nEND")
        );
    }

    #[test]
    fn prompts_round_trip() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut targets = vec![
            ch(),
            qft2(),
            toffoli(),
            phases(),
            Target::monomial(1, |j| (j ^ 1, 3)).expect("one qubit"),
            Target::monomial(6, |j| (j ^ 0b10_1010, u8::try_from(j % 8).expect("small")))
                .expect("six qubits"),
        ];

        targets.extend((0..4).map(|_| toffoli().dress(&Dressing::draw(3, &mut rng))));

        for target in &targets {
            let n = target.qubits();
            let previous = vec![
                random_circuit(&mut rng, n, 5),
                Circuit::default(),
                random_circuit(&mut rng, n, 12),
            ];

            for prev in [Vec::new(), previous] {
                for max_gates in [64, 0, 256] {
                    let text = render_circuit_solve_prompt(target, max_gates, &prev);
                    let expected = ParsedCircuitPrompt {
                        judge: false,
                        target: target.clone(),
                        max_gates,
                        previous: prev.clone(),
                        candidates: None,
                    };

                    assert_eq!(parse_circuit_prompt(&text).as_ref(), Some(&expected));
                    assert_eq!(
                        parse_circuit_prompt(&text.replace('\n', "\r\n")),
                        Some(expected)
                    );
                }
            }

            let (a, b) = (
                random_circuit(&mut rng, n, 7),
                random_circuit(&mut rng, n, 3),
            );
            let judged = parse_circuit_prompt(&render_circuit_judge_prompt(target, 64, &a, &b));

            assert_eq!(
                judged,
                Some(ParsedCircuitPrompt {
                    judge: true,
                    target: target.clone(),
                    max_gates: 64,
                    previous: Vec::new(),
                    candidates: Some((a, b)),
                })
            );
        }

        // Free instruction text, blank lines between lines and text after END are allowed,
        // as in the grid protocol.
        let text = render_circuit_solve_prompt(&ch(), 64, &[circuit("H 1")]);
        let loose = text
            .replacen(SOLVE_INTRO, "Any words.\n\nMore words.", 1)
            .replace("\nTARGET\n", "\n\nTARGET\n\n")
            .replace("\nH 1\n", "\n\nH 1\n\n")
            + "\nThanks.";

        assert_eq!(parse_circuit_prompt(&loose), parse_circuit_prompt(&text));
        assert!(parse_circuit_prompt(&text).is_some());
    }

    #[test]
    fn every_strict_prefix_is_refused() {
        let solve =
            render_circuit_solve_prompt(&qft2(), 64, &[circuit("H 0; CNOT 0 1"), circuit("T 1")]);
        let judge =
            render_circuit_judge_prompt(&phases(), 9, &circuit("H 0"), &circuit("CNOT 1 0; S 2"));

        for text in [&solve, &judge] {
            assert!(parse_circuit_prompt(text).is_some());

            for (i, _) in text.char_indices() {
                assert_eq!(
                    parse_circuit_prompt(&text[..i]),
                    None,
                    "prefix of {i} bytes"
                );
            }
        }
    }

    #[test]
    fn malformed_prompts_are_refused() {
        let solve = render_circuit_solve_prompt(&ch(), 64, &[circuit("H 1")]);
        let judge = render_circuit_judge_prompt(&ch(), 64, &circuit("H 1"), &circuit("X 0"));
        let broken = [
            ("header", solve.replace("CIRCUIT TASK", "CIRCUIT TASKS")),
            ("grid header", solve.replace("CIRCUIT TASK", "TASK")),
            ("no qubit count", solve.replace("QUBITS 2\n", "")),
            ("zero qubits", solve.replace("QUBITS 2", "QUBITS 0")),
            (
                "qubits above the cap",
                solve.replace("QUBITS 2", "QUBITS 7"),
            ),
            (
                "qubit count against the rows",
                solve.replace("QUBITS 2", "QUBITS 1"),
            ),
            ("signed count", solve.replace("QUBITS 2", "QUBITS +2")),
            ("gate set", solve.replace("GATES H S", "GATES H Y S")),
            (
                "gate cap above MAX_GATES",
                solve.replace("MAX GATES 64", "MAX GATES 257"),
            ),
            ("no gate cap", solve.replace("MAX GATES 64\n", "")),
            ("no TARGET line", solve.replace("TARGET\n", "")),
            (
                "input bits too short",
                solve.replace("|01> -> +|01>", "|1> -> +|01>"),
            ),
            (
                "input bits not binary",
                solve.replace("|01> -> +|01>", "|0x> -> +|01>"),
            ),
            ("term bits too long", solve.replace("-> +|01>", "-> +|001>")),
            ("ket without bracket", solve.replace("-> +|01>", "-> +|01")),
            (
                "rows out of order",
                solve.replace(
                    "|00> -> +|00>\n|01> -> +|01>",
                    "|01> -> +|01>\n|00> -> +|00>",
                ),
            ),
            ("a row missing", solve.replace("|01> -> +|01>\n", "")),
            (
                "an extra row",
                solve.replace(
                    "|11> -> +|10> -|11>\n",
                    "|11> -> +|10> -|11>\n|11> -> +|10> -|11>\n",
                ),
            ),
            (
                "not unitary",
                solve.replace("|11> -> +|10> -|11>", "|11> -> +|10> +|11>"),
            ),
            (
                "three terms",
                solve.replace("+|10> -|11>\nPREV", "+|10> -|11> +|00>\nPREV"),
            ),
            (
                "a repeated term",
                solve.replace("+|10> -|11>\nPREV", "+|10> -|10>\nPREV"),
            ),
            ("no terms", solve.replace("|00> -> +|00>", "|00> ->")),
            ("unknown phase symbol", solve.replace("-|11>", "-x|11>")),
            ("phase symbol without sign", solve.replace("-|11>", "i|11>")),
            ("doubled sign", solve.replace("-|11>", "+-|11>")),
            ("space inside a term", solve.replace("-|11>", "- |11>")),
            ("double space", solve.replace("+|10> -|11>", "+|10>  -|11>")),
            ("tab", solve.replace("+|10> -|11>", "+|10>\t-|11>")),
            ("arrow", solve.replace("|00> -> +|00>", "|00> => +|00>")),
            (
                "previous answer numbering",
                solve.replace("PREVIOUS ANSWER 1", "PREVIOUS ANSWER 2"),
            ),
            (
                "gate outside the register",
                solve.replace("\nH 1\nEND", "\nH 2\nEND"),
            ),
            (
                "unknown gate in a block",
                solve.replace("\nH 1\nEND", "\nY 1\nEND"),
            ),
            ("no END", solve.replace("\nEND", "")),
            (
                "candidate in a solve prompt",
                solve.replace("\nEND", "\nANSWER A\nH 0\nEND"),
            ),
            ("no candidate B", judge.replace("ANSWER B", "ANSWER C")),
            (
                "judge without candidates",
                judge.replace("ANSWER A\nH 1\nANSWER B\nX 0\n", ""),
            ),
            (
                "previous answer in a judge prompt",
                judge.replace("\nEND", "\nPREVIOUS ANSWER 1\nH 0\nEND"),
            ),
        ];

        assert!(parse_circuit_prompt(&solve).is_some() && parse_circuit_prompt(&judge).is_some());

        for (what, text) in &broken {
            assert!(text != &solve && text != &judge, "{what}: the edit applies");
            assert_eq!(parse_circuit_prompt(text), None, "{what}");
        }

        // A block of more than MAX_GATES gates is refused, even when every line is a gate.
        let long = vec!["X 0"; MAX_GATES + 1].join("\n");

        assert_eq!(
            parse_circuit_prompt(&solve.replace("\nH 1\nEND", &format!("\n{long}\nEND"))),
            None
        );
    }

    #[test]
    fn gate_lines_are_classified() {
        let gate = |g| GateLine::Gate(g);
        let cases = [
            ("H 0", gate(Gate::H(0))),
            (
                "  cx q0 Q1 ",
                gate(Gate::Cnot {
                    control: 0,
                    target: 1,
                }),
            ),
            ("SDAG 2", gate(Gate::Sdg(2))),
            ("tdag q3", gate(Gate::Tdg(3))),
            ("H 200", gate(Gate::H(200))),
            ("", GateLine::NotAGate),
            ("   ", GateLine::NotAGate),
            ("Done.", GateLine::NotAGate),
            ("The circuit is minimal.", GateLine::NotAGate),
            ("Hadamard first", GateLine::NotAGate),
            ("ANSWER", GateLine::NotAGate),
            ("CNOT 0 0", GateLine::Malformed),
            ("CNOT 0", GateLine::Malformed),
            ("H", GateLine::Malformed),
            ("H 0 1", GateLine::Malformed),
            ("H -1", GateLine::Malformed),
            ("H 256", GateLine::Malformed),
            ("H 1.5", GateLine::Malformed),
            ("H0", GateLine::Malformed),
            ("CNOT(0,1)", GateLine::Malformed),
            ("S† 1", GateLine::Malformed),
            ("T, 0", GateLine::Malformed),
            ("Y 0", GateLine::Malformed),
            ("RZ 1", GateLine::Malformed),
            ("CZ 0 1", GateLine::Malformed),
            ("SWAP q0 q1", GateLine::Malformed),
            ("2. CNOT 0 1", GateLine::Malformed),
            ("- H 0", GateLine::Malformed),
            ("Step 1", GateLine::Malformed),
            ("Y(0)", GateLine::Malformed),
            ("CZ(0,1)", GateLine::Malformed),
            ("SWAP(0,1)", GateLine::Malformed),
            ("(X 0)", GateLine::Malformed),
            ("`T 1`", GateLine::Malformed),
            ("**T 1**", GateLine::Malformed),
            ("cz q[0],q[1];", GateLine::Malformed),
            ("Rz(pi/4) q[0];", GateLine::Malformed),
            ("Ｔ １", GateLine::Malformed),
            ("0 X", GateLine::Malformed),
            ("(Note: the answer is final.)", GateLine::NotAGate),
        ];

        for (line, expected) in cases {
            assert_eq!(parse_gate_line(line), expected, "{line:?}");
        }
    }

    #[test]
    fn answers_follow_the_table() {
        let some = |text: &str| Some(circuit(text));
        let cases = [
            ("ANSWER\nH 0\nCNOT 0 1", some("H 0; CNOT 0 1")),
            (
                "Let me think.\nANSWER\nH 0\n\nNo, wait:\nANSWER\nX 1\nCNOT 1 0\n\nDone.",
                some("X 1; CNOT 1 0"),
            ),
            (
                "ANSWER\nh q0\ncx Q0 q1\nsdag 1\nTDAG 0\nSdg 1\ntdg q0",
                some("H 0; CNOT 0 1; SDG 1; TDG 0; SDG 1; TDG 0"),
            ),
            (
                "  ANSWER  \r\n\r\nT 1\r\nCNOT 1 0\r\n",
                some("T 1; CNOT 1 0"),
            ),
            ("ANSWER\nH 0\n\nCNOT 0 1", some("H 0")),
            ("ANSWER\nH 0\nThat is all.", some("H 0")),
            ("ANSWER\nH 1\nH 2", None),
            ("ANSWER\nCNOT 1 2", None),
            ("ANSWER\nH q2", None),
            ("ANSWER\nCNOT 0 0", None),
            ("ANSWER\nH 0\nCNOT(0,1)", None),
            ("ANSWER\nH 0\nCZ 0 1\nH 0", None),
            ("ANSWER\nH 0\n2. CNOT 0 1", None),
            ("ANSWER\nH 0\nThis uses 1 CNOT.", None),
            ("ANSWER\nH 1\nCNOT 0 1\nH 1\n(X 0)", None),
            ("ANSWER\nH 1\nCNOT 0 1\nH 1\nY(0)", None),
            ("ANSWER\nH 1\nCNOT 0 1\nH 1\nCZ(0,1)", None),
            ("ANSWER\nH 1\nCNOT 0 1\nH 1\n`X 0`", None),
            ("ANSWER\nH 1\nCNOT 0 1\nH 1\n(Note: the answer is final.)", some("H 1; CNOT 0 1; H 1")),
            ("ANSWER\nDone.", None),
            ("ANSWER", None),
            ("ANSWER\n\n\n", None),
            ("ANSWER\nH 0\nANSWER", None),
            ("H 0\nCNOT 0 1", None),
            ("ANSWER: H 0", None),
            ("ANSWER A\nH 0", None),
        ];

        for (text, expected) in cases {
            assert_eq!(parse_circuit_answer(text, 2, 64), expected, "{text:?}");
        }

        // The gate cap: max_gates, never above MAX_GATES; and at most MAX_H H gates.
        let lines = |g: &str, count: usize| format!("ANSWER\n{}", vec![g; count].join("\n"));
        let length =
            |text: &str, cap: usize| parse_circuit_answer(text, 2, cap).map(|c| c.gates().len());

        assert_eq!(length(&lines("H 0", 3), 3), Some(3));
        assert_eq!(length(&lines("H 0", 4), 3), None);
        assert_eq!(length(&lines("X 0", MAX_GATES), 1000), Some(256));
        assert_eq!(length(&lines("X 0", MAX_GATES + 1), 1000), None);
        assert_eq!(length(&lines("H 0", MAX_H), 256), Some(120));
        assert_eq!(length(&lines("H 0", MAX_H + 1), 256), None);

        // The index range is the qubit count given, never beyond MAX_QUBITS.
        assert_eq!(
            parse_circuit_answer("ANSWER\nH 11", MAX_QUBITS, 64),
            some("H 11")
        );
        assert_eq!(
            parse_circuit_answer("ANSWER\nH 12", MAX_QUBITS + 1, 64),
            None
        );
        assert_eq!(parse_circuit_answer("ANSWER\nH 0", 0, 64), None);

        let c = circuit("H 0; CNOT 0 1; T 1; SDG 0; X 1; Z 0; TDG 1; S 0; CNOT 1 0");

        assert_eq!(
            parse_circuit_answer(&render_circuit_answer(&c), 2, 64),
            Some(c)
        );
    }

    #[test]
    fn parsers_survive_garbage() {
        let solve = render_circuit_solve_prompt(&qft2(), 64, &[circuit("H 0; CNOT 0 1")]);
        let judge = render_circuit_judge_prompt(&toffoli(), 64, &circuit("H 2"), &circuit("T 0"));
        let many_gates = format!("ANSWER\n{}", "H 0\n".repeat(10_000));
        let mut texts: Vec<String> = [
            "",
            "\n\n\n",
            "CIRCUIT TASK",
            "CIRCUIT JUDGE\nEND",
            "CIRCUIT TASK\nQUBITS\nEND",
            "CIRCUIT TASK\nQUBITS 99999999999999999999999999\n",
            "CIRCUIT TASK\nQUBITS 6\nGATES H S SDG T TDG X Z CNOT\nMAX GATES 1\nTARGET\n",
            "ANSWER\nH 99999999999999999999999",
            "ANSWER\nCNOT q q",
            "ANSWER\nH q-1",
            "Grüße 🌍\nANSWER\nＨ ０\n٣",
            "\u{0}\u{1}\u{7f}\u{feff}ANSWER\u{2028}H 0",
            "|0> -> +|0>\n|1> -> -iw|1>",
            "ANSWER\n\u{a0}H 0",
        ]
        .map(String::from)
        .to_vec();

        texts.push(many_gates.clone());
        texts.push(format!("ANSWER\nH {}", "1".repeat(10_000)));
        texts.push(format!("CIRCUIT TASK\n{}", "x ".repeat(50_000)));
        texts.push(format!("{solve}\n{}", "PREVIOUS ANSWER 2\n".repeat(1000)));

        for text in [&solve, &judge] {
            texts.extend(text.char_indices().map(|(i, _)| text[..i].to_string()));
        }

        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let alphabet = [
            "ANSWER",
            "ANSWER A",
            "ANSWER B",
            "PREVIOUS ANSWER 1",
            "END",
            "CIRCUIT TASK",
            "CIRCUIT JUDGE",
            "QUBITS 1",
            "QUBITS 2",
            "MAX GATES 3",
            "TARGET",
            GATES_LINE,
            "H",
            "CNOT",
            "cx",
            "q",
            "0",
            "1",
            "7",
            "255",
            "256",
            " ",
            "\n",
            "\r\n",
            "|",
            ">",
            "->",
            "+",
            "-",
            "w",
            "i",
            "iw",
            "|0>",
            "|1>",
            "|00>",
            "|11>",
            "é",
            "🌍",
            "\t",
            "\u{0}",
        ];

        for _ in 0..2000 {
            let len = rng.random_range(1..80);

            texts.push(
                (0..len)
                    .map(|_| alphabet[rng.random_range(0..alphabet.len())])
                    .collect(),
            );
        }

        let edits = ['0', '1', '|', '+', '-', ' ', '\n', 'x', 'w', 'i'];

        for _ in 0..500 {
            let base = if rng.random_bool(0.5) { &solve } else { &judge };
            let mut chars: Vec<char> = base.chars().collect();
            let at = rng.random_range(0..chars.len());

            chars[at] = edits[rng.random_range(0..edits.len())];
            texts.push(chars.into_iter().collect());
        }

        let mut accepted = 0;

        for text in &texts {
            // Whatever parses renders back to the same prompt.
            if let Some(p) = parse_circuit_prompt(text) {
                accepted += 1;
                assert_eq!(
                    parse_circuit_prompt(&rendered(&p)).as_ref(),
                    Some(&p),
                    "{text:?}"
                );
            }

            for qubits in [0, 2, MAX_QUBITS] {
                let _ = parse_circuit_answer(text, qubits, 64);
            }

            for line in text.lines() {
                let _ = parse_gate_line(line);
            }
        }

        assert!(accepted > 2, "the corpus holds valid prompts too");
        assert_eq!(parse_circuit_answer(&many_gates, 2, 64), None);
    }
}
