//! The genes of circuit breeding (step 1, ticket 03): the seed Cascades, the phrase pool
//! the section operators draw from, the Mutator's brief, and the admission rule that keeps
//! written-out circuits out of genomes.
//!
//! The genes speak to the simulated designer (`designer.rs`) only through words. A Worker
//! knows a method family when its system prompt names one of the family's stems
//! (`designer::FAMILY_STEMS`). The seeds are weak on purpose: each names one family and
//! thinks little, so they reach few targets at their best known cost.
//!
//! The phrases describe methods in words and never list gates. Every family is named alone
//! in some phrase, some phrases combine families, and many name none. Only strategy and
//! residual phrases name a family. So role, format, check and style genes never change what
//! a Worker knows, and every residual phrase reviews the answer of an earlier stage.
//!
//! Pure computation: no files, no network, no clock.
#![forbid(unsafe_code)]

use crate::cascade::domain::GenePool;
use crate::cascade::genome::{Cascade, Section, Stage, Traits, Worker};
use crate::cascade::qubit::Gate;
use crate::cascade::variation::MutatorBrief;

/// The role of every seed Worker. It names the gate set and no method.
const SEED_ROLE: &str =
    "You design quantum circuits from the gates H, S, SDG, T, TDG, X, Z and CNOT.";

/// The answer format of every seed Worker. Its two examples are the only gate mentions a
/// seed carries.
const SEED_FORMAT: &str =
    "Answer with the line ANSWER, then one gate per line, such as H 0 or CNOT 0 1.";

/// The strategy of the linear seed: CNOT and X networks.
const SEED_LINEAR: &str =
    "Treat each CNOT as a row operation: it adds one bit into another by XOR.";

/// The strategy of the phase seed: phase polynomials.
const SEED_PHASE: &str = "Write the phase of each basis state as a sum over parities of the bits; one T or TDG per odd term.";

/// The strategy of the hadamard seed: a change of basis.
const SEED_HADAMARD: &str =
    "Use Hadamard gates to change the basis, so that bit flips become sign flips.";

/// The strategy of the search seed: word search.
const SEED_SEARCH: &str = "Try short sequences, one gate at a time.";

/// Model of every seed Worker: the small simulated designer.
const SEED_MODEL: &str = "sim:small";

/// Most gate mentions one section may hold. A section with more lists gates.
const MAX_GATE_MENTIONS: usize = 2;

/// Prompt-text gene pool for circuits: (section name, text). The section operators draw
/// section texts from here, as they draw from `variation::PHRASES` for grids.
///
/// The pool holds strategies that name one method family, strategies that combine two or
/// three, German variants, strategies and checks that name no family, residual reviews of
/// the previous answer (some name a method to repair it with), style lines, roles and
/// answer formats. The seeds' texts are among them. Every text passes the genome's text
/// rules and `no_gate_lists`.
///
/// The section operators keep one phrase per section name in a Worker, so its families
/// come from one strategy and one residual phrase. Such pairs name every route of the
/// designer's calibration.
pub const CIRCUIT_PHRASES: &[(&str, &str)] = &[
    // Roles.
    ("role", SEED_ROLE),
    (
        "role",
        "You are a careful designer of quantum circuits. Read every row of the target before you answer.",
    ),
    (
        "role",
        "Du entwirfst Quantenschaltungen aus den Gattern H, S, SDG, T, TDG, X, Z und CNOT.",
    ),
    // Strategies that name one family.
    ("strategy", SEED_LINEAR),
    (
        "strategy",
        "Write the map of the bits as a matrix and reduce it to the identity with as few row operations as possible.",
    ),
    ("strategy", SEED_PHASE),
    (
        "strategy",
        "Read the phases of a diagonal target as a phase polynomial: a multiple of pi/4 on each parity of the bits.",
    ),
    ("strategy", SEED_HADAMARD),
    (
        "strategy",
        "A Hadamard gate on the target bit turns a controlled bit flip into a controlled sign flip.",
    ),
    (
        "strategy",
        "Conjugate the target by one CNOT on both sides; what remains in between may be easier.",
    ),
    (
        "strategy",
        "Sandwich the target between two equal CNOT gates and solve the part in between.",
    ),
    ("strategy", SEED_SEARCH),
    (
        "strategy",
        "Search through all short gate sequences and stop at the first one that matches every row.",
    ),
    (
        "strategy",
        "Meet in the middle: grow sequences from the input and from the output until they fit together.",
    ),
    (
        "strategy",
        "Grow half a circuit from both ends and join two halves that agree.",
    ),
    (
        "strategy",
        "Cancel gates that undo each other and merge neighbouring phase gates on the same wire.",
    ),
    (
        "strategy",
        "Simplify your best circuit: move each gate past the gates it commutes with until something cancels.",
    ),
    // Strategies that combine two or three families.
    (
        "strategy",
        "Build the parities with an XOR network of CNOTs and put the phase of each parity on a wire that carries it.",
    ),
    (
        "strategy",
        "Change the basis of a flipped bit with Hadamard gates; the rest is a network of CNOT row operations.",
    ),
    (
        "strategy",
        "After a basis change on some qubits the target may only add phases; write them as a sum over parities.",
    ),
    (
        "strategy",
        "Change the basis with Hadamard gates where bits flip, write the phases over parities, and build the parities with XOR.",
    ),
    (
        "strategy",
        "Search from both ends at once and meet in the middle.",
    ),
    // German strategies.
    (
        "strategy",
        "Behandle jedes CNOT als Zeilenoperation, die ein Bit per XOR auf ein anderes addiert.",
    ),
    (
        "strategy",
        "Schreibe die Phasen als Phasenpolynom über die Paritäten der Bits.",
    ),
    (
        "strategy",
        "Nutze einen Basiswechsel mit Hadamard-Gattern, damit aus Bitflips Vorzeichenwechsel werden.",
    ),
    (
        "strategy",
        "Konjugiere das Ziel mit einem CNOT und löse den Teil dazwischen.",
    ),
    (
        "strategy",
        "Durchsuche kurze Gatterfolgen, ein Gatter nach dem anderen.",
    ),
    (
        "strategy",
        "Baue die Schaltung von beiden Enden her, bis sich die Hälften treffen.",
    ),
    (
        "strategy",
        "Vereinfache die Schaltung: kürze inverse Paare und fasse Phasengatter zusammen.",
    ),
    (
        "strategy",
        "Nutze einen Basiswechsel und schreibe die Phasen dann als Phasenpolynom über Paritäten.",
    ),
    // Strategies that name no family.
    (
        "strategy",
        "Count the T and TDG gates first, then the CNOT gates; of two exact circuits, the one with fewer T gates wins.",
    ),
    (
        "strategy",
        "Look at which inputs keep their bits and which only gain a phase before you place any gate.",
    ),
    (
        "strategy",
        "Split the target into a part that moves bits and a part that only adds phases.",
    ),
    // Checks of the Worker's own circuit.
    (
        "check",
        "Before you answer, follow every input basis state through your circuit and compare it with its row of the target.",
    ),
    (
        "check",
        "If your circuit fails on even one input, drop it and test the next idea.",
    ),
    (
        "check",
        "Count the T, TDG and CNOT gates of your circuit and keep the cheaper of two correct circuits.",
    ),
    (
        "check",
        "Make sure that only a global phase separates your circuit from the target, never a relative one.",
    ),
    (
        "check",
        "Prüfe deine Schaltung an jedem Eingang, bevor du antwortest.",
    ),
    // Residual reviews of the circuit an earlier stage chose.
    (
        "residual",
        "Read the previous answer first. If it implements the target exactly, repeat it; otherwise fix it.",
    ),
    (
        "residual",
        "Start from the earlier answer and change only the gates that break the target.",
    ),
    (
        "residual",
        "Run the previous circuit on every input; keep it if it fits, otherwise design a new one.",
    ),
    (
        "residual",
        "Start from the previous answer and simplify it: cancel inverse pairs and merge phase gates.",
    ),
    (
        "residual",
        "If the previous answer fails, conjugate the target by one CNOT and solve the middle again.",
    ),
    (
        "residual",
        "Keep the previous answer if it is exact; otherwise rebuild its CNOT network from XOR row operations.",
    ),
    (
        "residual",
        "Check the previous answer and recompute its phases as a sum over parities, with as few T gates as you can.",
    ),
    (
        "residual",
        "If the previous answer fails, put Hadamard gates around the bits that flip and solve the rest again.",
    ),
    (
        "residual",
        "Lies zuerst die vorige Antwort. Stimmt sie für jeden Eingang, gib sie unverändert zurück.",
    ),
    // Style.
    ("style", "Be concise."),
    (
        "style",
        "Think step by step, but keep the final answer short.",
    ),
    (
        "style",
        "Work precisely; one wrong phase makes the whole circuit wrong.",
    ),
    ("style", "Write as little as possible outside the circuit."),
    (
        "style",
        "Name your method in one short sentence before you write the circuit.",
    ),
    ("style", "Arbeite sorgfältig und knapp."),
    // Answer formats.
    ("format", SEED_FORMAT),
    (
        "format",
        "End with the line ANSWER and then your circuit: one gate per line, the gate name first, a CNOT with its control before its target.",
    ),
];

/// System prompt of every LLM mutator request on circuits.
const BRIEF_SYSTEM: &str = "You are the mutation operator of a research breeding system. \
You edit Cascades: JSON data that configures model Workers, which design quantum circuits. \
Return exactly one JSON object for the new Cascade and nothing else.";

/// What the circuit tasks are; the first paragraph of every request.
const BRIEF_CONTEXT: &str = "A Cascade designs quantum circuits. Each task shows a target \
on a few qubits: the image of every input basis state, as an equal-weight sum of basis \
states with phases. It asks for a circuit of the gates H, S, SDG, T, TDG, X, Z and CNOT, \
within a cap on the number of gates, that implements the target exactly up to one global \
phase. Fewer T and TDG gates are better, then fewer CNOT gates. A Worker meets many \
different targets, so its sections should teach methods, not circuits.\n\n";

/// The Cascade JSON format in words, mirroring `genome.rs`, with the admission rule of
/// `no_gate_lists`.
const BRIEF_FORMAT: &str = "Format: a Cascade is a JSON object. Every field below is \
required and no other field is allowed.\n\
- Cascade: {\"stages\": [Stage, ...]}. The stages run in order; every stage sees the task \
and the circuits chosen by the earlier stages.\n\
- Stage: {\"workers\": [Worker, ...]}. The Workers of one stage run in parallel on the same \
input.\n\
- Worker: {\"model\": string, \"sections\": [Section, ...], \"traits\": Traits}. The system \
prompt of a Worker is its sections in order, each under its name.\n\
- Section: {\"name\": string, \"text\": string}. Typical names are \"role\", \"strategy\", \
\"check\", \"residual\", \"style\" and \"format\"; texts are plain text and describe \
methods in words. A section that names three or more gates together with their qubits, \
such as a written-out circuit, is refused.\n\
- Traits: {\"think_tokens\": integer, \"temperature\": number, \"answer_tokens\": integer, \
\"self_check\": boolean}. think_tokens is the budget for reasoning before the answer (0 \
answers at once); answer_tokens bounds the visible answer; self_check makes the Worker run \
its circuit on every input basis state and compare it with the target before it \
answers.\n\n";

/// The task of a mutation request.
const BRIEF_MUTATE_TASK: &str = "Task: change the parent Cascade so that it finds exact \
circuits with fewer T gates, then fewer CNOT gates, on more tasks and at lower cost. You \
may rewrite, add or remove sections, change Traits and models, and add or remove Workers \
and stages. ";

/// The task of a crossover request.
const BRIEF_CROSSOVER_TASK: &str = "Task: merge Parent A and Parent B into one child \
Cascade that keeps what works in each, so that it finds exact circuits with fewer T gates, \
then fewer CNOT gates, on more tasks and at lower cost. You may take stages, Workers, \
sections and Traits from either parent and change them. ";

/// The brief of the circuit tasks, in the shape of `variation::GRID_BRIEF`. It names no
/// target and lists no gates.
pub static CIRCUIT_BRIEF: MutatorBrief = MutatorBrief {
    system: BRIEF_SYSTEM,
    context: BRIEF_CONTEXT,
    format: BRIEF_FORMAT,
    mutate_task: BRIEF_MUTATE_TASK,
    crossover_task: BRIEF_CROSSOVER_TASK,
};

/// Hand-written starting Cascades for circuits, named linear, phase, hadamard and search.
/// Each is one small Worker with 256 thinking tokens, temperature 0.2, 512 answer tokens
/// and no self-check. Its sections are the role, one strategy and the format; only the
/// strategy names a method family.
pub fn circuit_seed_cascades() -> Vec<(String, Cascade)> {
    let worker = |strategy: &str| Worker {
        model: SEED_MODEL.to_string(),
        sections: vec![
            section("role", SEED_ROLE),
            section("strategy", strategy),
            section("format", SEED_FORMAT),
        ],
        traits: Traits {
            think_tokens: 256,
            temperature: 0.2,
            answer_tokens: 512,
            self_check: false,
        },
    };

    let single = |w: Worker| Cascade {
        stages: vec![Stage { workers: vec![w] }],
    };

    [
        ("linear", SEED_LINEAR),
        ("phase", SEED_PHASE),
        ("hadamard", SEED_HADAMARD),
        ("search", SEED_SEARCH),
    ]
    .into_iter()
    .map(|(name, strategy)| (name.to_string(), single(worker(strategy))))
    .collect()
}

/// Refuse a Cascade when one of its sections lists gates: three or more gate mentions,
/// counted over the section's name and text as the system prompt shows them.
///
/// A gate mention is a gate name or alias of the circuit protocol, in any case, followed by
/// a qubit index such as 0 or q0. The text is split into words at every character that is
/// not a letter or digit, as the answer parser splits a gate-like line. So "H 0;", "h(q0)"
/// and "CX 0,1" all count once. Prose that names a gate or a qubit, or a gate on a qubit
/// once or twice, passes. The error says where the list is.
pub fn no_gate_lists(c: &Cascade) -> std::result::Result<(), String> {
    for (k, stage) in c.stages.iter().enumerate() {
        for (w, worker) in stage.workers.iter().enumerate() {
            for s in &worker.sections {
                let mentions = gate_mentions(&format!("{}\n{}", s.name, s.text));

                if let Some(first) = mentions.first()
                    && mentions.len() > MAX_GATE_MENTIONS
                {
                    return Err(format!(
                        "stage {k} worker {w}: section {:?} lists {} gates with their qubits, \
                         the first {first:?}; sections describe methods in words, not circuits",
                        s.name,
                        mentions.len()
                    ));
                }
            }
        }
    }

    Ok(())
}

/// The genes of circuit breeding: the seeds, the phrase pool, the brief and the rule that
/// refuses gate lists.
pub fn circuit_pool() -> GenePool {
    GenePool {
        seeds: circuit_seed_cascades,
        phrases: CIRCUIT_PHRASES,
        brief: &CIRCUIT_BRIEF,
        admit: no_gate_lists,
    }
}

/// A section named `name` with `text`.
fn section(name: &str, text: &str) -> Section {
    Section {
        name: name.to_string(),
        text: text.to_string(),
    }
}

/// The gate mentions in `text`, in order, each as its two words: "H 0", "cx q1". The words
/// are the pieces between characters that are not letters or digits.
fn gate_mentions(text: &str) -> Vec<String> {
    let words: Vec<&str> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();

    words
        .iter()
        .zip(words.iter().skip(1))
        .filter(|(name, index)| Gate::is_name(name) && is_qubit_index(index))
        .map(|(name, index)| format!("{name} {index}"))
        .collect()
}

/// A qubit index as the circuit protocol writes it: ASCII digits that fit a u8, with or
/// without a leading q or Q.
fn is_qubit_index(word: &str) -> bool {
    let digits = word.strip_prefix(['q', 'Q']).unwrap_or(word);

    !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) && digits.parse::<u8>().is_ok()
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::backend::{GenRequest, ModelBackend, StopReason};
    use crate::cascade::arena::worker_request;
    use crate::cascade::circuit_text::{
        CIRCUIT_SELF_CHECK_TEXT, parse_circuit_answer, render_circuit_solve_prompt,
    };
    use crate::cascade::designer::{FAMILY_STEMS, SimDesigner, knowledge};
    use crate::cascade::genome::{CascadeLimits, adapt_seeds};
    use crate::cascade::qubit::{Circuit, Column, Dressing, Target, check};
    use crate::cascade::variation::{llm_crossover_request_with, llm_mutate_request_with};
    use crate::util::derive_seed;

    /// The section names of the pool, the same as in the grid pool.
    const NAMES: [&str; 6] = ["role", "strategy", "check", "residual", "style", "format"];

    /// Sections whose phrases never name a family.
    const NEUTRAL_NAMES: [&str; 4] = ["role", "format", "check", "style"];

    /// The German stems of each family in `designer::FAMILY_STEMS`.
    const GERMAN: [(&str, &[&str]); 8] = [
        ("Linear", &["zeilenoperation"]),
        ("Phase", &["phasenpolynom"]),
        ("Hadamard", &["basiswechsel"]),
        ("Compose", &["konjug", "einrahm"]),
        ("Search", &["durchsuch", "durchprobier"]),
        ("Meet", &["beiden enden"]),
        ("Simplify", &["kürz", "vereinfach"]),
        ("Previous", &["vorige antwort", "vorige schaltung"]),
    ];

    /// Words that must never reach the Mutator: it learns nothing of evaluation it cannot
    /// see.
    const BANNED: [&str; 6] = [
        "hidden", "held-out", "held out", "holdout", "secret", "test set",
    ];

    /// Names of calibration targets. No gene and no brief carries one.
    const TARGET_NAMES: [&str; 8] = [
        "toffoli", "fredkin", "peres", "ccz", "ccix", "rtof", "qft", "fourier",
    ];

    /// The gate cap of the circuit domain's default config.
    const MAX_GATES: usize = 64;

    fn monomial(n: usize, image: impl Fn(u32) -> (u32, u8)) -> Target {
        Target::monomial(n, image).expect("unitary")
    }

    fn column(terms: Vec<(u32, u8)>) -> Column {
        Column::new(terms).expect("column")
    }

    /// The eleven calibration targets from their definitions, each with the (T, CNOT) of
    /// its best known circuit. Qubit 0 is the most significant bit, so on three qubits the
    /// input |abc> is 4a + 2b + c.
    fn targets() -> Vec<(&'static str, Target, (u32, u32))> {
        // |0><0| x I + |1><1| x H: H on qubit 1, controlled by qubit 0.
        let ch = Target::new(
            2,
            vec![
                Column::single(0, 0),
                Column::single(1, 0),
                column(vec![(2, 0), (3, 0)]),
                column(vec![(2, 0), (3, 4)]),
            ],
        )
        .expect("unitary");

        // The two-qubit Fourier transform: entry (b, j) is i^(b j) / 2.
        let fourier_column = |j: u32| {
            column(
                (0..4u32)
                    .map(|b| (b, u8::try_from(2 * j * b % 8).expect("phase")))
                    .collect(),
            )
        };
        let qft2 = Target::new(2, (0..4u32).map(fourier_column).collect()).expect("unitary");

        // |a,b,c> -> |a, a xor b, c xor ab>.
        let peres = |j: u32| {
            let (a, b, c) = (j >> 2, (j >> 1) & 1, j & 1);

            ((a << 2) | ((a ^ b) << 1) | (c ^ (a & b)), 0)
        };

        vec![
            // diag(1, 1, 1, -1).
            (
                "cz",
                monomial(2, |j| (j, if j == 3 { 4 } else { 0 })),
                (0, 1),
            ),
            // |ab> -> |ba>.
            (
                "swap",
                monomial(2, |j| (((j & 1) << 1) | (j >> 1), 0)),
                (0, 3),
            ),
            // diag(1, 1, 1, i).
            (
                "cs",
                monomial(2, |j| (j, if j == 3 { 2 } else { 0 })),
                (3, 2),
            ),
            ("ch", ch, (2, 1)),
            ("qft2", qft2, (3, 3)),
            // diag(1, ..., 1, -1).
            (
                "ccz",
                monomial(3, |j| (j, if j == 7 { 4 } else { 0 })),
                (7, 6),
            ),
            // |a,b,c> -> |a,b,c xor ab>.
            (
                "toffoli",
                monomial(3, |j| (if j >= 6 { j ^ 1 } else { j }, 0)),
                (7, 6),
            ),
            ("peres", monomial(3, peres), (7, 5)),
            // a = 1 swaps b and c: |101> and |110> trade places.
            (
                "fredkin",
                monomial(3, |j| (if j == 5 || j == 6 { j ^ 3 } else { j }, 0)),
                (7, 7),
            ),
            // The identity except |101> -> -|101>, |110> -> i|111>, |111> -> -i|110>.
            (
                "rtof",
                monomial(3, |j| match j {
                    5 => (5, 4),
                    6 => (7, 2),
                    7 => (6, 6),
                    _ => (j, 0),
                }),
                (4, 3),
            ),
            // The identity except |110> -> i|111>, |111> -> i|110>.
            (
                "ccix",
                monomial(3, |j| match j {
                    6 => (7, 2),
                    7 => (6, 2),
                    _ => (j, 0),
                }),
                (4, 4),
            ),
        ]
    }

    /// A published circuit for each target, in time order.
    fn published(name: &str) -> String {
        let nielsen_chuang = "H 2; CNOT 1 2; TDG 2; CNOT 0 2; T 2; CNOT 1 2; TDG 2; CNOT 0 2; \
                              T 1; T 2; H 2; CNOT 0 1; T 0; TDG 1; CNOT 0 1";
        let from_third = nielsen_chuang
            .splitn(3, "; ")
            .nth(2)
            .expect("third gate on");

        match name {
            "cz" => "H 1; CNOT 0 1; H 1".to_string(),
            "swap" => "CNOT 0 1; CNOT 1 0; CNOT 0 1".to_string(),
            "cs" => "T 0; T 1; CNOT 0 1; TDG 1; CNOT 0 1".to_string(),
            "ch" => "S 1; H 1; T 1; CNOT 0 1; TDG 1; H 1; SDG 1".to_string(),
            "qft2" => "H 0; CNOT 0 1; CNOT 1 0; TDG 1; CNOT 0 1; T 1; T 0; H 0".to_string(),
            "ccz" => nielsen_chuang.replacen("H 2; ", "", 2),
            "toffoli" => nielsen_chuang.to_string(),
            "peres" => nielsen_chuang.trim_end_matches("; CNOT 0 1").to_string(),
            "fredkin" => {
                format!("S 1; CNOT 2 1; SDG 1; H 2; S 2; H 2; S 2; {from_third}; CNOT 2 1")
            }
            "rtof" => "H 2; T 2; CNOT 1 2; TDG 2; CNOT 0 2; T 2; CNOT 1 2; TDG 2; H 2".to_string(),
            "ccix" => "H 2; CNOT 0 2; T 2; CNOT 1 2; TDG 2; CNOT 0 2; T 2; CNOT 1 2; TDG 2; H 2"
                .to_string(),
            other => panic!("no published circuit for {other}"),
        }
    }

    fn rng(seed: u64) -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(seed)
    }

    /// A Cascade of one stage with the one Worker `w`.
    fn alone(w: Worker) -> Cascade {
        Cascade {
            stages: vec![Stage { workers: vec![w] }],
        }
    }

    /// A Cascade of one small Worker whose only section is `name` with `text`.
    fn single(name: &str, text: &str) -> Cascade {
        alone(Worker {
            model: SEED_MODEL.to_string(),
            sections: vec![section(name, text)],
            traits: Traits::default(),
        })
    }

    /// The first phrase of section `name` whose families are exactly `families`.
    fn phrase_naming(name: &str, families: &[&str]) -> &'static str {
        let want: BTreeSet<&str> = families.iter().copied().collect();

        CIRCUIT_PHRASES
            .iter()
            .find(|(n, text)| {
                *n == name && knowledge(text).into_iter().collect::<BTreeSet<_>>() == want
            })
            .map(|(_, text)| *text)
            .unwrap_or_else(|| panic!("no {name} phrase names exactly {families:?}"))
    }

    /// A Worker of pool phrases at temperature 0: the seed role, the strategy phrase that
    /// names `strategy`, the residual phrase that names `residual` if one is asked for, and
    /// the seed format. `traits` is (model, thinking tokens, self-check).
    fn pool_worker(
        strategy: &[&str],
        residual: Option<&[&str]>,
        traits: (&str, u32, bool),
    ) -> Worker {
        let (model, think_tokens, self_check) = traits;
        let mut sections = vec![
            section("role", SEED_ROLE),
            section("strategy", phrase_naming("strategy", strategy)),
        ];

        if let Some(families) = residual {
            sections.push(section("residual", phrase_naming("residual", families)));
        }

        sections.push(section("format", SEED_FORMAT));

        Worker {
            model: model.to_string(),
            sections,
            traits: Traits {
                think_tokens,
                temperature: 0.0,
                answer_tokens: 512,
                self_check,
            },
        }
    }

    /// The request arena builds for `w`, with the circuit self-check text where the grid's
    /// would stand: `worker_request` of the Worker without its self-check Trait, then the
    /// check section in the form `worker_request` writes it.
    fn circuit_request(w: &Worker, prompt: String, seed: u64) -> GenRequest {
        let mut plain = w.clone();

        plain.traits.self_check = false;

        let mut req = worker_request(&plain, prompt, seed);

        if w.traits.self_check {
            req.system.push_str("## check\n");
            req.system.push_str(CIRCUIT_SELF_CHECK_TEXT);
            req.system.push('\n');
        }

        req
    }

    /// The circuit a Cascade of one-Worker stages chooses, as `arena::run_cascade` runs it:
    /// each stage sees the circuits the earlier stages chose, and a reply counts only when
    /// it ends normally and parses.
    fn run(c: &Cascade, target: &Target, seed: u64) -> Option<Circuit> {
        let mut chosen: Vec<Circuit> = Vec::new();

        for (k, stage) in c.stages.iter().enumerate() {
            let w = stage.workers.first().expect("a Worker");
            let model = SimDesigner::from_tag(&w.model).expect("a simulated designer");
            let prompt = render_circuit_solve_prompt(target, MAX_GATES, &chosen);
            let req = circuit_request(w, prompt, derive_seed(seed, &format!("stage{k}/worker0")));
            let reply = model.generate(&req).expect("generate");

            if reply.stop_reason == StopReason::Stop
                && let Some(answer) = parse_circuit_answer(&reply.text, target.qubits(), MAX_GATES)
            {
                chosen.push(answer);
            }
        }

        chosen.last().cloned()
    }

    /// True when the circuit `c` chooses implements `target` exactly with at most `cost`
    /// (T, CNOT).
    fn reached(c: &Cascade, target: &Target, (t, cnot): (u32, u32), seed: u64) -> bool {
        run(c, target, seed).is_some_and(|answer| {
            let checked = check(target, &answer).expect("check");

            checked.exact && checked.cost.t <= t && checked.cost.cnot <= cnot
        })
    }

    /// Every dressing of an n-qubit target: each relabelling of the qubits with each pair of
    /// X masks on the inputs and the outputs. The first is the identity.
    fn every_dressing(n: usize) -> Vec<Dressing> {
        let mut perms: Vec<Vec<u8>> = vec![Vec::new()];

        for _ in 0..n {
            perms = perms
                .iter()
                .flat_map(|head| {
                    (0..u8::try_from(n).expect("few qubits"))
                        .filter(|q| !head.contains(q))
                        .map(|q| [head.as_slice(), &[q]].concat())
                })
                .collect();
        }

        let mut out = Vec::new();

        for perm in perms {
            for x_in in 0..1u32 << n {
                for x_out in 0..1u32 << n {
                    out.push(Dressing {
                        perm: perm.clone(),
                        x_in,
                        x_out,
                    });
                }
            }
        }

        out
    }

    #[test]
    fn the_target_definitions_match_published_circuits() {
        for (name, target, (t, cnot)) in targets() {
            let c: Circuit = published(name).parse().expect("circuit text");
            let checked = check(&target, &c).expect("check");

            assert!(checked.exact, "{name}");
            assert_eq!((checked.cost.t, checked.cost.cnot), (t, cnot), "{name}");
        }

        assert_eq!(published("fredkin").split("; ").count(), 21);
    }

    #[test]
    fn phrases_pass_the_genome_text_rules_and_list_no_gates() {
        let limits = CascadeLimits::default();

        assert!(
            CIRCUIT_PHRASES.len() >= 30,
            "{} phrases",
            CIRCUIT_PHRASES.len()
        );

        for (k, (name, text)) in CIRCUIT_PHRASES.iter().enumerate() {
            let c = single(name, text);
            let lower = text.to_lowercase();

            assert!(
                c.check(&limits).is_ok(),
                "{name}: {text}: {:?}",
                c.check(&limits)
            );
            assert_eq!(no_gate_lists(&c), Ok(()), "{text}");
            assert!(NAMES.contains(name), "{name}");
            assert!(text.chars().count() <= 200, "{text}");
            assert!(
                text.chars()
                    .all(|ch| ch.is_ascii() || "äöüÄÖÜß".contains(ch)),
                "{text}"
            );
            assert!(
                CIRCUIT_PHRASES[..k].iter().all(|(_, t)| t != text),
                "duplicate: {text}"
            );

            // The self-check stays a Trait: no phrase switches it on.
            assert!(!text.contains(CIRCUIT_SELF_CHECK_TEXT), "{text}");

            for target in TARGET_NAMES {
                assert!(!lower.contains(target), "{target:?} in {text:?}");
            }
        }
    }

    #[test]
    fn phrases_mix_single_combined_german_and_neutral_texts() {
        let families = |text: &str| knowledge(text);

        for (family, _) in FAMILY_STEMS {
            assert!(
                CIRCUIT_PHRASES.iter().any(|(_, t)| families(t) == [family]),
                "no phrase names {family} alone"
            );
        }

        let combined = CIRCUIT_PHRASES
            .iter()
            .filter(|(_, t)| families(t).len() >= 2)
            .count();
        let neutral = CIRCUIT_PHRASES
            .iter()
            .filter(|(_, t)| families(t).is_empty())
            .count();

        assert!(combined >= 5, "{combined} combined phrases");
        assert!(neutral >= 8, "{neutral} neutral phrases");

        // A German phrase for every family. The stems listed here must be the designer's.
        for (family, stems) in GERMAN {
            let (_, known) = FAMILY_STEMS
                .iter()
                .find(|(name, _)| *name == family)
                .expect("a family of the designer");

            assert!(
                stems.iter().all(|s| known.contains(s)),
                "{family}: {stems:?}"
            );
            assert!(
                CIRCUIT_PHRASES
                    .iter()
                    .any(|(_, t)| stems.iter().any(|s| t.to_lowercase().contains(s))),
                "no German phrase names {family}"
            );
        }

        // Role, format, check and style genes never change what a Worker knows.
        for (name, text) in CIRCUIT_PHRASES {
            if NEUTRAL_NAMES.contains(name) {
                assert!(families(text).is_empty(), "{name}: {text}");
            }
        }

        // Every residual phrase reviews the answer of an earlier stage.
        let residual: Vec<&str> = CIRCUIT_PHRASES
            .iter()
            .filter(|(name, _)| *name == "residual")
            .map(|(_, t)| *t)
            .collect();

        assert!(residual.len() >= 3);
        assert!(
            residual.iter().all(|t| families(t).contains(&"Previous")),
            "{residual:?}"
        );
    }

    #[test]
    fn pool_phrases_reach_every_target_at_its_best_known_cost() {
        // Cascades of pool phrases only, as the section operators can build them. Each Worker
        // has the seed role and format, one strategy and at most one residual phrase, which
        // name the families of one route of the designer's calibration, and Traits in range.
        let small = |think: u32, check: bool| ("sim:small", think, check);
        let large = |think: u32, check: bool| ("sim:large", think, check);
        let hpl: &[&str] = &["Hadamard", "Phase", "Linear"];
        let basis = ["qft2", "toffoli", "peres", "rtof", "ccix"];
        let fredkin_first = pool_worker(hpl, Some(&["Previous", "Compose"]), large(2048, true));
        let routes: Vec<(&[&str], Vec<Worker>)> = vec![
            (
                &["swap"],
                vec![pool_worker(&["Linear"], None, small(256, false))],
            ),
            (
                &["cs"],
                vec![pool_worker(&["Phase"], None, small(256, false))],
            ),
            (
                &["cz"],
                vec![pool_worker(
                    &["Hadamard", "Linear"],
                    None,
                    small(256, false),
                )],
            ),
            (
                &["ccz"],
                vec![pool_worker(&["Phase", "Linear"], None, small(256, false))],
            ),
            (&basis, vec![pool_worker(hpl, None, small(1024, true))]),
            // The same three families from a pair and a residual review.
            (
                &basis,
                vec![pool_worker(
                    &["Hadamard", "Phase"],
                    Some(&["Previous", "Linear"]),
                    small(1024, true),
                )],
            ),
            (
                &["ch"],
                vec![pool_worker(&["Meet"], None, large(4096, true))],
            ),
            // Two stages: the first builds (7, 8), the second simplifies it to (7, 7).
            (
                &["fredkin"],
                vec![
                    fredkin_first.clone(),
                    pool_worker(&[], Some(&["Previous", "Simplify"]), large(2048, true)),
                ],
            ),
        ];
        let all = targets();
        let target = |name: &str| all.iter().find(|(n, _, _)| *n == name).expect("a target");
        let limits = CascadeLimits::default();
        let mut covered = BTreeSet::new();

        for (names, workers) in routes {
            let c = Cascade {
                stages: workers
                    .into_iter()
                    .map(|w| Stage { workers: vec![w] })
                    .collect(),
            };

            assert!(
                c.check(&limits).is_ok(),
                "{names:?}: {:?}",
                c.check(&limits)
            );
            assert_eq!(no_gate_lists(&c), Ok(()), "{names:?}");

            for &name in names {
                let (_, base, cost) = target(name);

                for d in 0..2 {
                    let dressed = if d == 0 {
                        base.clone()
                    } else {
                        base.dress(&Dressing::draw(base.qubits(), &mut rng(d)))
                    };

                    assert!(reached(&c, &dressed, *cost, d), "{name}, dressing {d}");
                }

                covered.insert(name);
            }
        }

        assert_eq!(covered.len(), all.len(), "{covered:?}");

        // The first Fredkin stage alone stops one CNOT short: the stage structure matters.
        let (_, fredkin, _) = target("fredkin");
        let first = run(&alone(fredkin_first), fredkin, 0).expect("a circuit");
        let checked = check(fredkin, &first).expect("check");

        assert!(checked.exact);
        assert_eq!((checked.cost.t, checked.cost.cnot), (7, 8));
    }

    #[test]
    fn the_seeds_are_valid_distinct_admitted_and_name_one_family_each() {
        let limits = CascadeLimits::default();
        let seeds = circuit_seed_cascades();
        let families = [
            ("linear", "Linear"),
            ("phase", "Phase"),
            ("hadamard", "Hadamard"),
            ("search", "Search"),
        ];

        // The default limits allow both simulated designers, so nothing needs adapting.
        assert!(limits.model("sim:small").is_some() && limits.model("sim:large").is_some());
        assert_eq!(adapt_seeds(seeds.clone(), &limits).expect("seeds"), seeds);

        let ids: BTreeSet<String> = seeds.iter().map(|(_, c)| c.id()).collect();

        assert_eq!(ids.len(), seeds.len(), "distinct ids");
        assert_eq!(seeds.len(), families.len());

        for ((name, c), (want, family)) in seeds.iter().zip(families) {
            assert_eq!(name, want);
            assert_eq!(no_gate_lists(c), Ok(()), "{name}");
            assert_eq!(c.stages.len(), 1, "{name}");
            assert_eq!(c.stages[0].workers.len(), 1, "{name}");

            let w = &c.stages[0].workers[0];
            let names: Vec<&str> = w.sections.iter().map(|s| s.name.as_str()).collect();

            assert_eq!(w.model, "sim:small", "{name}");
            assert_eq!(
                w.traits,
                Traits {
                    think_tokens: 256,
                    temperature: 0.2,
                    answer_tokens: 512,
                    self_check: false,
                },
                "{name}"
            );
            assert_eq!(names, ["role", "strategy", "format"], "{name}");

            // Every section is a pool phrase under its own name, and only the strategy
            // names a family.
            for s in &w.sections {
                assert!(
                    CIRCUIT_PHRASES
                        .iter()
                        .any(|(n, t)| *n == s.name && *t == s.text),
                    "{name}: {}",
                    s.text
                );
            }

            assert_eq!(knowledge(&w.system_prompt()), [family], "{name}");
        }
    }

    #[test]
    fn the_seeds_reach_only_swap_and_cs_on_every_dressing() {
        // Each seed's Worker answers a solve prompt as arena asks it, on every target, plain
        // and under every dressing: 4 seeds x (5 x 32 + 6 x 384) calls.
        let mut hits: BTreeMap<(String, &str), usize> = BTreeMap::new();
        let mut calls = 0usize;

        for (seed, c) in circuit_seed_cascades() {
            for (name, base, cost) in targets() {
                for (i, d) in (0u64..).zip(every_dressing(base.qubits())) {
                    calls += 1;

                    if reached(&c, &base.dress(&d), cost, i) {
                        *hits.entry((seed.clone(), name)).or_default() += 1;
                    }
                }
            }
        }

        let expected = BTreeMap::from([
            (("linear".to_string(), "swap"), 32),
            (("phase".to_string(), "cs"), 32),
        ]);

        assert_eq!(calls, 4 * (5 * 32 + 6 * 384));
        assert_eq!(hits, expected);
    }

    #[test]
    fn no_gate_lists_refuses_written_out_circuits_only() {
        assert_eq!(
            gate_mentions("cx q0 q1; H(2) and SDG Q3"),
            ["cx q0", "H 2", "SDG Q3"]
        );
        assert!(gate_mentions("x0 xor x1, T gates, qubit 0, H 256").is_empty());

        // Prose may name a gate, a qubit, or a gate on a qubit once or twice.
        for text in [
            "Use one CNOT, then look at qubit 0 again.",
            SEED_FORMAT,
            "A T gate on qubit 2 costs more than an S gate.",
            "Place T 1 before TDG 1.",
            "The phase is a sum over x0 xor x1 xor x2.",
            "Spend at most 7 T gates and 6 CNOT gates.",
        ] {
            assert_eq!(no_gate_lists(&single("strategy", text)), Ok(()), "{text}");
        }

        // Three mentions make a list, in any spelling the answer parser accepts.
        for text in [
            "H 0; CNOT 0 1; T 1",
            "h q0, cx q0 q1, tdg q1",
            "Apply CNOT(0,1), then H(2) and S(2).",
            "SDAG 1\nTDAG 2\nX 0",
            "Use T 0, T 1 and T 2.",
        ] {
            let reason = no_gate_lists(&single("strategy", text)).expect_err(text);

            assert!(reason.contains("section \"strategy\""), "{reason}");
        }

        // Each section counts on its own, together with its name.
        let (_, seed) = circuit_seed_cascades().remove(0);
        let worker = seed.stages[0].workers[0].clone();
        let mut spread = worker.clone();
        let mut named = worker.clone();

        spread
            .sections
            .push(section("check", "Compare H 0 with H 1."));
        spread
            .sections
            .push(section("style", "Write T 0 and T 1 last."));
        named.sections.push(section("H 0 T 1", "X 2 ends it."));

        assert_eq!(no_gate_lists(&alone(spread)), Ok(()));
        assert!(no_gate_lists(&alone(named)).is_err());

        // The reason says where the list is.
        let mut listing = worker.clone();

        listing
            .sections
            .push(section("residual", "Keep CNOT 0 1, CNOT 1 0 and CNOT 0 1."));

        let deep = Cascade {
            stages: vec![
                seed.stages[0].clone(),
                Stage {
                    workers: vec![worker, listing],
                },
            ],
        };

        assert_eq!(
            no_gate_lists(&deep),
            Err(
                "stage 1 worker 1: section \"residual\" lists 3 gates with their qubits, \
                 the first \"CNOT 0\"; sections describe methods in words, not circuits"
                    .to_string()
            )
        );
    }

    #[test]
    fn the_brief_names_no_banned_words_no_gates_and_no_targets() {
        let fields = [
            CIRCUIT_BRIEF.system,
            CIRCUIT_BRIEF.context,
            CIRCUIT_BRIEF.format,
            CIRCUIT_BRIEF.mutate_task,
            CIRCUIT_BRIEF.crossover_task,
        ];

        for text in fields {
            let lower = text.to_lowercase();

            for word in BANNED.iter().chain(&TARGET_NAMES) {
                assert!(!lower.contains(word), "{word:?} in {text:?}");
            }

            assert!(gate_mentions(text).is_empty(), "{text}");
            assert!(!lower.contains("grid"), "{text}");
        }

        // The shape of the grid brief: paragraphs end in a blank line, and the tasks end in
        // a space before the shared rules.
        assert!(CIRCUIT_BRIEF.system.contains("exactly one JSON object"));
        assert!(CIRCUIT_BRIEF.context.ends_with("\n\n"));
        assert!(CIRCUIT_BRIEF.format.ends_with("\n\n"));
        assert!(CIRCUIT_BRIEF.mutate_task.starts_with("Task: "));
        assert!(CIRCUIT_BRIEF.mutate_task.ends_with(". "));
        assert!(CIRCUIT_BRIEF.crossover_task.starts_with("Task: "));
        assert!(CIRCUIT_BRIEF.crossover_task.ends_with(". "));

        // Whole requests carry the circuit context, the parents and the task, and still
        // none of the banned words.
        let limits = CascadeLimits::default();
        let seeds: Vec<Cascade> = circuit_seed_cascades()
            .into_iter()
            .map(|(_, c)| c)
            .collect();
        let mutate = llm_mutate_request_with(&CIRCUIT_BRIEF, &seeds[0], 0.5, &limits, 1);
        let cross = llm_crossover_request_with(
            &CIRCUIT_BRIEF,
            (&seeds[1], 0.5),
            (&seeds[2], 0.25),
            &limits,
            2,
        );

        assert!(mutate.prompt.contains(CIRCUIT_BRIEF.mutate_task));
        assert!(cross.prompt.contains(CIRCUIT_BRIEF.crossover_task));

        for req in [mutate, cross] {
            let text = format!("{}\n{}", req.system, req.prompt).to_lowercase();

            assert_eq!(req.system, CIRCUIT_BRIEF.system);
            assert!(req.prompt.starts_with(CIRCUIT_BRIEF.context));
            assert!(req.prompt.contains(CIRCUIT_BRIEF.format));

            for word in BANNED {
                assert!(!text.contains(word), "{word:?} in a request");
            }
        }
    }

    #[test]
    fn the_circuit_pool_holds_the_circuit_genes() {
        let pool = circuit_pool();

        assert_eq!((pool.seeds)(), circuit_seed_cascades());
        assert_eq!(pool.phrases, CIRCUIT_PHRASES);
        assert!(std::ptr::eq(pool.brief, &CIRCUIT_BRIEF));

        for (_, c) in circuit_seed_cascades() {
            assert_eq!((pool.admit)(&c), Ok(()));
        }

        assert!((pool.admit)(&single("strategy", "H 0; CNOT 0 1; T 1")).is_err());
    }
}
