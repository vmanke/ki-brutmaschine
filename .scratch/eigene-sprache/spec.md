# Spec: models that invent their own language, chosen by overall efficiency

Status: needs-triage. A design for the owner. Nothing is built.

On 2026-11-05 the owner asked: "Also let the model invent its own language based on overall
efficiency." Asked which model, the owner answered: "Alles". So this note has three parts:

- **Part A**, the prompt Workers of the Cascades (the Soul model). It can be built and screened now.
- **Part B**, chemical reaction networks. It is a design. It waits for the simulator and the genome
  of `.scratch/chemische-netze/`.
- **Part C**, language models that use CNNs. It is a design. It waits for the Architect and the
  CNN tool of `.scratch/llm-nutzt-cnn/`.

`CONTEXT.md` avoids the word "language" for the DSL. This note therefore names the new things
plainly: a **Lexicon** is a Run's list of coined words. A **Coined word** is a phrase that the
Run made itself, with its meaning. A **Coining step** is the moment at which the Run coins or
retires words. A **Code gene** says how a chemical network writes a number into concentrations.

## 1 What the sources say

All sources below were checked on 2026-11-05 against their records (publisher, proceedings,
PubMed, ACL Anthology, OpenReview or arXiv). All are peer reviewed. Each finding is close to the
abstract's own words.

**Efficiency must be a cost, or an invented code goes the wrong way.**

- Neural agents that invent a code give the most frequent inputs the longest messages. Only a
  penalty on message length makes the code efficient. Chaabouni, Kharitonov, Dupoux and Baroni
  (2019), Anti-efficient encoding in emergent communication, NeurIPS 32, arXiv 1905.12561.
- Efficiency needs pressure on both sides: a lazy speaker and an impatient listener. Rita,
  Chaabouni and Dupoux (2020), "LazImpa": Lazy and Impatient neural agents learn to communicate
  efficiently, CoNLL 2020, 335 to 343, DOI 10.18653/v1/2020.conll-1.26.
- People optimise a small lexicon only when accuracy and efficiency both count. Kanwal, Smith,
  Culbertson and Kirby (2017), Zipf's Law of Abbreviation and the Principle of Least Effort:
  Language users optimise a miniature lexicon for efficient communication, Cognition 165, 45 to
  52, DOI 10.1016/j.cognition.2017.05.001.

**Structure comes from compression together with use.**

- Linguistic structure arises from a trade-off between compressibility, imposed while learning,
  and expressivity, imposed while communicating. Kirby, Tamariz, Cornish and Smith (2015),
  Cognition 141, 87 to 102, DOI 10.1016/j.cognition.2015.03.016.
- Colour names across languages are near the optimal trade-off between the complexity and the
  accuracy of the lexicon (information bottleneck). Zaslavsky, Kemp, Regier and Tishby (2018),
  PNAS 115(31), 7937 to 7942, DOI 10.1073/pnas.1800521115.
- A review: a pervasive pressure for efficiency shapes natural language, from words to syntax.
  Gibson, Futrell, Piantadosi, Dautriche, Mahowald, Bergen and Levy (2019), Trends in Cognitive
  Sciences 23(5), 389 to 407, DOI 10.1016/j.tics.2019.02.003.
- Agents form a discrete language with a vocabulary and a syntax. A soft penalty on the active
  vocabulary size helps it become compositional. Mordatch and Abbeel (2018), AAAI 32(1), DOI
  10.1609/aaai.v32i1.11492.

**Invented languages work, but they drift and become unreadable.**

- Most agent-invented languages reach near-perfect task reward, but they are not interpretable
  or compositional unless their communication is restricted. Kottur, Moura, Lee and Batra
  (2017), EMNLP 2017, 2962 to 2967, DOI 10.18653/v1/D17-1321.
- Negotiating agents that both kept learning diverged from human language; the authors froze one
  and mixed in supervised updates. Lewis, Yarats, Dauphin, Parikh and Batra (2017), Deal or No
  Deal? End-to-End Learning of Negotiation Dialogues, EMNLP 2017, 2443 to 2453, section 4, DOI
  10.18653/v1/D17-1259.

**A library of abstractions can be learned by compression.**

- The model with the shortest description of the data is the best estimate. Rissanen (1978),
  Modeling by shortest data description, Automatica 14(5), 465 to 471, DOI
  10.1016/0005-1098(78)90005-5.
- DreamCoder finds common parts of the programs it has solved and builds a growing library of
  abstractions from them. Ellis et al. (2021), PLDI 2021, 835 to 850, DOI 10.1145/3453483.3454080.
  The journal version states that its objective equals a compression objective: Ellis et al.
  (2023), Phil. Trans. R. Soc. A 381(2251), 20220050.
- Stitch learns such libraries 3 to 4 orders of magnitude faster, with equal or better
  compression. Bowers et al. (2023), Top-Down Synthesis for Library Learning, PACMPL 7(POPL),
  article 41, DOI 10.1145/3571234.
- LILO gives the abstractions to a language model. Unnamed abstractions made the model measurably
  worse; named and documented ones helped. Grand et al. (2024), ICLR 2024, arXiv 2310.19791.

**Machines can talk in codes of their own.**

- Agents on top of fixed CNN image features develop their own language in a referential game.
  Lazaridou, Peysakhovich and Baroni (2017), ICLR 2017, arXiv 1612.07182.
- Language models that debate through embeddings instead of tokens did 0.5 to 5.0 % better than
  in plain text. Pham et al. (2024), Let Models Speak Ciphers, ICLR 2024, arXiv 2310.06272.
- When agents may send continuous messages, their systems become more complex and in the end less
  efficient. Chaabouni, Kharitonov, Dupoux and Baroni (2021), PNAS 118(12), e2016569118, DOI
  10.1073/pnas.2016569118.
- One trade-off parameter pushes an agent's code toward lower complexity. Tucker, Levy, Shah and
  Zaslavsky (2022), Trading off Utility, Informativeness, and Complexity in Emergent
  Communication, NeurIPS 35.
- Learned gist tokens compress prompts up to 26 times and save up to 40 % of the FLOPs, with
  little loss. Mu, Li and Goodman (2023), NeurIPS 36, 19327 to 19352, arXiv 2304.08467.
- LLMLingua compresses prompts up to 20 times with little loss. Jiang, Wu, Lin, Yang and Qiu
  (2023), EMNLP 2023, 13358 to 13376, DOI 10.18653/v1/2023.emnlp-main.825.
- Language models that choose their own message format by prompting used up to 72.7 % fewer
  tokens between agents, with the same effect. Chen et al. (2024), Beyond Natural Language,
  Findings of EMNLP 2024, 10626 to 10641, DOI 10.18653/v1/2024.findings-emnlp.623.

**Chemistry can carry numbers in more than one code.**

- The fractional code writes a value as the share of type-1 molecules in two species. With it,
  networks compute polynomials that map [0, 1] to itself. Salehi, Parhi and Riedel (2017), ACS
  Synthetic Biology 6(1), 76 to 83, DOI 10.1021/acssynbio.5b00163.
- A binary-weight ReLU network trained in silico compiles to an equivalent chemical network that
  works independently of the reaction rates. Vasic, Chalk, Khurshid and Soloveichik (2020), ICML
  2020, PMLR 119, 9701 to 9711, arXiv 2003.13720.
- A formose reaction network in a flow reactor classified and forecast after a single trained
  linear read-out layer turned its ion signals into an output. Baltussen, de Jong, Duez, Robinson
  and Huck (2024), Nature 631(8021), 549 to 555, DOI 10.1038/s41586-024-07567-x.

No source here describes the construction below. It is our own.

## 2 Rules for all three parts

1. **Efficiency is a cost in the choice of the code.** Without it, an invented code is not
   efficient (Chaabouni 2019; Kanwal 2017).
2. **Every word and every code has a definition a person can read, and the log holds it.**
   Invented languages drift and become unreadable (Kottur 2017; Lewis 2017). A language model
   needs names to use abstractions well (Grand 2024).
3. **A random code of the same size is the control.** It separates having a code from inventing
   one that fits.
4. **Each part sits behind a switch.** When it is off, every byte stays as today. When it is on,
   `--verify` replays it and a resume continues it.
5. **A code changes what a member says or holds, never how it is scored.** Pass times, the
   solved counts and `rediscovery.json` keep their definitions.

## 3 Part A: a Lexicon for the prompt Workers

### 3.1 What is fixed today

- A Worker's text names gate families only through fixed stems. `designer::FAMILY_STEMS` lists
  them, in English and German (`src/cascade/designer.rs:126-170`). `known` finds the first stem
  of each family in the lower-cased system prompt and orders the families by position
  (`src/cascade/designer.rs:230-245`).
- The section operators draw texts from a fixed pool, `CIRCUIT_PHRASES`. With p = 0.7 the text
  comes from the phrases with the section's name, otherwise from the whole pool, each text alike
  (`src/cascade/variation.rs:1499-1528`).
- The pool has a rule: no phrase names more than two families, and none names two of Hadamard,
  Phase and Linear (`src/cascade/circuit_genes.rs:67-71`). So a Worker needs three sections, one
  for each, before it can pass. The screens of W and IG failed on this step.
- A call's prompt tokens are the words of the system prompt and the prompt
  (`src/cascade/designer.rs:2263`). The fitness charges only completion tokens and the fixed
  cost per call (`src/cascade/arena.rs:3949-3955`). A long prompt costs nothing today.

### 3.2 The Lexicon

- **Coining step.** Every 20 generations, after the commit, the breeder reads the Workers of its
  22 best members, in the archive's ranking. For a set M of two or three families, a Worker
  counts if, for each family of M, it holds a section that names that family and no other. One
  Coined word for M could replace those sections. It saves their words minus the words of the
  coined phrase. The saving of M is the sum over the Workers that count, minus the words of the
  phrase once, for the Lexicon. This is a description length in the sense of Rissanen (1978),
  counted in words, as the simulator counts tokens.
- **Coining.** A set that already has a word in the Lexicon is no candidate. If the best saving
  of the other sets is above 0, the Run coins one word for that set. A tie goes to the set with
  the lower family indices. At most one word is coined per step, and the Lexicon holds at most 8
  words. The coining step changes no member; it only adds a phrase to the pool. Since the members
  do not change, the same set could win again at the next step; the first rule prevents that.
- **The phrase.** A Coined word is a readable phrase from a fixed term per family, for example
  "Use a hadamard basis change, then the phase polynomial." The order is the order in which the
  most Workers name the families. A test checks that `named_families` of each phrase is exactly
  its meaning. Readable phrases follow rule 2 and the finding of Grand (2024).
- **Its section name.** The section name that the compressed sections carry most often. A tie
  goes to the name with the most fixed phrases in the gene pool, then to the alphabetically
  first. Ties are common: a Worker often holds one family under "strategy" and the other under a
  name that SectionReplace wrote it into. Under the name with the most texts, the word gets the
  smallest share of the same-name draws, so it spreads by selection and not because few texts
  share its name (ticket 01, Comments).
- **Use.** A coined phrase joins the gene pool. The section operators draw it like any other
  phrase. It is exempt from the pool's rule of section 3.1: naming two of Hadamard, Phase and
  Linear in one phrase is exactly what a coined word may do.
- **Retirement.** A word that no Worker of the 22 best members holds at two coining steps in a
  row leaves the pool. Workers that hold it keep it; their text still names its families.
- **Arm R, the control.** The same schedule, the same caps and the same retirement. At each
  coining step, arm R first applies L's rule to its own archive. If that rule would coin a set
  of k families, arm R coins a word for a set of k families drawn at random from the sets that
  have no word yet; otherwise it coins nothing. So both arms coin as often, and words of the same
  size, and they differ only in how the meaning is chosen. The draw uses its own stream,
  `lexicon/{g}`, so the generation's stream stays as it is.

### 3.3 Efficiency

- The Lexicon is chosen by the shortest description of the best members' texts. That is the
  efficiency of the language itself.
- The members must feel the cost too. Without a cost on length, codes do not become efficient
  (Chaabouni 2019), and efficiency needs pressure on both sides (Rita 2020).
- Under lexicase the fitness orders the members but does not draw the parents. So a separate
  switch `brevity` adds one case to lexicase. A member passes that case, with score 1, if the
  words of all its Workers' system prompts are at most the median of the archive; otherwise it
  scores 0. The case uses the filter of every other case (`src/cascade/selection.rs:107-119`).
  So when it is drawn and some remaining member passes, the longer ones drop out, like a lazy
  speaker. When no remaining member passes, they tie and all stay. With a lexicase epsilon of 1
  or more the case would remove no one, so such a Run is refused.
- `brevity` works without a Lexicon too. In the screen all three arms use it, so the arms differ
  only in their Lexicon.

### 3.4 What is logged

- An event `coined` with the word, its meaning, its section name and its saving, and an event
  `retired`. `--verify` recomputes both from the archive.
- A state-line group `lexicon`: the number of words, the words coined and retired so far, and
  how many of the 22 best members hold a coined word.
- The texts of every newborn are already in its Born event, so the words a Worker holds and the
  length of its prompt can be read from the log.

### 3.5 Risks

- A coined word for two of Hadamard, Phase and Linear removes the stepping stone the pool was
  built around. That is the aim. Arm R shows how much of the gain comes from compounds alone.
- The 22 best members often share ancestors. Then a word spreads the know-how of one lineage to
  the others, like culture. This is intended, but it can also spread an error.
- Coined words can crowd the pool. The cap of 8 words and the retirement bound that.
- `brevity` works against Workers with many sections, and the pass needs three of them. It can
  slow every arm. A Lexicon that compresses should suffer less from it; that is part of the
  hypothesis.
- A real language model may read a coined phrase differently from the simulator. The phrases are
  readable for that reason, and a real-model Run must check it.

## 4 Part B: chemical networks invent their code

### 4.1 What is fixed today

- Each task input is one dual-rail pair, set from the input value. Each task output is one
  dual-rail pair, read at a fixed time (`.scratch/chemische-netze/research.md`, section 8.2, the
  genome table).
- The research note concludes that a breeder must fix the semantics before the genome
  (`research.md`, the closing reading of section 3; `map.md`, Fog 1, "Semantics first"). These
  semantics are the reaction dynamics: rate-independent, mass action, or stochastic. They are
  not the code of the inputs and outputs, and this part leaves them as they are.
- The German draft fixes the read-out on purpose: "Die Auslese hat keine freie Verstärkung,
  damit die Chemie den Kontrast selbst erzeugen muss"
  (`.scratch/chemische-netze/entwurf-rekurrentes-chemisches-netz.md:759-760`). In its phase 1,
  however, the read-out learns an affine map (same file, line 877).

### 4.2 Proposal

- **Code genes for inputs.** Each input chooses one code from a small menu, with integer
  parameters: dual-rail as today, single-rail with an offset, the fractional code of Salehi
  (2017), or a population code over a few species. Each code costs its species.
- **Reporter genes for outputs.** The genome names a few reporter species. A linear read-out,
  fitted by ridge regression on the training split, maps their concentrations at the fixed read-out
  time to the logits, as in the formose reservoir (Baltussen 2024). The weights are fitted, not
  bred, and they count as parameters.
- **A fixed decoder, a free code.** The reaction semantics still come first; this part needs
  mass action with a fixed read-out time. Within them, the class of the decoder is fixed and
  checkable, a linear map at a fixed time, and the code inside the network is free. So the
  network may invent its own representation. This is a second choice beside Fog 1, not a
  replacement for it.
- **Efficiency.** The fitness keeps its form, the primary score − 0.005·log10(parameters) −
  0.002·log10(FLOPs) (German draft, lines 899-902). Parameters include the read-out weights and
  the code genes. Dissipation is reported; charging it needs an ADR, as the research note says.
- **Controls.** The same read-out on the raw inputs, without chemistry. A fixed random reservoir
  with the same read-out, the labelled baseline the research note already proposes. A random
  assignment of codes. The German draft's fixed read-out stays as a fourth arm, because a free
  read-out can do the work that the chemistry should do.

## 5 Part C: language models and CNNs

### 5.1 What is fixed today

- A CNN Worker reads and writes grids as text (`.scratch/llm-nutzt-cnn/spec.md`, Q84).
- The Architect writes a CNN genome as JSON, and a strict check refuses unknown fields and
  repairs nothing (Q102).
- The CNN tool takes `TOOL CNN` and a grid, and the next prompt carries `TOOL RESULT k` and a grid
  or `NONE` (Q106; the protocol section of the spec).
- Nothing of it is built.

### 5.2 A library for the Architect

- After each Architect round, the breeder compresses the accepted genomes of its best members into
  macros, named sub-architectures, as Stitch does for programs (Bowers 2023).
- Each macro gets a name and a one-line description, as LILO found necessary (Grand 2024).
- The next request lists the macros. An answer may use them. A fixed expander turns them into the
  JSON genome before the strict check, so the check stays as strict as today.
- Efficiency: tokens per Architect answer and validation score per answer. A macro that no answer
  uses for two rounds is retired; the library has a cap. The control offers random macros of the
  same size.

### 5.3 Formats for the CNN tool

- The tool's grid can be written in several lossless text formats: the full grid, runs of equal
  cells, or the difference to the input grid.
- The format is chosen per Run by measured efficiency: tokens per result at equal accuracy. A
  later variant lets the Worker name the format in its request, as in Chen (2024).
- The formats are lossless, so the information stays the same. Only the token cost changes.
- Embeddings, as in Pham (2024), are left out. Calls are recorded and replayed as text, and
  continuous messages became less efficient in Chaabouni (2021).

## 6 Order and open choices

1. Part A: ticket 01 builds the Lexicon; ticket 02 screens it. Both wait for the owner's approval.
2. Part B: ticket 03, after the simulator and the genome of `.scratch/chemische-netze/` (its
   tickets 01 and 02).
3. Part C: ticket 04 after the Architect (`.scratch/llm-nutzt-cnn/`, tickets 11 and 12), and
   ticket 05 after the CNN tool (tickets 15 and 16).

Open choices, for later:

- Opaque new words with a glossary instead of readable compounds. They are shorter, but a real
  model then needs the glossary in every prompt.
- A cost on prompt tokens in the fitness, in addition to `brevity`.
- Messages between Stages (`PREVIOUS ANSWER`) in a code of their own.
- For chemical networks, codes outside the menu, for example free linear maps over species.
