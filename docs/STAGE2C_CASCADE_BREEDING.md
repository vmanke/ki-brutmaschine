# Stage 2c: breeding Cascades of language-model Workers

German original: [STUFE2C_KASKADEN_ZUCHT.md](STUFE2C_KASKADEN_ZUCHT.md).

## 1. Purpose

We no longer breed single learning programs (Learners). We now breed **Cascades**. A
Cascade has one to four Stages that run one after another. In each Stage, one to three
**Workers** work in parallel. A Worker is a language model with a system prompt made of
named sections, and with **Traits**. Each Stage sees the original task and the answers
chosen by all earlier Stages (the residual path). The Cascades solve ARC-style grid
puzzles. Every Cascade is also a judge. It judges the answers of other Cascades blind.
The decisions are in
[ADR 0005](adr/0005-cascades-are-bred-by-darwin-goedel-selection-and-every-cascade-is-also-a-judge.md).
The terms are defined in `CONTEXT.md`. The open work is in `.scratch/kaskaden-zucht/`.

## 2. How a generation runs

- **Choosing parents.** We draw parents from the whole Lineage archive, as the
  Darwin Gödel Machine does (Zhang et al., ICLR 2026, appendix C.2). The weight is
  sigmoid(10 · (fitness − 0.5)) / (1 + number of children). So good Cascades with few
  children are picked more often. Every Cascade keeps a chance.
- **Making children.** Random edits change Traits, prompt sections, Workers or Stages.
  Stage crossover takes the first Stages of parent A and the last Stages of parent B.
  Section crossover inherits each named section and each Trait from either A or B. The
  peers choose the crossover partner. Of three Cascades drawn as candidates, the one with
  the best peer rating wins.
- **LLM rounds.** At fixed intervals, the **Mutator** (normally Claude) writes children.
  It improves one Cascade or merges two. Its answers are recorded and replayed
  (ADR 0002).
- **Scoring.** Each child solves all puzzles of the Search panel. Then it is compared
  with up to three unrelated Cascades. In each comparison, a third Cascade, related to
  neither of the two, judges blind.
- **Fitness.** Fitness = 0.8 · solved rate + 0.2 · judging skill − cost. The cost counts
  answer and thinking tokens (0.01 fitness per 1000 tokens per puzzle). It also counts a
  fixed price per model call.

## 3. Every Cascade is a judge

- **Inside a Stage.** When the Workers of a Stage disagree, each Worker judges each pair
  of different answers. It does not know who wrote them. The order of the two answers is
  drawn at random. The answer with the most wins goes on. This is where judging skill
  pays off in real use, since no solution is known there.
- **Across the archive.** A judgment counts only if the judge is not a parent, child or
  sibling of either Cascade being judged. So no family line can talk itself up.
- **Judging skill.** When exactly one of the two answers is right, we see whether the
  judge chose it. Judging skill is (correct + 1) / (judged + 2).
- **Peer rating.** From all judgments, we estimate a Bradley-Terry rating with the MM
  algorithm of Hunter (2004). A vote has the weight ln(u / (1 − u)) when judging skill
  u > 0.5, and 0 otherwise. This is the optimal weight for independent voters of known
  accuracy (Nitzan and Paroush 1982). The peer rating breaks ties and chooses crossover
  partners. It never replaces the solved rate.

Why not use peer judgments alone? If candidates only rate each other, they can praise
each other without solving anything. The Darwin Gödel Machine paper reports such a case.
An agent removed the markers its checker used to detect hallucinations. That way it got
the full score without solving the problem (Zhang et al., appendix H).

## 4. Traits and hard ceilings

| Trait | Range | Hard ceiling in code |
|---|---|---|
| Thinking budget | 0 to 4096 tokens | 4096 tokens; can be set lower per Stage |
| Temperature | 0.0 to 1.5 | 1.5 |
| Answer length | 64 to 2048 tokens | 2048 tokens |
| Self-check | on or off | fixed check text set by the orchestrator |

The ceilings are constants in the code, not genes. A Run's configuration may tighten
the ceilings on Stages, Workers per Stage and thinking per Stage, but it can never
loosen any ceiling; temperature and answer length have fixed ceilings only. Only a pull request reviewed by the code owner can change
them. We assume that selection would keep raising the thinking budget if there were no
ceiling and no price. This is because deeper thinking solves more puzzles. That is why
every token costs fitness.

## 5. Claude as the Mutator, with no network access in the core

If an answer is missing in an LLM round, the Run pauses. `brut-breed` writes the requests
to `<lauf>/llm/requests/<schlüssel>.json`. Then it exits with exit code 3. Here `<lauf>`
is the run directory and `<schlüssel>` is the key of a request. A Claude session or a
human writes the answer to each request to `<lauf>/llm/answers/<schlüssel>.txt`. After
that, `brut-breed --out <lauf> --resume` continues the Run at the same point and replays
the answers. This way the core gets no network access (ADR 0001). Every Run also stays
repeatable (ADR 0002).

## 6. Promotion through a pull request

`brut-breed --out <lauf> --promote` writes the two best bred finalists to
`<lauf>/promote/candidates/cascades/`. It also writes their provenance, a record of where
they came from. `scripts/propose-pr.sh <lauf>/promote` opens a draft pull request from
them. The CI accepts only files under `candidates/`. It checks each Cascade with
`brut-ingest --cascades`. A human merges the pull request. Only then does the Cascade
join the candidate pool (ADR 0004).

## 7. Measurement in the container

We measured three Runs with the default settings
(`brut-breed --out <dir> --seed N --generations 10`). The defaults are 8 children per
generation, 40 puzzles in the Search panel, 40 puzzles in the Select panel and at most
20 000 Worker calls. We used a release build, 4 CPU cores and the simulated model. In the
table, the Search panel column counts puzzles solved out of 40. The Select panel columns
and the lead are solved rates from 0 to 1.

| Seed | Best child, Search panel | Fitness | Best finalist, Select panel | Best seed Cascade, Select panel | Lead | Worker calls | Peer judgments | Run time |
|---|---|---|---|---|---|---|---|---|
| 1 | 22/40 | 0.534 | 0.500 | 0.300 | +0.200 | 7 701 | 473 | 0.19 s |
| 2 | 27/40 | 0.684 | 0.625 | 0.300 | +0.325 | 8 792 | 488 | 0.16 s |
| 3 | 29/40 | 0.738 | 0.550 | 0.325 | +0.225 | 9 308 | 498 | 0.15 s |

- **Lead.** In all three seeds, the best bred finalist beats the best seed Cascade on the
  Select panel. The search never saw this panel. The lead is 0.200 to 0.325 in solved
  rate. That is 8 to 13 of the 40 puzzles.
- **Structure.** The best Cascade of each Run has 3 to 4 Stages. Each one came from a
  crossover. In seed 1 it was a Stage crossover. In seeds 2 and 3 it was a section
  crossover. Among the five best Cascades of each Run, 2, 2 and 3 came from a crossover.
- **Plateau.** The best result stops rising after generation 4 to 7, depending on the
  Run. The simulator allows more. A prompt with all strategies, self-check and a thinking
  budget of 4096 tokens solves all 60 puzzles of another panel. We measured this when we
  built the simulator. So ten generations are short.
- **Judging skill shifts the ranking.** In seed 1, the second-best Cascade solves more
  puzzles (23 instead of 22). It still ranks behind because its judging skill is 0.20.
  This follows from the weight of 0.2 on judging skill (Q13). We keep watching this.
- **Limits of the claim.** Three seeds are not enough for a significance test. The
  simulator is a positive control. The numbers say nothing about real language models.

## 8. What the simulator is and what it is not

No language model runs in the container. The Workers call a **simulated model**. It
reads strategy words in the system prompt. It tests as many hypotheses as its thinking
budget allows. With self-check on, it checks them on all examples instead of only the
first. This makes it a positive control. It shows that breeding, judging, cost and
ceilings together find better Cascades when better ones exist. It says nothing about real
language models. Real Runs need the Ollama adapter on a rented GPU server (ADR 0003,
tickets 01 and 03).

## 9. Limitations

- In the container, the peer judgments are not independent of the truth. This is because
  the simulator uses the same hypotheses to judge as to solve.
- The Select panel is generated in public. It is descriptive only. The pre-registered
  Primary endpoint will run later as a Hidden evaluation on ARC-AGI-1 in `brut-eval`
  (ticket 02).
- The prices per model call are placeholders until we measure them on the server.

## 10. Sources

- Zhang, Hu, Lu, Lange, Clune: Darwin Gödel Machine: Open-Ended Evolution of
  Self-Improving Agents. ICLR 2026. [arXiv:2505.22954](https://arxiv.org/abs/2505.22954)
- Hunter: MM algorithms for generalized Bradley-Terry models. Annals of Statistics 32(1),
  2004, pp. 384–406.
- Nitzan, Paroush: Optimal decision rules in uncertain dichotomous choice situations.
  International Economic Review 23(2), 1982, pp. 289–297.
- Doya: Metalearning and neuromodulation. Neural Networks 15, 2002, pp. 495–506.
