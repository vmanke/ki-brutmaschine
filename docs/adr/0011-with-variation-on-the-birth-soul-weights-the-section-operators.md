# With variation on, the birth Soul weights the section operators

This record amends ADR 0009 for one opt-in setting. Its design is candidate W of
[.scratch/seelen-modell/spec.md](../../.scratch/seelen-modell/spec.md), section 4.1, and
ticket 01 of that folder builds it. The words "preference" and "lean" belong to that note.
`CONTEXT.md` gets them only if the owner adopts the model.

**The setting.** `SoulConfig` gets the field `variation`: `off`, `prefer` or `fixed`. Off is
the default, and it is left out of the JSON. So every config, settings hash, log and test of
ADR 0009 stays byte for byte as it was.

**Arm W, `prefer`.** The birth Soul of each parent Worker gives one preference per family:
w_f = 1 + Re u(p_f). It lies between 0 and 2, and its mean over uniform phases is 1. A text's
lean L adds up w_f − 1 over the families its words name, by the stems the simulated designer
reads. SectionReplace and SectionAdd choose the Worker by 1 + L of its whole prompt.
SectionReplace chooses the section to overwrite by 1 − L, so a Worker protects what it
prefers. Both choose the phrase by 1 + L; SectionAdd multiplies this by the protection of the
section that the phrase would overwrite. Section crossover takes the partner's section by
1 + L of each side, read by the first parent's Worker. Every weight is at least ε = 0.05, so
no choice becomes impossible.

**Today's rules stay.** A weight multiplies the chance that today's operator gives an option,
and the result is normalised again. SectionReplace still draws from its section's own name
with probability 0.7. Each choice takes one draw. A choice whose weights are all equal makes
today's draw, so with every preference at 1 the operators are exactly today's.

**Arm WF, `fixed`.** The same weights use a preference that comes from no Soul: 2 for Linear,
Phase and Hadamard, and 1 for every other family. It is the upper bound of the screen.

**What it supersedes in ADR 0009.** For Runs that set `variation`, it supersedes two points
of ADR 0009. First, ADR 0009 rejected genetic operators that see the Souls, for two reasons.
The first reason still holds: the operators get genes, the preferences come beside them, and
novelty still goes by gene id. The second reason is given up on purpose: arms with the same
genes no longer share every random number, since the weights act from generation 1. That is
the aim, because the pass of step 1 waits for an event of variation. Second, ADR 0009 lets a
Soul show itself only through its Character. With `variation` on, it also shapes the genome
through the choices of variation. Everything else of ADR 0009 holds. The Soul still reaches
no prompt, no request and no Mutator, and Souletics still writes every child's Soul.

**Logs and refusals.** The `soul` group of each `state` line gets `variation_choices` and
`variation_changed`: the choices that the weights decided in that generation's children, and
those whose pick differs from the uniform pick of the same draw. Both are left out when the
setting is off. A Run is refused when its gene pool cannot name the families of a text (grids
today), when `fixed` meets a Domain without Linear, Phase and Hadamard, and when a resume
changes the setting.

Status: proposed on 2026-11-02. It applies only to Runs that set `variation`. The owner
adopts it, or drops it, after the screen of ticket 02. That screen ended on 2026-11-03 with
stop: criteria 1, 2 and 3 failed (`.scratch/seelen-modell/screen-w.md`).
