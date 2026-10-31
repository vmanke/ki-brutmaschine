# 07 Upbringing at birth: imprinting and teaching

Status: needs-triage (waits for the owner's approval of the design; Q73 is decided in both parts; the step of imprinting follows Q69)
Blocked by: 05, 06
Spec: `../spec.md`, section 10

## What to do

- Imprinting [Q73 a]: a child's evaluation on the search panel runs its first task with the
  birth Souls, then imprints each Worker toward the current Souls of its source Workers (spec 10.1,
  the observed side of the rule of spec 8.3, one-sided), then runs the other tasks in parallel with
  the imprinted Souls. No model call; the parents read no answer and do not change. `imprint:
  false` keeps the birth Soul. The member keeps the imprinted Souls (`Member::imprinted`) and they
  start its current Souls. If the owner picks Q73 (c), this item changes: a Worker of each
  parent judges the child's first answer, which is an Observation of spec 8.2 (spec 10.1).
- Teaching [Q73 d] (spec 10.2), off by default: `Domain::example(task, answer)` renders
  one solved example; fixed code builds the section `upbringing` from each parent's best solved
  search task, within `MAX_SECTION_CHARS`; it ends every system prompt the child sends. It lives in
  the member and the `born` event, never in the genome. No lessons in this step.
- A test helper lists the family stems of both simulated models and their self-check texts, for the
  scan below.

## Done when

- `imprinting_off_keeps_the_birth_soul`: with `imprint: false`, or at κ = 0, every imprinted Soul
  equals the birth Soul.
- `the_first_task_uses_the_birth_soul`: with a scripted backend that reads phases and records them,
  the child's calls on task 0 carry its birth phases and its calls on every later task the imprinted
  ones; at κ = 1000 the imprinted phases equal the source Workers' current phases.
- `imprinting_leaves_the_parents_alone`: the parents' current Souls are the same before and after a
  child's imprinting.
- `teaching_is_never_a_gene`: the section is in the member and the `born` event and not in the
  genome; a grandchild's section is written from its own parents' examples.
- `the_teaching_text_holds_no_stem`: for 1000 sections from short Runs on both Domains, the text
  passes the genome's text rules, has at most `MAX_SECTION_CHARS` characters, and contains no stem
  of `FAMILY_STEMS`, no keyword of the grid model and no self-check text.
- `teaching_changes_no_simulated_answer`: a short circuit Run and a short grid Run with `teach` on
  give the same answers, solved counts and verdicts as with `teach` off.
- Without Souls nothing changes: the goldens of `tests/universe.rs` hold.

## Comments
