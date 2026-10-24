---
name: new-theory
description: Turning the user's handwritten mathematics (photos or scans of new results, proof sketches, diagrams, usually dropped in .temp/) into a Typst transcription and then iterating on it together: checking it, searching the literature, proposing variants, simplifying proofs, testing it numerically, and eventually moving it into docs/ or into an experiment. Use whenever the user uploads or points to handwritten or photographed math, or wants to discuss a new idea, statement or proof sketch.
---

# New theory

The user develops ideas by hand (a new theorem, a variant of a statement, a proof sketch, a diagram) and drops photos or scans of them in `.temp/`. This skill is the bridge from that paper to the repository: first a faithful transcription, then a conversation that checks and improves the idea, then, if it survives, its place in `docs/` or in an experiment.

## 1. Transcribe
- Read every page of the upload (image files in `.temp/`, or the paths the user gives) and write `.temp/<topic>.typ` from `template.typ` (next to this file: the project's layout and notation macros, a *Transcripción* and a *Notas* section), compiled with `typst compile --root . .temp/<topic>.typ`.
- Transcribe faithfully, in the user's order and words (Spanish), with the project's notation where the handwriting uses it. Do not correct or complete anything while transcribing: mark an unreadable symbol or step with *[?]* and a question, and a gap in an argument with *[falta]*. Diagrams become a short description or a small figure (synthetic figure code goes to `.temp/` too, `figures` skill).
- Report the transcription in a few lines and list every [?] for the user to resolve before going on.

## 2. Discuss
Only after the user confirms the transcription. Work through what is useful for the idea at hand, one question at a time, and write each conclusion into a second section of the note (`= Notas`), so that the file stays the shared record of the iteration:
- *Correctness*: check every step, hypotheses included; give a counterexample or the missing hypothesis when something fails, and say plainly whether the result holds.
- *Value*: whether the idea says something new about the project's questions, and what it would add to `docs/` (a new statement, a sharper bound, a simpler proof, a new measurement).
- *Literature*: search for the result or close relatives (arXiv, venues, the references in `docs/references.bib`) and report what exists with links; never present as new what is published.
- *Variants*: propose more general, sharper or more natural versions, and say which is worth pursuing and why.
- *Proof*: whether it can be simplified, reuse statements already in `docs/`, or be made more direct (the proof rules of the `writing` skill).
- *Numerical check*: a short script in `.temp/` (numpy, on synthetic data; never heavy computation on the local machine) that tests the statement or a bound before investing in a proof.
- *Experiment*: whether the idea predicts something measurable on the models, and how an experiment would test it (`experiment` skill).

## 3. Move it into the repository
Only when the user decides: the statement and its proof go into the document they belong to, following the `writing` skill (rigor, notation, Typst style, labels of the statements, and the `## Sources` of the experiments that use it); a measurement becomes a new case or analysis of an experiment, or a new experiment. The files in `.temp/` stay scratch: nothing in the repository depends on them.
