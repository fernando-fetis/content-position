---
name: writing
description: Writing and editing the project's document in docs/ (one Spanish Typst file: the theory, its validation and its applications; lib.typ, references.bib, figures/): mathematical rigor, notation, statements and proofs, how experiments are described, figures, bibliography, Typst style. Use for any change to those documents, their proofs, their figures or the bibliography, and when moving developed material into docs/.
---

# Writing

## The document
- The project has one document, a single Spanish Typst file in `docs/` with the title, the table of contents, the bibliography and everything we keep; each experiment's README names the sections that use it. Its order: the setup, the decomposition and its theory under RoPE with, next to each statement, the measurements that validate it; then the applications, each built on the statements it uses, which it cites by label; then the relation with prior work. A result is stated once, where it belongs, and cited elsewhere; a direct consequence of an earlier statement is a corollary with a short proof; when a statement is used later, its discussion says where (`la @sec:planes usa esta cota ...`). The evidence of an experiment enters as its results arrive: what is still running is stated plainly (which models or settings are pending), and the conclusions claim only what the finished part supports.
- The text never names a file, a folder or a document of the repository, nor an experiment by its number: an analysis is named by what it measures.
- The user polishes the document with coauthors: change them only on the user's instructions, never add results or claims on your own, and report every disagreement you find instead of fixing it silently.

## Language
Spanish, natural and precise. The attention vectors keep their English names with Spanish articles: la query, las keys, los values (never consulta, clave, valor; "valor" only in its numeric sense); the same for any term a Spanish-speaking ML researcher says in English (sink, attention sink, fine-tuning, partial RoPE). Names fixed by the project: ratio posicional $P_h$ and ratio simbólico $S_h = 1 - P_h$; the position-dependent variance written out, $Var(p_j) + Var(epsilon_(i j))$, and the content-dependent one, $Var(c_i) + Var(epsilon_(i j))$ (no $V_h$, $V^f$, energy or mean-key notation); "comportamiento" only for the point $b_h$ of the simplex $Delta^3$.

## Mathematical rigor
- Everything is defined before or at its first use, with its domain; no ambiguity anywhere: a reader must be able to reconstruct every quantity from the text alone.
- State dimensions whenever they are not obvious: vectors in $RR^(d_h)$ or $CC$, matrices in $cal(M)_(T-1)(RR)$, index sets ($[T-1]$, $[F]$).
- One symbol, one meaning: never reuse a symbol for two different or unrelated things, and do not introduce notation that is not used again (a symbol used once is written out instead).
- Natural indices: a sum or product up to $N$ is indexed by $n$, up to $K$ by $k$, and so on; every expectation states its distribution ($EE_(i tilde unif([T-1]))$, or the shorthands $EE_i$, $EE_j$, $EE_(i,j)$ defined in the setup section).
- Claims are exact: say "a lo sumo", "en la mediana", "en $k$ de los $n$ modelos", never more than what was proved or measured; a statement holds under its stated hypotheses only.

## Statements and proofs
- Environments from `docs/lib.typ`: `#definition`, `#proposition`, `#corollary` (one shared counter), `#proof`; label each one (`<def:...>`, `<prop:...>`, `<cor:...>`) and cite with `@label`. Code, configs and READMEs cite them by label, so adding or reordering statements breaks nothing; renaming or removing a label does. Do not chase references while iterating on a document: when a block of edits is finished, run `.claude/skills/audit-repo/scripts/labels.py` and fix the references it reports, or add them to `todo.md` if the document is still changing. When an experiment's definitions, methodology or results move to another file, update the `## Sources` of its README.
- Proofs are natural and complete: every step is shown and justified, small facts included (e.g. $EE[Re z] = Re EE[z]$ because $Re z = 1/2 (z + overline(z))$), and earlier results are reused instead of rederived. Write them as short paragraphs, each opening with a plain sentence that says what it shows ("Para el ítem 2, …"), with the key computations displayed; no "Paso N" headers or numbered steps, and earlier parts are referred to by their content, never "por el paso 3". No auxiliary notation inside proofs: write the expression out. Prefer the most direct argument.

## Experiments in the text
Every measurement is described so that a reader knows what was computed: the data (the tasks $D in cal(D)$ of the setup section, how many prompts, which query), the quantity (defined with the notation of its *Mediciones*: $Q(D)$ means over prompts, variances averaged before any ratio, $cal(H)(D)$ the non-sink heads, $rho_"S"$), how it is aggregated (medians over heads, ranges over models), the result, and a conclusion ("Se concluye que …") that claims no more than the numbers show, important exceptions included. Precise but concise: the prose gives the general conclusions; the numbers go in tables and figures and are not repeated in the prose (at most a key number or two per conclusion), never a case-by-case enumeration; the methodology says what was computed, not implementation details, which live in the code and the summaries. The document is as short as its content allows. Every number comes from an experiment's `results/summary.json`, checked model by model.

## Figures
From real results, included from the experiment's `figures/` folder (`image("/experiments/NN_name/figures/<name>.pdf")`); synthetic ones from `docs/figures/`. How to make them: the `figures` skill. Captions in Spanish, precise: what each panel shows, on which data, how aggregated, and what to look at.

## Typst style
- `docs/lib.typ` holds the shared layout and notation: the `note` template of the document, the statement environments, and the macros for recurring symbols (`#Ph`, `wt`, `zt`, `wr`, `zr`, `sz`, `sw`, `Dpos`, `Dsym`, `eij`, `budget`, `rhoS`, `Dwiki`, `Daguja`, `Var`, `Cov`, `EE`, `Re`, `iu`, `unif`). A symbol that recurs gets a macro there; never retype complex notation inline.
- One paragraph per line (never hard-wrap); displayed equations for anything that is not short; lists only for genuinely parallel items.
- Compile from the repository root with `make docs`; after every edit, compile and check that nothing overflows and no figure lands inside a proof.

## Bibliography
`docs/references.bib`: verify every new entry on arXiv or the venue page before adding it; cite prior work wherever it overlaps, never claim as new what is published, and keep only entries the document cites. The work the project builds on is cited as its lineage, never presented as a competitor.
