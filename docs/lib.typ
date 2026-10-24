// Notation macros and layout of the document in docs/; import with `#import "/docs/lib.typ": *` and compile from the repository root (`make docs`).

#let Re = math.op("Re")
#let Var = math.op($bb(V)$)
#let Cov = math.op("Cov")
#let EE = math.op($bb(E)$)
#let unif = math.op("Unif")
#let wt = $tilde(w)$
#let zt = $tilde(zeta)$
#let wr = $circle(w)$
#let zr = $circle(zeta)$
#let iu = $upright(i)$
#let Dpos = $D_"pos"$
#let Dsym = $D_"sym"$
#let Ph = $P_h$
#let sz = $sigma_(zeta,f)^2$
#let sw = $sigma_(w,f)^2$
#let eij = $epsilon_(i j)$
#let budget = $EE_i [ |w_i^f|^2 ]$
#let rhoS = $rho_"S"$
#let Dwiki = $D_"wiki"$
#let Daguja = $D_"aguja"$

#let names = (proposition: "Proposición", corollary: "Corolario", definition: "Definición", proof: "Demostración")
#let word(key) = names.at(key)
#let thm-counter = counter(figure.where(kind: "thm"))
#let statement(kind, name, body) = figure(kind: "thm", supplement: word(kind), numbering: "1", outlined: false, block(width: 100%, above: 1em, below: 1em, align(start)[*#word(kind) #context thm-counter.display()* (#name)*.* #body]))
#let proposition = statement.with("proposition")
#let corollary = statement.with("corollary")
#let definition = statement.with("definition")
#let proof(body) = block(width: 100%, above: 0.9em, below: 1.3em, inset: (left: 0.9em, y: 0.2em), stroke: (left: 0.6pt + luma(175)))[#set par(spacing: 1.05em, leading: 0.68em); #set math.equation(numbering: none); _#word("proof")._ #body #h(1fr) $square$]

#let base(body) = {
  show figure.where(kind: "thm"): set block(breakable: true)
  show figure.where(kind: "thm"): it => block(breakable: true, it.body)
  show figure.where(kind: image): set figure(placement: auto)
  show figure.where(kind: table): set figure(placement: auto)
  show figure.where(kind: table): set block(breakable: true)
  set figure(gap: 0.5em)
  show figure.caption: set text(size: 9pt)
  set place(clearance: 1em)
  set page(paper: "us-letter", margin: (x: 2.2cm, y: 2.2cm), numbering: "1")
  set text(font: "New Computer Modern", size: 10.5pt, lang: "es", costs: (orphan: 0%, widow: 0%))
  set par(justify: true, leading: 0.6em, spacing: 0.9em)
  set heading(numbering: "1.1")
  show heading: set block(above: 1.2em, below: 0.7em)
  show link: set text(fill: rgb("#1f4e9a"))
  show ref: set text(fill: rgb("#1f4e9a"))
  show cite: set text(fill: rgb("#1f4e9a"))
  show raw: set text(size: 9pt)
  show figure.where(kind: table): set figure.caption(position: top)
  set table(stroke: none, inset: (x: 5pt, y: 3pt))
  show table: set par(justify: false)
  set math.equation(numbering: "(1)")
  show math.equation.where(block: true): it => if it.has("label") or it.numbering == none { it } else { counter(math.equation).update(n => n - 1); math.equation(it.body, block: true, numbering: none) }
  body
}

#let note(title: "", body) = {
  show: base
  align(center, text(size: 17pt, weight: "bold", title))
  v(0.8em)
  body
}
