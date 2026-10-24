#import "/docs/lib.typ": *

#set document(title: "Contenido, posición e interacción en la atención RoPE: una descomposición exacta")
#show: note.with(title: [Contenido, posición e interacción en la atención RoPE: una descomposición exacta])

#block(width: 100%, inset: (x: 1.5em, y: 0.9em), stroke: (y: 0.5pt + luma(190)))[
  #set text(size: 9.5pt)
  #show outline.entry.where(level: 1): it => { v(0.7em, weak: true); strong(it) }
  #set outline.entry(fill: box(width: 1fr, repeat(text(fill: luma(170))[.], gap: 0.25em)))
  #outline(title: none, depth: 2, indent: 1.4em)
]
#v(0.6em)

= Marco teórico <sec:setup>

Fijemos un cabezal $h$ de dimensión $d_h$, con proyecciones de query y de key $W_Q, W_K in cal(M)_(d_h times d_"model")(RR)$, y una entrada $x_0, dots, x_T in RR^(d_"model")$, los vectores que entran al cabezal en las posiciones $0$ a $T$. Se estudia la query de la posición $T$ (una query en otra posición se trata igual) y su *contexto*, las posiciones $[T-1] := {1, dots, T-1}$: quedan fuera la propia query y la posición 0, el primer token, donde muchos cabezales ponen su atención cualquiera sea el texto (un attention sink, #cite(<xiao2024efficient>, form: "prose")). Un *cabezal sink*, que pone casi toda su atención en esa posición, tiene una matriz contrafactual (@def:counterfactual-logit-matrix) bien definida pero logits que apenas pesan en su salida. Las entradas del cabezal se mantienen fijas: la descomposición describe lo que el cabezal aporta a los logits de su query, no la reacción de las demás capas.

RoPE @su2021roformer divide cada vector en $F := d_h \/ 2$ planos y rota el plano $f in [F]$ en el ángulo $s theta_f$ en la posición $s$, con $theta_f := beta^(-(f-1)\/F)$ para una base $beta$ (Llama 3.1 y 3.2 reducen además las frecuencias lentas, @sec:models). Lo que sigue solo usa que $0 < theta_f < pi$, como en todos los modelos medidos (las construcciones de la @prop:three-pure-heads, que eligen la frecuencia, son la excepción). Identificando cada plano con $CC$, sean $q^f in CC$ el plano $f$ de la query sin rotar $W_Q x_T$ y $k^f (x) in CC$ el de la key sin rotar $W_K x$ (ambas con el sesgo de su proyección, si lo hay). Como el producto interno de dos planos $z, z' in CC$ es $Re[z overline(z')]$ y las rotaciones de la query y de la key se combinan en una por el desfase $T - j$, el logit que la query asigna a un contenido $x$ en la posición $j$ es
$ L(x, j) := 1/sqrt(d_h) sum_(f=1)^F Re[q^f overline(k^f (x)) e^(iu (T-j) theta_f)]. $ <eq:logit>
El factor $1\/sqrt(d_h)$ fija la escala de los logits; ningún ratio depende de él.

Los índices $i, j in [T-1]$ recorren contenidos (el token de la posición $i$ de la entrada) y posiciones; salvo indicación contraria son independientes y uniformes en $[T-1]$, con $EE_i := EE_(i tilde unif([T-1]))$, $EE_j := EE_(j tilde unif([T-1]))$ y $EE_(i,j)$ la esperanza bajo la ley producto, y las varianzas y covarianzas (como $Var(A_(i j))$ o las de las partes de la @sec:matrix) se toman bajo las mismas leyes.

== Modelos <sec:models>

Se usan siete modelos base de dos familias (@tab:models): Qwen2.5 @qwen2024qwen25, con $beta = 10^6$ y sesgo en las queries y las keys, y Llama 3 @grattafiori2024llama, con $beta = 5 dot 10^5$ y las frecuencias de período $2 pi \/ theta_f$ mayor que 8192 posiciones divididas por 32 (Llama-3.2) o por 8 (Llama-3.1), sin cambio bajo 2048 posiciones e interpoladas entre ambos umbrales. En todos, varios cabezales de query consecutivos comparten las proyecciones de key y value (grouped-query attention, @ainslie2023gqa); cada grupo es un *cabezal KV*.

#figure(table(columns: 6, align: (left, center, center, center, center, center), table.hline(), [*modelo*], [*capas*], [*cabezales de query por capa*], [*cabezales KV por capa*], [$d_h$], [*planos* $F$], table.hline(stroke: 0.5pt),
  [Qwen2.5-0.5B], [24], [14], [2], [64], [32],
  [Qwen2.5-1.5B], [28], [12], [2], [128], [64],
  [Qwen2.5-3B], [36], [16], [2], [128], [64],
  [Qwen2.5-7B], [28], [28], [4], [128], [64],
  [Llama-3.2-1B], [16], [32], [8], [64], [32],
  [Llama-3.2-3B], [28], [24], [8], [128], [64],
  [Llama-3.1-8B], [32], [32], [8], [128], [64],
  table.hline()), placement: none, caption: [Los siete modelos.]) <tab:models>

== Tareas <sec:tasks>

Las mediciones usan un conjunto finito $cal(X)$ de prompts en inglés (salvo el código), particionado en 27 *tareas* $D in cal(D)$ de un mismo tipo y longitud, de tres clases.

*Nueve tareas cortas* $cal(D)_"c" subset cal(D)$, de 128 prompts (@tab:tasks): un contexto de 10 a 16 elementos y una pregunta cuya respuesta es un token del contexto, fijada por una posición, un contenido, ambos o una mezcla, frente a un *distractor*, el token de una lectura equivocada.

#figure(table(columns: 4, align: (left, left, left, left), table.hline(), [*tarea*], [*símbolo*], [*la respuesta la fija*], [*ejemplo: contexto; pregunta → respuesta*], table.hline(stroke: 0.5pt),
  [índice], [$D_"índice"$], [una posición], [16 palabras: "meadow. needle. ..."; "What is the word at position 1?" → meadow],
  [salto de un paso], [$D_"salto"$], [una posición], [tokens que terminan en un número $n$; el token $n$ lugares antes del último],
  [recuperación], [$D_"recuperación"$], [un contenido], [16 pares: "M 27. Y 22. ..."; "What number follows M?" → 27],
  [asociación], [$D_"asociación"$], [un contenido], [16 frases: "Jack Silva likes the color gray. ..."; "What color does Jack Silva like the most?" → gray],
  [salto por letra], [$D_"letra"$], [un contenido], [tokens; el token que empieza por la segunda letra del último],
  [punto fijo], [$D_"fijo"$], [contenido y posición], [10 líneas "6: meadow. 5: needle. ..."; la palabra cuyo número es su posición → falcon],
  [salto de dos pasos], [$D_"salto doble"$], [contenido y posición], [como el de un paso, pero el token de llegada es otro número, que indica un segundo salto],
  [última aparición], [$D_"última"$], [una mezcla], [16 pares con una letra repetida; "What number follows I?" → el número de su segunda aparición],
  [sucesor], [$D_"sucesor"$], [una mezcla], [16 palabras; "Which word comes right after needle?" → breeze],
  table.hline()), placement: none, caption: [Las tareas cortas. Las tres de saltos son las de #cite(<urrutia2026positional>, form: "prose"), con sus instrucciones.]) <tab:tasks>

*Tres textos* (@tab:texts), de 32 ventanas de 1024 tokens cada uno, sin pregunta.

#figure(table(columns: 3, align: (left, left, left), table.hline(), [*símbolo*], [*corpus*], [*comienzo de una ventana*], table.hline(stroke: 0.5pt),
  [$Dwiki$], [el conjunto de prueba de WikiText-2 @merity2016pointer], ["The headquarters of the Philippine Coast Guard is located at the South Harbor ..."],
  [$D_"código"$], [código de la biblioteca estándar de Python], [`def default(self): return self._default ...`],
  [$D_"azar"$], [palabras sorteadas al azar entre las de las tareas cortas y las claves de las largas], ["Bob pink Frank falcon magnet Rachel Olivia ..."],
  table.hline()), placement: none, caption: [Los tres textos.]) <tab:texts>

*Quince tareas largas* $D^L$ (@tab:long): cinco tareas, cuatro de ellas de RULER @hsieh2024ruler, cada una a $L in {4000, 16000, 31000}$ tokens, con 50 prompts por longitud y la respuesta a cinco profundidades. Sus claves son nombres escritos en el texto, no keys de la atención.

#figure(table(columns: 3, align: (left, left, left), table.hline(), [*símbolo*], [*tarea*], [*ejemplo: frases clave en el contexto; pregunta → respuesta*], table.hline(stroke: 0.5pt),
  [$Daguja^L$], [una aguja en un pajar], ["One of the special magic numbers for quarry-napkin is: 7048764."; "What is the special magic number for quarry-napkin ...?" → 7048764],
  [$D_"multiclave"^L$], [una aguja entre tres agujas de otras claves], [además, "One of the special magic numbers for lantern-candle is: 6781565."; la misma pregunta → el número de quarry-napkin],
  [$D_"multivalor"^L$], [cuatro valores de una misma clave], [cuatro agujas de quarry-napkin; "What are all the special magic numbers for quarry-napkin ...?" → el primero de ellos],
  [$D_"variables"^L$], [seguimiento de variables, dos cadenas de cinco asignaciones], ["VAR SQEJE = 76048. ..."; "Find all variables that are assigned the value 76048 ..." → SQEJE],
  [$D_"clave–valor"^L$], [el valor de una clave entre pares clave–valor al azar], ["Key y0cq65 has value gmhmtr. ..."; "What is the value of key y0cq65?" → gmhmtr],
  table.hline()), placement: none, caption: [Las cinco tareas largas, cada una a tres longitudes $L$.]) <tab:long>

*Margen y exactitud.* En las tareas con pregunta, el *margen* de un prompt es el logit de salida de la respuesta menos el del distractor en su primer token distinto, con los tokens que comparten antes añadidos al prompt (teacher forcing); el último token del prompt así extendido es la query que se mide. La *exactitud* de una tarea es la fracción de prompts con margen positivo: una elección entre dos tokens tras el prefijo correcto, no la generación de la respuesta. En las tareas largas el distractor es otro valor o variable del mismo prompt (o, en la aguja sin otras agujas, un número al azar), y en $D_"multivalor"^L$ y $D_"variables"^L$ solo se compara el primero.

== Mediciones <sec:measurements>

En cada prompt se mide la última query, cuya posición hace de $T$. Para una cantidad $Q$ de un prompt $x$ y un conjunto de prompts $D subset.eq cal(X)$, $Q(D) := |D|^(-1) sum_(x in D) Q(x)$. Los cocientes se forman con las varianzas ya promediadas: el comportamiento $b_h (D)$ y el ratio posicional $Ph (D)$ (@def:behavior-and-positional-ratio) usan $Var(c_i)(D)$, $Var(p_j)(D)$ y $Var(eij)(D)$. Un cabezal es *sink* en $D$ si su atención media sobre el primer token supera 0.8, y $cal(H)(D)$ es el conjunto de los cabezales de un modelo que no lo son. La correlación de Spearman $rhoS in [-1, 1]$ de dos sucesiones reales de igual longitud es la de Pearson entre sus rangos (con empates al rango medio), invariante por transformaciones crecientes.

= Descomposición contenido-posición <sec:decomposition>

Una fila de logits no separa contenido de posición, porque cada token ocupa una sola posición. La @eq:logit da lo que haría el cabezal con cada token en otro lugar: llevar una key a otra posición solo cambia su rotación.

== Matriz contrafactual <sec:matrix>

#definition[matriz contrafactual de logits][La *matriz contrafactual de logits* del cabezal sobre la entrada es la matriz $A in cal(M)_(T-1)(RR)$ de entradas
$ A_(i j) := L(x_i, j), quad i, j in [T-1], $
el logit que recibiría el contenido de la posición $i$ si estuviera en la posición $j$. Su diagonal, $lambda_t := A_(t t)$ para $t in [T-1]$, contiene los logits que calcula el modelo.] <def:counterfactual-logit-matrix>

La matriz cuesta una pasada hacia adelante: basta guardar la query y las keys sin rotar. Es un contrafactual de un solo producto entre la query y una key, no una nueva ejecución del modelo con los tokens movidos: cada key se calcula en su propio contexto y solo su rotación se lleva a otras posiciones. Mide así la sensibilidad a la posición que RoPE añade en este cabezal; la posición heredada de capas anteriores cuenta como contenido.

Un cabezal *posicional* da el mismo logit a todo contenido en una posición dada (columnas constantes), y uno *simbólico* da a cada contenido el mismo logit en toda posición (filas constantes), nombres de esta descomposición y no de cómo se calcularon las keys; la @sec:metric los formaliza como subespacios. Cuánto de la variación de $A$ se debe a cada factor lo dice un análisis de varianza de dos factores (@fig:overview). #cite(<gu2026deconstructing>, form: "prose") separan también los logits en contenido y posición, pero suponiendo que cada representación es la suma de una parte de contenido y una de posición; la descomposición de $A$ no supone nada sobre las representaciones.

#figure(image("/docs/figures/overview.pdf"), caption: [La descomposición. (a) Una matriz contrafactual, con su diagonal (los logits del modelo) marcada. (b) Sus partes: constante (aquí cero), contenido (constante en cada fila), posición (constante en cada columna) e interacción. Rojo positivo, azul negativo.]) <fig:overview>

#definition[partes de contenido, posicional y de interacción][Las partes de $A$ son:
+ la parte *constante*, $mu := EE_(i,j)[A_(i j)]$;
+ la parte *de contenido*, $c_i := EE_j [A_(i j)] - mu$;
+ la parte *posicional*, $p_j := EE_i [A_(i j)] - mu$;
+ la parte *de interacción*, $eij := A_(i j) - mu - c_i - p_j$.
Por construcción,
$ A_(i j) = mu + c_i + p_j + eij quad "para todo" i, j in [T-1]. $] <def:content-positional-and-interaction-parts>

$c_i$ es la preferencia por el contenido $i$ promediada sobre las posiciones, el efecto de la key calculada en $i$; $p_j$ la preferencia por la posición $j$ promediada sobre los contenidos, el efecto de la fase que RoPE aplica en $j$ en este cabezal; y $eij$ la que depende de ambos. En la diagonal, $lambda_t = mu + c_t + p_t + epsilon_(t t)$, y la atención $alpha_t$ de la query (el softmax sobre todas las posiciones) cumple sobre el contexto
$ alpha_t prop e^(c_t) e^(p_t) e^(epsilon_(t t)), quad t in [T-1], $ <eq:attention-factors>
con una constante común, la única cantidad en que entran $mu$, el primer token y la propia query. Sin interacción, la atención es un producto de expertos @hinton2002training: un peso de contenido por un peso posicional común a todos los contenidos.

#proposition[descomposición bifactorial][
+ *Medias nulas.* $EE_i [c_i] = 0$ y $EE_j [p_j] = 0$; además, $EE_j [eij] = 0$ para todo $i$ y $EE_i [eij] = 0$ para todo $j$.
+ *Ortogonalidad.* $c_i$, $p_j$ y $eij$ son no correlacionados dos a dos, y por lo tanto
  $ Var(A_(i j)) = Var(c_i) + Var(p_j) + Var(eij). $
+ *Casos puros.* Todas las columnas de $A$ son constantes si y solo si $c = epsilon = 0$, y todas las filas de $A$ son constantes si y solo si $p = epsilon = 0$.] <prop:two-way-split>

#proof[Las medias nulas salen de que $i$ y $j$ son independientes y uniformes, de modo que $EE_i [EE_j [A_(i j)]] = EE_(i,j)[A_(i j)] = mu$. Así,
$ EE_i [c_i] = EE_i [EE_j [A_(i j)]] - mu = 0, quad EE_j [p_j] = EE_j [EE_i [A_(i j)]] - mu = 0. $
Para $i$ fijo, promediando la definición de $eij$ sobre $j$, y como $c_i$ no depende de $j$,
$ EE_j [eij] = EE_j [A_(i j)] - mu - c_i - EE_j [p_j] = (mu + c_i) - mu - c_i - 0 = 0, $
y el mismo cálculo sobre $i$, para $j$ fijo, da $EE_i [eij] = (mu + p_j) - mu - EE_i [c_i] - p_j = 0$.

Para la ortogonalidad, como las tres partes tienen media cero, sus covarianzas son las medias de sus productos. Por la independencia de $i$ y $j$, y condicionando en $i$ o en $j$ con las medias nulas anteriores,
$ EE_(i,j)[c_i p_j] = EE_i [c_i] EE_j [p_j] = 0, quad EE_(i,j)[c_i eij] = EE_i [c_i EE_j [eij]] = 0, quad EE_(i,j)[p_j eij] = EE_j [p_j EE_i [eij]] = 0. $
Como $EE_(i,j)[A_(i j)] = mu$, la varianza de $A_(i j)$ es la media del cuadrado de $c_i + p_j + eij$; al desarrollarlo, los productos cruzados tienen media cero y quedan las tres varianzas.

Para los casos puros, supongamos que cada columna $j$ es constante, es decir, $A_(i j) = A_(j j)$ para todo $i$. Entonces $EE_j [A_(i j)] = EE_j [A_(j j)]$ no depende de $i$, y como su promedio sobre $i$ es $mu$, vale $mu$; luego $c = 0$. Además $p_j = EE_i [A_(i j)] - mu = A_(j j) - mu$, así que
$ eij = A_(j j) - mu - 0 - (A_(j j) - mu) = 0. $
Recíprocamente, si $c = epsilon = 0$, entonces $A_(i j) = mu + p_j$ no depende de $i$. Las filas se tratan igual, con los papeles de $i$ y $j$ intercambiados: si cada fila es constante, $p = epsilon = 0$, y si $p = epsilon = 0$, $A_(i j) = mu + c_i$ no depende de $j$.]

== Interacción <sec:interaction>

Un cabezal es *aditivo* si $epsilon = 0$ y de *interacción pura* si $c = p = 0$. La diagonal sola no revela las partes (@fig:three): $A_(i j) = cos(2 pi (i - j)\/(T-1))$ es de interacción pura y de diagonal constante, así que su atención es uniforme, aunque da a cada contenido su mayor logit en su propia posición.

#figure(image("/docs/figures/three_parts.pdf"), placement: top, caption: [Un cabezal posicional, uno simbólico y una interacción pura de diagonal constante ($T - 1 = 12$, diagonal marcada, con la varianza $Var(lambda_t)$ de la diagonal).]) <fig:three>

Definida como residuo, la interacción tiene dos lecturas directas (@fig:profiles): es lo que separa el perfil posicional de cada contenido (su fila centrada) del perfil común, de modo que en un cabezal aditivo mover un token cambia su logit en lo mismo cualquiera sea su contenido, y se lee en cuatro celdas de la matriz, sin calcular antes $c$ ni $p$.

#figure(image("/docs/figures/interaction_profiles.pdf"), placement: none, caption: [Los perfiles posicionales (filas centradas, arriba) y de contenido (columnas centradas, a la derecha), con sus promedios $p_j$ y $c_i$, en una matriz aditiva (a) y en una con interacción (b), donde $eij$ es la distancia de cada perfil al común; (c) el contraste $xi$ de la matriz (b), la suma con signo de las cuatro celdas marcadas, nula en la aditiva ($T - 1 = 6$).]) <fig:profiles>

#proposition[lecturas de la interacción][
+ *Perfiles.* Para cada contenido $i$, sea $p_j^((i)) := A_(i j) - EE_(j')[A_(i j')]$ su *perfil posicional*, cuánto prefiere $i$ la posición $j$ respecto de su promedio sobre las posiciones. Entonces $p_j = EE_i [p_j^((i))]$ y $eij = p_j^((i)) - p_j$. En particular, $Var(eij) = EE_(i,j)[(p_j^((i)) - p_j)^2]$, y $epsilon = 0$ si y solo si todos los contenidos tienen el mismo perfil posicional. Lo mismo vale intercambiando filas y columnas: con el *perfil de contenido* $c_i^((j)) := A_(i j) - EE_(i')[A_(i' j)]$ de cada posición $j$, $c_i = EE_j [c_i^((j))]$ y $eij = c_i^((j)) - c_i$.
+ *Contraste de cuatro celdas.* Sean $i, i', j, j'$ independientes y uniformes en $[T-1]$, y sea
  $ xi := (A_(i j) - A_(i j')) - (A_(i' j) - A_(i' j')), $
  cuánto más gana en logit el contenido $i$ que el contenido $i'$ al pasar de la posición $j'$ a la posición $j$. Entonces, para $i$ y $j$ fijos, $EE_(i',j')[xi] = eij$, y $Var(xi) = 4 Var(eij)$.] <prop:readings-of-the-interaction>

#proof[Para los perfiles, por la @def:content-positional-and-interaction-parts, $EE_(j')[A_(i j')] = mu + c_i$, así que
$ p_j^((i)) = A_(i j) - mu - c_i = p_j + eij. $
Promediando sobre $i$ y usando que $EE_i [eij] = 0$ (@prop:two-way-split, ítem 1), $EE_i [p_j^((i))] = p_j$, y entonces $eij = p_j^((i)) - p_j$. La expresión de $Var(eij)$ sigue porque $epsilon$ tiene media cero, y $epsilon = 0$ equivale a $p_j^((i)) = p_j$ para todo $i, j$. El caso simétrico es el mismo argumento con filas y columnas intercambiadas: $EE_(i')[A_(i' j)] = mu + p_j$ da $c_i^((j)) = c_i + eij$, y como $EE_j [eij] = 0$, $EE_j [c_i^((j))] = c_i$.

Para el contraste, en cada diferencia de una misma fila se cancela el promedio de la fila, $A_(i j) - A_(i j') = p_j^((i)) - p_(j')^((i))$, y al escribir los perfiles como $p_j^((i)) = p_j + eij$ se cancela el perfil común:
$ xi = (eij - epsilon_(i j')) - (epsilon_(i' j) - epsilon_(i' j')). $
Con $i$ y $j$ fijos, la @prop:two-way-split, ítem 1, da $EE_(j')[epsilon_(i j')] = EE_(i')[epsilon_(i' j)] = EE_(i',j')[epsilon_(i' j')] = 0$, así que $EE_(i',j')[xi] = eij$; en particular, $xi$ tiene media cero.

Para la varianza, cada uno de los cuatro términos es $epsilon$ en un par uniforme de $[T-1]^2$, de media cuadrática $Var(eij)$, y son no correlacionados dos a dos: dos de ellos o bien son independientes, y su producto tiene media $EE_(i,j)[eij]^2 = 0$, o bien comparten un solo índice, y condicionando en él las medias nulas del mismo ítem anulan su producto, como en
$ EE_(i,j,j')[eij epsilon_(i j')] = EE_i [EE_j [eij] EE_(j')[epsilon_(i j')]] = 0. $
Por lo tanto $Var(xi) = EE_(i,i',j,j')[xi^2] = 4 Var(eij)$.]

$Var(eij)$ promedia sobre todos los pares de contenido y posición, no sobre la diagonal $epsilon_(t t)$ que ve la atención, cuyo softmax amplifica los logits mayores: una interacción pequeña en varianza puede decidir la atención, como muestra la proposición siguiente con una matriz que no tiene por qué ser la de un cabezal RoPE.

#proposition[interacción pequeña en varianza][Sea $Lambda > 0$. Hay una familia de matrices, una $A in cal(M)_(T-1)(RR)$ para cada $T >= 5$, con $|c_i|, |p_j| <= 1$ y $Var(eij) \/ Var(A_(i j)) <= (3 ln(T - 1) \/ (T - 1))^2$ tal que, con el softmax sobre las posiciones $0$ a $T$ y los logits de la posición 0 y de la query en $[-Lambda, Lambda]$, la atención sobre la posición 1 tiende a uno cuando $T -> infinity$, mientras que sin la interacción la mayor atención sobre una posición tiende a cero.] <prop:small-interaction>

#proof[La matriz pone una interacción concentrada en el contenido y la posición 1 sobre partes de contenido y posicional acotadas. Sean $c, p in RR^(T-1)$ de media cero con $|c_i|, |p_j| <= 1$ y $Var(c_i) + Var(p_j) >= 1$; por ejemplo, $c_i = cos(2 pi i \/ (T - 1))$ y $p_j = cos(2 pi j \/ (T - 1))$, de media cero y varianza $1\/2$, porque $cos^2 x = (1 + cos 2 x) \/ 2$ y $sum_(i=1)^(T-1) cos(2 pi k i \/ (T - 1)) = 0$ para $k in {1, 2}$, al ser $T - 1 >= 3$. Con $y_i := bb(1)_(i = 1) - 1\/(T - 1)$ para $i in [T-1]$, sea
$ A_(i j) := c_i + p_j + 3 ln(T - 1) y_i y_j. $
Como $sum_i y_i = 1 - (T - 1) dot 1\/(T - 1) = 0$, el último término tiene media cero sobre $i$ para todo $j$ y sobre $j$ para todo $i$, así que, por la @def:content-positional-and-interaction-parts, $mu = 0$, las partes de contenido y posicional son $c$ y $p$, y $eij = 3 ln(T - 1) y_i y_j$.

La fracción de interacción es pequeña. Como $sum_i y_i^2 = ((T - 2)^2 + (T - 2)) \/ (T - 1)^2 = (T - 2) \/ (T - 1)$ y $epsilon$ tiene media cero,
$ Var(eij) = (3 ln(T - 1))^2 (EE_i [y_i^2])^2 = (3 ln(T - 1))^2 (T - 2)^2 / (T - 1)^4 <= (3 ln(T - 1) / (T - 1))^2, $
y como $Var(A_(i j)) >= Var(c_i) + Var(p_j) >= 1$ (@prop:two-way-split, ítem 2), la fracción es a lo sumo $Var(eij)$.

La interacción decide la atención. En la diagonal, $epsilon_(1 1) = 3 ln(T - 1) (T - 2)^2 \/ (T - 1)^2$, $epsilon_(t t) = 3 ln(T - 1) \/ (T - 1)^2$ para $t >= 2$ y $|c_t + p_t| <= 2$. Fuera de la posición 1, el softmax ve $T - 2$ logits del contexto de a lo sumo $2 + 3 ln(T - 1) \/ (T - 1)^2$ y dos logits de a lo sumo $Lambda$, así que, como $((T - 2)^2 - 1) \/ (T - 1)^2 = (T - 3) \/ (T - 1) >= 1\/2$,
$ (1 - alpha_1) / alpha_1 <= (T e^(2 + Lambda + 3 ln(T - 1) \/ (T - 1)^2)) / e^(-2 + 3 ln(T - 1) (T - 2)^2 \/ (T - 1)^2) = T e^(4 + Lambda) (T - 1)^(-3 (T - 3) \/ (T - 1)) <= T e^(4 + Lambda) (T - 1)^(-3\/2) -> 0, $
y $alpha_1 -> 1$. Sin la interacción, los $T - 1$ logits del contexto están en $[-2, 2]$ y los otros dos en $[-Lambda, Lambda]$, así que la atención sobre cualquier posición es a lo sumo $e^(2 + Lambda) \/ ((T - 1) e^(-2)) = e^(4 + Lambda) \/ (T - 1)$, que tiende a cero.]

La fracción de interacción mide cuánto se aleja $A$ de las matrices aditivas en la distancia de la @sec:metric, no cuánto pesa en la atención; la prueba causal de la @sec:stability mide en los modelos el efecto de quitarla.

Es de esperar que un cabezal que lee representaciones que aún no unen contenido y posición necesite interacción cuando la posición del objetivo depende de un contenido (el value escrito después de una key dada, el punto fijo de una lista etiquetada); si el objetivo lo fija solo el contenido basta $c$, y si es un desfase fijo, $p$. Las capas anteriores pueden hacer esa unión, y entonces al cabezal que lee le basta $c$.

= Formas cerradas bajo RoPE <sec:rope>

Por la @eq:logit, $A = d_h^(-1\/2) sum_f A^f$, con las matrices de plano
$ A^f_(i j) := Re[w_i^f zeta_j^f], quad w_i^f := q^f overline(k^f (x_i)), quad zeta_j^f := e^(iu (T-j) theta_f). $
El *factor de contenido* $w_i^f$ depende solo del contenido ($Re[w_i^f]$ es el logit del plano sin posición, su módulo el peso del plano y su argumento el desfase que favorece), y la *fase* $zeta_j^f$ solo de la posición.

== Partes de un plano <sec:plane-parts>

Hasta la @sec:length el contexto está fijo. Cada matriz de plano tiene sus partes $mu^f, c^f, p^f, epsilon^f$, y por linealidad las de $A$ son $d_h^(-1\/2)$ veces sus sumas sobre los planos. Se escriben con los estadísticos de la @tab:statistics: el centro de masa (lo común) y la desviación (lo propio), con su varianza y su pseudovarianza.

#figure(table(columns: 3, align: (left, center, center), inset: (x: 8pt, y: 5pt), table.hline(), [], [*factores de contenido* $w_i^f$], [*fases* $zeta_j^f$], table.hline(stroke: 0.5pt),
  [centro de masa], [$wt^f := EE_i [w_i^f]$], [$zt^f := EE_j [zeta_j^f]$],
  [desviación], [$wr_i^f := w_i^f - wt^f$], [$zr_j^f := zeta_j^f - zt^f$],
  [varianza], [$sw := EE_i [ |wr_i^f|^2 ]$], [$sz := EE_j [ |zr_j^f|^2 ]$],
  [pseudovarianza], [$s_w^f := EE_i [(wr_i^f)^2]$], [$s_zeta^f := EE_j [(zr_j^f)^2]$],
  table.hline()), caption: [Los estadísticos de un plano.]) <tab:statistics>

Las desviaciones estándar son $sigma_(w,f), sigma_(zeta,f) >= 0$. Como $|zeta_j^f| = 1$, $sz = 1 - |zt^f|^2$. La varianza mide cuánto se dispersan las desviaciones, y la pseudovarianza en qué dirección.

#proposition[pseudovarianza y componentes principales][Sea $z$ un número complejo aleatorio de media cero, con varianza $sigma^2 := EE[ |z|^2 ]$ y pseudovarianza $s := EE[z^2]$. Visto como un vector aleatorio de $RR^2$, $z$ tiene sus componentes principales en los ángulos $arg(s) \/ 2$ y $arg(s) \/ 2 + pi \/ 2$, con varianzas $(sigma^2 + |s|) \/ 2$ y $(sigma^2 - |s|) \/ 2$; si $s = 0$, toda dirección es principal, con varianza $sigma^2 \/ 2$.] <prop:pseudo-variance-and-pca>

#proof[La proyección de $z$ sobre la dirección de ángulo $phi.alt$ es $Re[z e^(-iu phi.alt)]$. Como $Re w = 1/2 (w + overline(w))$, se tiene $(Re w)^2 = 1/2 |w|^2 + 1/2 Re[w^2]$, y con $w = z e^(-iu phi.alt)$ su varianza es
$ EE[(Re[z e^(-iu phi.alt)])^2] = 1/2 sigma^2 + 1/2 Re[s e^(-2 iu phi.alt)] = 1/2 sigma^2 + 1/2 |s| cos(2 phi.alt - arg(s)). $
Las componentes principales son las direcciones de mayor y de menor varianza proyectada; como el coseno va de $-1$ a $1$, son $phi.alt = arg(s) \/ 2$ y su perpendicular, con las varianzas del enunciado.]

La varianza es la suma de las varianzas de las dos componentes y $|s|$ su diferencia, así que $|s| <= sigma^2$, con igualdad si y solo si $z$ está sobre una recta que pasa por cero, y $s = 0$ si y solo si $z$ se dispersa igual en toda dirección: $|s| \/ sigma^2 in [0, 1]$ mide la anisotropía sin depender de la escala (@fig:statistics). Importa porque un logit es una parte real, la proyección de un número complejo sobre una dirección.

#figure(image("/docs/figures/statistics.pdf"), placement: top, caption: [Los estadísticos de la @tab:statistics: centro de masa (flecha), círculo de radio $sigma$ (discontinuo) y elipse de componentes principales, de semiejes $sqrt((sigma^2 plus.minus |s|) \/ 2)$ (@prop:pseudo-variance-and-pca). Arriba, factores de contenido de igual varianza, de alineados (a) a isotrópicos (c); abajo, las fases de un plano lento (d), uno intermedio (e) y uno rápido (f) sobre 32 posiciones.]) <fig:statistics>

#proposition[formas cerradas por plano][
+ *Formas cerradas de las partes.* Para todo plano $f$,
  $ mu^f = Re[wt^f zt^f], quad c_i^f = Re[wr_i^f zt^f], quad p_j^f = Re[wt^f zr_j^f], quad eij^f = Re[wr_i^f zr_j^f]. $
+ *Formas cerradas de las varianzas.* Para todo plano $f$,
  $ Var(c_i^f) = 1/2 sw |zt^f|^2 + 1/2 Re[s_w^f (zt^f)^2], $
  $ Var(p_j^f) = 1/2 |wt^f|^2 sz + 1/2 Re[(wt^f)^2 s_zeta^f], $
  $ Var(eij^f) = 1/2 sw sz + 1/2 Re[s_w^f s_zeta^f]. $] <prop:the-parts-of-a-plane>

#proof[Como $Re z = 1/2 (z + overline(z))$ y la media es lineal y conmuta con la conjugación, $EE[Re z] = Re EE[z]$ para todo número complejo aleatorio $z$.

Para las formas cerradas de las partes, escribiendo $w_i^f = wt^f + wr_i^f$ y $zeta_j^f = zt^f + zr_j^f$,
$ A^f_(i j) = Re[wt^f zt^f] + Re[wr_i^f zt^f] + Re[wt^f zr_j^f] + Re[wr_i^f zr_j^f]. $
Como $EE_i [wr_i^f] = 0$, $EE_j [zr_j^f] = 0$ y la media conmuta con la parte real, todo término con un factor $wr_i^f$ tiene media cero sobre $i$, y todo término con un factor $zr_j^f$ tiene media cero sobre $j$. Promediando sobre $i$ y $j$ solo queda el primer término, $mu^f = Re[wt^f zt^f]$. Promediando solo sobre $j$ quedan los dos primeros, y restando $mu^f$ se obtiene $c_i^f = Re[wr_i^f zt^f]$; promediando solo sobre $i$ quedan el primero y el tercero, y se obtiene $p_j^f = Re[wt^f zr_j^f]$. Lo que resta es el cuarto término, $eij^f = Re[wr_i^f zr_j^f]$.

Para las varianzas, las tres partes tienen media cero (@prop:two-way-split, ítem 1, aplicada a $A^f$), así que cada varianza es la media del cuadrado de su forma cerrada. Elevando al cuadrado $Re z = 1/2 (z + overline(z))$,
$ (Re z)^2 = 1/4 (z^2 + 2 |z|^2 + overline(z)^2) = 1/2 |z|^2 + 1/2 Re[z^2], quad "de donde" quad EE[(Re z)^2] = 1/2 EE[ |z|^2 ] + 1/2 Re EE[z^2]. $
Para $c_i^f$, con $z = wr_i^f zt^f$,
$ EE_i [ |wr_i^f zt^f|^2 ] = sw |zt^f|^2, quad EE_i [(wr_i^f zt^f)^2] = s_w^f (zt^f)^2. $
Para $p_j^f$, con $z = wt^f zr_j^f$,
$ EE_j [ |wt^f zr_j^f|^2 ] = |wt^f|^2 sz, quad EE_j [(wt^f zr_j^f)^2] = (wt^f)^2 s_zeta^f. $
Para $eij^f$, con $z = wr_i^f zr_j^f$, la independencia de $i$ y $j$ factoriza las medias:
$ EE_(i,j)[ |wr_i^f zr_j^f|^2 ] = sw sz, quad EE_(i,j)[(wr_i^f zr_j^f)^2] = s_w^f s_zeta^f. $]

Cada parte combina lo común o lo propio del contenido con lo común o lo propio de la posición (@fig:plane-parts). Un plano solo prefiere unas posiciones a otras, igual para todos los contenidos, a través de $wt^f$, la dirección que comparten sus keys vistas por la query: si se cancelan, no tiene parte posicional aunque rote. Es el perfil común $Re[wt^f zeta_j^f] = mu^f + p_j^f$ con que TriAttention @mao2026triattention, a partir de los centros no nulos de las queries y las keys sin rotar, predice a qué distancias atiende la query para comprimir la KV cache. La interacción necesita a la vez que las keys difieran ($wr^f != 0$) y que el plano rote ($zr^f != 0$). Acotando $(Re z)^2$ por $|z|^2$ se obtienen cotas que solo usan centros de masa y varianzas.

#corollary[cotas por plano][Para todo plano $f$:
+ *Posición.* $Var(p_j^f) <= |wt^f|^2 sz$.
+ *Interacción.* $Var(eij^f) <= sw sz$.
+ *Contenido.* $Var(c_i^f) <= sw (1 - sz)$.] <cor:where-each-part-can-live>

#proof[Las tres partes tienen media cero (@prop:two-way-split, ítem 1, aplicada a $A^f$), así que cada varianza es la media del cuadrado de la forma cerrada de la @prop:the-parts-of-a-plane, ítem 1, y en cada caso se usa $(Re z)^2 <= |z|^2$.
+ $Var(p_j^f) = EE_j [(Re[wt^f zr_j^f])^2] <= EE_j [ |wt^f|^2 |zr_j^f|^2 ] = |wt^f|^2 sz$.
+ $Var(eij^f) = EE_(i,j)[(Re[wr_i^f zr_j^f])^2] <= EE_(i,j)[ |wr_i^f|^2 |zr_j^f|^2 ] = EE_i [ |wr_i^f|^2 ] EE_j [ |zr_j^f|^2 ] = sw sz$, por la independencia de $i$ y $j$.
+ $Var(c_i^f) = EE_i [(Re[wr_i^f zt^f])^2] <= EE_i [ |wr_i^f|^2 ] |zt^f|^2 = sw |zt^f|^2 = sw (1 - sz)$, pues $|zt^f|^2 = 1 - sz$.]

#figure(image("/docs/figures/plane_parts.pdf"), placement: none, caption: [Las cuatro partes de un plano: cada celda es la parte real del producto de lo común o lo propio del contenido (columna izquierda) por lo común o lo propio de la posición (fila superior), y las cuatro suman $A^f$.]) <fig:plane-parts>

Las cotas son productos de dos repartos, el de los factores de contenido en lo común y lo propio y el de las fases (@fig:part-bounds). Un plano es *lento* si sus fases se dispersan poco, $sz approx 0$, y *rápido* si se reparten por el círculo, $sz approx 1$ (@prop:phase-statistics). La *varianza dependiente de la posición* de un plano, $Var(p_j^f) + Var(eij^f)$, es a lo sumo $(|wt^f|^2 + sw) sz$; ambas son reglas de selección de planos de la @sec:planes: un plano lento lleva casi solo contenido o una constante que el softmax ignora. En uno rápido no hay espacio para el contenido, y posición e interacción se reparten en la proporción $|wt^f|^2 : sw$, posicional si las keys comparten una dirección y de interacción si están dispersas; como además $s_zeta^f approx 0$ (@fig:dispersion), sus varianzas son casi la mitad de esas cotas (@prop:the-parts-of-a-plane). Como $wt^f$ promedia las keys de la entrada y $sz$ depende de la longitud del contexto, las partes de un cabezal dependen de ambos.

#figure(image("/experiments/01_measure/figures/plane_bounds_real.pdf"), placement: bottom, caption: [Las cotas del @cor:where-each-part-can-live en los planos 31, 28 y 26 de Qwen2.5-7B en $Dwiki$, de más lento a más rápido ($sz approx 0.19$, 0.56 y 0.90), promediadas sobre los cabezales no sink y divididas por $budget$. Las columnas reparten $budget$ en $|wt^f|^2$ y $sw$, y las filas reparten $1$ en $|zt^f|^2$ y $sz$: el área de cada celda es la cota de la parte que cruza esa columna con esa fila (gris, la constante), el área sólida su varianza medida y el número, su cociente con la cota.]) <fig:part-bounds>

== Momentos de las fases <sec:phase-moments>

=== Planos lentos y rápidos <sec:slow-fast>

Las fases no dependen del modelo ni de la entrada: para un contexto dado, sus momentos son funciones explícitas de la frecuencia, a través de una suma geométrica.

#proposition[momentos de las fases y núcleo de Dirichlet][Para $u in (-2 pi, 2 pi)$,
$ EE_j [e^(iu (T-j) u)] = e^(iu T u \/ 2) delta_(T-1) (u), quad delta_(T-1) (u) := sin((T-1) u \/ 2) / ((T-1) sin(u \/ 2)), quad delta_(T-1) (0) := 1, $
con $delta_(T-1)$ el núcleo de Dirichlet normalizado. Por lo tanto, para un plano con $0 < theta_f < pi$:
+ *Centro de masa.* $zt^f = e^(iu T theta_f \/ 2) delta_(T-1) (theta_f)$.
+ *Varianza.* $sz = 1 - delta_(T-1) (theta_f)^2$.
+ *Pseudovarianza.* $s_zeta^f = e^(iu T theta_f) delta_(T-1) (2 theta_f) - (zt^f)^2$.] <prop:phase-statistics>

#proof[La fórmula general es una suma geométrica. Si $u = 0$, ambos lados valen uno. Si $u != 0$, entonces $e^(iu u) != 1$, porque $u in (-2 pi, 2 pi)$, y como el desfase $T - j$ recorre $1, dots, T - 1$ cuando $j$ recorre $[T-1]$,
$ sum_(j=1)^(T-1) e^(iu (T-j) u) = e^(iu u) (e^(iu (T-1) u) - 1) / (e^(iu u) - 1) = e^(iu u) (e^(iu (T-1) u \/ 2) sin((T-1) u \/ 2)) / (e^(iu u \/ 2) sin(u \/ 2)) = e^(iu T u \/ 2) sin((T-1) u \/ 2) / sin(u \/ 2), $
donde la segunda igualdad usa $e^(iu x) - 1 = 2 iu e^(iu x\/2) sin(x\/2)$ en el numerador y en el denominador, y $sin(u\/2) != 0$ porque $u\/2 in (-pi, pi) without {0}$. Dividiendo por $T - 1$ se obtiene la fórmula.

Para el ítem 1, $zt^f = EE_j [e^(iu (T-j) theta_f)]$ es la fórmula con $u = theta_f in (0, pi)$. Para los ítems 2 y 3, desarrollando los cuadrados y usando $EE_j [zeta_j^f] = zt^f$,
$ sz = EE_j [ |zeta_j^f|^2 ] - |zt^f|^2, quad s_zeta^f = EE_j [(zeta_j^f)^2] - (zt^f)^2. $
Como $|zeta_j^f| = 1$ y $delta_(T-1)$ es real, la primera es $1 - delta_(T-1) (theta_f)^2$. Como $(zeta_j^f)^2 = e^(iu (T-j) 2 theta_f)$ y $2 theta_f in (0, 2 pi)$, la fórmula con $u = 2 theta_f$ da $EE_j [(zeta_j^f)^2] = e^(iu T theta_f) delta_(T-1) (2 theta_f)$.]

#figure(image("/experiments/01_measure/figures/phase_moments_real.pdf"), caption: [Los momentos de las fases en los planos de los siete modelos. (a) $delta_(T-1) (u)$ para tres valores de $T - 1$. (b–g) $|zt^f|$, $sz$ y $|s_zeta^f|$ en función de $theta_f$, en $Dwiki$ (arriba) y $Daguja^31000$ (abajo): curva, la @prop:phase-statistics; puntos, los planos, con $sz$ medido; discontinua, $theta_f = 2 pi \/ (T - 1)$; sombreado, los planos lentos ($sz < 0.1$), y rayado, los rápidos ($sz > 0.9$).]) <fig:dispersion>

Un plano cuyo período es mucho más largo que el contexto gira poco sobre él ($|zt^f| approx 1$, $sz approx 0$: lento); uno que da una vuelta o más reparte sus fases por el círculo ($|zt^f| approx 0$, $sz approx 1$: rápido). La transición está cerca de $theta_f = 2 pi \/ (T - 1)$, donde $delta_(T-1) (theta_f)$ cae de uno a casi cero, y solo cerca de ella es grande la pseudovarianza; con un contexto más largo se mueve hacia frecuencias menores, y un mismo plano pasa de lento a rápido (@sec:length). En los siete modelos, la varianza medida de las fases coincide con la de la proposición, y cada cabezal tiene de 13 a 27 planos rápidos en $Dwiki$ y de 17 a 41 en $Daguja^31000$ (@fig:dispersion).

=== Partes de los planos <sec:plane-measures>

Para un plano $f$ de un cabezal $h in cal(H)(D)$, la fracción de cada parte $K in {c, p, epsilon}$ en la varianza del plano y su razón a su cota del @cor:where-each-part-can-live son
$ psi_K^f := (Var(K^f)(D)) / ((Var(c^f) + Var(p^f) + Var(epsilon^f))(D)), quad R_K^f := (Var(K^f)(D)) / (B_K^f (D)), $
con $B_c^f = sw (1 - sz)$, $B_p^f = |wt^f|^2 sz$ y $B_epsilon^f = sw sz$ evaluadas en $D$, y $R_K^f$ definida cuando $B_K^f (D) > 0$. Por la @prop:the-parts-of-a-plane, ítem 2, y como $|s_w^f| <= sw$ (@prop:pseudo-variance-and-pca), $R_p^f$ y $R_epsilon^f$ están en $1/2 (1 plus.minus |s_zeta^f| \/ sz)$, que fijan solo las fases, y $R_c^f$ en $1/2 (1 plus.minus |s_w^f| \/ sw)$, que fijan las keys. La @fig:planes-exp muestra ambas en $Dwiki$, en función del período del plano sobre la longitud del contexto, $2 pi \/ ((T - 1) theta_f)$.

Ninguna cota se viola más allá del redondeo. Las fracciones siguen a las cotas (@fig:planes-exp a): la mediana de la de contenido es menor que 0.01 hasta un período de tres cuartos del contexto, donde posición e interacción llevan cerca de dos tercios y un tercio, y de al menos 0.97 desde 13 veces el contexto; el reparto varía mucho entre cabezales en los planos rápidos, según sus keys, y casi nada en los lentos. Las razones quedan en la mitad (@fig:planes-exp b): mediana 0.50 para la posición y la interacción en los planos rápidos y cerca de 0.50 para el contenido en los lentos, salvo en la transición (los planos de período entre 2 y 13 veces el contexto), donde la pseudovarianza de las fases sube la de la posición hasta 0.56–0.65 en la mediana. Con las varianzas promediadas sobre los cabezales antes del cociente (@fig:part-bounds), la mediana sobre los planos lentos de la razón de contenido en $Dwiki$ baja a 0.37–0.45 según el modelo: los cabezales de mayor presupuesto quedan bajo la mitad.

#figure(image("/experiments/01_measure/figures/plane_fractions_ci.pdf"), caption: [Las partes de los planos en $Dwiki$, sobre los pares de cabezal no sink y plano de los siete modelos, por intervalos de un cuarto de década del período sobre la longitud del contexto. (a) Las fracciones $psi_K^f$: banda, el rango intercuartílico; gruesa, la mediana; finas, la de cada modelo. (b) Las razones $R_K^f$ (banda y mediana), con el intervalo $1/2 (1 plus.minus |s_zeta^f| \/ sz)$ de $R_p^f$ y $R_epsilon^f$ (gris). Sombreado, los planos lentos, y rayado, los rápidos.]) <fig:planes-exp>

Se concluye que, como imponen las cotas, la frecuencia de un plano frente al contexto decide qué lleva: contenido en los planos lentos, y posición o interacción, según las keys de cada cabezal, en los rápidos, en estos modelos sobre todo posición, en línea con #cite(<barbero2025round>, form: "prose") (@sec:barbero) y con que los cabezales de recuperación a larga distancia apenas usen las dimensiones que más rotan @chiang2025rotary; y que, fuera de la transición, el cabezal típico usa la mitad del espacio de sus cotas.

== Del plano al cabezal <sec:plane-to-head>

Esta sección suelta el contexto y el plano: cómo cambian las partes de un plano con la longitud (@sec:length) y cómo se suman en las de un cabezal (@sec:head-from-planes).

=== Longitud del contexto <sec:length>

La longitud del contexto lleva planos de un régimen al otro (@fig:dispersion, filas). Para aislar ese efecto, la varianza dependiente de la posición de un plano se divide por una cantidad que no dependa de la rotación (no la varianza total: al rotar, el contenido le cede su parte a la posición y a la interacción).

Esa cantidad es el *presupuesto* del plano, el módulo cuadrático medio de sus factores de contenido, $budget = |wt^f|^2 + sw = |q^f|^2 EE_i [ |k^f (x_i)|^2 ]$: acota la varianza total del plano, pues $|A^f_(i j)| <= |w_i^f|$, no depende de las fases, incluye el centro de masa del que sale la parte posicional, y el cociente con él no cambia al escalar la query o las keys. El corolario siguiente determina, salvo un margen, la fracción del presupuesto que es varianza dependiente de la posición, en función solo de la frecuencia y de la longitud del contexto.

#corollary[dependencia de la longitud del contexto][Sea $f$ un plano de frecuencia $0 < theta_f < pi$ con $budget > 0$. La fracción de su presupuesto que es varianza dependiente de la posición queda entre dos valores que dependen solo de $theta_f$ y de $T$:
$ 1/2 (sz - |s_zeta^f|) <= (Var(p_j^f) + Var(eij^f)) / (budget) <= 1/2 (sz + |s_zeta^f|). $
En consecuencia:
+ *Contextos cortos.* La fracción es menor que $((T-1) theta_f)^2 \/ 12$.
+ *Contextos largos.* La fracción tiende a $1\/2$ cuando $T -> infinity$.] <cor:slow-planes-and-long-contexts>

#proof[La varianza dependiente de la posición tiene forma cerrada. Sumando las de $Var(p_j^f)$ y $Var(eij^f)$ de la @prop:the-parts-of-a-plane, ítem 2,
$ Var(p_j^f) + Var(eij^f) = 1/2 (|wt^f|^2 + sw) sz + 1/2 Re[((wt^f)^2 + s_w^f) s_zeta^f] = 1/2 budget sz + 1/2 Re[EE_i [(w_i^f)^2] s_zeta^f], $
porque, desarrollando los cuadrados con $EE_i [w_i^f] = wt^f$, $budget = |wt^f|^2 + sw$ y $EE_i [(w_i^f)^2] = (wt^f)^2 + s_w^f$. Como $|Re z| <= |z|$ y $|EE_i [(w_i^f)^2]| <= budget$, el segundo término está entre $-1/2 budget |s_zeta^f|$ y $1/2 budget |s_zeta^f|$, lo que da las dos cotas.

Para el ítem 1, como $|s_zeta^f| <= sz$ (@prop:pseudo-variance-and-pca), la cota superior es a lo sumo $sz$. Para toda constante $a in CC$, desarrollando el cuadrado y usando $EE_j [zeta_j^f - zt^f] = 0$,
$ EE_j [ |zeta_j^f - a|^2 ] = sz + |zt^f - a|^2 >= sz. $
Con $a = e^(iu T theta_f \/ 2)$, como la distancia entre dos puntos del círculo unitario es a lo sumo la diferencia de sus ángulos, $|zeta_j^f - a| <= theta_f |T\/2 - j|$, y por lo tanto
$ sz <= theta_f^2 EE_j [(j - T\/2)^2] = theta_f^2 ((T-1)^2 - 1) / 12, $
pues $j$ es uniforme en ${1, dots, T-1}$, con media $T\/2$, y la varianza de la ley uniforme sobre $T - 1$ enteros consecutivos es $((T-1)^2 - 1)\/12$. Como $(T-1)^2 - 1 < (T-1)^2$, la fracción es menor que $((T-1) theta_f)^2 \/ 12$; con una décima de vuelta, $(T-1) theta_f = pi \/ 5$ y esta cota es $pi^2 \/ 300 < 0.033$.

Para el ítem 2, por la @prop:phase-statistics, $sz = 1 - delta_(T-1) (theta_f)^2$ y $|s_zeta^f| <= |delta_(T-1) (2 theta_f)| + delta_(T-1) (theta_f)^2$. Para $u in (-2 pi, 2 pi)$ no nulo, $|delta_(T-1) (u)| <= 1 \/ ((T-1) |sin(u\/2)|)$, porque $|sin((T-1) u\/2)| <= 1$, así que, con $u = theta_f$ y con $u = 2 theta_f$, $sz -> 1$ y $s_zeta^f -> 0$ cuando $T -> infinity$, y las dos cotas tienden a $1\/2$.]

#figure(image("/experiments/01_measure/figures/budget_share.pdf"), placement: none, caption: [El @cor:slow-planes-and-long-contexts en los planos medidos, con las varianzas y el presupuesto promediados sobre los prompts y los cabezales de $cal(H)(D)$. (a) El presupuesto de seis planos de Qwen2.5-7B en $Daguja^4000$ y $Daguja^31000$, repartido entre las tres partes y el resto (la constante y lo que pierde la parte real), con el intervalo del corolario bajo cada barra y $Phi^f$ a la derecha. (b) $Phi^f (D)$ en función de las vueltas $(T - 1) theta_f \/ 2 pi$, en las tareas largas de los siete modelos, con cada plano de Qwen2.5-7B en $Daguja^L$ unido a través de las tres longitudes (gris): sombreado, el intervalo con $T = 4000$; discontinua, la cota de contextos cortos; punteada, el límite $1\/2$; rayado, lo que ningún plano alcanza.]) <fig:budget-exp>

La varianza dependiente de la posición de un plano es así su presupuesto, que fijan los pesos y la entrada, por una fracción que fijan $theta_f$ y $T$: menos de 3.3% con una décima de vuelta y la mitad con muchas vueltas, porque un logit es una parte real; las keys solo deciden cómo se reparte esa mitad entre posición e interacción. Al crecer el contexto, un plano lento gana espacio para la posición, lo que usan la @sec:long-contexts, para el ratio posicional, y la @sec:planes, para la longitud de calibración de la selección de planos.

*Fracción del presupuesto en los modelos.* Para un plano $f$ y una tarea $D$, $Phi^f (D)$ es la media sobre $h in cal(H)(D)$ de $Var(p_j^f)(D) + Var(eij^f)(D)$ dividida por la media de su presupuesto, que queda también en el intervalo del corolario si los prompts de $D$ tienen la misma longitud. Los planos salen de él a lo sumo en $2.8 dot 10^(-3)$ del presupuesto, y solo en tareas cuyos prompts no tienen exactamente la misma longitud; $Phi^f$ es a lo sumo 0.54: los planos de muchas vueltas quedan en $1\/2$, y solo el margen de la pseudovarianza, cerca de una vuelta, pasa de la mitad (@fig:budget-exp b). Al crecer el contexto, cada plano avanza por el intervalo y su presupuesto pasa del contenido a la posición (@fig:budget-exp a). Se concluye que la varianza dependiente de la posición de los planos entrenados es la fracción del presupuesto que fija la rotación, salvo el margen de la pseudovarianza, la única dependencia de las keys que queda.

=== Agregación sobre planos <sec:head-from-planes>

Las varianzas de las partes de un cabezal son sumas de covarianzas entre sus planos, que los momentos de las fases acotan y que, entre planos distintos, se anulan al crecer el contexto.

#proposition[agregación sobre planos][
+ *Covarianzas entre planos.* La varianza de cada parte del cabezal es $d_h^(-1)$ veces la suma sobre los pares de planos $f, g$ de las covarianzas de sus partes, y para $f != g$ las de las partes posicional y de interacción tienen forma cerrada:
  $ Cov(p_j^f, p_j^g) = 1/2 Re[wt^f overline(wt^g) EE_j [zr_j^f overline(zr_j^g)]] + 1/2 Re[wt^f wt^g EE_j [zr_j^f zr_j^g]], $
  $ Cov(eij^f, eij^g) = 1/2 Re[EE_i [wr_i^f overline(wr_i^g)] EE_j [zr_j^f overline(zr_j^g)]] + 1/2 Re[EE_i [wr_i^f wr_i^g] EE_j [zr_j^f zr_j^g]]. $
  En particular, $|Cov(p_j^f, p_j^g)| <= |wt^f| |wt^g| sigma_(zeta,f) sigma_(zeta,g)$ y $|Cov(eij^f, eij^g)| <= sigma_(w,f) sigma_(w,g) sigma_(zeta,f) sigma_(zeta,g)$.
+ *Contextos largos.* Si las frecuencias de los planos son distintas, las medias de las fases $EE_j [zr_j^f overline(zr_j^g)]$ y $EE_j [zr_j^f zr_j^g]$ dependen solo de $theta_f$, $theta_g$ y $T$, y tienden a cero cuando $T -> infinity$. Si además $|wt^f|$ y $sw$ se mantienen acotados en todo plano $f$ (y con ellos, por la desigualdad de Cauchy–Schwarz, $|s_w^f|$ y las medias cruzadas $EE_i [wr_i^f overline(wr_i^g)]$ y $EE_i [wr_i^f wr_i^g]$), la varianza dependiente de la posición del cabezal tiende a la suma de las de sus planos, y su interacción, a la mitad de la suma de las cotas del @cor:where-each-part-can-live, ítem 2, ambas divididas por $d_h$:
  $ (Var(p_j) + Var(eij)) - 1/d_h sum_f (Var(p_j^f) + Var(eij^f)) -> 0, quad Var(eij) - 1/(2 d_h) sum_f sw sz -> 0. $] <prop:head-as-a-sum-of-planes>

#proof[Para el ítem 1, las partes del cabezal son sumas de las de sus planos. Por la @eq:logit, $A = d_h^(-1\/2) sum_f A^f$, y cada parte es una media de entradas de la matriz menos otras medias, así que es lineal en la matriz: $c_i = d_h^(-1\/2) sum_f c_i^f$, y lo mismo para $mu$, $p$ y $epsilon$. La varianza de una suma es la suma de las covarianzas de todos los pares de sumandos.

Las formas cerradas salen como las de la @prop:the-parts-of-a-plane. Como $Re z = 1/2 (z + overline(z))$, para todo par de números complejos $Re[z] Re[z'] = 1/2 Re[z overline(z')] + 1/2 Re[z z']$. Por la @prop:the-parts-of-a-plane, ítem 1, $p_j^f = Re[wt^f zr_j^f]$ y $eij^f = Re[wr_i^f zr_j^f]$, y todas las partes tienen media cero, así que tomando medias con $z = wt^f zr_j^f$ y $z' = wt^g zr_j^g$, y con $z = wr_i^f zr_j^f$ y $z' = wr_i^g zr_j^g$, donde la independencia de $i$ y $j$ factoriza cada media, se obtienen las dos fórmulas.

Las cotas salen de la desigualdad de Cauchy–Schwarz. Para las fases, como $|zr_j^g| = |overline(zr_j^g)|$,
$ |EE_j [zr_j^f overline(zr_j^g)]| <= sqrt(EE_j [|zr_j^f|^2] EE_j [|zr_j^g|^2]) = sigma_(zeta,f) sigma_(zeta,g), $
y lo mismo para $|EE_j [zr_j^f zr_j^g]|$; para los factores de contenido, del mismo modo, $|EE_i [wr_i^f overline(wr_i^g)]|$ y $|EE_i [wr_i^f wr_i^g]|$ son a lo sumo $sigma_(w,f) sigma_(w,g)$. Con $|Re z| <= |z|$, cada término de las fórmulas queda acotado por la mitad del producto correspondiente, y los dos términos suman la cota.

Para el ítem 2, las medias de las fases se desarrollan con $EE_j [zeta_j^f] = zt^f$:
$ EE_j [zr_j^f overline(zr_j^g)] = EE_j [zeta_j^f overline(zeta_j^g)] - zt^f overline(zt^g), quad EE_j [zr_j^f zr_j^g] = EE_j [zeta_j^f zeta_j^g] - zt^f zt^g, $
con $zeta_j^f overline(zeta_j^g) = e^(iu (T-j)(theta_f - theta_g))$ y $zeta_j^f zeta_j^g = e^(iu (T-j)(theta_f + theta_g))$, que dependen solo de las frecuencias y de $T$. Como $theta_f, theta_g in (0, pi)$ son distintos, $theta_f - theta_g in (-pi, pi) without {0}$ y $theta_f + theta_g in (0, 2 pi)$, así que, por la @prop:phase-statistics, la media de $e^(iu (T-j) u)$ en esos puntos tiene módulo $|delta_(T-1) (u)| <= 1 \/ ((T-1) |sin(u\/2)|)$, porque $|sin((T-1) u\/2)| <= 1$, y tiende a cero; y $|zt^f|^2 = delta_(T-1) (theta_f)^2$ y $|zt^g|^2 = delta_(T-1) (theta_g)^2$ tienden a cero por la misma cota.

En las formas cerradas del ítem 1, cada media de las fases de dos planos distintos multiplica un factor de contenido, $wt^f overline(wt^g)$, $wt^f wt^g$, $EE_i [wr_i^f overline(wr_i^g)]$ o $EE_i [wr_i^f wr_i^g]$, de módulo a lo sumo $|wt^f| |wt^g|$ o $sigma_(w,f) sigma_(w,g)$ por la desigualdad de Cauchy–Schwarz, y por lo tanto acotado; así, toda covarianza entre dos planos distintos tiende a cero. Por el ítem 1,
$ Var(p_j) + Var(eij) = 1/d_h sum_(f, g) (Cov(p_j^f, p_j^g) + Cov(eij^f, eij^g)), $
y los términos con $f = g$ son $Var(p_j^f) + Var(eij^f)$, así que la primera diferencia es la suma de los términos con $f != g$ y tiende a cero. Del mismo modo, la segunda es la suma de las covarianzas de la interacción entre planos distintos más $1/d_h sum_f (Var(eij^f) - 1/2 sw sz)$, y por la @prop:the-parts-of-a-plane, ítem 2, $Var(eij^f) - 1/2 sw sz = 1/2 Re[s_w^f s_zeta^f]$, que tiende a cero porque $|s_w^f| <= sw$ (@prop:pseudo-variance-and-pca) está acotado y $|s_zeta^f| <= |delta_(T-1) (2 theta_f)| + delta_(T-1) (theta_f)^2$ por la @prop:phase-statistics y $delta_(T-1)$ tiende a cero en $theta_f$ y en $2 theta_f in (0, 2 pi)$.]

Como en un plano, la entrada interviene por los estadísticos de los factores de contenido y la rotación por las medias de las fases de dos planos, que dependen solo de las frecuencias y de $T$ (@fig:aggregation-exp a, b). En contextos largos, la varianza dependiente de la posición de un cabezal es la suma de las de sus planos, salvo entre planos vecinos de la transición, y su interacción, la mitad de la suma de las cotas del @cor:where-each-part-can-live, ítem 2.

#figure(image("/experiments/01_measure/figures/aggregation_real.pdf"), placement: bottom, caption: [(a, b) El módulo de la covarianza de las fases de dos planos, $|EE_j [zr_j^f overline(zr_j^g)]|$, con las frecuencias de Qwen2.5-7B ordenadas de rápida a lenta, en $Dwiki$ y $Daguja^31000$: la diagonal es $sz$, y las discontinuas separan los planos rápidos, los de la transición y los lentos. (c) Las distribuciones de $kappa_0$ y $kappa_1$ sobre los cabezales no sink de los siete modelos, por longitud media de los prompts, con su mediana (punto) y sus cuartiles (barra).]) <fig:aggregation-exp>

*Agregación en los modelos.* Para un cabezal $h in cal(H)(D)$ se miden los cocientes entre su varianza dependiente de la posición y su interacción y las sumas de las de sus planos,
$ kappa_0 (h, D) := ((Var(p_j) + Var(eij))(D)) / (1/d_h sum_f (Var(p_j^f) + Var(eij^f))(D)), quad kappa_1 (h, D) := (Var(eij)(D)) / (1/d_h sum_f Var(eij^f)(D)), $
que valen uno si los términos cruzados entre planos se anulan (@fig:aggregation-exp c). En la interacción casi se anulan ya en las tareas cortas (mediana de $kappa_1$ de 1.10) y se anulan en las largas (1.00); en la varianza dependiente de la posición, más lentamente: la mediana de $kappa_0$ baja de 2.2 en las tareas cortas a cerca de uno en las largas, con una cola bajo la suma a 31000 tokens. La cola es compatible con los planos de la transición, cuyas frecuencias vecinas difieren en mucho menos que $2 pi \/ (T - 1)$: $delta_(T-1) (theta_f - theta_g) approx 1$ y sus fases siguen correlacionadas aun a 31000 tokens (@fig:aggregation-exp a, b), con un signo que fijan las keys. Entre planos rápidos, la covarianza de las fases cae en más de un orden de magnitud de $Dwiki$ a $Daguja^31000$, y los lentos no aportan. Se concluye que los términos cruzados se anulan al crecer el contexto, como predice la proposición, antes en la interacción que en la varianza dependiente de la posición.

*Interacción de los cabezales.* Para un cabezal $h in cal(H)(D)$ se miden su fracción de interacción, $b_(h,3) (D) = Var(eij)(D) \/ Var(A_(i j))(D)$ (@def:behavior-and-positional-ratio), y su cociente con la suma de las cotas por plano,
$ kappa_2 (h, D) := (Var(eij)(D)) / (1/d_h sum_f (sw sz)(D)), $
que vale $1\/2$ si las interacciones de planos distintos no están correlacionadas y $Re[s_w^f s_zeta^f] = 0$ en todo plano (@prop:the-parts-of-a-plane, ítem 2, y @prop:head-as-a-sum-of-planes, ítem 1). En cada tarea y modelo, la mediana de $b_(h,3) (D)$ está entre 0.017 y 0.083 y a lo sumo 1.2% de los cabezales pasa de 0.2, y la de $kappa_2$ queda cerca de $1\/2$, algo por encima en las tareas cortas (@fig:interaction-exp). Los cabezales son, así, casi aditivos en varianza, lo que no dice cuánto pesa la interacción en la atención (@prop:small-interaction).

#figure(image("/experiments/01_measure/figures/interaction_real.pdf"), caption: [La interacción de los cabezales no sink, una fila por modelo: las densidades sobre los pares $(h, D)$ de las tareas cortas, de los textos y de las tareas largas, con sus medianas, de (a) la fracción de interacción $b_(h,3) (D)$ y (b) $kappa_2$, con las referencias $1\/2$ y 1.]) <fig:interaction-exp>

= Comportamiento posicional y simbólico <sec:behavior>

== Comportamiento y ratio posicional <sec:ratio>

Las tres varianzas de la @prop:two-way-split forman dos sumas, números que no dependen de $i$ ni de $j$ (el índice de $Var$ dice sobre qué variable se toma): la *varianza dependiente de la posición*, $Var(p_j) + Var(eij)$, y la *varianza dependiente del contenido*, $Var(c_i) + Var(eij)$. La interacción entra en ambas, así que suman $Var(A_(i j)) + Var(eij)$.

#definition[comportamiento y ratio posicional][Supongamos que $A$ no es constante. El *comportamiento* del cabezal $h$ es el punto
$ b_h := (Var(c_i), Var(p_j), Var(eij)) / Var(A_(i j)) $
del símplex
$ Delta^3 := {y in [0,1]^3 : y_1 + y_2 + y_3 = 1}. $
Su *ratio posicional* y su *ratio simbólico* son
$ Ph := (Var(p_j) + Var(eij)) / ((Var(p_j) + Var(eij)) + (Var(c_i) + Var(eij))), quad S_h := 1 - Ph. $] <def:behavior-and-positional-ratio>

Los vértices de $Delta^3$ son el cabezal simbólico $(1, 0, 0)$, el posicional $(0, 1, 0)$ y la interacción pura $(0, 0, 1)$ (@prop:two-way-split, ítem 3, y @sec:interaction); la @sec:vertices construye un cabezal en cada uno. En los modelos, los cabezales quedan cerca de la arista sin interacción, repartidos entre los vértices simbólico y posicional, y la nube se mueve con la tarea (@fig:simplex-exp).

#figure(image("/experiments/01_measure/figures/simplex_main.pdf"), placement: none, caption: [El comportamiento $b_h (D)$ de los cabezales no sink de Qwen2.5-7B en $D_"índice"$, $D_"recuperación"$, $Dwiki$ y $Daguja^31000$, coloreados por la profundidad relativa de su capa; punteada, $Ph = 1\/2$.]) <fig:simplex-exp>

Los ratios no cambian si los logits se escalan o se desplazan, así que no dependen de la temperatura del softmax. Suman uno, la interacción pura queda en $Ph = 1\/2$, y en la arista sin interacción $Ph = Var(p_j) \/ (Var(c_i) + Var(p_j))$; la @prop:the-positional-ratio-as-a-relative-distance justifica esta elección.

=== Propiedades métricas <sec:metric>

Los tipos puros son subespacios de $cal(M)_(T-1)(RR)$: las matrices *posicionales* $cal(P)$, cuyas entradas dependen solo de $j$, y las *simbólicas* $cal(S)$, que dependen solo de $i$, ambos de dimensión $T - 1$ y con intersección las constantes, de modo que las *aditivas* $cal(P) + cal(S) = {B : B_(i j) = gamma_1 (i) + gamma_2 (j)}$ tienen dimensión $2T - 3$. Sean $d(A, B) := (EE_(i,j)[(A_(i j) - B_(i j))^2])^(1\/2)$, la distancia de Frobenius dividida por $T - 1$, que es la norma de $A - B$ para el producto interno $EE_(i,j)[A_(i j) B_(i j)]$, y $d(A, cal(C)) := min_(B in cal(C)) d(A, B)$ para $cal(C) in {cal(P), cal(S), cal(P) + cal(S)}$, alcanzado en la proyección ortogonal.

Como el softmax ignora una constante común, la geometría que importa es la del cociente por las constantes: sean $cal(K)$ el subespacio de las matrices constantes, $[A]$ la clase de $A$ en $cal(M)_(T-1)(RR) \/ cal(K)$, $cal(E)$ el subespacio de las matrices cuyas filas y columnas tienen media cero y, para un vector no nulo $v$ y un subespacio $cal(U)$, $cos^2(v, cal(U))$ la norma al cuadrado de la proyección de $v$ sobre $cal(U)$ dividida por la de $v$.

#proposition[ratio posicional y distancias][Para todo cabezal y toda entrada:
+ *Producto interno.* $chevron.l [A], [B] chevron.r := Cov(A_(i j), B_(i j))$ es un producto interno sobre $cal(M)_(T-1)(RR) \/ cal(K)$, con norma $||[A]||^2 = Var(A_(i j))$, y $d(A, B)^2 = ||[A] - [B]||^2 + (EE_(i,j)[A_(i j) - B_(i j)])^2$.
+ *Descomposición ortogonal.* El cociente es la suma directa ortogonal
  $ cal(M)_(T-1)(RR) \/ cal(K) = cal(S) \/ cal(K) plus.o cal(P) \/ cal(K) plus.o [cal(E)], $
  de subespacios de dimensiones $T - 2$, $T - 2$ y $(T - 2)^2$, y las componentes de $[A]$ en ellos son las clases de $(c_i)_(i j)$, $(p_j)_(i j)$ y $(eij)_(i j)$.
+ *Matrices más cercanas.* Las matrices posicional, simbólica y aditiva más cercanas a $A$ son $(mu + p_j)_(i j)$, $(mu + c_i)_(i j)$ y $(mu + c_i + p_j)_(i j)$, a distancias
  $ d(A, cal(P))^2 = Var(c_i) + Var(eij), quad d(A, cal(S))^2 = Var(p_j) + Var(eij), quad d(A, cal(P) + cal(S))^2 = Var(eij), $
  y cada $d(A, cal(C))$ es la distancia de $[A]$ a $cal(C) \/ cal(K)$.
+ *Comportamiento.* Si $A$ no es constante, su comportamiento es el vector de cosenos al cuadrado
  $ b_h = (cos^2([A], cal(S) \/ cal(K)), cos^2([A], cal(P) \/ cal(K)), cos^2([A], [cal(E)])). $
+ *Ratio posicional.* Si $A$ no es constante,
  $ Ph = d(A, cal(S))^2 / (d(A, cal(S))^2 + d(A, cal(P))^2). $
  En particular, $Ph = 1$ si y solo si el cabezal es posicional ($A in cal(P)$), $Ph = 0$ si y solo si es simbólico ($A in cal(S)$), y $Ph > 1\/2$ si y solo si el cabezal está más cerca de las matrices posicionales que de las simbólicas.] <prop:the-positional-ratio-as-a-relative-distance>

#proof[Para el ítem 1, la covarianza no cambia si a $A$ o a $B$ se les suma una constante, así que el producto está bien definido en el cociente; es bilineal y simétrico, y $chevron.l [A], [A] chevron.r = Var(A_(i j))$ se anula si y solo si $A$ es constante, es decir, si y solo si $[A] = 0$. La fórmula de $d$ es la de la media del cuadrado de una variable, $EE[U^2] = Var(U) + EE[U]^2$, con $U = A_(i j) - B_(i j)$ y $Var(U) = ||[A] - [B]||^2$.

Para el ítem 2, los tres subespacios son ortogonales. Toda clase de $cal(S) \/ cal(K)$ tiene un representante $gamma_1 (i)$ de media cero, toda clase de $cal(P) \/ cal(K)$ uno $gamma_2 (j)$ de media cero, y toda clase de $[cal(E)]$ uno $B in cal(E)$, también de media cero; para esos representantes los productos internos son las medias de los productos, que se anulan por la independencia de $i$ y $j$:
$ EE_(i,j)[gamma_1 (i) gamma_2 (j)] &= EE_i [gamma_1 (i)] EE_j [gamma_2 (j)] = 0, \ EE_(i,j)[gamma_1 (i) B_(i j)] &= EE_i [gamma_1 (i) EE_j [B_(i j)]] = 0, \ EE_(i,j)[gamma_2 (j) B_(i j)] &= EE_j [gamma_2 (j) EE_i [B_(i j)]] = 0. $
Generan el cociente: por la @def:content-positional-and-interaction-parts, $A_(i j) = mu + c_i + p_j + eij$ con $mu$ constante, así que $[A]$ es la suma de las clases de $(c_i)_(i j) in cal(S)$, $(p_j)_(i j) in cal(P)$ y $(eij)_(i j)$, que está en $cal(E)$ por la @prop:two-way-split, ítem 1. La suma es entonces directa y esas clases son las componentes de $[A]$. Para las dimensiones, $cal(S)$ y $cal(P)$ tienen dimensión $T - 1$, pues una matriz simbólica queda dada por una columna y una posicional por una fila, y contienen a $cal(K)$, así que sus cocientes tienen dimensión $T - 2$; como el cociente tiene dimensión $(T - 1)^2 - 1$, $[cal(E)]$ tiene dimensión $(T - 1)^2 - 1 - 2 (T - 2) = (T - 2)^2$.

Para el ítem 3, la distancia a cada subespacio se calcula en el cociente. Como $cal(K) subset.eq cal(C)$, sumar a $B in cal(C)$ la constante $EE_(i,j)[A_(i j) - B_(i j)]$ lo deja en $cal(C)$ y en la misma clase y anula el segundo término del ítem 1, así que $d(A, cal(C))$ es la distancia de $[A]$ a $cal(C) \/ cal(K)$, y $B in cal(C)$ la alcanza si y solo si $[B]$ es la proyección ortogonal de $[A]$ sobre $cal(C) \/ cal(K)$ y $EE_(i,j)[B_(i j)] = mu$. Por el ítem 2, esa proyección es la clase de $(p_j)_(i j)$ para $cal(P)$, la de $(c_i)_(i j)$ para $cal(S)$ y la de $(c_i + p_j)_(i j)$ para $cal(P) + cal(S)$, cuyo cociente es $cal(S) \/ cal(K) plus.o cal(P) \/ cal(K)$; como $c$ y $p$ tienen media cero, la única matriz de esa clase con media $mu$ es la del enunciado. La diferencia entre $[A]$ y su proyección es la suma de las componentes restantes, y por la ortogonalidad y el ítem 1 su norma al cuadrado es la suma de sus varianzas; por ejemplo, para $cal(P)$,
$ d(A, cal(P))^2 = ||[(c_i)_(i j)] + [(eij)_(i j)]||^2 = Var(c_i) + Var(eij). $

Para el ítem 4, la proyección de $[A]$ sobre cada sumando del ítem 2 es su componente, cuya norma al cuadrado es la varianza de la parte, y $||[A]||^2 = Var(A_(i j))$ es positiva si $A$ no es constante; los cocientes son los de la @def:behavior-and-positional-ratio.

Para el ítem 5, por el ítem 3, el numerador de $Ph$ en la @def:behavior-and-positional-ratio es $d(A, cal(S))^2$ y su denominador es $d(A, cal(S))^2 + d(A, cal(P))^2$, que es al menos $Var(c_i) + Var(p_j) + Var(eij) = ||[A]||^2$, positivo si $A$ no es constante. Como $d(A, cal(C)) = 0$ si y solo si $A in cal(C)$, $Ph = 1$ si y solo si $A in cal(P)$ y $Ph = 0$ si y solo si $A in cal(S)$; y $Ph > 1\/2$ si y solo si $d(A, cal(S))^2 > d(A, cal(P))^2$, es decir, si y solo si $d(A, cal(P)) < d(A, cal(S))$.]

Así, $Ph = 1 \/ (1 + d(A, cal(P))^2 \/ d(A, cal(S))^2)$ si $A in.not cal(S)$ (@fig:geometry), y el comportamiento y los ratios dependen solo de la dirección de $[A]$. La varianza dependiente de la posición, en cambio, es el cambio cuadrático medio de los logits contrafactuales al reemplazar $A$ por la matriz simbólica más cercana (la @sec:removing-rotation la convierte en el costo de quitar la rotación): mide cuánto usa el cabezal la posición, y $Ph$, cuán cerca está de cada tipo puro, así que pueden ordenar los cabezales de forma distinta. $Ph$ no mide la interacción, que se reporta aparte, y es una coordenada continua que depende de la entrada (@sec:stability) y de su longitud (@cor:slow-planes-and-long-contexts).

#figure(image("/docs/figures/distances.pdf"), placement: bottom, caption: [La geometría del ratio posicional (salvo constantes): las tres partes son direcciones ortogonales, y las matrices posicional, simbólica y aditiva más cercanas a $A$ son sus proyecciones, a distancias al cuadrado $Var(c_i) + Var(eij)$, $Var(p_j) + Var(eij)$ y $Var(eij)$.]) <fig:geometry>

En la esfera unitaria del cociente, donde $[A] \/ ||[A]||$ tiene proyecciones sobre los tres subespacios de normas $(b_(h,1), b_(h,2), b_(h,3))^(1\/2)$, los cabezales aditivos están sobre el arco $b_(h,3) = 0$, la interacción los aleja de él en el ángulo $arcsin(b_(h,3)^(1\/2))$ y $Ph$ los ordena a lo largo de él. En los modelos forman una banda pegada al arco que al crecer el contexto se corre hacia la posición en los siete modelos y, en los Qwen2.5, se aleja un poco de él: en Qwen2.5-7B, de $Dwiki$ a $Daguja^31000$, la mediana de $b_(h,3)$ pasa de 0.051 a 0.075 y la fracción de cabezales con $Ph > 0.8$, de 6% a 14% (@fig:simplex-exp).

=== Estabilidad y efecto causal <sec:stability>

*Estabilidad del ratio posicional.* Sobre los cabezales $cal(H)^* := inter.big_(D in cal(D)) cal(H)(D)$ que no son sink en ninguna tarea se miden la *confiabilidad* de una tarea, la $rhoS$ media entre los $Ph$ de dos mitades de sus prompts sorteadas al azar; el *acuerdo entre tareas*, la $rhoS$ media entre los $Ph$ de dos tareas distintas; la *fracción entre cabezales* $eta in [0, 1]$, la fracción de la suma de cuadrados de los $Ph (D)$ centrados, sobre $cal(H)^* times cal(D)_"c"$, que explican las medias de cada cabezal; y la *dispersión*, la mediana sobre los cabezales de la desviación estándar de su $Ph (D)$ en las tareas cortas. La confiabilidad es al menos 0.988, y el acuerdo y $eta$, al menos 0.81 (@tab:stability): el orden de los cabezales casi no depende de los prompts ni de la tarea. Pero con una dispersión de 0.08–0.09, menos de la mitad de los cabezales quedan en una misma clase, simbólica ($Ph (D) < 0.2$), posicional ($Ph (D) > 0.8$) o mixta, en las nueve tareas cortas (@fig:stability-exp): $Ph$ es una propiedad del cabezal desplazada por la tarea, que sirve para ordenar cabezales y no para clasificarlos con un umbral.

#figure(table(columns: 5, align: (left, center, center, center, center), table.hline(), [*modelo*], [*acuerdo entre tareas*], [$eta$], [*dispersión*], [*misma clase*], table.hline(stroke: 0.5pt),
[Qwen2.5-0.5B], [0.87], [0.90], [0.09], [41%],
  [Qwen2.5-1.5B], [0.87], [0.85], [0.09], [43%],
  [Qwen2.5-3B], [0.88], [0.88], [0.08], [45%],
  [Qwen2.5-7B], [0.87], [0.87], [0.08], [45%],
  [Llama-3.2-1B], [0.90], [0.87], [0.08], [41%],
  [Llama-3.2-3B], [0.87], [0.81], [0.09], [42%],
  [Llama-3.1-8B], [0.85], [0.84], [0.08], [41%],
  table.hline()), placement: none, caption: [La estabilidad de $Ph$ sobre $cal(H)^*$; misma clase: la fracción de cabezales en la misma clase en las nueve tareas cortas.]) <tab:stability>

#figure(image("/experiments/01_measure/figures/stability.pdf"), caption: [La estabilidad de $Ph$ sobre $cal(H)^*$. (a) La confiabilidad de cada tarea, un punto por modelo. (b) $rhoS (P(D), P(D'))$ para cada par de tareas, en Qwen2.5-7B, con $P(D) := (Ph (D))_(h in cal(H)^*)$. (c) Cada cabezal de Qwen2.5-7B en las nueve tareas cortas: el rango de sus $Ph (D)$ (gris) y su media (negro), en el orden de su media, con los umbrales 0.2 y 0.8.]) <fig:stability-exp>

*Edición de las partes.* La prueba causal edita los logits de un grupo $G$ de cabezales en cinco modelos (todos salvo Qwen2.5-0.5B y Llama-3.2-1B): para $h in G$ y cada query $r$, con $A$ la matriz contrafactual de $h$ en $r$ y $K subset.eq {c, p, epsilon}$ las partes que se conservan, el logit de la key en la posición $s in [r - 1]$ se reemplaza por
$ tilde(lambda)_s := mu + bb(1)_(c in K) c_s + bb(1)_(p in K) p_s + bb(1)_(epsilon in K) epsilon_(s s), $
sin cambiar el primer token, la query ni los demás cabezales. *Quitar la posición* ($K = {c}$, la matriz simbólica más cercana) se aplica al 25% de los cabezales no sink de mayor $Ph$, $G_"pos"$; *quitar el contenido* ($K = {p, epsilon}$), al 10% de menor $Ph$, $G_"sim"$; y *quitar la interacción* ($K = {c, p}$), al 25% de mayor $b_(h,3)$, $G_"int"$, todos medidos en ventanas de 4096 tokens de WikiText-2. Cada edición se aplica también a cabezales al azar y, las dos primeras, al extremo opuesto del mismo tamaño y a un *control*: entre 4000 grupos al azar del mismo tamaño, que pueden compartir cabezales con él, aquel cuya varianza quitada ($sum_h (Var(p_j) + Var(eij))$ o $sum_h Var(c_i)$ sobre sus cabezales) es la más cercana a la del grupo.

*Efecto.* En las tareas que el modelo intacto resuelve con una exactitud de al menos 0.9 ($D_"recuperación"$, $D_"asociación"$, $Daguja^4000$ y $D_"clave–valor"^4000$), la *caída del margen* es la media sobre los prompts de las tareas cortas (o largas) de $-(m'(x) - m(x)) \/ sigma_(D(x))$, con $m(x)$ el margen de un prompt $x$ (@sec:tasks), $m'(x)$ el mismo con la edición y $sigma_(D(x))$ la desviación estándar de los márgenes intactos de su tarea; frente al control se toma la diferencia de las caídas, pareada por prompt, con un intervalo BCa al 95% @efron1987better. En texto, el efecto es el cambio de la pérdida media por token en otras ventanas de WikiText-2, y frente al control, el cociente de los cambios.

*Resultados* (@fig:dissociation-exp). Quitar la posición daña donde $Ph$ es mayor, y quitar el contenido, donde es menor: en las tareas cortas, el margen cae al menos 3.5 desviaciones estándar al quitar la posición de $G_"pos"$ y 1.4 al quitar el contenido de $G_"sim"$, y a lo sumo 0.63 y 0.33 en los extremos opuestos. Ambos grupos cuestan más que cabezales al azar en las tres medidas (en la media) y que su control en ambas clases de tareas, con todo intervalo por encima de cero, en los cinco modelos, salvo al quitar el contenido en Qwen2.5-7B y, frente al control, al quitar la posición en las tareas largas de Qwen2.5-1.5B; en texto, quitar la posición también cuesta más que su control en los cinco. En las tareas largas, quitar la posición de los cabezales de menor $Ph$ cuesta más que en las cortas en cuatro modelos: en contextos largos, también los cabezales simbólicos llevan posición que el modelo usa. Quitar la interacción de $G_"int"$ cambia la exactitud en las tareas cortas en menos de 0.02, aunque en tres modelos su caída del margen supera, con intervalo por encima de cero, a la de cabezales al azar.

*Alcance del control.* Salvo en Qwen2.5-1.5B, el control quita menos varianza que el grupo (0.59–0.92 de la suya). Dividir sus efectos por esa fracción, la *referencia proporcional* (rombos), solo equivale a igualar la varianza si el costo crece linealmente con ella; frente a esa referencia, quitar la posición sigue costando más en texto en los cinco modelos pero en el margen solo en tres, y quitar el contenido, en el margen en cuatro y en texto en tres.

Se concluye que la parte que nombra $Ph$ es la que el cabezal usa, con una disociación doble entre posición y contenido. Que el daño venga de qué cabezales se editan y no solo de cuánta varianza se quita lo sostienen sin excepción Qwen2.5-1.5B, cuyo control es exacto, en las tareas cortas, y el texto al quitar la posición; en los demás casos, solo la referencia proporcional, y no en todos. La interacción, pequeña en varianza en casi todos los cabezales, puede quitarse con poco efecto en estas tareas.

#figure(image("/experiments/03_causal/figures/causal_overview.pdf"), caption: [Las ediciones en los cinco modelos, coloreadas por la parte que quitan; hacia arriba o a la derecha, más costo. (a–c) La caída del margen en las tareas cortas y largas y el cambio de la pérdida en texto en los grupos extremos de cada edición: finas, cada modelo; gruesas, la media (geométrica en texto). (d–f) Cada grupo frente a su control (cabezales al azar al quitar la interacción), con el intervalo BCa al 95% en (d, e) (hueco si incluye el cero); rombos, frente a la referencia proporcional.]) <fig:dissociation-exp>

*Cabezales lectores.* Los *lectores* $cal(L)(D) subset cal(H)(D)$ de una tarea son el 5% de los cabezales con más atención sobre la posición de la respuesta. En las tareas que el modelo resuelve (exactitud de al menos 0.9) y en que sus lectores atienden a la respuesta al menos el doble que al distractor, casi todas de respuesta fijada por un contenido o largas y ninguna de respuesta fijada por una posición, la media de $Ph (D)$ sobre $cal(H)(D)$ supera a la de los lectores en 0.27–0.40 según el modelo (@tab:roles): los cabezales que leen la respuesta son más simbólicos que el cabezal medio.

#figure(table(columns: 7, align: (left, center, center, center, center, center, center), table.hline(), table.cell(rowspan: 2, align: horizon)[*modelo*], table.cell(rowspan: 2, align: horizon)[*lectores*], table.cell(rowspan: 2, align: horizon)[*grupos de key (azar)*], table.cell(colspan: 2)[*fracción por tercio, primero / último*], table.cell(rowspan: 2, align: horizon)[*token anterior*], table.cell(rowspan: 2, align: horizon)[*inducción*], [*posicionales*], [*simbólicos*], table.hline(stroke: 0.5pt),
  [Qwen2.5-0.5B], [0.31], [0.21 (0.08)], [0.22 / 0.00], [0.05 / 0.50], [0.92 / 0.86], [0.014 / 0.041],
  [Qwen2.5-1.5B], [0.30], [0.34 (0.10)], [0.14 / 0.01], [0.09 / 0.51], [0.95 / 0.90], [0.020 / 0.063],
  [Qwen2.5-3B], [0.30], [0.30 (0.07)], [0.14 / 0.01], [0.11 / 0.37], [0.93 / 0.84], [0.014 / 0.058],
  [Qwen2.5-7B], [0.27], [0.44 (0.12)], [0.13 / 0.02], [0.12 / 0.43], [0.94 / 0.87], [0.028 / 0.071],
  [Llama-3.2-1B], [0.40], [0.54 (0.27)], [0.18 / 0.06], [0.09 / 0.34], [0.93 / 0.83], [0.018 / 0.046],
  [Llama-3.2-3B], [0.32], [0.68 (0.37)], [0.07 / 0.05], [0.12 / 0.32], [0.90 / 0.78], [0.015 / 0.066],
  [Llama-3.1-8B], [0.30], [0.61 (0.29)], [0.04 / 0.02], [0.14 / 0.28], [0.93 / 0.86], [0.023 / 0.078],
  table.hline()), placement: top, caption: [Los cabezales por su función. Lectores: la media de $Ph (D)$ sobre $cal(H)(D)$ menos la de $cal(L)(D)$, en promedio sobre las tareas que cuentan. Grupos de key: la fracción de la variación de $Ph (D)$ en cada capa que explican sus grupos de key, en media sobre las tareas cortas (entre paréntesis, con grupos al azar). Fracción de posicionales y simbólicos de $cal(H)(Dwiki)$ por tercio de las capas. Token anterior e inducción: el $Ph$ mediano, en los tokens repetidos / en $Dwiki$.]) <tab:roles>

*Profundidad y grupos de key.* En $Dwiki$, los cabezales posicionales ($Ph > 0.8$) son más frecuentes en el primer tercio de las capas y los simbólicos ($Ph < 0.2$) en los dos últimos, en los siete modelos (@tab:roles). Los cabezales de query que comparten keys (@sec:models) se parecen: en todo modelo, sus grupos de key explican más de la variación de $Ph (D)$ dentro de cada capa que grupos al azar del mismo tamaño.

=== Vértices del símplex <sec:vertices>

Los tres vértices de $Delta^3$ se alcanzan con cabezales de un solo plano, cada uno de los cuales resuelve la tarea que su parte fija (@fig:single-plane-heads). Los casos posicional y simbólico reformulan construcciones de #cite(<urrutia2026decoupling>, form: "prose") (sus Teoremas 3 y 5); el de interacción pura es nuevo. Aquí la frecuencia $theta$ es libre, y $theta = 0$ es un plano congelado.

#proposition[cabezales de un solo plano][Sean $T >= 4$, un cabezal cuya query solo es no nula en el plano $f$ ($q^g = 0$ para $g != f$), y $r > 0$. Escribimos $theta := theta_f$, $q := q^f$ y $k(x) := k^f (x)$.
+ *Simbólica: recuperación.* Sea $theta = 0$. Se asigna a cada contenido $x$ un ángulo $phi.alt(x) in [0, 2 pi)$ y se toma $k(x) := e^(-iu phi.alt(x))$ y $q := r e^(-iu phi.alt(x_*))$ para un contenido consultado $x_*$. Entonces
  $ A_(i j) = d_h^(-1\/2) r cos(phi.alt(x_i) - phi.alt(x_*)) $
  tiene filas constantes; el comportamiento es $(1, 0, 0)$ siempre que estos valores no sean todos iguales, y los logits alcanzan su mayor valor posible, $d_h^(-1\/2) r$, exactamente en los contenidos con $phi.alt(x_i) = phi.alt(x_*)$, dondequiera que estén.
+ *Posicional: $n$ lugares atrás.* Sean $theta in (0, 2 pi \/ (T - 2))$, $n in [T-1]$, $k(x) := 1$ para todo contenido y $q := r e^(-iu n theta)$. Entonces
  $ A_(i j) = d_h^(-1\/2) r cos((T - j - n) theta) $
  tiene columnas constantes, el comportamiento es $(0, 1, 0)$, y el mayor logit va a la posición de desfase $n$, y solo a ella.
+ *Interacción pura: el punto fijo.* Sea $theta = 2 pi \/ (T - 1)$. Se asigna a cada contenido una etiqueta $ell(x) in [T-1]$ de modo que las etiquetas del contexto son una permutación de $[T-1]$, y se toma $k(x) := e^(-iu ell(x) theta)$ y $q := r e^(-iu theta)$. Entonces
  $ A_(i j) = d_h^(-1\/2) r cos((ell(x_i) - j) theta), $
  $c = p = 0$ y el comportamiento es $(0, 0, 1)$; los logits observados alcanzan su mayor valor posible, $d_h^(-1\/2) r$, exactamente en las posiciones $t$ con $ell(x_t) = t$. Con $ell(x_i) = i$ es, salvo escala, la matriz coseno de la @sec:interaction (@fig:three).] <prop:three-pure-heads>

#proof[Como solo el plano $f$ tiene query no nula, la @eq:logit y la @def:counterfactual-logit-matrix dan
$ A_(i j) = d_h^(-1\/2) Re[q overline(k(x_i)) e^(iu (T - j) theta)], $
y cuando $j$ recorre $[T-1]$, el desfase $T - j$ recorre también $[T-1]$.

Para la recuperación, con $theta = 0$ la fase vale uno en toda posición, y como $overline(e^(-iu phi.alt(x_i))) = e^(iu phi.alt(x_i))$,
$ q overline(k(x_i)) = r e^(-iu phi.alt(x_*)) e^(iu phi.alt(x_i)) = r e^(iu (phi.alt(x_i) - phi.alt(x_*))). $
La parte real da la fórmula del enunciado, que no depende de $j$: las filas son constantes, y por la @prop:two-way-split, ítem 3, $p = epsilon = 0$. Entonces $Var(A_(i j)) = Var(c_i)$, positiva si los valores no son todos iguales, y el comportamiento es $(1, 0, 0)$. Como $cos <= 1$, con igualdad si y solo si el ángulo es un múltiplo de $2 pi$, y ambos ángulos están en $[0, 2 pi)$, el mayor logit lo reciben exactamente los contenidos con $phi.alt(x_i) = phi.alt(x_*)$, en cualquier posición $j$.

Para el desfase, con $k(x) = 1$ se tiene $q overline(k(x_i)) = q = r e^(-iu n theta)$ para todo $i$, y
$ A_(i j) = d_h^(-1\/2) r Re[e^(iu (T - j - n) theta)] = d_h^(-1\/2) r cos((T - j - n) theta), $
que no depende de $i$: las columnas son constantes y, por la @prop:two-way-split, ítem 3, $c = epsilon = 0$. Si $T - j = n$, el coseno vale uno; si no, $1 <= |T - j - n| <= T - 2$, así que $0 < |T - j - n| theta <= (T - 2) theta < 2 pi$ y el coseno es menor que uno. El mayor logit va a la única posición con $T - j = n$, y como $T >= 4$ hay otras posiciones, de modo que $A$ no es constante y el comportamiento es $(0, 1, 0)$.

Para el punto fijo, con $k(x) = e^(-iu ell(x) theta)$, y como $(T - 1) theta = 2 pi$,
$ q overline(k(x_i)) = r e^(iu (ell(x_i) - 1) theta), quad e^(iu (T - j) theta) = e^(iu (T - 1) theta) e^(iu (1 - j) theta) = e^(iu (1 - j) theta), $
y su producto es $r e^(iu (ell(x_i) - j) theta)$, cuya parte real da la fórmula del enunciado. Las filas y las columnas tienen media cero, porque los $e^(-iu j theta)$, $j in [T-1]$, son las $T - 1$ raíces $(T - 1)$-ésimas de la unidad, de suma nula, y las etiquetas $ell(x_i)$, $i in [T-1]$, son una permutación de $[T-1]$:
$ sum_(j=1)^(T-1) e^(iu (ell(x_i) - j) theta) = e^(iu ell(x_i) theta) sum_(j=1)^(T-1) e^(-iu j theta) = 0, quad sum_(i=1)^(T-1) e^(iu (ell(x_i) - j) theta) = e^(-iu j theta) sum_(l=1)^(T-1) e^(iu l theta) = 0. $
Tomando parte real, $EE_j [A_(i j)] = 0$ para todo $i$ y $EE_i [A_(i j)] = 0$ para todo $j$, así que, por la @def:content-positional-and-interaction-parts, $mu = 0$, $c = p = 0$ y $epsilon = A$. La matriz no es nula, pues su entrada en la fila $i$ y la columna $j = ell(x_i)$ vale $d_h^(-1\/2) r$; como su media es cero, no es constante, y el comportamiento es $(0, 0, 1)$.

En la diagonal, por último, $lambda_t = A_(t t) = d_h^(-1\/2) r cos((ell(x_t) - t) theta)$ vale a lo sumo $d_h^(-1\/2) r$, con igualdad si y solo si $(ell(x_t) - t) theta = 2 pi (ell(x_t) - t) \/ (T - 1)$ es un múltiplo de $2 pi$, es decir, si y solo si $T - 1$ divide a $ell(x_t) - t$. Como $ell(x_t)$ y $t$ están en $[T-1]$, $|ell(x_t) - t| <= T - 2$, y la única posibilidad es $ell(x_t) = t$.]

#figure(image("/docs/figures/single_plane_heads.pdf"), caption: [Los cabezales de la @prop:three-pure-heads con $T = 11$: (a) simbólico, plano congelado; (b) posicional, tres lugares atrás, $theta = pi \/ 6$; (c) interacción pura, $theta = 2 pi \/ (T - 1)$. Arriba, la query (flecha) y las keys rotadas, rotuladas por su posición (anillo, el mayor logit); al centro, la matriz contrafactual; abajo, los logits observados.]) <fig:single-plane-heads>

*Cabezales en los vértices.* Tres tareas sintéticas de 30 tokens de contexto fijan la respuesta por una parte cada una: la *posicional*, el símbolo $n$ posiciones antes de la query, con $n$ dado por la query; la *simbólica*, el valor de una clave entre 30 pares (clave, valor), cada par un solo token; la *de interacción*, el único número de una lista igual a su posición. Los modelos son una capa de atención con RoPE (16 planos), sin normalización ni MLP, entrenada con Adam @kingma2015adam, y resuelven su tarea si su exactitud, la fracción de ejemplos en que la respuesta tiene el mayor logit de salida, es al menos 0.9 (@fig:toy-exp). Los cabezales *construidos* como en la @prop:three-pure-heads, repartidos sobre los 16 planos porque ninguno es congelado ni tiene período 30, la resuelven cerca de su vértice. Un cabezal *entrenado solo* resuelve la tarea posicional en las cinco semillas, la de interacción en una y la simbólica en ninguna, aunque la solución existe. En modelos de tres cabezales entrenados en la mezcla de las tareas, el *lector* de cada tarea, el cabezal con más atención sobre la respuesta, queda cerca de su vértice en la posicional y la simbólica, pero en la de interacción, en tres de las cinco semillas, cerca de 40% de su varianza es contenido; en las tres tareas, otros cabezales también atienden a la respuesta: estos modelos no reparten las tareas de forma nítida. Si los logits de todos los cabezales conservan solo algunas partes, la tarea posicional se resuelve con solo $p$ y la simbólica con solo $c$, y la de interacción cae casi al azar sin $epsilon$, aunque con solo $epsilon$ recupera poco más de la mitad de su exactitud, como predice la factorización de la atención de un cabezal sin interacción (@eq:attention-factors). Los tres vértices son así alcanzables, y cada uno corresponde a una forma de resolver una tarea.

#figure(image("/experiments/02_reading/figures/toy_overview.pdf"), caption: [Las tres tareas sintéticas. Izquierda: el comportamiento de los cabezales construidos (estrellas), de los entrenados solos que resuelven su tarea (rellenos) o no (huecos) y de los lectores de los modelos de tres cabezales (cuadrados). Derecha: la exactitud de los modelos de tres cabezales, en media sobre las semillas, cuando los logits conservan solo las partes marcadas (la primera columna es el modelo intacto), con el azar bajo cada tarea; enmarcadas, las ediciones que conservan la parte de la tarea.]) <fig:toy-exp>

*Cabezales de token anterior y de inducción.* En secuencias de 64 tokens al azar repetidas una vez, con las varianzas promediadas sobre las secuencias y sobre ocho queries de la segunda copia, un cabezal es *de token anterior* si en la segunda copia pone en promedio más de 0.5 de su atención en el token previo, y *de inducción* si no lo es y pone más de 0.3 en el token que siguió a la aparición anterior del token actual. En los siete modelos, el $Ph$ mediano de los cabezales no sink de token anterior es al menos 0.89, y el de los de inducción, a lo sumo 0.03, algo menos extremos en $Dwiki$ (@tab:roles), aunque Llama-3.2-3B y Llama-3.1-8B tienen solo dos cabezales de token anterior cada uno: el circuito de inducción copia el token previo con un cabezal casi posicional y busca la aparición anterior del token con uno casi simbólico.

== Ratio posicional en contextos largos <sec:long-contexts>

Con los estadísticos de contenido fijos, el ratio de un plano en contextos largos depende solo de cómo se reparten sus keys entre lo común y lo propio.

#corollary[ratio posicional en contextos largos][Sea $f$ un plano de frecuencia $0 < theta_f < pi$ con $sw > 0$, y sea $P^f$ el ratio posicional de su matriz $A^f$ (@def:behavior-and-positional-ratio). Si $|wt^f|^2$ y $sw$ se mantienen fijos, cuando $T -> infinity$
$ P^f -> (|wt^f|^2 + sw) / (|wt^f|^2 + 2 sw) in [1\/2, 1). $
El límite vale $1\/2$ si las keys no comparten ninguna dirección ($wt^f = 0$) y se acerca a uno cuanto más domina su centro de masa sobre su dispersión.] <cor:ratio-in-long-contexts>

#proof[Por el @cor:slow-planes-and-long-contexts, ítem 2, la varianza dependiente de la posición del plano dividida por su presupuesto tiende a $1\/2$, y como el presupuesto $budget = |wt^f|^2 + sw$ está fijo, la varianza tiende a $1/2 (|wt^f|^2 + sw)$. La varianza dependiente del contenido sale de sumar las formas cerradas de $Var(c_i^f)$ y $Var(eij^f)$ de la @prop:the-parts-of-a-plane, ítem 2:
$ Var(c_i^f) + Var(eij^f) = 1/2 sw (|zt^f|^2 + sz) + 1/2 Re[s_w^f ((zt^f)^2 + s_zeta^f)] = 1/2 sw + 1/2 Re[s_w^f EE_j [(zeta_j^f)^2]], $
porque $|zt^f|^2 + sz = EE_j [ |zeta_j^f|^2 ] = 1$ y $(zt^f)^2 + s_zeta^f = EE_j [(zeta_j^f)^2]$. Por la @prop:phase-statistics, $|EE_j [(zeta_j^f)^2]| = |delta_(T-1) (2 theta_f)|$, que tiende a cero, y como $|s_w^f| <= sw$ (@prop:pseudo-variance-and-pca), el segundo término está acotado por $1/2 sw |delta_(T-1) (2 theta_f)|$, así que esta varianza tiende a $1/2 sw$. El cociente de la @def:behavior-and-positional-ratio tiende entonces a $(|wt^f|^2 + sw) \/ (|wt^f|^2 + 2 sw)$, que es al menos $1\/2$ porque $|wt^f|^2 >= 0$, menor que uno porque $sw > 0$, e igual a $1\/2$ si y solo si $wt^f = 0$.]

Un plano lento tiene un ratio cercano a cero, salvo que su varianza dependiente del contenido también sea pequeña (@cor:slow-planes-and-long-contexts, ítem 1). Al crecer el contexto, cada plano que se vuelve rápido gana varianza dependiente de la posición, que en contextos largos se suma sobre los planos (@prop:head-as-a-sum-of-planes), y pierde su contenido, pues $c_i^f = Re[wr_i^f zt^f]$ y $|zt^f| -> 0$: así crece el ratio de un cabezal con la longitud. Cuando todos los planos se vuelven rápidos, tiene un límite.

#corollary[ratio posicional de un cabezal en contextos largos][Supongamos que las frecuencias de los planos son distintas y que, cuando $T -> infinity$, $|wt^f|^2$ y $sw$ convergen en todo plano $f$, con $sum_f lim_(T -> infinity) (|wt^f|^2 + sw) > 0$. Entonces $Ph$ está definido para $T$ grande y
$ Ph -> (sum_f lim_(T -> infinity) (|wt^f|^2 + sw)) / (sum_f lim_(T -> infinity) (|wt^f|^2 + 2 sw)) in [1\/2, 1]. $
El límite vale $1$ si y solo si $sw -> 0$ en todo plano, y $1\/2$ si y solo si $wt^f -> 0$ en todo plano.] <cor:head-ratio-in-long-contexts>

#proof[Para la varianza dependiente de la posición, las hipótesis incluyen las de la @prop:head-as-a-sum-of-planes, ítem 2, porque una sucesión convergente es acotada, así que esa varianza menos $d_h^(-1) sum_f (Var(p_j^f) + Var(eij^f))$ tiende a cero. Por el @cor:slow-planes-and-long-contexts, el sumando de cada plano está entre $1/2 (sz - |s_zeta^f|) budget$ y $1/2 (sz + |s_zeta^f|) budget$ (si $budget = 0$ es nulo, por el @cor:where-each-part-can-live), y como en la demostración de su ítem 2, $sz -> 1$ y $s_zeta^f -> 0$, mientras que $budget = |wt^f|^2 + sw$ converge. Por lo tanto
$ Var(p_j) + Var(eij) -> 1/(2 d_h) sum_f lim_(T -> infinity) (|wt^f|^2 + sw). $

Para la varianza dependiente del contenido, la @prop:head-as-a-sum-of-planes, ítem 2, da $Var(eij) - (2 d_h)^(-1) sum_f sw sz -> 0$, y como $sz -> 1$, $Var(eij) -> (2 d_h)^(-1) sum_f lim_(T -> infinity) sw$. La parte de contenido tiende a cero: $c_i = d_h^(-1\/2) sum_f Re[wr_i^f zt^f]$ (@prop:the-parts-of-a-plane, ítem 1, y la linealidad de las partes, @prop:head-as-a-sum-of-planes), tiene media cero, y como $(Re z)^2 <= |z|^2$ y el cuadrado de una suma de $F$ términos es a lo sumo $F$ veces la suma de sus cuadrados, con $F \/ d_h = 1\/2$ y $|zt^f| = |delta_(T-1) (theta_f)|$ (@prop:phase-statistics),
$ Var(c_i) = EE_i [c_i^2] <= F/d_h sum_f EE_i [ |wr_i^f|^2 ] |zt^f|^2 = 1/2 sum_f sw delta_(T-1) (theta_f)^2 -> 0, $
pues $sw$ está acotada y $delta_(T-1) (theta_f) -> 0$ (@prop:head-as-a-sum-of-planes, demostración del ítem 2).

Así, el numerador de $Ph$ tiende a $(2 d_h)^(-1) sum_f lim (|wt^f|^2 + sw) > 0$, de modo que para $T$ grande es positivo, $A$ no es constante y $Ph$ está definido, y el denominador tiende a $(2 d_h)^(-1) sum_f lim (|wt^f|^2 + 2 sw)$, lo que da el límite. Es al menos $1\/2$ porque $sum_f lim |wt^f|^2 >= 0$, con igualdad si y solo si esa suma es nula, y a lo sumo $1$ porque $sum_f lim sw >= 0$, con igualdad si y solo si esa suma es nula.]

Con keys de estadísticos convergentes, en contextos suficientemente largos ningún cabezal se mantiene bajo un valor menor que $1\/2$: parte del corrimiento hacia la posición al crecer el contexto es geométrica, de las fases, y no un cambio de las keys. El límite exige que todo plano sea rápido, y a 31000 tokens los planos lentos aún no lo son (@fig:dispersion).

*Ratio de los planos en contextos largos.* En cada tarea larga, el ratio posicional $P^f$ de cada plano rápido ($sz > 0.9$), con sus varianzas promediadas sobre los prompts y los cabezales no sink, se compara con el límite del @cor:ratio-in-long-contexts evaluado con $|wt^f|^2$ y $sw$ promediados del mismo modo. A 31000 tokens la diferencia absoluta tiene mediana $1.1 dot 10^(-5)$, y baja con las vueltas del plano (@fig:long-exp a): el ratio de los planos rápidos ya está en su límite, fijado solo por el reparto de sus keys entre lo común y lo propio.

*Longitud del contexto.* En cada tarea larga $D^L$, el aumento del ratio de un cabezal no sink a las tres longitudes es $Delta_h := Ph (D^31000) - Ph (D^4000)$, y su predicción $hat(Delta)_h$ recalcula solo las fases: fija los estadísticos de contenido de cada plano a 4000 tokens, calcula $sz$ a cada longitud (@prop:phase-statistics) y suma sobre los planos las varianzas de la @prop:the-parts-of-a-plane sin los términos de pseudovarianza ni los cruzados entre planos. La media de $Delta_h$ es positiva en las cinco tareas largas de los siete modelos, como sugieren el @cor:ratio-in-long-contexts y el @cor:head-ratio-in-long-contexts, y la de $hat(Delta)_h$ acierta su signo y su orden de magnitud pero no su tamaño, que sobrestima en las tareas de la aguja y en general subestima en el seguimiento de variables (@fig:long-exp b). Entre cabezales, $rhoS (Delta, hat(Delta))$ es 0.34–0.65, y la predicción yerra el ratio mismo a 4000 tokens en 0.04–0.13, del mismo orden que el aumento. La sola varianza de las fases da así el signo y el orden de magnitud del crecimiento y, en parte, qué cabezales crecen más, pero no su tamaño; cuánto aporta el cambio de las keys con la longitud no se midió.

#figure(image("/experiments/01_measure/figures/long_contexts.pdf"), caption: [Los contextos largos en los siete modelos y las cinco tareas largas. (a) La diferencia absoluta entre el ratio $P^f$ de cada plano y su límite del @cor:ratio-in-long-contexts, en función de las vueltas del plano, con su mediana por intervalos a cada longitud. (b) La media de $Delta_h$ frente a la de $hat(Delta)_h$, un punto por modelo y tarea, con la recta $y = x$.]) <fig:long-exp>

= Eliminación de la rotación <sec:removing-rotation>

Quitar la rotación de algunos planos (partial RoPE), de un cabezal o de una capa entera deja keys que no dependen de su posición, y la descomposición da en forma cerrada lo que cuesta. Cuando la decisión concierne a un cabezal KV (@sec:models), los puntajes de sus cabezales de query se agregan como se indica en cada caso; los cabezales sink quedan fuera solo de las reglas que usan el ratio posicional. Para una permutación $tau$ de $[T-1]$, la entrada *reordenada* por $tau$ ubica el contenido $tau(t)$ en la posición $t$ con las keys sin rotar fijas: su matriz contrafactual es $A$ con la fila $t$ igual a la fila $tau(t)$, y sus logits observados son $A_(tau(t), t)$.

#proposition[costo de eliminar la rotación][Sea $R subset.eq [F]$ un conjunto de planos que pierden su rotación. Cada plano de $R$ da entonces a todo contenido el logit que daría sin posición, $Re[w_i^f]$, y los logits contrafactuales pasan a ser
$ A^R_(i j) := A_(i j) - 1/sqrt(d_h) sum_(f in R) (A^f_(i j) - Re[w_i^f]). $
+ *Descomposición del cambio.* El cambio es $A_(i j) - A^R_(i j) = Pi_(i j) + chi_i$, con
  $ Pi_(i j) := 1/sqrt(d_h) sum_(f in R) (p_j^f + eij^f), quad chi_i := 1/sqrt(d_h) sum_(f in R) Re[w_i^f (zt^f - 1)], $
  y sus dos partes son ortogonales:
  $ EE_(i,j) [(A_(i j) - A^R_(i j))^2] = EE_(i,j) [Pi_(i j)^2] + EE_i [chi_i^2], quad EE_(i,j) [Pi_(i j)^2] = 1/d_h sum_(f, g in R) (Cov(p_j^f, p_j^g) + Cov(eij^f, eij^g)). $
+ *Un solo plano.* Si $R = {f}$, el primer término es $d_h^(-1) (Var(p_j^f) + Var(eij^f)) = d_h^(-1) d(A^f, cal(S))^2$: $d_h^(-1)$ veces la varianza dependiente de la posición del plano, la distancia al cuadrado de $A^f$ a las matrices simbólicas. El segundo término cumple
  $ EE_i [chi_i^2] <= 1/d_h budget |1 - zt^f|^2, quad |1 - zt^f| <= (T - 1) theta_f, $
  así que es pequeño en los planos que apenas rotan sobre el contexto.
+ *Logits observados.* Para $tau$ uniforme sobre las permutaciones de $[T-1]$, los logits observados de la entrada reordenada por $tau$ cambian en promedio lo mismo:
  $ EE_tau EE_(t tilde unif([T-1])) [(A_(tau(t), t) - A^R_(tau(t), t))^2] = EE_(i,j) [(A_(i j) - A^R_(i j))^2]. $] <prop:the-cost-of-removing-a-rotation>

#proof[Primero, el cambio de un plano. Un plano $f in R$ que pierde su rotación reemplaza su fase $zeta_j^f$ por $1$, y su contribución al logit pasa de $A^f_(i j) = Re[w_i^f zeta_j^f]$ a $Re[w_i^f]$. Sumando y restando $zt^f$,
$ A^f_(i j) - Re[w_i^f] = Re[w_i^f (zeta_j^f - zt^f)] + Re[w_i^f (zt^f - 1)]. $
Como $zeta_j^f - zt^f = zr_j^f$ y $w_i^f = wt^f + wr_i^f$, la @prop:the-parts-of-a-plane, ítem 1, da para el primer término
$ Re[w_i^f zr_j^f] = Re[wt^f zr_j^f] + Re[wr_i^f zr_j^f] = p_j^f + eij^f. $
Multiplicando por $d_h^(-1\/2)$ y sumando sobre $f in R$ se obtiene $A_(i j) - A^R_(i j) = Pi_(i j) + chi_i$.

Los dos términos son ortogonales. Como $chi_i$ no depende de $j$, desarrollar el cuadrado da
$ EE_(i,j) [(Pi_(i j) + chi_i)^2] = EE_(i,j) [Pi_(i j)^2] + 2 EE_i [chi_i EE_j [Pi_(i j)]] + EE_i [chi_i^2]. $
En cada plano, $EE_j [p_j^f] = 0$ y $EE_j [eij^f] = 0$ para todo $i$ (@prop:two-way-split, ítem 1, aplicada a $A^f$), así que $EE_j [Pi_(i j)] = 0$ y el término cruzado se anula.

El término posicional es una suma de covarianzas. Desarrollando el cuadrado de la suma sobre $R$,
$ EE_(i,j) [Pi_(i j)^2] = 1/d_h sum_(f, g in R) (EE_(i,j)[p_j^f p_j^g] + EE_(i,j)[p_j^f eij^g] + EE_(i,j)[eij^f p_j^g] + EE_(i,j)[eij^f eij^g]). $
Los términos mixtos se anulan, pues $EE_(i,j)[p_j^f eij^g] = EE_j [p_j^f EE_i [eij^g]] = 0$ por la @prop:two-way-split, ítem 1, aplicada a $A^g$, y lo mismo con $f$ y $g$ intercambiados. Como todas las partes tienen media cero, los dos términos restantes son las covarianzas $Cov(p_j^f, p_j^g)$ y $Cov(eij^f, eij^g)$, lo que completa el primer ítem.

Para el ítem 2, con $R = {f}$ la suma anterior se reduce a $d_h^(-1) (Var(p_j^f) + Var(eij^f))$, y la @prop:the-positional-ratio-as-a-relative-distance, ítem 3, cuya demostración vale para toda matriz de $cal(M)_(T-1)(RR)$, aplicada a $A^f$, identifica $Var(p_j^f) + Var(eij^f)$ con $d(A^f, cal(S))^2$. Para el segundo término, $(Re z)^2 <= |z|^2$ y $|w_i^f (zt^f - 1)| = |w_i^f| |zt^f - 1|$ dan
$ chi_i^2 <= 1/d_h |w_i^f|^2 |1 - zt^f|^2, $
y promediando sobre $i$ se obtiene la primera cota. Para la segunda, $1 - zt^f = EE_j [1 - zeta_j^f]$, y como la distancia entre dos puntos del círculo unitario es a lo sumo la diferencia de sus ángulos, $|1 - zeta_j^f| <= (T - j) theta_f <= (T - 1) theta_f$ para todo $j in [T-1]$; por la desigualdad triangular,
$ |1 - zt^f| <= EE_j [ |1 - zeta_j^f| ] <= (T - 1) theta_f. $

Para el ítem 3, la entrada reordenada por $tau$ tiene los logits observados $A_(tau(t), t)$, y después de la edición, $A^R_(tau(t), t)$. Si $tau$ es uniforme sobre las permutaciones y $t$ es uniforme en $[T-1]$ e independiente de $tau$, la probabilidad de que $tau(t) = i$ y $t = j$ es la de $t = j$ por la de $tau(j) = i$, es decir, $(T-1)^(-2)$: el par $(tau(t), t)$ es uniforme en $[T-1]^2$. El promedio de $(A_(tau(t), t) - A^R_(tau(t), t))^2$ es entonces $EE_(i,j) [(A_(i j) - A^R_(i j))^2]$.]

La parte $Pi$ del cambio es la dependencia de la posición que la edición quita; $chi$ es constante en cada fila, un cambio de contenido que deja intacta la preferencia entre posiciones. Sin los términos cruzados entre planos, que la @prop:head-as-a-sum-of-planes acota, $EE_(i,j)[Pi_(i j)^2]$ es $d_h^(-1)$ veces la suma sobre $R$ de las varianzas dependientes de la posición de los planos. Con ellos, elegir qué planos pierden la rotación es un problema cuadrático.

#corollary[elección de los planos][Sean $Gamma in cal(M)_F (RR)$ la matriz de entradas
$ Gamma_(f g) := 1/d_h EE_(i,j) [(A^f_(i j) - Re[w_i^f]) (A^g_(i j) - Re[w_i^g])], quad f, g in [F], $
y $C(R) := EE_(i,j) [(A_(i j) - A^R_(i j))^2]$ el costo de quitar la rotación de $R subset.eq [F]$ (@prop:the-cost-of-removing-a-rotation).
+ *Forma cuadrática.* $Gamma$ es simétrica y semidefinida positiva, $C(R) = sum_(f, g in R) Gamma_(f g)$ y
  $ Gamma_(f f) = 1/d_h (Var(p_j^f) + Var(eij^f) + EE_i [(Re[w_i^f (zt^f - 1)])^2]). $
+ *Planos poco correlacionados.* Sea $omega >= 0$ con $|Gamma_(f g)| <= omega sqrt(Gamma_(f f) Gamma_(g g))$ para todo $f != g$. Para todo $R$ no vacío,
  $ (1 - (|R| - 1) omega) sum_(f in R) Gamma_(f f) <= C(R) <= (1 + (|R| - 1) omega) sum_(f in R) Gamma_(f f). $
  Si $(|R| - 1) omega < 1$, $R_Gamma$ reúne $|R|$ planos de menor $Gamma_(f f)$ y $R_*$ minimiza $C$ entre los conjuntos de $|R|$ planos, entonces
  $ C(R_Gamma) <= (1 + (|R| - 1) omega) / (1 - (|R| - 1) omega) C(R_*). $
+ *Nuestra regla (@sec:planes).* Si además $EE_i [(Re[w_i^f (zt^f - 1)])^2] <= upsilon (Var(p_j^f) + Var(eij^f))$ en todo plano, con $upsilon >= 0$, y $R_"var"$ reúne $|R|$ planos de menor varianza dependiente de la posición, entonces
  $ C(R_"var") <= (1 + upsilon) (1 + (|R| - 1) omega) / (1 - (|R| - 1) omega) C(R_*). $] <cor:choosing-planes>

#proof[Para el ítem 1, por la definición de $A^R$, $A_(i j) - A^R_(i j) = d_h^(-1\/2) sum_(f in R) (A^f_(i j) - Re[w_i^f])$, y desarrollando el cuadrado de la suma, $C(R) = sum_(f, g in R) Gamma_(f g)$. $Gamma$ es simétrica por definición, y para todo $r in RR^F$,
$ sum_(f, g) r_f r_g Gamma_(f g) = 1/d_h EE_(i,j) [(sum_f r_f (A^f_(i j) - Re[w_i^f]))^2] >= 0. $
La diagonal es $Gamma_(f f) = C({f})$, que por la @prop:the-cost-of-removing-a-rotation, ítems 1 y 2, es $d_h^(-1) (Var(p_j^f) + Var(eij^f)) + EE_i [chi_i^2]$, con $chi_i = d_h^(-1\/2) Re[w_i^f (zt^f - 1)]$.

Para el ítem 2, por el ítem 1, $C(R)$ es $sum_(f in R) Gamma_(f f)$ más la suma de $Gamma_(f g)$ sobre los pares ordenados de planos distintos de $R$, cuyo valor absoluto es, por la hipótesis, a lo sumo
$ omega sum_(f != g in R) sqrt(Gamma_(f f) Gamma_(g g)) = omega ((sum_(f in R) sqrt(Gamma_(f f)))^2 - sum_(f in R) Gamma_(f f)) <= omega (|R| - 1) sum_(f in R) Gamma_(f f), $
donde $Gamma_(f f) >= 0$ porque $Gamma$ es semidefinida positiva, y la desigualdad es la de Cauchy–Schwarz, $(sum_(f in R) sqrt(Gamma_(f f)))^2 <= |R| sum_(f in R) Gamma_(f f)$. Esto da las dos cotas. Para la comparación, la cota superior en $R_Gamma$, que $R_Gamma$ minimiza la suma de la diagonal entre los conjuntos de $|R|$ planos y la cota inferior en $R_*$ dan
$ C(R_Gamma) <= (1 + (|R| - 1) omega) sum_(f in R_Gamma) Gamma_(f f) <= (1 + (|R| - 1) omega) sum_(f in R_*) Gamma_(f f) <= (1 + (|R| - 1) omega) / (1 - (|R| - 1) omega) C(R_*). $

Para el ítem 3, por el ítem 1 y la hipótesis, $d_h^(-1) (Var(p_j^f) + Var(eij^f)) <= Gamma_(f f) <= (1 + upsilon) d_h^(-1) (Var(p_j^f) + Var(eij^f))$ en todo plano. Como $R_"var"$ minimiza la suma de las varianzas dependientes de la posición entre los conjuntos de $|R|$ planos, la cota superior del ítem 2 en $R_"var"$ y la inferior en $R_*$ dan
$ C(R_"var") <= (1 + (|R| - 1) omega) (1 + upsilon) / d_h sum_(f in R_"var") (Var(p_j^f) + Var(eij^f)) <= (1 + (|R| - 1) omega) (1 + upsilon) / d_h sum_(f in R_*) (Var(p_j^f) + Var(eij^f)) $
$ <= (1 + (|R| - 1) omega) (1 + upsilon) sum_(f in R_*) Gamma_(f f) <= (1 + upsilon) (1 + (|R| - 1) omega) / (1 - (|R| - 1) omega) C(R_*). $]

Lo mismo vale para los puntajes de calibración, sumas de costos sobre queries y cabezales de query: sus matrices $Gamma$ son sumas de matrices de Gram, también semidefinidas positivas. La garantía es más exigente cuantos más planos se quitan, pues pide $omega < 1 \/ (|R| - 1)$, y $omega$ y $upsilon$ dependen del modelo y de la entrada.

*Evaluación.* Las reglas de la @sec:planes y de la @sec:nope se evalúan en los siete modelos con la edición exacta: en los planos de $R$, la query y las keys de todas las posiciones pierden su rotación (los logits $A^R$), en todas las capas y cabezales KV a la vez.
- *Calibración.* Todo puntaje se calibra en 16 ventanas de 1024 tokens de la primera mitad del conjunto de prueba de WikiText-2, como media sobre 256 queries, salvo la regla de normas, que promedia todos los tokens, y EliteKV, que recorre todos los pares de query y key; las calibraciones largas usan ventanas de 4096 o de 16000 tokens del mismo texto. El puntaje de un cabezal KV suma el de sus cabezales de query, sinks incluidos.
- *Pérdida.* El aumento de la NLL media del siguiente token (nats por token) respecto del modelo intacto, en 16 ventanas de 1024 o de 4096 tokens de la segunda mitad del conjunto de prueba de WikiText-2 o de código de la biblioteca estándar de Python: cuatro pares de corpus y longitud. Dos reglas se comparan por la media de la diferencia de sus aumentos, pareada por ventana, la nuestra menos la otra, con un intervalo BCa al 95%.
- *Tareas.* La exactitud de la @sec:tasks en prompts nuevos: 32 de $D_"recuperación"$ y de $D_"asociación"$ y 50 de $Daguja^4000$ y de $Daguja^16000$.

== Selección de planos para partial RoPE <sec:planes>

La multi-head latent attention (MLA) @deepseek2024v2 guarda en la KV cache un vector latente por token, del que las keys salen por una aplicación lineal que se absorbe en la proyección de la query; como una rotación entre ambas lo impediría, MLA lleva la posición en pocas dimensiones aparte con RoPE, y convertir un modelo preentrenado a MLA obliga a conservar la rotación solo en pocas dimensiones de cada cabezal (partial RoPE). La conversión empieza por elegir esas dimensiones: MHA2MLA @ji2025towards y MHA2MLA-VLM @fan2026mha2mlavlm eligen planos (@tab:plane-rules) antes de comprimir y hacer fine-tuning, y aquí se comparan bajo la misma edición, sin comprimir; TransMLA @meng2025transmla concentra antes la información posicional en pocas dimensiones, y el RPD de #cite(<shi2026implicit>, form: "prose") puntúa cada plano por el cambio de la atención al quitar su rotación, como el efecto directo lo hace en los logits. Por la @prop:the-cost-of-removing-a-rotation, sin términos cruzados, el término posicional del costo es mínimo cuando se conservan los planos de mayor varianza dependiente de la posición, y $chi$ es pequeño en los que apenas rotan (ítem 2): nuestra regla conserva esos planos, con la varianza sumada sobre los cabezales de query del cabezal KV, y las demás estiman partes del mismo costo o lo ignoran. Cada regla conserva, en cada cabezal KV, la rotación de un dieciseisavo, un octavo o un cuarto de los $F$ planos, los de mayor puntaje; las tareas se evalúan a un cuarto.

#figure(text(size: 8pt, table(columns: (2.4cm, 1fr, 4.4cm), align: left, inset: (x: 3pt, y: 3pt), stroke: none, table.hline(), [*regla*], [*puntaje del plano $f$*], [*relación con el costo*], table.hline(stroke: 0.5pt),
  [nuestra], [$Var(p_j^f) + Var(eij^f)$], [el término posicional con $R = {f}$ (ítem 2)],
  [nuestra, 4096 y 16000], [el mismo, en las calibraciones largas], [el mismo, en contextos más largos (@cor:slow-planes-and-long-contexts)],
  [cota], [$(|wt^f|^2 + sw) sz$], [su cota (@cor:where-each-part-can-live)],
  [efecto directo], [$d_h^(-1) EE_(t tilde unif([T-1])) [(A^f_(t t) - Re[w_t^f])^2]$, el cambio cuadrático medio de los logits observados cuando solo el plano pierde su rotación], [el costo completo con $R = {f}$, $Pi$ y $chi$, sobre el orden de los contenidos del texto (ítem 3)],
  [EliteKV @zhou2025elitekv], [el lugar del plano en el orden voraz de RoPElite: desde ningún plano, se agrega el que, rotando con los ya agregados, deja los puntajes previos al softmax del grupo más cerca en norma $L^1$ de los de RoPE completo], [el costo del conjunto, con términos cruzados y $chi$, en $L^1$],
  [normas (MHA2MLA @ji2025towards)], [la norma media de la query sin rotar del plano por la de sus keys sin rotar], [aproximadamente la raíz del presupuesto $|wt^f|^2 + sw$, la cota sin el factor $sz$ (@sec:barbero)],
  [KL (MHA2MLA-VLM @fan2026mha2mlavlm)], [en cada query, la divergencia KL de su atención sobre las posiciones $0$ a $T$ respecto de la atención con el plano anulado en la query y en las keys], [el cambio de la atención, que cuenta también el contenido del plano],
  [planos más rápidos (p-RoPE @barbero2025round)], [$theta_f$], [solo la frecuencia],
  table.hline())), caption: [Las reglas de selección de planos: cada una conserva, en cada cabezal KV, la rotación de los planos de mayor puntaje.]) <tab:plane-rules>

Se miden además el $rhoS$ entre los puntajes de nuestra regla y de las demás sobre los planos de un modelo, y la fracción de los planos conservados a un cuarto con $sz < 1\/2$ a la longitud de evaluación (@prop:phase-statistics).

*Predicciones.* Nuestra regla y la cota deberían perder menos que la regla de normas, que ignora $sz$, y que el criterio KL, con al menos la misma exactitud en las tareas; el efecto directo y EliteKV, ordenar los planos casi igual y perder casi lo mismo, salvo si importan los términos cruzados y $chi$. La mayoría de los planos de la regla de normas deberían tener $sz < 1\/2$, y por lo tanto una varianza dependiente de la posición menor que la mitad de su presupuesto (@cor:where-each-part-can-live), con una desventaja mayor en los modelos en que conserva más de esos planos y menor a 4096 que a 1024 tokens (@cor:slow-planes-and-long-contexts). Las calibraciones largas deberían cambiar parte de los planos por otros con $sz < 1\/2$ a 1024 tokens, sin que las formas cerradas digan si eso mejora el modelo a esas longitudes.

=== Sin entrenamiento <sec:planes-training-free>

*Nuestra regla frente a las publicadas.* Nuestra regla pierde menos que la regla de normas y que el criterio KL en todos los modelos, pares de corpus y longitud y fracciones, con todo intervalo bajo cero (@tab:plane-comparisons), por 0.17 a 4.9 nats por token a un cuarto de los planos en los dos corpus (WikiText-2 en @fig:planes-models); la cota pierde del mismo orden que ella, algo menos en Llama-3.2-1B. En las tareas conserva la recuperación, la asociación y la aguja a 4000 tokens cerca del modelo intacto, mientras que esas dos reglas bajan mucho las dos primeras (@fig:planes-tasks); las excepciones son Llama-3.2-1B, donde pierde parte de las dos tareas cortas, y la aguja en Qwen2.5-0.5B, que ninguna regla calibrada a 1024 tokens conserva. A 16000 tokens conserva la aguja solo en Llama-3.2-3B y Llama-3.1-8B, y en Qwen2.5-3B y 7B el criterio KL y EliteKV la conservan mejor.

#figure(text(size: 8pt, table(columns: 7, align: (left,) + (center,) * 6, inset: (x: 3pt, y: 3pt), stroke: none, table.hline(), [*fracción*], [*normas*], [*KL*], [*efecto directo*], [*EliteKV*], [*nuestra, 4096*], [*nuestra, 16000*], table.hline(stroke: 0.5pt),
  [1/16], [28 / 0 / 0], [28 / 0 / 0], [14 / 9 / 5], [16 / 10 / 2], [17 / 8 / 3], [26 / 0 / 2],
  [1/8], [28 / 0 / 0], [28 / 0 / 0], [18 / 3 / 7], [25 / 0 / 3], [19 / 1 / 8], [28 / 0 / 0],
  [1/4], [28 / 0 / 0], [28 / 0 / 0], [18 / 4 / 6], [20 / 0 / 8], [19 / 2 / 7], [27 / 0 / 1],
  table.hline(stroke: 0.5pt),
  [$rhoS$], [0.02–0.16], [$-0.03$–0.17], [0.998–0.999], [0.92–0.97], [0.96–0.98], [0.86–0.94],
  table.hline())), caption: [Nuestra regla frente a las demás, por fracción de planos conservados: en cuántos de los 28 pares de modelo, corpus y longitud pierde menos, lo mismo y más (intervalo bajo cero, con el cero o sobre cero). Última fila: $rhoS$ entre sus puntajes y los de cada regla sobre los planos de un modelo, el rango sobre los siete.]) <tab:plane-comparisons>

#figure(image("/experiments/04_planes/figures/models.pdf"), caption: [El aumento de la pérdida al conservar la rotación de un cuarto de los planos de cada cabezal KV, en WikiText-2 a 1024 (a) y a 4096 tokens (b), en escala logarítmica: una columna de puntos por modelo, uno por regla; huecos, nuestra regla calibrada a 4096 y a 16000 tokens.]) <fig:planes-models>

#figure(image("/experiments/04_planes/figures/tasks.pdf"), caption: [La exactitud, en porcentaje, del modelo intacto y de cada regla a un cuarto de los planos, por tarea (paneles), regla (filas) y modelo (columnas).]) <fig:planes-tasks>

*Estimadores del mismo costo.* El efecto directo y la cota ordenan los planos casi exactamente como nuestra regla ($rhoS$ de 0.998–0.999 ambos), EliteKV casi igual, y la regla de normas y el criterio KL no (@tab:plane-comparisons). A un cuarto y a un octavo de los planos, nuestra regla pierde menos que EliteKV y no más que el efecto directo salvo a 4096 tokens en los Qwen, donde EliteKV y el efecto directo pierden menos en varios modelos, y en Qwen2.5-7B, donde el efecto directo, que suma $EE_i [chi_i^2]$, pierde menos a un octavo también a 1024 tokens. A un dieciseisavo las cuatro reglas pierden de 2.8 a 5.8 nats por token y se separan poco. En las 256 queries de la calibración, nuestro puntaje cuesta de 36 a 190 veces menos tiempo que el orden de EliteKV, que recorre todos los pares de query y key. Se concluye que los tres estiman el mismo costo y que la búsqueda voraz solo lo mejora a 4096 tokens en los Qwen, en parte por la longitud de calibración; con pocos planos, en Qwen2.5-7B importa lo que nuestro puntaje descarta. Por el @cor:choosing-planes, ítem 3, nuestra regla solo puede quedar lejos del costo mínimo si los términos cruzados o $chi$ son grandes frente a las varianzas dependientes de la posición, una lectura posible de esas diferencias y no una medición, pues $omega$ y $upsilon$ no se midieron y la pérdida no es $C$.

*Planos con $sz < 1\/2$.* A 1024 tokens tienen $sz < 1\/2$ del 62 al 77% de los planos que conservan la regla de normas y el criterio KL, frente a a lo sumo el 8% de los de nuestra regla (@fig:planes-sigma-zeta2 a). A 4096 tokens esa fracción baja en la regla de normas, y su desventaja frente a la nuestra también, en los siete modelos en WikiText-2 y en cinco en código (@fig:planes-sigma-zeta2 b); entre modelos, en cambio, la desventaja no crece con esa fracción, al revés de lo predicho ($rhoS$ negativo en ambos corpus y longitudes). Se concluye que la regla de normas conserva sobre todo planos sin espacio para la posición y que su desventaja baja cuando esos planos lo ganan al crecer el contexto, como predicen el @cor:where-each-part-can-live y el @cor:slow-planes-and-long-contexts, pero su tamaño entre modelos no sigue a esa fracción.

#figure(image("/experiments/04_planes/figures/sigma_zeta2_planes.pdf"), caption: [(a) La fracción de planos con $sz < 1\/2$ a 1024 tokens entre los que conserva cada regla a un cuarto de los planos, un punto por modelo. (b) El cociente entre lo que pierden la regla de normas y nuestra regla a un cuarto de los planos, a 1024 y a 4096 tokens, una línea por modelo y corpus.]) <fig:planes-sigma-zeta2>

*Longitud de calibración.* Las calibraciones a 4096 y a 16000 tokens conservan la mayoría de los planos de nuestra regla y cambian los demás por planos con $sz < 1\/2$ a 1024 tokens (@fig:planes-sigma-zeta2 a). A un cuarto y a un octavo de los planos, pierden más a 1024 tokens en todo modelo y corpus; a 4096 tokens la calibración a 4096 pierde menos en cuatro de los siete modelos en WikiText-2 (@fig:planes-models), y la de 16000 recupera la aguja a 16000 tokens en Qwen2.5-1.5B, 3B y 7B (@fig:planes-tasks). También EliteKV, donde gana, conserva más planos con $sz < 1\/2$ que nuestra regla: su ventaja apunta a los planos que solo rotan más allá de la longitud de calibración, salvo en Qwen2.5-7B, donde el efecto directo, que suma $EE_i [chi_i^2]$, recupera buena parte de ella. Se concluye que calibrar más largo cambia texto en contextos cortos por exactitud en la aguja larga, como sugiere el @cor:slow-planes-and-long-contexts, así que la longitud de calibración debe corresponder a los contextos que el modelo ha de atender; si baja también la pérdida a 4096 tokens depende del modelo.

=== Con fine-tuning <sec:finetune>

La @prop:the-cost-of-removing-a-rotation mide cuánto se aleja el modelo editado del preentrenado, el punto de partida del fine-tuning que sigue a la conversión, no lo que ese fine-tuning reconstruye. En Qwen2.5-0.5B y Qwen2.5-1.5B se entrena el modelo entero bajo partial RoPE, con un cuarto de los planos de cada cabezal KV elegidos por una regla y fijos, una corrida por regla (en Qwen2.5-0.5B, la nuestra, la de normas, el criterio KL y los planos más rápidos; en Qwen2.5-1.5B, la nuestra, sus calibraciones a 4096 y a 16000 tokens y la de normas), y otra del modelo intacto, con RoPE completo, sobre los mismos tokens.
- *Edición diferenciable.* En los planos sin rotación, a la query y a la key ya rotadas se les aplica la rotación inversa en su posición, y la atención se calcula con el producto escalar escalado de PyTorch @paszke2019pytorch; antes de la primera actualización da la NLL de la edición exacta, a menos de $10^(-4)$ nats por token en cada ventana.
- *Entrenamiento.* AdamW @loshchilov2019decoupled sobre todos los parámetros, con tasa máxima $5 dot 10^(-5)$, calentamiento lineal y decaimiento coseno, en lotes de 32 secuencias de 2048 tokens y con la pérdida de predicción del siguiente token.
- *Datos.* Los primeros 50 millones de tokens de la muestra sample-10BT de FineWeb-Edu @penedo2024fineweb, que todas las corridas ven en el mismo orden.
- *Evaluación.* Como sin entrenamiento, bajo la edición diferenciable, tras 0, 4.98, 12.5, 25.0 y 49.9 millones de tokens. La *brecha* de una corrida en un corpus y una longitud es su NLL por token menos la del modelo intacto con fine-tuning sobre los mismos tokens, lo que descuenta el efecto del fine-tuning por sí solo; una corrida se compara con la nuestra por su brecha menos la nuestra, pareada por ventana y con un intervalo BCa al 95%.

*Predicciones.* Antes de la primera actualización, el orden de las reglas es el de sin entrenamiento. Si la elección de los planos importa también con entrenamiento, las brechas deben achicarse sin que la nuestra deje de estar por debajo de la regla de normas y del criterio KL; las calibraciones largas deberían acertar más en $Daguja^16000$ si el fine-tuning no reconstruye lo que llevan los planos que solo rotan en contextos largos.

*Resultados.* Antes de la primera actualización el orden es el de sin entrenamiento. Todas las corridas se acercan luego al modelo intacto, sobre todo en los primeros 4.98 millones de tokens (@fig:finetune), salvo en código en Qwen2.5-1.5B, donde la brecha de la regla de normas sube después y, a 4096 tokens, también la de las demás corridas, pero la regla de normas, el criterio KL y los planos más rápidos quedan por encima de la nuestra en las cinco evaluaciones y los cuatro pares de corpus y longitud, con todo intervalo por encima de cero; al final, la brecha de la regla de normas en WikiText-2 a 1024 tokens es 12 veces la nuestra en Qwen2.5-0.5B y 8 en Qwen2.5-1.5B. La recuperación y la asociación, que la regla de normas rompe antes del entrenamiento, vuelven con él, y la nuestra queda junto al modelo intacto con el mismo fine-tuning (@tab:finetune-tasks). En $Daguja^16000$, la ventaja inicial de la calibración a 16000 tokens en Qwen2.5-1.5B (0.98, frente a 0.60), que predice el @cor:slow-planes-and-long-contexts, casi desaparece con el fine-tuning, y al final, en el texto, las calibraciones largas quedan a lo sumo 0.012 nats por detrás de la nuestra, y en WikiText-2 a 4096 tokens, por delante; en Qwen2.5-0.5B ninguna regla recupera la aguja, mientras que el modelo intacto la conserva.

#figure(text(size: 8pt, table(columns: 5, align: (left, left) + (center,) * 3, inset: (x: 3pt, y: 3pt), stroke: none, table.hline(), [], [*corrida*], [$D_"recuperación"$], [$D_"asociación"$], [$Daguja^4000$], table.hline(stroke: 0.5pt),
  [Qwen2.5-0.5B], [intacto], [1.00 → 0.97], [0.81 → 0.88], [1.00 → 1.00],
  [], [nuestra], [0.94 → 0.94], [0.88 → 0.81], [0.56 → 0.98],
  [], [planos más rápidos], [0.94 → 0.88], [0.59 → 0.81], [0.50 → 0.94],
  [], [KL], [0.75 → 0.97], [0.53 → 0.84], [0.58 → 1.00],
  [], [normas], [0.75 → 0.97], [0.44 → 0.84], [0.56 → 0.96],
  table.hline(stroke: 0.3pt),
  [Qwen2.5-1.5B], [intacto], [1.00 → 0.94], [1.00 → 0.97], [1.00 → 1.00],
  [], [nuestra], [0.94 → 0.97], [0.97 → 0.97], [1.00 → 1.00],
  [], [nuestra, 4096], [0.94 → 0.94], [1.00 → 0.97], [1.00 → 1.00],
  [], [nuestra, 16000], [0.84 → 0.94], [0.88 → 0.94], [1.00 → 1.00],
  [], [normas], [0.66 → 0.97], [0.53 → 0.94], [0.56 → 1.00],
  table.hline())), caption: [La exactitud de cada corrida antes del fine-tuning → tras 49.9 millones de tokens; "intacto" es el modelo con RoPE completo y el mismo fine-tuning, y "nuestra, 4096" y "nuestra, 16000", nuestra regla calibrada a esas longitudes.]) <tab:finetune-tasks>

#figure(image("/experiments/05_finetune/figures/finetune.pdf"), caption: [Partial RoPE con fine-tuning en Qwen2.5-0.5B (arriba) y Qwen2.5-1.5B (abajo), en función de los tokens de fine-tuning: la brecha de cada corrida, en escala logarítmica, en WikiText-2 a 1024 (a) y a 4096 tokens (b) y en código a 1024 tokens (c); (d) la exactitud en $Daguja^16000$, con el modelo intacto en negro.]) <fig:finetune>

*Alcance.* Son dos modelos pequeños, con una corrida y un orden de los datos por regla, y las tareas no tienen intervalos. Se concluye que la elección de los planos importa también tras el fine-tuning: las reglas que parten más lejos del modelo no lo alcanzan en 49.9 millones de tokens, mientras que la ventaja de las calibraciones largas en la aguja desaparece, lo que indica que el fine-tuning reconstruye lo que llevaban esos planos. Que ninguna regla recupere la aguja en Qwen2.5-0.5B apunta al entrenamiento, con secuencias mucho más cortas que ella, más que a la elección de los planos, aunque no se probó con secuencias más largas.

== Cabezales y capas sin rotación <sec:nope>

Las arquitecturas híbridas alternan capas con RoPE y capas sin codificación posicional @yang2025rope, AdaRoPE @wang2026adarope aprende en el preentrenamiento frecuencias propias de cada cabezal, y DroPE @gelberg2025drope quita la rotación de todo un modelo preentrenado y, con una breve recalibración, extiende así su contexto. Con todos los planos de un cabezal, el costo tiene una forma simple.

#corollary[cabezales sin rotación][Para todo cabezal, quitar la rotación de todos sus planos, $R = [F]$ en la @prop:the-cost-of-removing-a-rotation, cambia los logits contrafactuales en media cuadrática en
  $ Var(p_j) + Var(eij) + EE_i [chi_i^2], quad "con" quad Var(p_j) + Var(eij) = d(A, cal(S))^2, $
  la distancia al cuadrado de $A$ a las matrices simbólicas.] <cor:heads-without-rotation>

#proof[Con $R = [F]$, la @prop:the-cost-of-removing-a-rotation da $Pi_(i j) = d_h^(-1\/2) sum_f (p_j^f + eij^f)$, y como las partes del cabezal son $d_h^(-1\/2)$ veces las sumas de las de sus planos (@sec:plane-parts), $Pi_(i j) = p_j + eij = A_(i j) - (mu + c_i)$. Por la @prop:the-positional-ratio-as-a-relative-distance, ítem 3, $mu + c_i$ es la matriz simbólica más cercana a $A$, así que $EE_(i,j)[Pi_(i j)^2] = d(A, cal(S))^2 = Var(p_j) + Var(eij)$, y la @prop:the-cost-of-removing-a-rotation, ítem 1, suma a esto $EE_i [chi_i^2]$.]

Una capa pierde la rotación de todos sus cabezales, y su costo en logits es la suma de los de sus cabezales: la regla es quitar la rotación a los cabezales KV, o a las capas, de menor varianza dependiente de la posición sumada sobre sus cabezales de query, una cantidad con escala, a diferencia de $Ph$ (@sec:ratio). La regla deja fuera $EE_i [chi_i^2] = Var(chi_i) + (EE_i [chi_i])^2$, que con $R = [F]$ no tiene por qué ser pequeño, pues en los planos rápidos $|zt^f|^2 = 1 - sz approx 0$ y $|1 - zt^f| approx 1$; de él, el reparto de la atención dentro del contexto solo ve la parte centrada $chi_i - EE_i [chi_i]$, de media cuadrática $Var(chi_i)$, pues la constante $EE_i [chi_i]$ se suma a todos los logits observados del contexto. Quitar la rotación de un cabezal entero no reduce la cache; decide dónde un modelo híbrido puede prescindir de la posición.

*Reglas para cabezales y capas.* Con la calibración de 1024 tokens, un cabezal KV se puntúa por $Var(p_j) + Var(eij)$ sumada sobre sus cabezales de query, o por el mayor $Ph$ entre sus cabezales de query no sink, pues basta uno que atienda por posición para que el grupo necesite la rotación de sus keys; un grupo de solo cabezales sink va primero. Una capa se puntúa por la varianza sumada sobre sus cabezales de query. La edición quita la rotación de todos los planos de los cabezales KV elegidos (la del @cor:heads-without-rotation):
- *Cabezales KV*: un cuarto o la mitad de los cabezales KV del modelo: los de menor varianza (nuestra regla), los de menor $Ph$, los de mayor varianza (control) y tres sorteos uniformes; las tareas, a un cuarto.
- *Capas*: un cuarto de las capas: las de menor varianza (nuestra regla), una de cada cuatro (el patrón de los modelos híbridos), las primeras, las últimas y tres sorteos uniformes.
- *Una capa a la vez*: cada capa sola, en ambos corpus a 1024 tokens, y la $rhoS$ sobre las capas del modelo entre su aumento de la pérdida y su varianza sumada.

*Predicciones.* Por el @cor:heads-without-rotation, nuestra regla de cabezales debería perder menos que los sorteos y que el control en todo caso y conservar las tareas, y $Ph$ perder más, porque ignora la escala del costo. En capas, nuestra regla debería superar a los patrones fijos y a los sorteos en la mayoría de los modelos, y el costo de una capa sola seguir a su varianza sumada; una correlación débil señalaría a $EE_i [chi_i^2]$ o a las capas que siguen a la editada.

Las mediciones de cabezales y capas cubren seis de los siete modelos; las de Llama-3.1-8B están pendientes.

*Cabezales KV.* Nuestra regla pierde menos que los sorteos y que el control en los seis modelos, los cuatro pares de corpus y longitud y ambas fracciones, con todo intervalo bajo cero salvo dos (@tab:nope-heads). A un cuarto de los cabezales KV conserva las tareas en los cuatro modelos más grandes (0.91–1.00), mientras que el control las baja en todos (0.47–0.75); en Qwen2.5-0.5B pierde parte de la asociación y de la aguja a 16000 tokens, y en Llama-3.2-1B de la recuperación y de la asociación, por debajo del mejor sorteo.

#figure(text(size: 8pt, table(columns: 9, align: (left,) + (center,) * 8, inset: (x: 3pt, y: 3pt), stroke: none, table.hline(), [], table.cell(colspan: 4)[*un cuarto de los cabezales KV*], table.cell(colspan: 4)[*la mitad de los cabezales KV*], table.hline(stroke: 0.5pt), [*modelo*], [*varianza*], [$Ph$], [*mayor varianza*], [*al azar*], [*varianza*], [$Ph$], [*mayor varianza*], [*al azar*], table.hline(stroke: 0.5pt),
  [Qwen2.5-0.5B], [0.084], [0.18], [2.2], [0.34–1.0], [0.46], [0.58], [2.6], [1.6–2.0],
  [Qwen2.5-1.5B], [0.076], [0.072], [6.4], [0.24–1.0], [0.29], [0.32], [6.3], [1.0–4.3],
  [Qwen2.5-3B], [0.074], [0.075], [6.0], [0.16–2.7], [0.27], [0.33], [6.1], [1.2–2.7],
  [Qwen2.5-7B], [0.063], [0.071], [4.7], [0.21–0.62], [0.25], [0.28], [5.8], [1.2–1.5],
  [Llama-3.2-1B], [0.15], [3.1], [4.1], [0.33–1.9], [0.79], [3.2], [4.9], [2.2–3.4],
  [Llama-3.2-3B], [0.090], [1.6], [3.9], [0.47–2.0], [0.33], [2.0], [4.3], [2.3–3.0],
  table.hline())), caption: [El aumento de la pérdida, en nats por token, en WikiText-2 a 1024 tokens al quitar la rotación de un cuarto o de la mitad de los cabezales KV, elegidos por menor varianza dependiente de la posición (nuestra regla), por menor $Ph$, por mayor varianza o al azar (el rango de tres sorteos).]) <tab:nope-heads>

*El ratio posicional.* Elegir por $Ph$ pierde mucho más que por la varianza en los dos Llama, y no en los Qwen, donde a un cuarto de los cabezales pierde lo mismo o menos en 11 de los 16 pares de modelo, corpus y longitud, y a la mitad pierde más en WikiText-2 (@tab:nope-heads). Las dos reglas ordenan los cabezales KV de forma parecida ($rhoS$ de 0.82–0.86 sobre los que tienen algún cabezal de query no sink); lo que separa a las familias son los cabezales KV de solo cabezales sink, que la regla de $Ph$ quita primero y que solo existen en los Llama, donde su varianza mediana está en los percentiles 83 y 59 de la del modelo. Se concluye que la pérdida de $Ph$ en los Llama coincide con esos cabezales, así que no basta para atribuirla a la escala que $Ph$ ignora, y que el resultado en los Qwen debilita la predicción.

*Capas.* Nuestra regla pierde menos que el patrón de una de cada cuatro, que las primeras capas y que los sorteos en los seis modelos y los cuatro pares de corpus y longitud, con todo intervalo bajo cero salvo un sorteo que pierde lo mismo (@tab:nope-layers), y conserva las tareas salvo la asociación en Qwen2.5-0.5B y Llama-3.2-1B y la aguja a 16000 tokens en Qwen2.5-0.5B. Las últimas capas son la excepción: pierden menos que nuestra regla en esos dos modelos (en Qwen2.5-0.5B, solo en código) y conservan las tareas al menos igual de bien en casi todo modelo; como en ellos también la suma de lo que pierde cada una de esas capas sola es menor que la de las capas elegidas, la falla está en el orden de las capas solas por la varianza y no en cómo se suman sus costos. Lo que pierde una capa sola sigue a su varianza sumada en todos los modelos, débilmente solo en código en Qwen2.5-7B (@tab:nope-layers). Se concluye que la predicción se cumple en la mayoría de los modelos, no como una ventaja uniforme: sus excepciones son los dos modelos más pequeños de los seis medidos, los mismos en que los cabezales KV elegidos por la varianza pierden parte de las tareas.

#figure(text(size: 8pt, table(columns: 9, align: (left, left) + (center,) * 7, inset: (x: 3pt, y: 3pt), stroke: none, table.hline(), [*modelo*], [*capas de nuestra regla*], [*varianza*], [*una de cuatro*], [*primeras*], [*últimas*], [*al azar*], [$rhoS$*, WikiText-2*], [$rhoS$*, código*], table.hline(stroke: 0.5pt),
  [Qwen2.5-0.5B], [11, 16, 17, 20–22 (de 24)], [0.14], [0.35], [0.99], [0.17], [0.41–0.97], [0.86], [0.65],
  [Qwen2.5-1.5B], [16, 20, 21, 23, 25–27 (de 28)], [0.088], [0.53], [6.0], [0.14], [0.14–6.0], [0.91], [0.92],
  [Qwen2.5-3B], [7, 8, 21, 22, 25, 31–34 (de 36)], [0.094], [0.23], [6.5], [0.17], [0.21–5.8], [0.78], [0.78],
  [Qwen2.5-7B], [2, 9, 14, 16, 22, 23, 26 (de 28)], [0.10], [0.48], [5.7], [0.19], [0.24–4.6], [0.72], [0.35],
  [Llama-3.2-1B], [7, 8, 10, 14 (de 16)], [0.46], [1.8], [3.5], [0.33], [2.0–2.6], [0.80], [0.81],
  [Llama-3.2-3B], [12, 15, 17, 18, 23, 24, 26 (de 28)], [0.31], [1.7], [3.0], [0.71], [0.67–1.6], [0.75], [0.84],
  table.hline())), caption: [Capas sin rotación: las capas que elige nuestra regla (numeradas desde 0) y el aumento de la pérdida, en nats por token, en WikiText-2 a 1024 tokens al quitar la rotación de un cuarto de las capas según cada regla (al azar, el rango de tres sorteos); a la derecha, $rhoS$ sobre las capas entre lo que pierde cada capa sola y su varianza sumada.]) <tab:nope-layers>

= Relación con trabajos previos <sec:other>

Las normas de los planos, la diagonal de la matriz contrafactual y el ángulo entre las filas de los pesos, tres medidas previas de cuánto usa la posición un cabezal o un plano, son piezas de la matriz contrafactual o de sus formas cerradas; la entropía de frecuencia de #cite(<oka2026probing>, form: "prose") mide cuánto se usa cada dimensión de RoPE sin separar contenido, posición e interacción.

== Barbero et al. <sec:barbero>

#cite(<barbero2025round>, form: "prose") miden el uso de cada plano de Gemma 7B @gemma2024gemma por la norma media de la query y de las keys y encuentran que se concentra en los planos lentos, que conjeturan canales semánticos, y que los cabezales posicionales usan los rápidos; construyen cabezales posicionales, prueban que con queries y keys gaussianas de media cero los logits no decaen en promedio con la distancia y que un solo plano no sostiene la atención sobre un objetivo en contextos largos, y proponen p-RoPE, que quita la rotación de los planos más lentos.
- *La norma acota, la fase reparte.* La norma de un plano es $EE_i [ |w_i^f| ]$, y su cuadrado es a lo sumo el presupuesto $|wt^f|^2 + sw$, la cota de la varianza dependiente de la posición sin el factor $sz$: dice cuánto puede pesar el plano, no qué parte lleva; promediada sobre las queries es el puntaje de la regla de normas, y la @sec:planes mide lo que cuesta ignorar $sz$. Que un plano lento solo pueda llevar contenido o una constante (@cor:where-each-part-can-live), como se mide en los modelos (@fig:planes-exp a), es una versión formal de su conjetura.
- *Cabezales posicionales.* Su construcción es la de la @prop:three-pure-heads, ítem 2 ($sw = 0$, el vértice $(0, 1, 0)$ a cualquier frecuencia), y que los entrenados usen planos rápidos es el @cor:where-each-part-can-live (@fig:planes-exp).
- *Decaimiento.* El logit promedio sobre los contenidos en la posición $j$ es $mu + p_j$, con $p_j = d_h^(-1\/2) sum_f Re[wt^f zr_j^f]$: solo puede decaer a través de los centros de masa, y con keys de media cero ($wt^f = 0$) no decae, la forma en una entrada de su resultado gaussiano.
- *Contextos largos.* Su resultado sobre un solo plano es, en los logits, el @cor:slow-planes-and-long-contexts, ítem 2.
- *p-RoPE.* Quitar la rotación de los planos más lentos los congela: $zr^f = 0$ y $p^f = epsilon^f = 0$, canales exactos de contenido, a un costo que, plano por plano, acotan el @cor:slow-planes-and-long-contexts, ítem 1, en su término posicional y la @prop:the-cost-of-removing-a-rotation, ítem 2, en $chi$; la @sec:planes mide esta elección por frecuencia.

*Planos de los cabezales de token anterior.* En $Dwiki$, con $cal(F) := {f : sz > 1\/2}$, se miden las fracciones del presupuesto y de la varianza dependiente de la posición de un cabezal que están en $cal(F)$,
$ nu_h := (sum_(f in cal(F)) (|wt^f|^2 + sw)) / (sum_f (|wt^f|^2 + sw)), quad pi_h := (sum_(f in cal(F)) (Var(p_j^f) + Var(eij^f))) / (sum_f (Var(p_j^f) + Var(eij^f))) $
en los cabezales de token anterior, en los de inducción (@sec:vertices) y en los demás de $cal(H)(Dwiki)$. Los de token anterior ponen en $cal(F)$ una parte de su presupuesto (mediana de $nu_h$ de 6–21% según el modelo, frente a a lo sumo 3% en los demás) y casi toda su varianza dependiente de la posición (mediana de $pi_h$ de 98–100%) (@fig:fast-planes-exp): son posicionales porque ponen norma en los planos rápidos, mientras que los demás la concentran en los lentos (@cor:where-each-part-can-live).

#figure(image("/experiments/01_measure/figures/fast_planes.pdf"), placement: top, caption: [La fracción acumulada de (a) el presupuesto $|wt^f|^2 + sw$ y (b) la varianza dependiente de la posición de un cabezal sobre sus planos, del más rápido al más lento, en función del período del plano sobre la longitud del contexto, en media sobre los cabezales de cada rol en $Dwiki$, una línea por modelo y rol; discontinua, un período por contexto.]) <fig:fast-planes-exp>

== Urrutia et al. <sec:urrutia>

#cite(<urrutia2026decoupling>, form: "prose") llaman posicional a un cabezal si $L(x_s, t) = L(x_t, t)$ y simbólico si $L(x_t, s) = L(x_t, t)$ para todo $s, t$, es decir, $A in cal(P)$ y $A in cal(S)$, y gradúan la distancia a cada tipo con $d_"pos" (i,j) := A_(i j) - A_(j j)$ (el contenido $i$ en $j$ frente a lo que está en $j$) y $d_"sym" (i,j) := A_(i j) - A_(i i)$ (el contenido $i$ llevado de su posición a $j$). Con ellas definimos:

#definition[desviaciones posicional y simbólica][Las *desviaciones posicional* y *simbólica* del cabezal son
$ Dpos := EE_(i,j) [d_"pos" (i,j)^2], quad Dsym := EE_(i,j) [d_"sym" (i,j)^2]. $
Si $A$ no es constante, su *medida normalizada* es $Dsym \/ (Dpos + Dsym)$.] <def:deviations>

Las desviaciones usan la diagonal de $A$, es decir, el orden del texto, y las partes no. Sean $Dpos^tau$, $Dsym^tau$ las desviaciones de la entrada reordenada por una permutación $tau$ de $[T-1]$ (@sec:removing-rotation) y $V^tau := Var_(t tilde unif([T-1])) (A_(tau(t), t))$ la varianza de sus logits observados, con $V := V^"id" = Var_t (lambda_t)$, nula exactamente cuando la atención sobre el contexto es uniforme. Promediar sobre $tau$ uniforme quita la dependencia del orden.

#proposition[desviaciones y partes][
+ *Una entrada.* Con $t tilde unif([T-1])$,
  $ Dpos = Var(c_i) + Var(eij) + EE_t [(c_t + epsilon_(t t))^2], quad Dsym = Var(p_j) + Var(eij) + EE_t [(p_t + epsilon_(t t))^2]. $
+ *Promedio sobre reordenamientos.* Para $tau$ uniforme sobre las permutaciones de $[T-1]$,
  $ EE_tau [Dpos^tau] = 2 d(A, cal(P))^2, quad EE_tau [Dsym^tau] = 2 d(A, cal(S))^2, $
  y, si $A$ no es constante, $Ph = EE_tau [Dsym^tau] \/ EE_tau [Dpos^tau + Dsym^tau]$.
+ *Varianza de los logits observados.* Si $T >= 3$, para el mismo $tau$,
  $ EE_tau [V^tau] = Var(c_i) + Var(p_j) + (T - 3)/(T - 2) Var(eij), $
  y por lo tanto $EE_tau [V^tau] = 1/2 EE_tau [Dpos^tau + Dsym^tau] - (T - 1)/(T - 2) Var(eij)$.] <prop:deviations-in-terms-of-the-parts>

#proof[Para el ítem 1, por la @def:content-positional-and-interaction-parts, $A_(i j) = mu + c_i + p_j + eij$ y $A_(j j) = mu + c_j + p_j + epsilon_(j j)$, de modo que
$ d_"pos" (i,j) = A_(i j) - A_(j j) = (c_i + eij) - (c_j + epsilon_(j j)). $
Para $j$ fijo, el primer término tiene media cero sobre $i$, pues $EE_i [c_i] = 0$ y $EE_i [eij] = 0$ (@prop:two-way-split, ítem 1), y el segundo no depende de $i$, así que al desarrollar el cuadrado el término cruzado se anula:
$ EE_i [d_"pos" (i,j)^2] = EE_i [(c_i + eij)^2] + (c_j + epsilon_(j j))^2. $
Promediando sobre $j$, el primer término da $Var(c_i) + Var(eij)$, porque $EE_(i,j) [c_i eij] = EE_i [c_i EE_j [eij]] = 0$, y el segundo, llamando $t$ al índice, da $EE_t [(c_t + epsilon_(t t))^2]$: esta es la fórmula de $Dpos$. Del mismo modo, $d_"sym" (i,j) = (p_j + eij) - (p_i + epsilon_(i i))$; para $i$ fijo, el primer término tiene media cero sobre $j$ y el segundo no depende de $j$, así que
$ EE_j [d_"sym" (i,j)^2] = EE_j [(p_j + eij)^2] + (p_i + epsilon_(i i))^2, $
y promediando sobre $i$ se obtiene la fórmula de $Dsym$.

Para el ítem 2, la entrada reordenada por $tau$ tiene matriz contrafactual $A^tau$, con $A^tau_(t j) = A_(tau(t), j)$. Como $tau$ es una biyección de $[T-1]$, promediar sobre $t$ una función de $tau(t)$ es promediar sobre los contenidos, así que las partes de $A^tau$ son $mu$, $c_(tau(t))$, $p_j$ y $epsilon_(tau(t), j)$, con las mismas tres varianzas que las de $A$, y su diagonal es $A_(tau(t), t)$. El ítem 1 aplicado a $A^tau$ da
$ Dpos^tau = Var(c_i) + Var(eij) + EE_t [(c_(tau(t)) + epsilon_(tau(t), t))^2], quad Dsym^tau = Var(p_j) + Var(eij) + EE_t [(p_t + epsilon_(tau(t), t))^2]. $
Si $tau$ es uniforme, para cada $t$ fijo $tau(t)$ es uniforme en $[T-1]$, y el mismo cálculo del ítem 1 da
$ EE_tau EE_t [(c_(tau(t)) + epsilon_(tau(t), t))^2] = EE_t EE_i [(c_i + epsilon_(i t))^2] = Var(c_i) + Var(eij), $
así que $EE_tau [Dpos^tau] = 2 (Var(c_i) + Var(eij)) = 2 d(A, cal(P))^2$ por la @prop:the-positional-ratio-as-a-relative-distance, ítem 3. Del mismo modo, $EE_tau [Dsym^tau] = 2 (Var(p_j) + Var(eij)) = 2 d(A, cal(S))^2$ por el mismo ítem, y si $A$ no es constante su ítem 5 da
$ Ph = d(A, cal(S))^2 / (d(A, cal(S))^2 + d(A, cal(P))^2) = EE_tau [Dsym^tau] / EE_tau [Dpos^tau + Dsym^tau]. $

Para el ítem 3, la varianza observada se separa en dos términos. Para cualquier constante, la varianza es la media cuadrática alrededor de ella menos el cuadrado de su distancia a la media; con la constante $mu$,
$ V^tau = EE_t [(A_(tau(t), t) - mu)^2] - (EE_t [A_(tau(t), t)] - mu)^2. $
Como $tau(t)$ es uniforme para cada $t$, el promedio sobre $tau$ del primer término es $EE_t EE_i [(A_(i t) - mu)^2] = Var(A_(i j)) = Var(c_i) + Var(p_j) + Var(eij)$ (@prop:two-way-split, ítem 2).

El segundo término solo depende de la interacción. Como $sum_t c_(tau(t)) = sum_i c_i = 0$ y $sum_t p_t = 0$,
$ EE_t [A_(tau(t), t)] - mu = 1/(T - 1) sum_t epsilon_(tau(t), t). $
En el cuadrado de esta suma, los términos con $t = t'$ tienen media $EE_i [epsilon_(i t)^2]$, porque $tau(t)$ es uniforme, y suman $(T - 1) Var(eij)$. Para $t != t'$, el par $(tau(t), tau(t'))$ es uniforme sobre los $(T - 1) (T - 2)$ pares ordenados de contenidos distintos, y como las columnas de $epsilon$ suman cero (@prop:two-way-split, ítem 1),
$ EE_tau [epsilon_(tau(t), t) epsilon_(tau(t'), t')] = 1/((T - 1) (T - 2)) sum_(i != i') epsilon_(i t) epsilon_(i' t') = - 1/((T - 1) (T - 2)) sum_i epsilon_(i t) epsilon_(i t'). $
Como también las filas de $epsilon$ suman cero,
$ sum_(t != t') sum_i epsilon_(i t) epsilon_(i t') = sum_i ((sum_t epsilon_(i t))^2 - sum_t epsilon_(i t)^2) = - sum_(i, t) epsilon_(i t)^2 = - (T - 1)^2 Var(eij), $
así que los términos con $t != t'$ suman $(T - 1) Var(eij) \/ (T - 2)$, y, con $T >= 3$,
$ EE_tau [(EE_t [A_(tau(t), t)] - mu)^2] = 1/(T - 1)^2 ((T - 1) + (T - 1)/(T - 2)) Var(eij) = Var(eij)/(T - 2). $
Restando,
$ EE_tau [V^tau] = Var(c_i) + Var(p_j) + (T - 3)/(T - 2) Var(eij). $
Por el ítem 2 y la @prop:the-positional-ratio-as-a-relative-distance, ítem 3, $1/2 EE_tau [Dpos^tau + Dsym^tau] = Var(c_i) + Var(p_j) + 2 Var(eij)$, y la diferencia es $((T - 3)\/(T - 2) - 2) Var(eij) = - (T - 1)\/(T - 2) dot Var(eij)$.]

La medida normalizada depende así del orden del texto a través de los términos diagonales, y promediada sobre reordenamientos es $Ph$.

Su principio de exclusión, $V <= Dpos + Dsym$, dice que un cabezal puede ser posicional y simbólico a la vez solo con atención uniforme. De la @def:deviations sale, por la desigualdad de Minkowski, una cota por ambos lados y una superior más fina.

#corollary[principio de exclusión fuerte][Para todo cabezal y toda entrada,
$ 1/2 (sqrt(Dpos) - sqrt(Dsym))^2 <= V <= 1/2 (sqrt(Dpos) + sqrt(Dsym))^2 <= Dpos + Dsym, $
y la última desigualdad es una igualdad si y solo si $Dpos = Dsym$.] <cor:strong-exclusion-principle>

#proof[La varianza de la diagonal es la mitad de la media del cuadrado de la diferencia de dos copias. Si $i$ y $j$ son independientes y uniformes en $[T-1]$, $A_(i i)$ y $A_(j j)$ son copias independientes de $lambda_t$ con $t tilde unif([T-1])$, y para dos variables independientes $U$ y $U'$ con la misma ley, $EE[(U - U')^2] = EE[U^2] + EE[U'^2] - 2 EE[U] EE[U'] = 2 Var(U)$. Sumando y restando $A_(i j)$, esa diferencia se escribe con las desviaciones:
$ V = 1/2 EE_(i,j) [(A_(j j) - A_(i i))^2], quad A_(j j) - A_(i i) = (A_(i j) - A_(i i)) - (A_(i j) - A_(j j)) = d_"sym" (i,j) - d_"pos" (i,j). $

Las dos primeras desigualdades son la desigualdad triangular en $L^2$ de la ley uniforme sobre $[T-1]^2$ (la de Minkowski) y su forma inversa. Por la @def:deviations, $sqrt(Dpos)$ y $sqrt(Dsym)$ son las normas de $d_"pos"$ y $d_"sym"$, así que
$ |sqrt(Dsym) - sqrt(Dpos)| <= (EE_(i,j) [(d_"sym" (i,j) - d_"pos" (i,j))^2])^(1\/2) <= sqrt(Dsym) + sqrt(Dpos). $
Las tres cantidades son no negativas, y al elevarlas al cuadrado y multiplicarlas por $1\/2$ el término central es $V$.

La última desigualdad sale de desarrollar el cuadrado:
$ Dpos + Dsym - 1/2 (sqrt(Dpos) + sqrt(Dsym))^2 = 1/2 (sqrt(Dpos) - sqrt(Dsym))^2 >= 0, $
con igualdad si y solo si $Dpos = Dsym$.]

La cota superior mejora la de Urrutia et al. salvo si $Dpos = Dsym$, y por la @prop:deviations-in-terms-of-the-parts, ítem 3, en cabezales casi aditivos se espera, en promedio sobre reordenamientos, $V approx 1/2 (Dpos + Dsym)$ (en una entrada, $V$ contiene además $2 EE_t [c_t p_t]$, de cualquier signo).

*Comparación con Urrutia et al.* En las tareas cortas y los tres textos, sobre los cabezales de $cal(H)(D)$, la medida normalizada, $Dsym (D) \/ (Dpos (D) + Dsym (D))$, casi coincide con $Ph (D)$ ($rhoS$ de al menos 0.997; a lo sumo 2.7% de los cabezales a lados opuestos de $1\/2$), y la mediana de $V(D) \/ (Dpos (D) + Dsym (D))$ es 0.41–0.54 (@fig:measure-exp). Se concluye que en los modelos entrenados la medida normalizada casi coincide con $Ph$, como predice la @prop:deviations-in-terms-of-the-parts cuando los términos diagonales quedan cerca de su promedio sobre los reordenamientos, y que $V$ queda cerca de la mitad de la cota de Urrutia et al., el valor que da su ítem 3 cuando la interacción es pequeña.

#figure(image("/experiments/01_measure/figures/measure.pdf"), placement: none, caption: [Los cabezales no sink de Qwen2.5-7B en $D_"índice"$, $D_"recuperación"$, $D_"fijo"$ y $Dwiki$, un color por tarea: (a) la medida normalizada frente a $Ph (D)$, con la recta $y = x$; (b) la densidad de $V(D) \/ (Dpos (D) + Dsym (D))$, con $1\/2$.]) <fig:measure-exp>

== Chen y Yan <sec:chen>

#cite(<chen2024what>, form: "prose") miden en cada plano el coseno entre las dos filas de la proyección de la query o de la key que lo generan, y argumentan que filas casi colineales fijan la dirección del vector de cada token, de modo que el cabezal atiende a ciertos desfases (información sintáctica), y que filas ortogonales dejan que dependa del token (información semántica). Filas colineales sin sesgo ponen las keys del plano sobre una recta que pasa por el origen.

#corollary[keys colineales][Sea $f$ un plano.
+ *Sin centro de masa.* Si $wt^f = 0$, entonces $mu^f = 0$ y $p^f = 0$: el plano no tiene parte posicional.
+ *Keys colineales.* Si las keys $k^f (x_i)$ están sobre una recta que pasa por el origen y $wt^f != 0$, existen reales $a_i$ con $EE_i [a_i] = 1$ y $w_i^f = a_i wt^f$ para todo $i$. Entonces
  $ A^f_(i j) = a_i (mu^f + p_j^f), quad c_i^f = (a_i - 1) mu^f, quad eij^f = (a_i - 1) p_j^f, $
  y por lo tanto
  $ Var(c_i^f) = Var(a_i) (mu^f)^2, quad Var(eij^f) = Var(a_i) Var(p_j^f). $] <cor:keys-on-a-line>

#proof[Para el ítem 1, por la @prop:the-parts-of-a-plane, ítem 1, $mu^f = Re[wt^f zt^f]$ y $p_j^f = Re[wt^f zr_j^f]$, que se anulan si $wt^f = 0$.

Para el ítem 2, cada $w_i^f$ es un múltiplo real de $wt^f$. Como $wt^f = q^f overline(EE_i [k^f (x_i)]) != 0$, no todas las keys son nulas, así que la recta es el conjunto de los múltiplos reales de un número complejo no nulo, y cada key es uno de esos múltiplos. Como el conjugado de un múltiplo real de un número es el mismo múltiplo real de su conjugado, cada $w_i^f = q^f overline(k^f (x_i))$ es un múltiplo real de un mismo número, y su media $wt^f$ es la media de esos múltiplos por ese número; como $wt^f != 0$, esa media no es nula, y dividiendo existe $a_i in RR$ con $w_i^f = a_i wt^f$. Promediando sobre $i$, $EE_i [a_i] = 1$. Como $a_i$ es real, $Re[a_i z] = a_i Re[z]$ para todo $z in CC$, y con $zeta_j^f = zt^f + zr_j^f$ y la @prop:the-parts-of-a-plane, ítem 1,
$ A^f_(i j) = Re[a_i wt^f zeta_j^f] = a_i (Re[wt^f zt^f] + Re[wt^f zr_j^f]) = a_i (mu^f + p_j^f). $
Como $EE_j [p_j^f] = 0$ (@prop:two-way-split, ítem 1, aplicada a $A^f$), $EE_j [A^f_(i j)] = a_i mu^f$, y por la @def:content-positional-and-interaction-parts,
$ c_i^f = (a_i - 1) mu^f, quad eij^f = A^f_(i j) - mu^f - c_i^f - p_j^f = (a_i - 1) p_j^f. $
Cada parte tiene media cero, así que su varianza es la media de su cuadrado. Como $EE_i [a_i] = 1$, $EE_i [(a_i - 1)^2] = Var(a_i)$, de donde $Var(c_i^f) = Var(a_i) (mu^f)^2$, y la independencia de $i$ y $j$ factoriza la media de la interacción: $Var(eij^f) = Var(a_i) Var(p_j^f)$.]

Todos los contenidos ponderan entonces el mismo perfil $p^f$, escalado por $a_i$: las keys colineales fijan dónde mira el plano, como dicen Chen y Yan, pero no que mire allí para todo token. Es el caso de PoPE @gopalakrishnan2026decoupling, que separa el qué del dónde con magnitudes no negativas en las queries y las keys y fases que solo dependen de la posición: sus keys sin rotar están, en cada plano, sobre una semirrecta que sale del origen, así que su interacción no desaparece, sino que cada contenido escala un mismo perfil posicional. Las filas ortogonales dejan variar la dirección de $w_i^f$, y esa variación, $wr_i^f$, es contenido en los planos lentos e interacción en los rápidos (@cor:where-each-part-can-live).

El ángulo entre las filas no es, sin embargo, una cantidad de los logits: rotar las dos filas del plano en $W_Q$ y en $W_K$ por un mismo ángulo deja intactos los $w_i^f$ y cambia el coseno. Lo que sí fijan las keys sobre una recta que pasa por el origen es $|s_w^f| = sw$, una segunda componente principal nula de los factores de contenido (@prop:pseudo-variance-and-pca, @fig:statistics a).

*Ángulo entre las filas en los modelos entrenados.* En $Dwiki$, para cada cabezal $h in cal(H)(Dwiki)$ y cada plano $f$, se compara el coseno absoluto entre las dos filas de $W_K$ (o de $W_Q$) que generan el plano con $iota_(h f) := |s_w^f (Dwiki)| \/ sw (Dwiki) in [0, 1]$, que vale uno si y solo si las desviaciones $wr_i^f$ están sobre una recta que pasa por cero (@prop:pseudo-variance-and-pca). Las filas son casi ortogonales (coseno absoluto mediano de 0.04–0.07), $iota_(h f)$ pasa de 0.9 en a lo sumo el 1.0% de los pares $(h, f)$ y apenas sigue a los cosenos ($rhoS$ de 0.18–0.34), y el coseno medio de un cabezal, sobre sus planos y en la query y la key, tiende a bajar al crecer su $Ph (Dwiki)$ ($rhoS$ de −0.22 a −0.50), aunque no de forma monótona (@fig:chen-exp). Se concluye que las keys sobre una recta del @cor:keys-on-a-line son raras en los modelos entrenados y que el ángulo entre las filas predice mal cuán cerca de una recta están los factores de contenido.

#figure(image("/experiments/01_measure/figures/chen_angle.pdf"), caption: [El ángulo de Chen y Yan en $Dwiki$, por deciles, una línea por modelo: (a) la mediana de $iota_(h f) = |s_w^f| \/ sw$ (cuartiles sombreados) en función del coseno absoluto entre las dos filas de $W_K$ de un par de cabezal y plano; (b) el coseno absoluto medio de un cabezal en función de $Ph$.]) <fig:chen-exp>

#place.flush()
#bibliography("references.bib", style: "apa")
