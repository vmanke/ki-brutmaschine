# Schritt 2: Seele und Kopplungsstärke

Englische Fassung: [STEP2_SOUL.md](STEP2_SOUL.md).

## 1. Zweck

Schritt 2 der Bauordnung (Q47) gibt jedem Worker eine **Seele (Phasenzustand)**: verborgene,
vererbte Phasen, eine je Hypothesenfamilie der Domäne eines Laufs (Q42, Q43, Q52). Kein
Modell sieht die Seele. Sie zeigt sich nur in ihrem **Charakter** auf jeder Aufgabe (Q51,
Q57). In einer Stufe addieren sich die Stimmen der Worker als komplexe Zahlen, und das
Betragsquadrat entscheidet (Q46). Jedes Urteil ist eine **Beobachtung**: Es rückt die Seelen
beider Seiten aufeinander zu, so stark, wie die **Kopplungsstärke des Beobachters** κ es sagt
(Q44, Q60, Q62). Seelen gehen durch die **Seeletik** an die Kinder (Q64 bis Q66), und die
**Erziehung** formt sie bei der Geburt (Q59). Jeder Teil hat einen Schalter. Ohne Seele, der
Voreinstellung, schreibt jeder Lauf dieselben Bytes wie vor Schritt 2.

Schritt 2 endet mit einem vorab festgelegten Vergleich auf Schaltungen mit dem simulierten
Entwerfer, als Positivkontrolle: die Seele (Arm S) gegen dieselbe Seele, als Vorzeichen
gelesen (Arm V), und gegen κ = 0 (Arm K). Dieser Bericht beschreibt, was gebaut ist und wie
man einen Lauf rechnet und prüft. Es folgen der Pilot (Ticket 11), die Vorab-Registrierung
(Ticket 10) und die gezählten Läufe mit ihren Ergebnissen (Ticket 13); diese Ergebnisse
tragen wir hier nach. Die Entscheidungen stehen in
[ADR 0009](adr/0009-workers-carry-an-inherited-soul-whose-votes-interfere.md), der Entwurf in
[.scratch/seele-und-kappa/spec.md](../.scratch/seele-und-kappa/spec.md) und die Begriffe in
`CONTEXT.md`.

## 2. Was Schritt 2 gebaut hat

- **Die Seele im Genom** (Ticket 01, `src/cascade/genome.rs`). `Traits::phase_state`, der
  Code-Name aus Q43, hält die Seele: eine Phase je Familie, als ganze Zahl p von 0 bis
  65 535, die für den Winkel 2πp/65 536 steht. `Traits::coupling` hält κ in Tausendsteln,
  von 0 bis 1000. Beide Felder fehlen im JSON, solange sie leer sind. So lesen sich alle
  Genome aus der Zeit vor Schritt 2 Byte für Byte gleich zurück, mit derselben Kennung.
  `Cascade::gene_id()` ist die Kennung ohne beide Felder. Ganze Zahlen halten das JSON, die
  Kennung und jede Änderung einer Seele exakt; Gleitkomma steht nur im Charakter, in der
  Abstimmung und in den Protokollen.
- **Drei Seelen eines Workers.** Die Geburtsseele gehört zum Genom und ändert sich nie; nur
  sie gibt ein Elternteil weiter (Q65). Die geprägte Seele steht bei der Geburt fest
  (Abschnitt „Erziehung“ unten). Die aktuelle Seele liegt im Archiv-Mitglied, und
  Beobachtungen ändern sie (Q56).
- **Der Mutator sieht keine Seele.** Seine Anfragen zeigen nur die Gene der Eltern. Eine
  Antwort mit einem der beiden Felder lehnt der Züchter ab („not admitted: the Mutator does not
  write Souls“). `--promote` lehnt Läufe mit Seelen ab: Eine Seele gehört zu ihrem Lauf.
- **Familienstärken** (Ticket 02, `src/cascade/domain.rs`, `circuit.rs`, `sim.rs`). Die Domäne
  nennt ihre Familien und sagt je Aufgabe, wie gut jede passt. Schaltungen haben die 8
  Familien des Entwerfers: Linear, Phase, Hadamard, Compose, Search, Meet, Simplify und
  Previous. Ein Ziel gibt jeder Familie eines Wegs, den die Kalibrierung von Schritt 1 bis zur
  Referenz festhält, die Stärke 1 und jeder anderen 0 [Q67 a]. Gitter haben die 7 Familien des
  Gittermodells: Mirror, Rotate, Transpose, Gravity, Colour, Compose und Previous. Ein Rätsel
  gibt die Stärke 1 der Familie seines geometrischen Schritts, Colour bei einer Farbtabelle und
  Compose bei beidem. Gleiche Stärken (1 für jede Familie) sind die Kontrolle. Die Stärken
  erreichen nie einen Prompt und nie eine Anfrage.
- **Der Charakter** (Ticket 02, `src/cascade/soul.rs`). Auf Aufgabe t ist er
  χ = |Σ_f s_f(t) · u(p_f)|². Dabei ist u(p) der Einheitszeiger e^(2πip/65 536), in Arm V das
  Vorzeichen +1 oder −1 des näheren von 0 und einer halben Drehung [Q70 a]. Die Protokolle
  nutzen den normierten Charakter χ / Σ s_f², dessen Mittel über unabhängige, gleichverteilte
  Seelen 1 ist. Kein Datensatz speichert einen Charakter (Q55); jede Nutzung rechnet ihn aus
  der Seele. Für unabhängige Seelen ist der Kontrast des Charakters, Standardabweichung durch
  Mittel, nach unserer Rechnung K_C = √(1 − Σ s⁴ / (Σ s²)²) mit komplexen Phasen und
  K_R = √2 · K_C mit Vorzeichen (Q57). Ein Test prüft beide Formeln auf 2^20 frischen Seelen je
  Domäne mit der Toleranz 0.01.
- **Die komplexe Abstimmung** (Ticket 04, `src/cascade/arena.rs`, `soul.rs`). Die blinden
  Paarurteile einer Stufe bleiben, wie sie sind; nur die Zählung ändert sich. Der Zeiger von
  Worker j ist v_j = Σ_f s_f · u(p_f) seiner aktuellen Seele, sein Betragsquadrat also sein
  Charakter [Q68 a]. Mit w_j(x), den Urteilen von j für Antwort x, ist die
  **Gruppencharakteristik der Stufe** Π(x) = |Σ_j w_j(x) · v_j|² (Q61). Die Antwort mit dem
  größten Π gewinnt. Bei Gleichstand entscheidet die Zahl der Urteile, dann die Reihenfolge der
  Antworten [Q70 a]. Mit gleichen Zeigern ist das die heutige Zählung, und eine Abstimmung, in
  der alle Worker gleich urteilen, geht nie verloren. Jedes Mitglied hält die Urheber seiner
  Antworten und eine Zählung seiner Abstimmungen fest.
- **Beobachtungen und κ** (Ticket 05). Ein Urteil mit Antwort, in der Abstimmung einer Stufe
  oder in der Peer-Runde, ist eine Beobachtung (Q62); die Auswahlphase beobachtet nie. Je
  Familie rückt der Beobachtete um round(κ · Δ / 1000) Phaseneinheiten auf den Beobachter zu
  und der Beobachter um round(κ · Δ / 2000) zurück, in ganzen Zahlen [Q69 a]. Alle
  Beobachtungen einer Generation wirken beim Festschreiben, in der Reihenfolge des Protokolls.
  Ein `souls`-Ereignis nennt jeden Worker, dessen Seele sich änderte. Ein Mitglied stimmt im
  Auswahl-Panel mit seinen aktuellen Seelen ab; ein Fortsetzen um weitere Generationen bewertet
  daher jedes Mitglied neu, dessen Seele sich nach einer früheren Auswahlphase geändert hat.
- **Seeletik** (Ticket 03). Die genetischen Operatoren sehen nur Gene und ziehen so dieselben
  Zufallszahlen wie vor Schritt 2. Ein Kind-Worker kopiert die Geburtsseele seines
  Quell-Workers oder mischt zwei, Phase für Phase. Danach rückt jede Phase um einen zufälligen
  Schritt von höchstens `phase_drift` Einheiten. Mit der Wahrscheinlichkeit θ / (θ + N) bekommt
  er stattdessen eine ganz neue Seele, wie die neue Farbe in Hoppes Urne (Q58, Q64); N zählt
  die Worker im Archiv und die zuvor gezogenen. κ folgt denselben Quellen, mit einem Schritt
  von höchstens `kappa_drift`, und ist kein Gen [Q71 a]. Die Seeletik zieht aus eigenen
  Zufallsströmen. Ein Kind muss nach seinen Genen neu sein, und die Modellaufrufe nehmen ihren
  Seed aus der Gen-Kennung. So teilen Arme mit gleichen Genen jede Zufallszahl.
- **Erziehung** (Ticket 07, `src/cascade/upbringing.rs`). Prägung: Nach seiner ersten
  Such-Aufgabe rückt jeder Worker eines Kindes auf die aktuelle Seele seines Quell-Workers zu,
  nach der Regel für den Beobachteten [Q73 a]. Das kostet keinen Aufruf, und die Eltern ändern
  sich nicht. Lehre: Fester Code schreibt aus den besten gelösten Beispielen der Eltern einen
  Abschnitt in die Prompts des Kindes [Q73 d]. Die simulierten Modelle lesen ihn nicht, und im
  Vergleich ist die Lehre aus.
- **Die Seele in den simulierten Modellen** (Ticket 06, `src/backend.rs`, `designer.rs`,
  `sim.rs`). Anfragen an simulierte Modelle tragen die Phasen des Workers (contract-v3 von
  [STAGE2_BACKEND_CONTRACT.md](STAGE2_BACKEND_CONTRACT.md)), nie in einem Peer-Urteil und nie
  an ein echtes Modell. Der Entwerfer wählt nach der Seele nur zwischen gleich teuren, exakten
  Schaltungen, die zwei oder mehr Familien fanden, und nur mit Selbstprüfung; das Gittermodell
  nur im Urteil, wenn beide Antworten erklärt sind [Q72 a]. Kosten, Noten und die Kalibrierung
  von Schritt 1 bleiben. Antworten, in denen die Phasen entschieden, zählt `sim_decided`.
- **Die Gruppe `soul` der `state`-Zeile** (Ticket 08, `src/cascade/observe.rs`), siehe
  Abschnitt 4.
- **Programm, Prüfung und Auswertung** (Ticket 09). `brut-breed --soul s|v|k`, `--verify` für
  Läufe mit Seelen auf beiden Domänen und `examples/soul_character.rs`, siehe Abschnitt 3.
- **Ohne Seele ändert sich nichts.** Ohne die Einstellung `soul` schreibt kein Lauf ein neues
  Feld, zieht keine neue Zufallszahl und ändert keine Anfrage. Die Golden-Tests von
  `tests/universe.rs` und `tests/mlp_golden.rs` bestehen mit unveränderten Fixtures, und der
  Lauf über 60 Generationen auf Pilot-Seed 101 von Schritt 1 endet in seiner festgehaltenen
  Prüfsumme e93ed521… (Release-Test `souls_off_reproduces_the_step1_pilot`). Am Ende des Baus
  haben wir den gezählten Lauf A-1 von Schritt 1 von Hand wiederholt: Seine fünf Dateien haben
  die Prüfsummen, die [step1-counted/hashes.txt](step1-counted/hashes.txt) festhält.

## 3. Ein Lauf mit Seelen und seine Prüfung

```sh
cargo build --release

# Arm S, die Seele (Pilot-Seed 901, 60 Generationen)
./target/release/brut-breed --out runs/s901-S --domain circuits --search 33 --select 22 \
  --children 8 --generations 60 --seed 901 --max-calls 200000 --soul s

# Die Kontrollen: Arm V liest dieselbe Seele als Vorzeichen, Arm K hält κ bei 0
./target/release/brut-breed --out runs/s901-V --domain circuits --search 33 --select 22 \
  --children 8 --generations 60 --seed 901 --max-calls 200000 --soul v
./target/release/brut-breed --out runs/s901-K --domain circuits --search 33 --select 22 \
  --children 8 --generations 60 --seed 901 --max-calls 200000 --soul k

# Protokoll, Bericht, rediscovery.json, Archiv und Seelen prüfen, ohne Modellaufruf
./target/release/brut-breed --out runs/s901-S --verify

# Den Charakter jedes Workers aus dem Protokoll nachrechnen
cargo run --release --example soul_character -- runs/s901-S
```

`--soul` setzt die Einstellungen eines Arms aus Abschnitt 12.2 des Entwurfs:

| Feld | Bedeutung | Arm S | Arm V | Arm K |
|---|---|---|---|---|
| `readout` | komplex oder Vorzeichen (Q42, Q70) | complex | real | complex |
| `strengths` | Stärken der Domäne oder gleiche (Q63) | domain | domain | domain |
| `vote` | komplexe Abstimmung (Q46) | an | an | an |
| `simulator` | Phasen an die simulierten Modelle (Q72) | an | an | an |
| `observe_vote`, `observe_peer` | Urteile der Abstimmung, der Peer-Runde sind Beobachtungen (Q62) | an | an | an |
| `kappa_seed` | κ der Startprogramme, in Tausendsteln (Q71) | 200* | 200* | 0 |
| `kappa_fixed` | κ ändert sich nie (Q44) | aus | aus | an |
| `kappa_drift` | größter Schritt von κ bei einer Geburt, in Tausendsteln | 20* | 20* | 20 |
| `copy`, `mix` | Kinder kopieren Seelen; zwei Quellen mischen (Q64) | an | an | an |
| `phase_drift` | größter Phasenschritt bei einer Geburt, in Einheiten (Q64) | 512* | 512* | 512* |
| `new_soul_theta` | Hoppes θ (Q64) | 1* | 1* | 1* |
| `imprint`, `teach` | Prägung; Lehre (Q59) | an; aus | an; aus | an; aus |

Ein Stern markiert einen Platzhalter, den der Pilot festlegt (Ticket 11); dann ändern sich
auch die Zeilen unten. Feinere Einstellungen kommen über `--config <datei>`: eine vollständige
Laufkonfiguration mit einem Objekt `soul`, in dem jedes der 15 Felder stehen muss. `--soul`
neben einer solchen Datei ist ein Bedienfehler (Exit-Code 2). `--soul` legt das Experiment
fest; ein Fortsetzen lehnt es ab, wie `--domain` und `--selection`. Die Seeds 901 bis 923 sind
Pilot-Seeds von Schritt 2 und zählen nie.

Ein Lauf druckt dieselben Zeilen wie in Schritt 1. Für Arm S auf Seed 901 lautet die Zeile zu
`rediscovery.json`:

```
brut-breed: wrote runs/s901-S/rediscovery.json: the seeds solve 2 of 11 targets, bred Cascades 2 of the other 9; D > 1/2: no; calibrated
```

und `--verify` meldet:

```
brut-breed: 2046 events, hash chain intact, head e38339538f19d85fccd1d1a7a1c3a0435e1989eec7838e6b6e0c5a46e0e692a1; checkpoint at entry 2046 matches; settings match; report recomputed and equal; rediscovery recomputed and equal
```

Für Arm V lautet die Prüfsumme 3516f3d2b956a6002eddb95d1340d3ffbd59b84963bf89e60aec8301069ea41d,
für Arm K e431bae607b74d20f2e810075f121c8a25ba8c2e579717125aec6274b4107afa; beide haben
ebenfalls 2046 Ereignisse und dieselbe Zeile zu `rediscovery.json`. Wir messen 3.9 bis 4.2 s
je Lauf (Release-Build, 4 Threads); der Debug-Build schreibt dieselben Bytes.

**Was `--verify` prüft.** Alles, was es in Schritt 1 prüft (Hash-Kette, Anker des Checkpoints,
Einstellungen, Bericht, bei Schaltungen `rediscovery.json`). Bei einem fertigen Lauf mit Seelen
vergleicht es dazu jedes Archiv-Mitglied mit seinem `born`-Ereignis, auch in den fünf Feldern
von Schritt 2: geprägte Seelen, Herkunft jeder Seele, Lehre, Urheber der Antworten und Zählung
der Abstimmungen. Die Geburtsseelen und κ stehen im Genom und werden mit ihm verglichen. Die
aktuelle Seele jedes Workers muss seinem letzten Eintrag in den `souls`-Ereignissen gleichen,
oder seiner geprägten Seele, wenn kein Ereignis ihn nennt. Das gilt nun auch für Gitterläufe
mit Seelen; deren Zeile endet mit „archive and Souls match the log“. Ein Lauf mit Seelen behält
alle Antworten des Auswahl-Panels. Hat er einen Bericht, den sein Checkpoint nicht nachrechnen
kann, weil der Checkpoint unfertig ist oder Antworten des Auswahl-Panels fehlen, ist das daher
auch auf Gittern ein Fehler, wie bei einem Schaltungslauf. Ohne Bericht nennt die Zeile, warum
nicht geprüft wurde: „archive and Souls not checked: the run is not finished“ oder, bei einem
fertigen Lauf ohne seinen Bericht, „… there is no report.json; resume the run to write it“.
Ein Gitterlauf ohne Seele wird wie bisher geprüft und meldet dieselbe Zeile wie vor Schritt 2.
Jede Abweichung beendet `--verify` mit Exit-Code 1 und einer Meldung „integrity: …“.
`--verify` ruft kein Modell auf und rechnet die Beobachtungen daher nicht nach; ein erneuter
Lauf wiederholt sie Byte für Byte, und die Tests prüfen das bei 1 und 4 Threads.

**Der Charakter aus dem Protokoll.** `examples/soul_character.rs` liest `config.json`, den
Checkpoint und den Laufbeginn im Protokoll. Für jedes Mitglied, jeden Worker und jedes Ziel
(bei Gittern jede Regelart: der geometrische Schritt, mit „+colours“ bei einer Farbtabelle)
druckt es den Charakter und den normierten Charakter der aktuellen Seele, mit der Ablesung und
den Stärken des Laufs. Die Ausgabe ist eine Tabelle mit Tabulatoren; Zeilen mit „#“ nennen
die Familien und je Ziel die Zahl der Such-Aufgaben und die Stärken. Die Zahlen stehen in der
kürzesten Form, die genau dieselbe f64 zurückgibt. Ein Test vergleicht jede Zeile exakt mit
`character()` und `character_norm()` auf den Seelen des Checkpoints.

## 4. Was die Gruppe `soul` bedeutet

In Läufen mit Seelen trägt die `state`-Zeile nach jeder festgeschriebenen Generation eine
siebte Gruppe `soul`. Ohne Seele fehlt sie. Wie die anderen sechs Gruppen hängt sie nur vom
festgeschriebenen Zustand ab, sodass ein angehaltener und fortgesetzter Lauf dieselben Zeilen
schreibt wie ein durchgehender. Mittel über Aufgaben laufen über das Such-Panel, jede Aufgabe
einmal; eine Statistik des Charakters rechnen wir je verschiedenem Stärkevektor einmal und
gewichten sie mit der Zahl seiner Aufgaben. Ein undefinierter Wert fehlt, statt NaN zu sein.

| Feld | Bedeutung |
|---|---|
| `souls` | Worker im Archiv: die Seelengrundgesamtheit (Q58); sie fällt nie |
| `souls_born`, `souls_new` | Worker, die in dieser Generation geboren wurden, und davon die mit ganz neuer Seele (Q64) |
| `origins`, `origin_entropy_bits` | verschiedene Ursprünge der Seelen im Archiv, Hoppes Farben, und die Entropie ihrer Häufigkeiten |
| `character_norm_mean` | Mittel des normierten Charakters über die Worker und die Such-Aufgaben, aus den aktuellen Seelen (Q51) |
| `contrast` | Mittel über die Such-Aufgaben des Kontrasts des Charakters über die Worker (Q57) |
| `contrast_independent` | dasselbe Mittel der Formel K_C bzw. K_R: der Wert für unabhängige Seelen |
| `order`, `order_mean` | je Familie die Länge des mittleren abgelesenen Zeigers über alle Worker (1: alle gleich; nahe 0: verteilt), und ihr Mittel |
| `moved_mean` | mittlerer Kreisabstand zwischen aktueller und Geburtsseele über Worker und Familien, in Grad: was Erfahrung bewirkte |
| `kappa` | Quantile von κ über die Worker, als Bruch |
| `observations_vote`, `observations_peer` | Beobachtungen dieses Festschreibens, nach Art (Q62) |
| `shift_mean` | mittlere Phasenänderung der Worker, die sich bei diesem Festschreiben bewegten, in Grad (Q56) |
| `votes`, `votes_flipped`, `votes_tied` | komplexe Abstimmungen der Neugeborenen dieser Generation auf dem Such-Panel; die, deren Sieger von dem der Zählung abweicht; die, in denen zwei oder mehr Antworten das größte Π teilten |
| `coherence_mean` | mittlere Kohärenz dieser Abstimmungen, \|Σ_j v_j\| / Σ_j \|v_j\|: 1, wenn alle Zeiger gleich zeigen, nahe 0, wenn sie sich aufheben |
| `sim_decided` | Antworten an die Neugeborenen dieser Generation, in denen die Phasen die Wahl eines simulierten Modells entschieden |

Ein Beispiel aus den drei Läufen von Abschnitt 3, gemessen in Generation 60:

| Feld | Arm S | Arm V | Arm K |
|---|---|---|---|
| `souls`, `origins` | 1554, 10 | 1554, 10 | 1554, 10 |
| `character_norm_mean` | 0.973 | 0.678 | 1.017 |
| `contrast` (Formel) | 0.534 (0.655) | 1.217 (0.927) | 0.546 (0.655) |
| `order_mean` | 0.663 | 0.651 | 0.544 |
| `moved_mean` | 14.4° | 14.4° | 0° |
| `kappa`, Median | 0.206 | 0.206 | 0 |

Über alle 61 Generationen zählen wir in jedem Arm 32 komplexe Abstimmungen, 64 Beobachtungen in
Abstimmungen und 2937 in Peer-Runden. Keine Abstimmung kippte gegen die Zählung, und
`sim_decided` blieb 0. In Arm V entschied bei 12 Abstimmungen die Gleichstandsregel. Die drei
Arme bringen auf diesem Seed dieselben 484 Neugeborenen hervor, Mitglied für Mitglied gleich
in Genen, Operator, gelösten Aufgaben, Aufgabenwerten und Antworten: Bis Generation 60
änderte die Seele hier kein Ergebnis. In Arm S und Arm K liegt der Kontrast unter der Formel,
wie gemeinsame Abstammung und, in Arm S, Kopplung es erwarten lassen. In Arm V liegt er
darüber, anders als Abschnitt 13.1 des Entwurfs es erwartet. Bei Vorzeichen drücken
gemeinsame Phasen den Charakter vieler Worker auf 0: Auf den beiden Zielen mit zwei passenden
Familien, CH und CCZ, messen wir mit `examples/soul_character.rs` bei 70.5 % und 90.3 % der
1554 Worker den Charakter 0, bei unabhängigen Seelen wären es 50 %. Dort liegt der Kontrast bei
1.55 und 3.06 statt 1.00. Das ist eine Messung auf einem Pilot-Seed, kein Ergebnis; der Pilot
untersucht es auf 12 Seeds.

**Die Ereignisse `born` und `souls`.** Ohne Seele unverändert. Mit Seelen trägt jedes
`born`-Ereignis fünf Felder mehr: `imprinted` (die geprägte Seele jedes Workers),
`soul_sources` (Quell-Worker, ob die Seele neu ist, und ihr Ursprung), `teaching` (der
Abschnitt der Lehre, wenn es einen gibt), `authors` (je Such-Aufgabe die Worker, deren Antwort
gewählt wurde) und `votes` (die Zählung der Abstimmungen). Das Genom im Ereignis hält die
Geburtsseele und κ. Nach jedem `generation`-Ereignis nennt ein `souls`-Ereignis jeden Worker,
dessen aktuelle Seele sich beim Festschreiben änderte, mit seinen neuen Phasen.

## 5. Wie es weitergeht

- **Pilot (Ticket 11).** Die Arme S, V und K, dazu E (gleiche Stärken) und N (ohne Seele, das
  ist Arm A von Schritt 1), auf den Pilot-Seeds 901 bis 912 über 1000 Generationen. Er legt die
  Platzhalter fest und misst, wie oft die Seele ein Ergebnis ändert.
- **Vorab-Registrierung (Ticket 10).** Zwei getrennte Aussagen, „S vor V“ und „S vor K“, nach
  der Bestehenszeit von Schritt 1, jede höchstens mit Wahrscheinlichkeit 1/200 falsch, beide
  zusammen höchstens 1/100 [Q74 a]. Sie wird nach dem Pilot geschrieben und mit dem
  Zusammenführen ihres Pull-Requests eingefroren.
- **Gezählte Läufe (Ticket 13).** Ihre Ergebnisse tragen wir in diesen Bericht nach.

## 6. Grenzen

- **Ein Modell.** Seele, Charakter und Kopplungsstärke sind Begriffe eines Modells. Nichts
  hier ist physikalisch quantenhaft, und der Vergleich sagt nichts über Seelen in der Natur
  (Q48 bleibt offen).
- **Eine Positivkontrolle.** Der simulierte Entwerfer ist kein Sprachmodell. Was wir mit ihm
  finden, sagt nichts über echte Modelle.
- **Schwache Wirkung im Simulator.** Der Entwerfer lässt die Seele nur zwischen gleich teuren,
  exakten Schaltungen wählen, die zwei oder mehr Familien fanden, und nur mit Selbstprüfung
  [Q72 a]. Die Seele wirkt daher vor allem über Abstimmung, Beobachtung und Erziehung. Auf
  Seed 901 änderte sie bis Generation 60 kein Ergebnis (Abschnitt 4). Ob der Vergleich genug
  Trennschärfe hat, muss der Pilot zeigen.
- **Einstimmige Urteile.** Der Entwerfer urteilt durch Nachrechnen, sodass die Worker einer
  Stufe meist gleich urteilen. Eine solche Abstimmung geht in keinem Arm verloren. Die komplexe
  Abstimmung ändert einen Sieger nur, wo die Urteile auseinandergehen.
- **Handgeschriebene Stärken.** Die Stärken der Schaltungen stammen aus den Wegen, die Schritt 1
  festhielt. Bei Swap und CS passt eine Familie, dort ist jeder Charakter 1.
- **Die Formel des Kontrasts** gilt für unabhängige Seelen. In einem Lauf machen Abstammung und
  Kopplung Seelen ähnlich, und 4 Startprogramme geben in Generation 0 nur eine grobe Schätzung.
- **Lehre ohne Wirkung.** Die simulierten Modelle lesen den Abschnitt der Lehre nicht; ihre
  Wirkung braucht echte Modelle.
- **Plattform.** Die Zeiger nutzen `cos` und `sin` der Standardbibliothek. Ihr letztes Bit kann
  zwischen Plattformen abweichen. Wir vergleichen Läufe darum auf einer Werkzeugkette und
  Plattform, rustc 1.97.0 auf x86_64 Linux, wie in Schritt 1.

## 7. Quellen

Die geprüften Quellen des Entwurfs stehen in Abschnitt 19 der
[Spezifikation](../.scratch/seele-und-kappa/spec.md). Dieser Bericht stützt sich auf:

- Acebrón, Bonilla, Pérez Vicente, Ritort, Spigler (2005). The Kuramoto model: A simple paradigm
  for synchronization phenomena. Reviews of Modern Physics 77, 137.
  https://doi.org/10.1103/RevModPhys.77.137 (Name der Kopplungsstärke)
- Goodman (1976). Some fundamental properties of speckle. Journal of the Optical Society of
  America 66(11), 1145–1150. https://doi.org/10.1364/JOSA.66.001145 (Kontrast 1 bei voll
  entwickeltem Speckle)
- Goodman (2008). Speckle with a finite number of steps. Applied Optics 47(4), A111–A118.
  https://doi.org/10.1364/AO.47.00A111 (nur die Zusammenfassung gesehen; √(1 − 1/k) haben wir
  selbst nachgerechnet)
- Hoppe (1984). Pólya-like urns and the Ewens' sampling formula. Journal of Mathematical Biology
  20, 91–94. https://doi.org/10.1007/BF00275863 (nur die Zusammenfassung gesehen)
- Pothos, Busemeyer (2013). Can quantum probability provide a new direction for cognitive
  modeling? Behavioral and Brain Sciences 36(3), 255–274.
  https://doi.org/10.1017/S0140525X12001525
- Renou et al. (2021). Quantum theory based on real numbers can be experimentally falsified.
  Nature 600, 625–629. https://doi.org/10.1038/s41586-021-04160-4 (die Frage hinter Arm V)
