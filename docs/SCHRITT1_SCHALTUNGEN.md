# Schritt 1: Zucht auf Quantenschaltungen mit exakter Prüfung

Englische Fassung: [STEP1_CIRCUITS.md](STEP1_CIRCUITS.md).

## 1. Zweck

Schritt 1 der zweiten Bauordnung (Q47, Tickets 01 bis 03 in `.scratch/universum-erfinden/`)
öffnet die Kaskaden-Zucht der Stufe 2c für eine zweite Aufgabenart: Quantenschaltungen aus
den Gattern H, S, S†, T, T†, X, Z und CNOT, deren Richtigkeit ein exakter Simulator
entscheidet. Später soll die Zucht erfinden: Schaltungen, die exakt richtig sind, eine vorab
festgelegte Referenz schlagen und neu sind (Q38). Zuerst prüfen wir mit einer
Positivkontrolle, ob sie die bekannten besten T- und CNOT-Zahlen wiederfindet, welche ihre
Startprogramme verfehlen (Q40). Außerdem schreibt jeder Lauf nach jeder Generation
Kennzahlen, mit denen wir ihn als Universum untersuchen (Q32, Q33). Die Entscheidungen stehen
in [ADR 0006](adr/0006-cascades-breed-on-domains-and-circuits-are-checked-exactly.md), das
vorab festgelegte Experiment in [PREREG_STEP1_CIRCUITS.md](PREREG_STEP1_CIRCUITS.md), die
Pilotläufe in [pilot-step1/](pilot-step1/README.md) und die Begriffe in `CONTEXT.md`.
Das gezählte Experiment ist gelaufen und hat beide vorab festgelegten Nachweise erbracht,
den Hauptnachweis bei Paar 15 und den zur Auswahl bei Paar 27 (Abschnitt 7). Beide
betreffen den Züchter als Positivkontrolle mit dem simulierten Entwerfer; über echte
Sprachmodelle sagen sie nichts. Die Daten des Experiments liegen in
[step1-counted/](step1-counted/README.md).

## 2. Was Schritt 1 gebaut hat

- **Eine Zuchtschleife für jede Aufgabenart.** Die Bewerterseite einer Aufgabenart
  (`Domain`, `src/cascade/domain.rs`) erzeugt die Panels, schreibt die Prompts, liest und
  bewertet die Antworten und sagt, welche von zwei Antworten besser ist. Sie sieht nie eine
  Kaskade, sodass keine Kaskade ändern kann, wie sie bewertet wird. Die Züchterseite
  (`GenePool`) liefert die Startprogramme, die Satzbausteine der Abschnitts-Operatoren, den
  Auftrag an den Mutator und eine Aufnahmeregel. Gitter-Rätsel und Schaltungen teilen sich
  dieselbe Schleife. Golden-Tests, aufgenommen auf dem Gitter-Code vor dem Umbau, halten
  einen Gitter-Lauf, einen Lauf mit LLM-Runden, die Schlüssel der Mutator-Anfragen und die
  Prüfsummen der Einstellungen fest; einzig neu im Protokoll eines Gitter-Laufs ist die
  `state`-Zeile (`tests/universe.rs`).
- **Universums-Kennzahlen** (Ticket 01, `src/cascade/observe.rs`). Nach jeder
  festgeschriebenen Generation schreibt jeder Lauf eine `state`-Zeile ins Protokoll
  (Abschnitt 4).
- **Exakter Simulator** (Ticket 02, `src/cascade/qubit.rs`). Er speichert jede Amplitude als
  x/√2^k mit x in Z[ω], ω = e^(iπ/4), in vier 64-Bit-Ganzzahlen und rechnet jedes Gatter
  ganzzahlig. „Richtig“ heißt exakt gleich bis auf eine globale Phase; eine Toleranz gibt es
  nicht. Die Fidelity berichten wir nur, denn die Schwelle F ≥ 1 − 1e-9 aus Q38 lässt
  Näherungen zu: Nach unserer Rechnung erreicht eine Schaltung auf zwei Qubits, deren
  einziger Fehler eine relative Phase von 7e-5 rad auf einem Basiszustand ist, diese Schwelle
  noch. Die Grenzen stehen im Code: 12 Qubits für Zustände, 6 für ganze Unitaries, 256 Gatter
  und 120 H-Gatter. Mit höchstens 120 H-Gattern bleibt jeder Koeffizient unter 2^60, sodass
  keine 64-Bit-Zahl überläuft; geprüft wird jede Operation trotzdem.
- **Zweiter Simulator als Gegenprobe.** `src/quantum.rs` (PR #12) rechnet in Gleitkomma und
  teilt keinen Code mit `qubit.rs`. Sein Qubit 0 ist das niedrigste Bit, das von `qubit.rs`
  das höchste; die Tests bilden darum Qubit q auf n − 1 − q ab. In `tests/circuits.rs`
  entscheiden beide Simulatoren 900 Paare zufälliger Schaltungen auf 1 bis 4 Qubits gleich:
  300 Paare sind nach Bauart gleich, 300 verschieden und 300 unabhängig gezogen. Für die
  15 Beleg-Schaltungen der Bibliothek liefern beide dieselbe Matrix bis auf 1e-12.
- **Zielbibliothek und Aufgaben** (Ticket 03, `src/cascade/circuit.rs`). 11 Ziele tragen T-
  und CNOT-Zahlen, welche die Recherche in der Literatur nachgelesen hat (Tabelle 1). Vier
  weitere (W-Gatter, √SWAP, Quanten-OR, 1-Bit-Addierer) haben keine geprüfte Referenz; sie
  kommen nie auf ein Panel und dienen nur als Testfälle des Prüfers. Jedes Ziel ist aus
  seiner Definition gebaut, nie aus einer Schaltung. Eine Aufgabe ist ein verkleidetes Ziel:
  Die Qubits sind umbenannt, an Ein- und Ausgängen stehen X-Gatter. Die Verkleidung erhält
  die optimalen T- und CNOT-Zahlen, aber eine auswendig gelernte Schaltung passt nicht mehr.
  Ein Prompt zeigt nur die Wirkung auf jeden Basiszustand, nie Namen, Kennung, Quellen oder
  Referenz. Das Such-Panel hat 33 Aufgaben (drei Verkleidungen je Ziel), das Auswahl-Panel
  22 (zwei je Ziel).
- **Bewertung.** Eine nicht exakte Schaltung erhält 0, eine exakte
  0.5 + 0.5/(1 + ΔT + ΔC/(1 + ΔC)), wobei ΔT und ΔC die T- und CNOT-Zahlen über der Referenz
  sind (0, wenn sie sie nicht übersteigen). Ein T-Gatter mehr kostet so stets mehr als
  beliebig viele CNOT-Gatter mehr, und der Wert ist genau dann 1, wenn die Schaltung beide
  Referenzzahlen erreicht: Dann gilt die Aufgabe als gelöst. Die Fitness nimmt den mittleren
  Wert statt der Lösungsquote, also 0.8 · mittlerer Wert + 0.2 · Urteilskraft − Kosten. Ein
  Richter wird gewertet, wenn die Aufgabenart eine der beiden Antworten streng vorzieht.
- **Integritätsalarm und Erfindungskandidat.** Eine exakte Antwort unter einer bewiesenen
  oder abgeleiteten Schranke (Proven, Derived) hält den Lauf mit einem Integritätsfehler
  an, denn dann irrt der Prüfer oder die Referenz; die Generation wird nicht
  festgeschrieben. Eine exakte Antwort unter einem besten bekannten Wert (BestKnown: 7 CNOT
  für Fredkin, 4 CNOT für CCiX) ist ein Erfindungskandidat; `rediscovery.json` hält den
  billigsten je Ziel fest, mit `literature_checked = false`. Neu nennen wir sie erst, wenn
  ein Mensch die Literatur geprüft hat (Q38).
- **Positivkontrolle** (`src/cascade/designer.rs`, `src/cascade/circuit_genes.rs`). Im
  Container schreibt kein Sprachmodell Schaltungen. Ein simulierter Entwerfer unter den
  Modellnamen `sim:small` und `sim:large` beantwortet die Schaltungs-Prompts (Abschnitt 5).
  Die Gene der Schaltungs-Zucht sind vier Startprogramme, 50 Satzbausteine, der Auftrag an
  den Mutator und die Aufnahmeregel `no_gate_lists`: Sie lehnt einen Abschnitt mit drei
  oder mehr Gatter-Nennungen wie „CNOT 0 1“ ab, sodass kein einzelner Abschnitt eine
  Schaltung ausschreibt. Über mehrere Abschnitte verteilt, je zwei Gatter, besteht eine
  Schaltung die Regel noch (`tests/circuits.rs`); dagegen steht die Verkleidung jeder
  Aufgabe.
- **Programm** (`src/bin/brut-breed.rs`). `--domain circuits` startet einen Schaltungslauf.
  Ein fertiger Lauf schreibt vor seinem Bericht `rediscovery.json`, die vorab festgelegte
  Zählung (Abschnitt 4). `--verify` rechnet Bericht und Zählung ohne Modellaufruf aus dem
  Checkpoint nach. `--promote` lehnt Schaltungsläufe mit Exit-Code 1 ab, denn die Beförderung
  von Schaltungs-Kaskaden gehört nicht zu Schritt 1.
- **Auswertung** (`examples/circuit_positive_control.rs`). Sie liest die fertigen Läufe
  beider Arme, paarweise nach Seed, bestimmt für jeden Lauf die Bestehenszeit und wendet
  die vorab festgelegte Regel über die Seeds an (Abschnitt 7). Nach jedem Paar nennt sie
  den nächsten Seed oder das Urteil. Eine Aussage trifft sie nur für die ersten Seeds der
  festen Liste 1 bis 100, dann 111 bis 130, ohne Lücke, höchstens 120 Paare, mit 1000
  Generationen und der Aufrufgrenze 13 000 000.
- **Tests.** `tests/circuits.rs` hat 18 Tests: Schaltungsläufe sind über Prozesse hinweg
  wiederholbar und bestehen `--verify`; Stopp und Fortsetzen ergeben dieselbe Abstammung
  und dieselben `state`-Zeilen wie ein durchgehender Lauf; `rediscovery.json` entsteht erst
  mit einer fertigen Auswahlphase; 1 und 4 Threads ergeben byte-gleiche Dateien;
  `--promote` wird abgelehnt; die Zeile zu D stimmt auch bei einem nicht kalibrierten Lauf;
  `rediscovery.json` gleicht einer Nachzählung aus den protokollierten Antworten; eine
  Änderung an der Datei lässt `--verify` scheitern, ebenso ein Mitglied des Archivs im
  Checkpoint, das in einem Feld seines `born`-Ereignisses abweicht, und ein Mitglied zu
  viel oder zu wenig; `--verify` nennt, warum es die Datei auslässt, und lehnt einen
  Bericht ab, den sein Checkpoint nicht nachrechnen kann; ein Quelltext-Scan zeigt, dass
  `designer.rs` die Bibliothek nicht kennt; die Aufnahmeregel zählt Gatter-Nennungen je
  Abschnitt; Tests legen die Kalibrierung fest und zwei Wege mit weniger Familien an der
  Obergrenze der Traits; dazu kommen die beiden Gegenproben mit `quantum.rs`.

Tabelle 1: die 11 Ziele mit geprüfter Referenz. Die vollständigen Quellen stehen je Eintrag
in `LIBRARY` in `src/cascade/circuit.rs`. „Bewiesen“ und „abgeleitet“ sind untere Schranken;
der beste bekannte Wert ist nur die beste veröffentlichte Konstruktion.

| Ziel | Qubits | Wirkung | T | CNOT | Quellen |
|---|---|---|---|---|---|
| CZ | 2 | diag(1, 1, 1, −1) | 0 (bewiesen) | 1 (bewiesen) | Amy, Maslov, Mosca, Roetteler 2013 |
| SWAP | 2 | tauscht zwei Qubits | 0 (bewiesen) | 3 (bewiesen) | Vatan, Williams 2004 |
| CS | 2 | diag(1, 1, 1, i) | 3 (bewiesen) | 2 (abgeleitet) | Amy et al. 2013; Gheorghiu, Mosca, Mukhopadhyay 2022 |
| CH | 2 | gesteuertes H | 2 (abgeleitet) | 1 (bewiesen) | Amy et al. 2013 |
| QFT2 | 2 | Quanten-Fourier-Transformation auf zwei Qubits | 3 (bewiesen) | 3 (bewiesen) | Gheorghiu et al. 2022; Shende, Bullock, Markov 2004 |
| CCZ | 3 | diag(1, …, 1, −1) | 7 (bewiesen) | 6 (bewiesen) | Gosset et al. 2014; Shende, Markov 2009 |
| Toffoli | 3 | kippt c, wenn a = b = 1 | 7 (bewiesen) | 6 (bewiesen) | Gosset et al. 2014; Shende, Markov 2009; Maslov 2016 |
| Peres | 3 | Toffoli, dann CNOT von a auf b | 7 (bewiesen) | 5 (bewiesen) | Shende, Markov 2009; Mosca, Mukhopadhyay 2022; Di Matteo, Mosca 2016 |
| Fredkin | 3 | a = 1 tauscht b und c | 7 (bewiesen) | 7 (bester bekannter Wert) | Gosset et al. 2014; Niroula, Nam 2021 |
| RTOF | 3 | Toffoli bis auf relative Phasen (Maslov) | 4 (abgeleitet) | 3 (bewiesen) | Maslov 2016 |
| CCiX | 3 | doppelt gesteuertes iX (Selinger) | 4 (abgeleitet) | 4 (bester bekannter Wert) | Selinger 2013; Maslov 2016 |

## 3. Ein Schaltungslauf und seine Prüfung

```sh
cargo build --release

# Schaltungslauf, Arm A: Eltern nach Darwin-Gödel (Pilot-Seed 101, 60 Generationen)
./target/release/brut-breed --out runs/c101-A --domain circuits --search 33 --select 22 \
  --children 8 --generations 60 --seed 101 --max-calls 200000

# Kontrollarm R: Eltern gleichverteilt
./target/release/brut-breed --out runs/c101-R --domain circuits --search 33 --select 22 \
  --children 8 --generations 60 --seed 101 --max-calls 200000 --selection uniform

# Protokoll, Bericht und rediscovery.json prüfen, ohne Modellaufruf
./target/release/brut-breed --out runs/c101-A --verify
./target/release/brut-breed --out runs/c101-R --verify

# Eingefrorene Auswertung; für Pilot-Seeds ohne Aussage
cargo run --example circuit_positive_control -- --a runs/c101-A --r runs/c101-R
```

Arm A wählt die Eltern wie die Darwin-Gödel-Maschine (Zhang et al., ICLR 2026, Anhang
C.2), Arm R gleichverteilt; sonst gleichen sich beide Arme. Die Seeds 1 bis 100 und 111
bis 130 sind dem vorab festgelegten Experiment vorbehalten; die Beispiele nutzen darum den
Pilot-Seed 101. Ein Lauf druckt je Generation eine Zeile (beste Kaskade, Fitness, gelöste
Such-Aufgaben, Archivgröße, Aufrufe), dann eine Zeile zu `rediscovery.json` und am Ende
die besten Kaskaden und den Vergleich auf dem Auswahl-Panel. Für Arm A auf Seed 101 lautet
die Zeile zu `rediscovery.json`:

```
brut-breed: wrote runs/c101-A/rediscovery.json: the seeds solve 2 of 11 targets, bred Cascades 7 of the other 9; D > 1/2: yes; calibrated
```

`--verify` meldet bei diesem Lauf „brut-breed: 1362 events, hash chain intact, head
e93ed521…; checkpoint at entry 1362 matches; settings match; report recomputed and equal;
rediscovery recomputed and equal“. Wir haben Arm A auf Seed 101 erneut laufen lassen:
`events.jsonl` und `rediscovery.json` sind byte-gleich mit dem Pilotlauf, und der Lauf
dauerte 3.2 s (Release-Build, 2 Threads).

**Was `--verify` prüft.** Wie in Stufe 2c prüft es die Hash-Kette, den Anker des
Checkpoints und die Einstellungen. Bei einem fertigen Lauf mit seiner `report.json` baut es
den Lauf aus dem Checkpoint neu auf, rechnet den Bericht nach und leitet den Stoppgrund aus
dem Checkpoint ab, statt ihn dem Bericht zu glauben („all generations done“ oder „call
budget reached“). Keine Prüfsumme deckt den Checkpoint, daher verlangt es bei einem
solchen Schaltungslauf zudem, dass das Archiv des Checkpoints je `born`-Ereignis des
Protokolls ein Mitglied hält, in derselben Reihenfolge, jedes mit allen Feldern, die sein
Ereignis festhält (Kennung, Kaskade, Eltern, Generation, Operator, Zahl der gelösten
Aufgaben, Summe der Aufgabenwerte, Verbrauch und Antworten), und dass `rediscovery.json`
Byte für Byte der Neuberechnung aus diesem Archiv gleicht. Was ein Mitglied in späteren
Generationen hinzugewinnt (Kinder, Urteile, Fitness, Peer-Wertung), steht nicht in seinem
`born`-Ereignis und wird nicht geprüft. Weicht das Archiv oder die Datei ab oder fehlt die
Datei, endet `--verify` mit Exit-Code 1 und einer Meldung „integrity: …“. Ebenso endet
es, wenn ein Schaltungslauf einen Bericht hat, den sein Checkpoint nicht nachrechnen kann,
weil der Checkpoint unfertig ist oder Antworten des Auswahl-Panels fehlen: Ein
Schaltungslauf behält alle, also wurde der Checkpoint gekürzt oder bearbeitet.
Sonst nennt es, warum es die Datei nicht geprüft hat: „rediscovery.json not checked: the
run is not finished“ oder, bei einem fertigen Lauf ohne seinen Bericht, „… there is no
report.json; resume the run to write it“. `timings.jsonl` gehört nicht zum geprüften
Protokoll.

**Einstellungen.** Ohne `--config` nimmt `--domain circuits` alle 11 Ziele, verkleidet, mit
höchstens 64 Gattern je Antwort. Andere Schaltungs-Einstellungen kommen über
`--config <datei>`: eine vollständige Laufkonfiguration im Format von `config.json`, deren
Feld `domain` etwa `{"circuits": {"targets": ["cz", "ccz"], "dress": true, "max_gates": 32}}`
lautet. Die Gatter-Obergrenze darf sinken, aber nie über 256 steigen; ein unbekanntes oder
ungeprüftes Ziel lehnt der Lauf ab. Fortsetzen, Not-Aus, Sperre und Exit-Codes funktionieren
wie in Stufe 2c ([STUFE2C_KASKADEN_ZUCHT.md](STUFE2C_KASKADEN_ZUCHT.md), Abschnitt 5).
`--domain` und `--selection` legen das Experiment fest; ein Fortsetzen lehnt sie ab. Ein
Fortsetzen auf mehr Generationen entfernt `report.json` und `rediscovery.json`, bis die
Auswahlphase wieder fertig ist.

## 4. Was die Kennzahlen bedeuten

**Die `state`-Zeile.** Nach jeder festgeschriebenen Generation folgt auf deren
Zusammenfassung genau eine `state`-Zeile mit sechs Gruppen:

| Gruppe | Felder | Bedeutung |
|---|---|---|
| Leistung | `solved_best`, `solved_median`, `solved_best_born`, `fitness_best`, `fitness_median`, `quality_best`, `quality_median` | beste und mittlere Lösungsquote auf dem Such-Panel, die beste der neu Geborenen, die Fitness; `quality` ist der mittlere Aufgabenwert und erscheint nur bei Schaltungen |
| Vielfalt | `shapes`, `blocks`, `behaviours`, Entropien in Bit | Bauformen (Breiten der Stufen, etwa „1-2-1“), Bausteine (Paare aus Abschnittsname und Text), verschiedene Antwortlisten, Entropie der Modellnamen; `parent_entropy_bits` ist die Entropie der Elternwahl der nächsten Generation, 2 hoch diesem Wert die effektive Zahl der Eltern |
| Abstammung | `depth_max`, `depth_mean`, `branching_mean`, `branching_max`, `leaf_share`, `seed_lineages_alive`, `surviving_share` | Tiefe des Stammbaums (Startprogramme haben Tiefe 0), Verzweigung, Anteil ohne Kinder, Startprogramme mit Nachkommen in dieser Generation |
| Ehrlichkeit | `fakes_planted`, `fakes_caught`, `false_alarms`, `confirmed_collusion` | bleibt 0, bis es Aufseher gibt (Ticket 05) |
| Richter | `comparisons_new`, `scored_new`, `accuracy_new`, `skill`, `trusted_share`, `vote_weight_mean` | Vergleiche dieser Generation, davon gewertete und richtig entschiedene; Quantile der Urteilskraft; Anteil mit Urteilskraft über 1/2 |
| Kosten | `calls`, `prompt_tokens`, `completion_tokens`, `model_cost`, `think_tokens_mean`, `workers_mean`, `stages_mean` | Verbrauch dieser Generation (Neugeborene auf dem Such-Panel und Peer-Runde) und Mittel über das Archiv |

Die Zeile hängt nur vom festgeschriebenen Zustand ab, nie von kumulierten Aufrufzahlen,
welche auch Rücklagen abgebrochener Versuche enthalten. Darum schreibt ein gestoppter und
fortgesetzter Lauf dieselben Zeilen wie ein durchgehender. Ein Median ist der untere Median.
Undefinierte Werte fehlen, statt NaN zu sein. Die Laufzeit je Generation steht in
`timings.jsonl`, außerhalb der Hash-Kette.

Ein Beispiel aus den Pilotläufen (`pilot.json`, jede zehnte Generation): In Arm A auf
Seed 101 löst die beste Kaskade 3 der 33 Such-Aufgaben in Generation 0, 9 in Generation 20,
24 in Generation 40 und 25 in Generation 60; die Zahl der verschiedenen Antwortlisten
steigt von 4 auf 30. `parent_entropy_bits` steigt dort von 1.66 auf 8.43 Bit. In Arm R ist
dieser Wert stets der Zweierlogarithmus der Archivgröße, von 2 Bit bei 4 Startprogrammen bis
8.92 Bit bei 484 Kaskaden, wie es die gleichverteilte Elternwahl ergibt.

**`rediscovery.json`.** Für jedes Ziel (`targets`) stehen dort die Referenz mit ihrem
Status, die Zahl der Such-Aufgaben des Ziels (`tasks`), die Kosten der billigsten exakten
Antwort der Startprogramme und der gezüchteten Kaskaden (`best_seed`, `best_bred`: T, CNOT,
H, Gatter, Tiefe, T-Tiefe), ob Startprogramme und gezüchtete Kaskaden die Referenz erreicht
haben (`reached_by_seed`, `reached_by_bred`), die früheste Generation einer gezüchteten
Kaskade, die das Ziel löste (`first_generation`), der Text der billigsten exakten
gezüchteten Antwort (`circuit`; er setzt die verkleidete Aufgabe um, nicht das Ziel in
seiner Grundform) und der billigste Erfindungskandidat, falls es einen gibt
(`below_best_known`, `candidate`). Darunter stehen |L| (`library`), |R_seed|
(`reached_by_seeds`), |R_bred \ R_seed| (`reached_by_bred_only`), der Anteil D (`share`),
die Schranke 3 · |R_seed| ≤ |L| (`calibrated`) und `passes` (kalibriert und D > 1/2). Die
Begriffe definiert die Vorab-Registrierung, Abschnitt 7.

**`report.json`.** Wie in Stufe 2c: die besten Kaskaden mit ihren gelösten Such-Aufgaben,
die Finalisten mit ihren gelösten Auswahl-Aufgaben (bei Schaltungen dazu `select_quality`,
die Summe der Aufgabenwerte) und die besten Quoten der Finalisten und der Startprogramme auf
dem Auswahl-Panel.

## 5. Kalibrierung der Positivkontrolle

**Der Entwerfer.** Er liest nur die Anfrage. Im Systemprompt schalten Wortstämme auf Deutsch
und Englisch acht Methodenfamilien frei; die Reihenfolge ihrer ersten Nennung ist die
Reihenfolge der Methoden. Linear baut CNOT-Netze für affine Permutationen, Phase baut
Phasenpolynome (nach Amy und Mosca 2019), Hadamard wechselt auf einem Teil der Qubits die
Basis, Compose rahmt das Ziel mit einem CNOT ein, Search durchsucht Gatterfolgen in der
Breite, Meet sucht von beiden Enden (Meet in the Middle nach Amy et al. 2013), Simplify
kürzt und fasst zusammen, Previous prüft und übernimmt die Antworten früherer Stufen. Er
kennt keine Schaltung, keine Vorlage und keine Zielliste; ein Quelltext-Scan in
`tests/circuits.rs` prüft, dass `designer.rs` die Bibliothek nicht nennt. Jede Arbeit kostet
Schritte, die er bezahlt, bevor er sie tut: 16 freie Schritte je Aufruf und einen Schritt je
4 Denk-Tokens bei `sim:small`, je 2 bei `sim:large`, höchstens 2^13 = 8 192 Schritte je
Aufruf. Bei 4096 Denk-Tokens, der Obergrenze der Traits, hat ein Aufruf höchstens
2 064 Schritte; die Obergrenze von 8 192 Schritten greift also nur bei direkten Anfragen.
Mit Selbstprüfung prüft er einen Kandidaten auf allen Basiszuständen, ohne Selbstprüfung
nur auf dem ersten und dem letzten, sodass ihn eine billige falsche Schaltung täuschen
kann. Er antwortet mit dem billigsten geprüften
Kandidaten nach T, dann CNOT, dann Gatterzahl. Alle Konstanten stehen in der
Vorab-Registrierung, Abschnitt 4.

**Startprogramme.** Jedes ist ein `sim:small`-Worker mit 256 Denk-Tokens, Temperatur 0.2,
512 Antwort-Tokens und ohne Selbstprüfung; nur seine Strategie nennt eine Familie (linear,
phase, hadamard, search). Auf den 33 Such-Aufgaben, welche Seed 101 zieht, löst das lineare
Startprogramm SWAP, das Phasen-Startprogramm CS und die beiden anderen nichts
(`designer_calibration_is_pinned` in `tests/circuits.rs`). Also ist R_seed = {SWAP, CS},
2 von 11 Zielen, und 3 · 2 = 6 ≤ 11 erfüllt die Schranke. Dasselbe gilt in jeder der 32
Verkleidungen je Ziel auf zwei Qubits und 384 auf drei (`circuit_genes.rs`).

**Starker Prompt.** Ein `sim:large`-Worker mit 4096 Denk-Tokens, Temperatur 0 und
Selbstprüfung, dessen Strategie Hadamard, Phase, Linear, Compose, Simplify, Search und Meet
in dieser Reihenfolge nennt, löst alle 33 Aufgaben zur Referenz. Kein Ziel ist also
unerreichbar, und L umfasst alle 11 Ziele. Der Prompt besteht die Regeln des Genoms und die
Aufnahmeregel. Er ist ein von Hand geschriebener Text, den ein Lauf ohne LLM-Runden nicht
schreiben kann; Kaskaden aus Satzbausteinen erreichen aber jedes Ziel zur Referenz,
unverkleidet und in einer Verkleidung
(`pool_phrases_reach_every_target_at_its_best_known_cost` in `circuit_genes.rs`).

**Wege zu den Zielen.** Die Tests in `designer.rs` legen für die genannten Familien fest,
wie viele Denk-Tokens die Antwort mindestens braucht; ein Denk-Token weniger verfehlt sie.

| Ziel | genannte Familien | Selbstprüfung | Modell | Denk-Tokens mindestens | exakte Antwort (T, CNOT) |
|---|---|---|---|---|---|
| SWAP | Linear | aus | `sim:small` | 0 | (0, 3), gelöst |
| CS | Phase | aus | `sim:small` | 0 | (3, 2), gelöst |
| CCZ | Phase | aus | `sim:small` | 8 | (7, 10), nicht gelöst |
| CCZ | Phase, Linear | aus | `sim:small` | 196 | (7, 6), gelöst |
| CZ | Search | an | `sim:large` | 104 | (0, 1), gelöst |
| RTOF, CCiX, Peres, Toffoli | Hadamard, Phase, Linear | an | `sim:large` | 78, 92, 124, 158 | Referenz |
| Fredkin | Hadamard, Phase, Linear, Compose | an | `sim:large` | 654 | (7, 8), nicht gelöst |
| Fredkin | Compose, Hadamard, Phase, Linear, Simplify | an | `sim:large` | 1 566 | (7, 7), gelöst |
| CH | Search, Meet | an | `sim:large` | 3 124 | (2, 1), gelöst |

Bei 4096 Denk-Tokens auf `sim:large` lösen außerdem Hadamard und Linear CZ, Hadamard, Phase
und Linear QFT2 und Meet allein CH; Phase allein liefert für CZ nur (0, 2).

**Satzbausteine.** Die 50 Bausteine verteilen sich auf die Abschnitte `strategy` (25),
`residual` (9), `style` (6), `check` (5), `role` (3) und `format` (2); das haben wir aus
dem Quelltext gezählt. Kein Baustein nennt zwei von Hadamard, Phase und Linear. Ein
Startprogramm kennt höchstens eine dieser drei Familien (das Such-Startprogramm keine), ein
zufälliger Schritt mit einem Baustein lehrt es höchstens eine weitere. Von den 704 Kindern,
welche ein solcher Schritt aus den Startprogrammen macht, lösen 40 ein Ziel über SWAP und
CS hinaus (20 CZ, 20 CCZ) und keines zwei. QFT2, Toffoli, Peres, RTOF und CCiX brauchen
darum mindestens zwei Schritte: Die 60 Kinder, deren letzter Worker zwei der drei kennt,
haben 384 Kinder, die alle drei kennen, und jedes davon löst mit den Traits der
Startprogramme CZ, QFT2, Toffoli, Peres, RTOF und CCiX. Diese Zahlen legen die Tests in
`circuit_genes.rs` fest, je Ziel unverkleidet und in drei Verkleidungen. Fredkin verfehlen
die drei allein; erst mit Compose fällt es (`designer.rs`). Mit den meisten Denk-Tokens,
dem größeren Modell und Selbstprüfung genügen für zwei dieser Ziele weniger Familien: Bei
4096 Denk-Tokens auf `sim:large` mit Selbstprüfung löst Meet allein QFT2 in 4 seiner 32
Verkleidungen, und Hadamard, Phase und Simplify lösen CCiX in 128 seiner 384
(`tests/circuits.rs`). Ein solcher Worker unterscheidet sich von jedem Startprogramm in
Modell, Denkbudget und Selbstprüfung, liegt also ebenfalls mehrere Schritte entfernt.

## 6. Pilotläufe

Pilot-Seeds zählen nie. Die ersten waren die Seeds 101 und 102; vier spätere Reihen folgen
unten (Tabelle 4). Im ersten Pilot haben wir beide Arme auf beiden Seeds 60 Generationen
lang laufen lassen, mit `--search 33 --select 22 --children 8`, Release-Build, rustc
1.97.0, 2 Threads. Das Programm des Commits, der diesen Bericht enthält, wiederholt jeden
dieser Pilotläufe byte-gleich (siehe Wiederholbarkeit unten). Die voreingestellte Grenze
von 20 000 Aufrufen beendete Arm A auf Seed 101 nach Generation 20 (18 451 Aufrufe); darum
nutzen alle vier Läufe `--max-calls 200000`. Danach haben wir jeden Lauf mit 200
Generationen wiederholt. Alle Zahlen stammen aus [pilot.json](pilot-step1/pilot.json); wir
haben alle neun Lauf-Verzeichnisse erneut mit `--verify` geprüft und die Auswertung erneut
ausgeführt.

Tabelle 2: Pilotläufe mit 60 Generationen.

| Arm | Seed | Aufrufe | Laufzeit | R_seed | D | erste gezüchtete Lösung je Ziel (Generation) |
|---|---|---|---|---|---|---|
| A | 101 | 60 140 | 2.96 s | SWAP, CS | 7/9 = 0.778 | CZ 1; QFT2, Toffoli, Peres, RTOF, CCiX 32; CCZ 55 |
| A | 102 | 53 996 | 2.29 s | SWAP, CS | 2/9 = 0.222 | CCZ 6; CZ 29 |
| R | 101 | 49 716 | 5.77 s | SWAP, CS | 1/9 = 0.111 | CZ 35 |
| R | 102 | 40 484 | 2.37 s | SWAP, CS | 2/9 = 0.222 | CZ 26; CCZ 40 |

Tabelle 3: D nach Generationen in den Läufen mit 200 Generationen (`--max-calls 1000000`).
Jeder dieser Läufe wiederholt seinen Lauf mit 60 Generationen bis Generation 60 genau, da nur
die Stopp-Prüfung `--generations` liest.

| Arm | Seed | Gen. 30 | Gen. 40 | Gen. 60 | Gen. 100 | Gen. 200 | Aufrufe | Laufzeit |
|---|---|---|---|---|---|---|---|---|
| A | 101 | 1/9 | 6/9 | 7/9 | 7/9 | 7/9 | 214 049 | 20.9 s |
| A | 102 | 2/9 | 2/9 | 2/9 | 2/9 | 2/9 | 193 928 | 16.8 s |
| R | 101 | 0/9 | 1/9 | 1/9 | 2/9 | 2/9 | 178 684 | 25.9 s |
| R | 102 | 1/9 | 2/9 | 2/9 | 2/9 | 2/9 | 153 624 | 12.6 s |

- **Kalibrierung.** In allen Läufen lösen die Startprogramme genau SWAP und CS; die Schranke
  hält. Alle Läufe bestehen `--verify`; keiner meldet einen Integritätsalarm oder einen
  Erfindungskandidaten.
- **Erste Lösungen.** In diesen Läufen kam jede erste Lösung bis Generation 90, keine in den
  Generationen 91 bis 200. CH und Fredkin löste kein Lauf.
- **Engpass.** In Arm A auf Seed 101 fielen fünf Ziele in Generation 32, der ersten, in der
  ein Worker Hadamard, Phase und Linear zugleich nannte. In den drei anderen Läufen entstand
  in 200 Generationen kein solcher Worker. Diese Generationen (`first_worker_naming` in
  `pilot.json`) haben wir außerhalb des Programms ermittelt, durch eine Suche nach den
  Wortstämmen in den Abschnittstexten.
- **Auswertung.** Für die vier Läufe mit 60 Generationen traf die Auswertung in ihrer ersten
  Fassung, mit der Regel „9 von 10“, keine Aussage, da nur zwei Seed-Paare vorlagen.
  D_A > 1/2 gilt auf einem von zwei Seeds (Median 1/2), D_A > D_R in einem von zwei Paaren.
  In der Variante nur nach T lösen die Startprogramme zusätzlich CCZ (7 T, 10 CNOT); dann
  ist D_A 6/8 und 1/8, D_R 3/8 und 3/8.
- **Auswahl-Panel.** In Arm A auf Seed 101 löst der beste Finalist 16 der 22
  Auswahl-Aufgaben, das beste Startprogramm 2. In den drei anderen Läufen löst der beste
  Finalist 4 (`report.json` der Pilotläufe).
- **Wiederholbarkeit.** Wir haben Arm A auf Seed 101 erneut laufen lassen; `events.jsonl` und
  `rediscovery.json` sind byte-gleich mit dem Pilotlauf. Nach den Korrekturen aus der
  Prüfung, die nur `--verify`, eine gedruckte Zeile, die Auswertung und Tests ändern, haben
  wir alle neun Pilotläufe mit dem Programm des Commits wiederholt, der diesen Bericht
  enthält: `config.json`, `events.jsonl`, `snapshot.json`, `report.json` und
  `rediscovery.json` sind byte-gleich mit dem Pilot, und alle neun bestehen sein
  `--verify`.

**Spätere Pilotreihen.** Nach dem ersten Pilot haben wir vier weitere Reihen laufen lassen,
beide Arme mit den Einstellungen der gezählten Läufe außer Seed, Generationen und
Aufrufgrenze (Tabelle 4). Aus ihren Daten entstand die Regel des gezählten Experiments;
darum zählen ihre Seeds nie. Die Läufe liegen nicht im Repository:
`.scratch/universum-erfinden/sequential-rule/pilot.json` hält die aus ihnen gelesenen
Werte, und [pilot-step1/](pilot-step1/README.md) nennt die Bestehenszeit jedes Laufs. Das
ist die erste Generation, in der die gezüchteten Kaskaden mehr als die Hälfte der Ziele
lösen, welche die Startprogramme verfehlen, hier 5 der 9.

Tabelle 4: spätere Pilotreihen. „Besteht“ heißt hier: bis zur letzten Generation des Laufs;
in Klammern die Bestehenszeiten.

| Seeds | Generationen | `--max-calls` | Arm A besteht | Arm R besteht |
|---|---|---|---|---|
| 101 bis 110 | 200 | 2 600 000 | 1 von 10 (32) | 1 von 10 (48) |
| 201 bis 210 | 200 | 2 600 000 | 5 von 10 (78 bis 177) | 2 von 10 (118, 149) |
| 301 bis 309 | 1000 | 13 000 000 | 8 von 9 (7 bis 810) | 1 von 9 (380) |
| 401, 402 | 2000 | 26 000 000 | 1 von 2 (10) | 1 von 1 (720) |

- **Unvollständig.** Der Lauf A-310 brach nach Generation 922 ab; er ist unvollständig und
  fehlt in allen Zahlen. R-310 und R-402 liefen nicht.
- **Kalibrierung.** In allen 61 vollständigen Läufen lösen die Startprogramme genau SWAP und
  CS, die Schranke hält, und der Lauf endete mit allen Generationen. Keiner meldete einen
  Integritätsalarm oder einen Erfindungskandidaten, und keiner löste CH oder Fredkin.
- **Späte Erfolge.** Bis Generation 200 bestand Arm A auf 6 der 20 Seeds 101 bis 110 und
  201 bis 210, bis Generation 1000 in 9 der 11 Läufe, die so weit liefen. Erfolge kamen in
  Arm A noch in Generation 810 (Seed 309), in Arm R in Generation 720 (Seed 401). Dass nach
  Generation 90 kein Ziel mehr hinzukommt, wie der erste Pilot nahelegte, gilt also nicht.
- **Auswahl.** In den 10 Paaren, deren Arme beide bis Generation 1000 liefen, bestand Arm A
  8-mal zuerst, Arm R einmal (Seed 401), und einmal bestand keiner (Seed 301).
- **Laufzeit.** Ein Lauf bis Generation 1000 dauerte bei 2 Threads in Arm A 6.01 bis
  8.57 min (Mittel 7.38 min, 11 Läufe) und in Arm R 4.54 bis 6.12 min (Mittel 5.38 min,
  10 Läufe), ein Paar 10.57 bis 14.31 min (Mittel 12.68 min, 10 Paare).

## 7. Das gezählte Experiment

**Ergebnis.** Das gezählte Experiment lief vom 2026-10-30 um 22:32 UTC bis zum 2026-10-31
um 12:44 UTC, nach der eingefrorenen Vorab-Registrierung. Die Regel hielt nach 27
Seed-Paaren an. Der Hauptnachweis (Ticket 03) ist bei Paar 15 erbracht. Er besagt, dass mehr
als die Hälfte aller Seeds bis Generation 1000 besteht. Der zweite Nachweis ist bei Paar 27
erbracht. Er besagt, dass Arm A unter den Paaren ohne Gleichstand öfter zuerst besteht als
Arm R. Beide Nachweise betreffen den Züchter als Positivkontrolle mit dem simulierten
Entwerfer; über echte Sprachmodelle sagen sie nichts. Die Daten und Skripte liegen in
[step1-counted/](step1-counted/README.md).

Aus den Pilotläufen folgen die Einstellungen der gezählten Läufe: `--generations 1000` und
`--max-calls 13000000`. Bei einer Quote von 0.30, wie bis Generation 200, scheitert die
Regel unten mit Wahrscheinlichkeit 1.000, nach 9.1 Paaren im Mittel; die Obergrenze konnte
also nicht bei 200 Generationen bleiben. 1000 ist der längste Horizont, den die meisten
Pilotläufe in beiden Armen erreichten; nur drei liefen bis 2000 weiter. Nach unserer
Rechnung aus den Aufrufgrenzen in `arena.rs` braucht ein Lauf mit 1000 Generationen
höchstens 12 723 412 Aufrufe, sodass die Grenze keinen gezählten Lauf vorzeitig beendet;
die Pilotläufe brauchten bis Generation 1000 höchstens 1 161 738 Aufrufe.

**Die Regel.** Das Experiment entscheidet selbst, wie viele Seeds es braucht. Die Regel,
ihre Begründung und ihre Quellen stehen in der Vorab-Registrierung, Abschnitt 8. Die
gezählten Seeds sind die feste Liste 1 bis 100, dann 111 bis 130; sie lässt die
Pilot-Seeds 101 bis 110 aus. Sie laufen in dieser Reihenfolge, je in beiden Armen und
stets bis Generation 1000. Ein Lauf besteht, wenn seine Bestehenszeit T höchstens 1000
ist. Nach jedem Paar ändern sich zwei Belegwerte, die bei 1 beginnen:

- **E1.** Es wächst um den Faktor 3/2, wenn Arm A besteht; sonst halbiert es sich.
- **E2.** Es wächst um den Faktor 3/2, wenn Arm A zuerst besteht, und halbiert sich, wenn
  Arm R zuerst besteht; bei Gleichstand bleibt es gleich.

Erreicht E1 den Wert 100, ist der Hauptnachweis erbracht: Mehr als die Hälfte aller Seeds
besteht bis Generation 1000. Dafür braucht es mindestens 12 Paare. Fällt E1 auf 1/20, ist
er gescheitert. Für E2 gelten dieselben Schwellen. Den zweiten Nachweis, dass Arm A unter
den Paaren ohne Gleichstand öfter zuerst besteht als Arm R, beanspruchen wir aber nur,
wenn der Hauptnachweis erbracht ist. Das Experiment endet, sobald der Hauptnachweis
scheitert, sobald er erbracht ist und E2 eine Schwelle erreicht hat, oder nach 120 Paaren.
Gleich wann es endet, meldet es einen falschen Nachweis höchstens mit Wahrscheinlichkeit
1/100; für den Hauptnachweis sind es im ungünstigsten Fall rund 0.0079. Die alte Regel
„9 von 10“ lag bei 11/1024 = 0.0107; die neue Schranke ist also etwas strenger. Sie gilt
nur, wenn die Liste der Seeds und ihre Reihenfolge fest sind, kein Seed der Liste
ausgelassen oder ersetzt wird und sich nach Seed 1 nichts ändert.

**Durchführung.** Wir haben das Programm beim eingefrorenen Commit
`15cb15100a39db9d815674b0778ab6be30932e56` gebaut, mit rustc 1.97.0 (2d8144b78 2026-07-07)
auf x86_64 Linux (Ubuntu 24.04.5 LTS). Dieser Commit entstand am 2026-10-30 um 21:50 UTC
beim Zusammenführen von Pull-Request #19. Dort bestanden `cargo test --lib` (610 Tests;
keiner der 7 ausgelassenen ist ein Kalibrierungstest), `cargo test --test circuits` (18) und
`cargo test --example circuit_positive_control` (31). Vor Seed 1 ergab die Abnahmeprüfung
auf den Pilotpaaren 301 bis 309 genau die Bestehenszeiten und Belegwerte, welche die
Vorab-Registrierung in Abschnitt 6, Schritt 2 nennt, in der Auswertung wie in
`reference_rule.py`; die Auswertung traf dort, wie verlangt, keine Aussage. Dann ließ ein
Treiber-Skript ([step1-driver.sh](step1-counted/step1-driver.sh)) die Paare in der
Reihenfolge der Liste laufen, einen Prozess nach dem anderen. Nach jedem Paar prüfte es
beide Läufe mit `--verify`, hielt die Prüfsummen ihrer fünf Dateien fest und rief die
eingefrorene Auswertung auf, die den nächsten Seed nannte. Gleich nach Paar 1 lief Seed 1
in beiden Armen noch einmal mit 1 und mit 4 Threads. Für die gezählten Läufe setzte der
Treiber `RAYON_NUM_THREADS` nicht; rayon nimmt dann einen Thread je CPU, und der Container
meldete nach dem Experiment 4 CPUs.

**Urteile.**

- **Hauptnachweis (Ticket 03): erbracht bei Paar 15.** Nach 15 Paaren mit 14 Erfolgen von
  Arm A erreichte E1 = 3^14/2^15 = 145.9646 erstmals 100. Beim Stopp nach 27 Paaren ist
  E1 = 3^24/2^27 = 2104.2640; Arm A bestand auf 24 der 27 Seeds.
- **Zweiter Nachweis (Auswahl): erbracht bei Paar 27.** Beim Stopp ist
  E2 = 3^20/2^25 = 103.9143. Unter den 25 Paaren ohne Gleichstand bestand Arm A 20-mal
  zuerst und Arm R 5-mal (Paare 5, 9, 11, 20 und 25). In den Paaren 10 und 19 bestand
  keiner der beiden Läufe; sie zählen als Gleichstand. Der zweite Nachweis zählt, weil der
  Hauptnachweis erbracht ist; so verlangt es die feste Reihenfolge aus Abschnitt 8 der
  Vorab-Registrierung.

Damit treffen wir die vorab festgelegte Aussage (Vorab-Registrierung, Abschnitt 11): Die
Zucht setzt Prompts und Traits zusammen, die für die meisten Bibliotheksziele, welche die
Startprogramme verfehlen, exakte Schaltungen zu den Referenzzahlen freischalten, und das
in mehr als der Hälfte aller Seeds bis Generation 1000. Die Darwin-Gödel-Auswahl kommt
dorthin früher als eine gleichverteilte Elternwahl mit gleich vielen Kindern: Unter den
Paaren ohne Gleichstand besteht Arm A öfter zuerst als Arm R. Beides sind Aussagen über den
Züchter als Positivkontrolle, nicht über echte Modelle und nicht über Erfindungen.

**Stopp.** Die Regel hielt nach Paar 27 an, nach der zweiten ihrer drei Stopp-Regeln: Der
Hauptnachweis war erbracht, und E2 hatte eine Schwelle erreicht. Nach Paar 26 stand E2 bei
69.2762. In Paar 27 bestand Arm A in Generation 126 und Arm R nie, sodass E2 um den Faktor
3/2 auf 103.9143 stieg. Nach dem Stopp lief kein Paar mehr. Wir berichten beide Belegwerte
beim Stopp, nicht ihre größten Werte unterwegs.

**Je Paar.** Tabelle 5 gibt, was Abschnitt 8 der Vorab-Registrierung je Paar verlangt.
Paar i hat den Seed i. D ist der Anteil der 9 Ziele, welche die Startprogramme verfehlen,
den gezüchtete Kaskaden bis Generation 1000 lösen. „nie“ heißt: keine Bestehenszeit bis
Generation 1000. X ist 1, wenn Arm A besteht, sonst 0. Y nennt den Arm, der zuerst bestand.
E1 und E2 gelten nach dem Paar, auf vier Nachkommastellen gerundet; ihre Brüche stehen in
[analysis.txt](step1-counted/analysis.txt).

Tabelle 5: die 27 gezählten Paare.

| Paar | R_seed | D_A | D_R | T_A | T_R | X | Y | E1 | E2 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | SWAP, CS | 7/9 | 2/9 | 574 | nie | 1 | A zuerst | 1.5000 | 1.5000 |
| 2 | SWAP, CS | 7/9 | 6/9 | 199 | 376 | 1 | A zuerst | 2.2500 | 2.2500 |
| 3 | SWAP, CS | 7/9 | 7/9 | 278 | 665 | 1 | A zuerst | 3.3750 | 3.3750 |
| 4 | SWAP, CS | 7/9 | 2/9 | 65 | nie | 1 | A zuerst | 5.0625 | 5.0625 |
| 5 | SWAP, CS | 7/9 | 6/9 | 227 | 61 | 1 | R zuerst | 7.5938 | 2.5312 |
| 6 | SWAP, CS | 7/9 | 2/9 | 296 | nie | 1 | A zuerst | 11.3906 | 3.7969 |
| 7 | SWAP, CS | 8/9 | 3/9 | 557 | nie | 1 | A zuerst | 17.0859 | 5.6953 |
| 8 | SWAP, CS | 7/9 | 2/9 | 211 | nie | 1 | A zuerst | 25.6289 | 8.5430 |
| 9 | SWAP, CS | 7/9 | 7/9 | 184 | 126 | 1 | R zuerst | 38.4434 | 4.2715 |
| 10 | SWAP, CS | 2/9 | 2/9 | nie | nie | 0 | gleich | 19.2217 | 4.2715 |
| 11 | SWAP, CS | 7/9 | 7/9 | 56 | 18 | 1 | R zuerst | 28.8325 | 2.1357 |
| 12 | SWAP, CS | 7/9 | 7/9 | 107 | 916 | 1 | A zuerst | 43.2488 | 3.2036 |
| 13 | SWAP, CS | 7/9 | 2/9 | 47 | nie | 1 | A zuerst | 64.8732 | 4.8054 |
| 14 | SWAP, CS | 7/9 | 7/9 | 20 | 220 | 1 | A zuerst | 97.3098 | 7.2081 |
| 15 | SWAP, CS | 7/9 | 2/9 | 28 | nie | 1 | A zuerst | 145.9646 | 10.8122 |
| 16 | SWAP, CS | 7/9 | 3/9 | 610 | nie | 1 | A zuerst | 218.9469 | 16.2183 |
| 17 | SWAP, CS | 7/9 | 7/9 | 204 | 618 | 1 | A zuerst | 328.4204 | 24.3274 |
| 18 | SWAP, CS | 7/9 | 2/9 | 62 | nie | 1 | A zuerst | 492.6306 | 36.4912 |
| 19 | SWAP, CS | 3/9 | 3/9 | nie | nie | 0 | gleich | 246.3153 | 36.4912 |
| 20 | SWAP, CS | 3/9 | 7/9 | nie | 116 | 0 | R zuerst | 123.1577 | 18.2456 |
| 21 | SWAP, CS | 7/9 | 2/9 | 225 | nie | 1 | A zuerst | 184.7365 | 27.3684 |
| 22 | SWAP, CS | 7/9 | 2/9 | 304 | nie | 1 | A zuerst | 277.1047 | 41.0526 |
| 23 | SWAP, CS | 7/9 | 7/9 | 158 | 182 | 1 | A zuerst | 415.6571 | 61.5788 |
| 24 | SWAP, CS | 7/9 | 7/9 | 23 | 37 | 1 | A zuerst | 623.4856 | 92.3682 |
| 25 | SWAP, CS | 7/9 | 6/9 | 629 | 439 | 1 | R zuerst | 935.2285 | 46.1841 |
| 26 | SWAP, CS | 7/9 | 2/9 | 923 | nie | 1 | A zuerst | 1402.8427 | 69.2762 |
| 27 | SWAP, CS | 7/9 | 2/9 | 126 | nie | 1 | A zuerst | 2104.2640 | 103.9143 |

**Fehlerschranken.** Nach Abschnitt 8 der Vorab-Registrierung gilt, gleich wann das
Experiment endet: Jeder der beiden Nachweise wird höchstens mit Wahrscheinlichkeit 1/100
fälschlich erbracht, und irgendein Nachweis ebenfalls höchstens mit 1/100. Das folgt aus
Villes Ungleichung für Test-Supermartingale und aus der festen Reihenfolge der beiden
Nachweise; für den Hauptnachweis liegt der ungünstigste Fall bei 0.0079. Die Schranken
verlangen, dass die Liste der Seeds und ihre Reihenfolge fest sind, kein Seed der Liste
ausgelassen oder ersetzt wird, die Belegwerte der Reihenfolge der Liste folgen und sich nach
Seed 1 nichts ändert. Wir haben diese Bedingungen eingehalten. Die Seeds 1 bis 27 liefen in
dieser Reihenfolge. Alle 54 `config.json` haben genau die Einstellungen aus Abschnitt 6 der
Vorab-Registrierung, mit dem Seed ihres Paars und in Arm R der gleichverteilten Elternwahl.
Der Treiber lief bei beiden Starts auf dem eingefrorenen Commit. A-12 lief nach einer
Unterbrechung auf demselben Seed von vorn (Abweichung 1, unten). Nicht prüfen lässt sich
die Annahme, dass die Seeds wie unabhängige Ziehungen wirken. Wir nehmen sie an, da die
Zufallsströme eines Laufs aus einem Hash seines Seeds folgen.

**Nachzählung.** Vor diesem Bericht hat `reference_rule.py`, eine unabhängige Umsetzung der
Regel, das Ergebnis aus den Lauf-Verzeichnissen nachgezählt
([recount.txt](step1-counted/recount.txt)). Es ergab dieselben 27 Zeilen (T_A, T_R, X, Y,
E1 und E2), denselben Stopp nach Paar 27 und dieselben Urteile wie die eingefrorene
Auswertung; einen Unterschied gibt es nicht. Ein drittes Skript,
[make_counted.py](step1-counted/make_counted.py), rechnet diese Zahlen aus
`rediscovery.json` noch einmal nach, vergleicht sie mit beiden Ausgaben und mit den
festgehaltenen Prüfsummen und schreibt [counted.json](step1-counted/counted.json). Es bricht
bei jedem Unterschied ab; auf den 54 Läufen lief es durch. Alle 270 Prüfsummen (27 Paare,
2 Arme, 5 Dateien) stimmen mit den Dateien überein.

**Ungültigkeitsgründe.** Keiner der sechs Gründe aus Abschnitt 9 der Vorab-Registrierung
trifft zu:

1. Kein gezählter Lauf meldete einen Integritätsalarm. Alle 54 endeten mit Exit-Code 0 und
   schrieben `rediscovery.json`.
2. Die Thread-Prüfung fand kein abweichendes Byte. `config.json`, `events.jsonl`,
   `snapshot.json`, `report.json` und `rediscovery.json` von Seed 1 sind in beiden Armen
   bei 1 und bei 4 Threads gleich denen der gezählten Läufe. Wir haben die 20 Dateien nach
   dem Experiment noch einmal verglichen.
3. Alle Kalibrierungstests aus Abschnitt 5 der Vorab-Registrierung bestanden beim
   eingefrorenen Commit (Durchführung, oben).
4. Jedes Paar erfüllt die Schranke 3 · |R_seed| ≤ |L|: In allen 54 Läufen lösen die
   Startprogramme genau SWAP und CS, und 3 · 2 = 6 ≤ 11.
5. Alle 54 gezählten Läufe bestanden `--verify`; die letzte Zeile endete jeweils auf
   „report recomputed and equal; rediscovery recomputed and equal“. Dasselbe gilt für die
   vier Läufe der Thread-Prüfung.
6. Alle 54 Läufe haben 1000 Generationen und die Aufrufgrenze 13 000 000, und alle endeten
   mit „all generations done“; keiner endete an der Grenze.

**Beschreibende Ergebnisse.** Die folgenden Zahlen tragen kein Urteil (Vorab-Registrierung,
Abschnitt 8).

- **Median von D_A** (beschreibend). Über die 27 Paare ist er 7/9 = 0.778. D_A war 23-mal
  7/9, einmal 8/9 (Seed 7), zweimal 3/9 (Seeds 19 und 20) und einmal 2/9 (Seed 10).
- **Variante nur nach T** (beschreibend). Die Startprogramme lösen dann in jedem Lauf auch
  CCZ, mit 7 T und 10 CNOT, sodass 8 Ziele offen sind. D_A ist dann 15-mal 7/8, 9-mal 6/8,
  einmal 4/8 (Seed 10) und zweimal 3/8 (Seeds 19 und 20); D_R ist 6-mal 7/8, 5-mal 6/8,
  einmal 5/8 (Seed 2), einmal 4/8 (Seed 13) und 14-mal 3/8. D_A > 1/2 gilt damit in 24 der
  27 Seeds, D_R > 1/2 in 12. Die Werte je Seed stehen in `analysis.txt` und `counted.json`.
- **Erste Lösung je Ziel** (beschreibend). Tabelle 6 zählt, auf wie vielen Seeds ein
  gezüchtetes Mitglied jedes Arms das Ziel löste, und nennt den Median der Generation, in
  der es zuerst fiel; die Werte je Seed stehen in `analysis.txt`. CH löste nur ein Lauf, A-7
  in Generation 594. Fredkin löste kein Lauf zur Referenz; exakte Fredkin-Schaltungen mit
  7 T und 8 bis 12 CNOT fanden 15 Läufe von Arm A und 7 von Arm R.
- **Aufrufe bis zur ersten Lösung** (beschreibend). Wir nehmen dafür `calls_used` der
  Generation, in der ein Ziel zuerst fiel. Tabelle 6 gibt den Median, `counted.json` den
  Wert je Lauf und Ziel.
- **Aufrufgrenze und Erfindungen** (beschreibend). Kein Lauf endete an der Aufrufgrenze,
  und kein Lauf fand einen Erfindungskandidaten.
- **Beste Abdeckung einer Kaskade** (beschreibend). Das Mitglied des Archivs mit den meisten
  gelösten Such-Aufgaben löst in Arm A 12 bis 27 der 33 (Median 27), in Arm R 9 bis 27
  (Median 12). Das beste Mitglied nach Fitness löst in Arm A 9 bis 27 (Median 27), in Arm R
  3 bis 27 (Median 9); in 32 der 54 Läufe löst es ebenso viele.
- **Auswahl-Panel** (beschreibend). Der beste Finalist löst in Arm A 6 bis 18 der 22
  Auswahl-Aufgaben (Median 18), in Arm R 4 bis 18 (Median 6). Das beste Startprogramm löst
  in jedem Lauf 2.
- **Verläufe** (beschreibend). Die `state`-Zeilen jeder zehnten Generation stehen je Lauf in
  `counted.json`.

Tabelle 6: die Ziele in den 27 Seeds je Arm (beschreibend). Die Mediane gelten für die
Läufe, die das Ziel lösten; die Aufrufe zählen bis zum Ende der Generation der ersten
Lösung. Ein Stern markiert SWAP und CS, welche schon die Startprogramme lösen; die Spalten
zählen hier gezüchtete Kaskaden.

| Ziel | Seeds mit Lösung, A | Seeds mit Lösung, R | erste Generation, Median A | erste Generation, Median R | Aufrufe bis dahin, Median A (Tausend) | Aufrufe bis dahin, Median R (Tausend) |
|---|---|---|---|---|---|---|
| CZ | 27 | 27 | 12 | 5 | 6.8 | 2.9 |
| SWAP* | 27 | 27 | 1 | 1 | 0.5 | 0.5 |
| CS* | 27 | 27 | 1 | 1 | 0.5 | 0.5 |
| CH | 1 | 0 | 594 | – | 814.7 | – |
| QFT2 | 24 | 12 | 201.5 | 201 | 186.4 | 151.2 |
| CCZ | 27 | 27 | 24 | 37 | 16.4 | 22.2 |
| Toffoli | 24 | 10 | 207.5 | 298 | 195.5 | 241.0 |
| Peres | 24 | 12 | 201.5 | 201 | 186.4 | 151.2 |
| Fredkin | 0 | 0 | – | – | – | – |
| RTOF | 24 | 11 | 201.5 | 182 | 186.4 | 131.3 |
| CCiX | 26 | 15 | 107 | 220 | 102.4 | 171.2 |

**Aufrufe und Laufzeit.** Je Lauf stehen beide in `counted.json`; hier die Zusammenfassung.
Ein Lauf von Arm A brauchte 963 146 bis 1 409 998 Aufrufe (Mittel 1 080 499), einer von
Arm R 780 867 bis 958 373 (Mittel 853 058). Bis Generation 1000 waren es höchstens
1 409 602 (A-18), weit unter der Schranke von 12 723 412 aus Abschnitt 6 der
Vorab-Registrierung. Als Laufzeit messen wir die Zeit zwischen den Zeilen „start“ und
„done“ eines Laufs im Protokoll des Treibers, auf die Sekunde genau. Ein Lauf dauerte in
Arm A 4.87 bis 11.18 min (Mittel 7.36 min), in Arm R 4.33 bis 5.92 min (Mittel 5.01 min),
ein Paar 10.30 bis 15.95 min (Mittel 12.37 min). Die 54 Läufe dauerten zusammen
5.57 Stunden, die vier Läufe der Thread-Prüfung 33.6 min: A-1 mit 1 Thread 14.42 min, mit
4 Threads 8.00 min, R-1 6.23 und 4.95 min. Vorab hatten wir im Mittel 26 bis 46 Paare
erwartet, also 5.8 bis 10.1 Stunden einschließlich 0.42 Stunden für die Thread-Prüfung;
für diese hatten wir die Dauer von zwei Paaren bei 2 Threads angenommen. Gemessen haben wir
5.57 Stunden für 27 Paare und 0.56 Stunden für die Thread-Prüfung, zusammen 6.13 Stunden.
Die Pilotläufe liefen mit 2 Threads, die gezählten mit der Voreinstellung; ihre Laufzeiten
sind darum nicht unmittelbar vergleichbar.

**Abweichung 1.** Ein Neustart des Containers zwischen dem 2026-10-31 um 01:18:49 UTC und
etwa 09:11 UTC unterbrach den Lauf A-12. Er hatte um 01:18:49 UTC begonnen und noch keine
`report.json`. Die Paare 1 bis 11 waren fertig und geprüft, und ihre 110 Prüfsummen
stimmten nach dem Neustart noch. Der Treiber startete um 09:20:01 UTC erneut
(`driver.log`); `deviations.txt` nennt 09:20:04 UTC, die Zeit der zweiten Prüfung von R-1 in
`driver.log`. Nach Abschnitt 6, Schritt 3 der Vorab-Registrierung löschte der Treiber um
09:20:09 UTC das unfertige Verzeichnis und ließ A-12 auf demselben Seed von vorn laufen,
bis 09:27:40 UTC. Kein fertiges Paar lief erneut oder änderte sich. Beim Neustart prüfte
der Treiber Paar 1 noch einmal mit `--verify` und schrieb dessen zehn Prüfsummen ein
zweites Mal in [hashes.txt](step1-counted/hashes.txt); beide Abschriften sind gleich
([deviations.txt](step1-counted/deviations.txt), [driver.log](step1-counted/driver.log)).

**Warum 100 und 120 Paare.** Am 2026-10-30 hat die Eigentümerin bzw. der Eigentümer zuerst
die Schwelle 20 mit höchstens 60 Paaren freigegeben. Noch am selben Tag verlangte sie bzw.
er die wissenschaftlich tragende Fassung und nannte keine Zahlen. Claude, der
KI-Assistent, der an diesem Repository arbeitet, prüfte darauf 16 Entwürfe: die Schwellen
20, 40, 1024/11 und 100, jede mit höchstens 60, 80, 100 und 120 Paaren. Die Prüfung
rechnete die Fehlerraten und Teststärken exakt aus, zählte sie unabhängig nach, testete
sie mit einer Monte-Carlo-Simulation und prüfte die Quellen. Ihre drei unabhängigen
Gutachter empfahlen je einen anderen Entwurf: 40 mit höchstens 100 Paaren, 1024/11 mit 120
und 100 mit 120. Ein Gegenprüfer rechnete die Zahlen nach und hielt 1024/11 mit 120 Paaren
(Schranke 11/1024 = 0.0107) für richtig, sofern die Liste der gezählten Seeds fest ist.
Claude wählte die Schwelle 100 mit 120 Paaren und die Liste der Seeds oben. Die
Eigentümerin bzw. der Eigentümer bestätigte diese Wahl mit dem Zusammenführen von
Pull-Request #19, das die Vorab-Registrierung beim Commit 15cb151 einfror; erst danach
begann Seed 1.

Drei der 16 Entwürfe im Vergleich, nach
`.scratch/universum-erfinden/sequential-rule/designs.txt`; die letzten drei Spalten geben
den Hauptnachweis bei der Kaplan-Meier-Quote 0.710:

| Nachweisschwelle, Höchstzahl | Falscher Nachweis, ungünstigster Fall | Erbracht bei 0.710 | Gescheitert bei 0.710 | Ohne Urteil bei 0.710 |
|---|---|---|---|---|
| 100, 120 Paare (gewählt) | 0.0079 | 0.8605 | 0.1144 | 0.0252 |
| 1024/11, 120 Paare | 0.0086 | 0.8620 | 0.1142 | 0.0238 |
| 20, 60 Paare (zuerst freigegeben) | 0.0385 | 0.8211 | 0.0988 | 0.0801 |

Statistisch sind der gewählte Entwurf und 1024/11 mit 120 Paaren fast gleich. Von beiden
ist die Schwelle 100 etwas strenger, und nur ihre Schranke ist rund. Vier weitere Gründe
tragen die Wahl:

- **Das Niveau.** Die Schranke von 1 % ist einseitig, wie die Nachweise. Für bestätigende
  Nachweise wie diese ist eine einseitige Grenze von 5 % großzügig: Nach der
  regulatorischen Leitlinie ICH E9 (1998, Abschnitt 5.5) ist es in regulatorischen
  Verfahren vorzuziehen, den Fehler erster Art eines einseitigen Tests auf die Hälfte des
  üblichen zweiseitigen zu setzen, also auf 2.5 %. Die Schranke von 1 % ist um den Faktor
  2.5 strenger; nach derselben Halbierung entspricht sie einem zweiseitigen Test zu 2 %.
  Die übrigen Tests des Projekts behalten ihre vorab festgelegte einseitige Grenze von
  5 %.
- **Die Belege.** Auf Jeffreys' Skala in der Fassung von Kass und Raftery (1995) sowie
  Vovk und Wang (2021) gilt ein Belegwert über 100 als „entscheidend“ (decisive), einer
  von 20 nur als „stark“ (strong). Die leicht abgewandelte eigene Skala von Kass und
  Raftery nennt 20 bis 150 „stark“; auf ihr fallen 20 und jeder Belegwert bei einem
  Nachweis in eine Klasse, da ein solcher Wert zwischen 100 und 150 liegt.
- **Die Teststärke.** Bei der vorsichtigsten Schätzung aus dem Pilot, der
  Kaplan-Meier-Quote 0.710, gelingt der Hauptnachweis mit Wahrscheinlichkeit 0.86, über
  der üblichen Teststärke von 0.80 nach Cohen (1992).
- **Die Abwägung der Fehler.** Bei der Kaplan-Meier-Quote gibt dieser Entwurf unter den 16
  das kleinste 4 · alpha + beta, wobei alpha der falsche Nachweis im ungünstigsten Fall
  ist und beta = 1 - Teststärke. Das Gewicht 4 ist das Verhältnis beta : alpha von 4 zu 1
  (0.20 zu 0.05), das Cohen (1992) aus seinen Konventionen alpha = 0.05 und Teststärke
  0.80 ableitet.

Der Gewinn hat eine Grenze: Häufiger als der frühere Entwurf gelingt der neue nur oberhalb
einer Quote von etwa 0.67. Über die Unsicherheit der Pilotquote gemittelt (Beta(10, 3): 9
Erfolge in 11 Läufen mit gleichverteilter Vorannahme) ist die Chance auf einen wahren
Nachweis etwa gleich: 0.827 gegenüber 0.829. Was die Änderung bringt, ist ein um etwa den
Faktor 5 seltenerer falscher Erfolg, um den Preis von mehr Rechenzeit.

## 8. Grenzen

- **Positivkontrolle.** Der Entwerfer ist kein Sprachmodell. Seine Zahlen zeigen, ob die
  Zucht Gene findet, von denen wir wissen, dass es sie gibt; über echte Modelle sagen sie
  nichts. Das gilt auch für die beiden Nachweise des gezählten Experiments.
- **Freigabe und Einfrieren.** Die Eigentümerin bzw. der Eigentümer hat am 2026-10-30
  sechs Punkte bestätigt: exakte Gleichheit als Kriterium (sie verschärft Q38), die
  Bewertung erst nach T und dann nach CNOT, die `state`-Zeile auch in Gitter-Läufen, keine
  Beförderung von Schaltungs-Kaskaden in Schritt 1, die Regel über die Seeds aus Abschnitt 7
  anstelle der Regel „9 von 10“ und die Panelgrößen 33 und 22. Noch am selben Tag
  verlangte sie bzw. er die wissenschaftlich tragende Fassung. Nach einer Prüfung von 16
  Entwürfen wählte Claude, der KI-Assistent, die Nachweisschwelle 100, höchstens 120 Paare
  und die Liste der Seeds aus Abschnitt 7. Die Vorab-Registrierung setzt diese Punkte um.
  Sie fror beim Commit 15cb151 ein, zu dem ihr Pull-Request #19 beim Zusammenführen in den
  Hauptzweig zusammengefasst wurde. Mit diesem Zusammenführen bestätigte die Eigentümerin
  bzw. der Eigentümer Claudes Wahl. Erst danach liefen die gezählten Läufe (Abschnitt 7).
  Zahlen aus gezählten Läufen stehen nur in den Abschnitten 1, 7 und 8; alle übrigen
  Zahlen aus Läufen stammen aus Pilotläufen.
- **Was das gezählte Experiment zeigt.** Der Hauptnachweis verwirft die Nullhypothese, dass
  höchstens die Hälfte aller Seeds bis Generation 1000 besteht. Der zweite Nachweis
  verwirft die Nullhypothese, dass Arm A unter den Paaren ohne Gleichstand höchstens so oft
  zuerst besteht wie Arm R. Die Regel erbringt jeden der beiden Nachweise höchstens mit
  Wahrscheinlichkeit 1/100 fälschlich, unter den Annahmen aus Abschnitt 8 der
  Vorab-Registrierung. Diese Annahmen gelten, wenn die Seeds wie unabhängige Ziehungen
  wirken; das nehmen wir an.
- **Was es nicht zeigt.** Das Experiment schätzt die Erfolgsquote nicht. 24 Erfolge in 27
  Seeds sind eine beschreibende Zahl. Da der Stopp von den Daten abhing, gilt ein
  Konfidenzintervall für eine feste Zahl von Seeds hier nicht, und wir geben keines an.
  Ebenso sagt der zweite Nachweis nicht, um wie viele Generationen Arm A früher besteht.
  E2 erreichte die Schwelle 100 erst bei Paar 27, mit dem die Regel anhielt.
- **Was der Pilot zeigt.** Die Kalibrierung hält in jedem fertigen Pilotlauf. Bis Generation
  1000 bestand Arm A in 9 von 11 Läufen, Arm R in 2 von 10, und in 8 der 10 Paare, deren
  Arme beide so weit liefen, bestand Arm A zuerst. Die neun Läufe des ersten Pilots bestehen
  `--verify`.
- **Was der Pilot nicht zeigt.** Die Pilotläufe schätzen die Erfolgsquote nur grob: 9 von 11
  hat ein Konfidenzintervall zu 95 % von 0.48 bis 0.98, und die Kaplan-Meier-Schätzung über
  alle 31 vollständigen Läufe von Arm A ergibt 0.710. Die frühen Reihen weichen ab: Bis
  Generation 200 bestand Arm A auf 6 der 20 Seeds 101 bis 110 und 201 bis 210, in den
  späteren Reihen in 7 von 11 Läufen (Fisher-Test, p = 0.128). Ein Erfolg hängt an einem
  Ereignis, einem Worker, der Hadamard, Phase und Linear zugleich nennt. Darum streuen die
  Bestehenszeiten weit, und ein Erfolg nach Generation 1000 zählt als Misserfolg. Bei der
  Quote 0.710 endet die Regel mit Wahrscheinlichkeit 0.114 als gescheitert und mit 0.025
  ohne Urteil; ein gescheiterter Hauptnachweis beweist keine Quote von höchstens 1/2.
  Liegt die wahre Quote nur bei 0.6, gelingt der Hauptnachweis mit Wahrscheinlichkeit
  0.18. Mit den jetzigen Genen hätte das gezählte Experiment also durchaus scheitern können;
  es hat beide Nachweise erbracht (Abschnitt 7). Die Gene blieben, wie sie waren; jede
  Änderung verlangte neue Pilotläufe und eine neue Vorab-Registrierung.
- **CH und Fredkin.** Kein Pilotlauf löste sie. Von den 54 gezählten Läufen löste nur A-7
  CH, in Generation 594, und keiner Fredkin zur Referenz. Die Wege, welche die Tests zu
  ihnen festlegen, sind lang: CH fällt mit Search und Meet ab 3 124 Denk-Tokens auf
  `sim:large`, Fredkin mit Compose außer Hadamard, Phase und Linear, und 7 CNOT erreicht es
  erst mit Simplify.
- **Abgeleitete Schranken.** Bei vier Zahlen ist die Schranke abgeleitet: CS mit 2 CNOT, CH
  mit 2 T, RTOF und CCiX mit je 4 T. Die Zahl steht in einer Quelle; dass es nicht besser
  geht, folgert die Recherche aus veröffentlichten Ergebnissen. Unterbietet eine exakte
  Antwort eine solche Zahl, stoppt der Lauf mit einem Integritätsalarm.
- **Gemeinsame Fehlerquelle.** Entwerfer und Prüfer nutzen beide `qubit.rs`; ein Fehler dort
  könnte beide gleich täuschen. Dagegen stehen die Gegenprobe mit `quantum.rs`, die aus ihren
  Definitionen gebauten Ziele und die Beleg-Schaltungen, die meisten aus der Literatur, die
  übrigen von der Recherche zu Schritt 1 gebaut und geprüft.
- **Plattform.** Die Prüfsummen des ersten Pilots und der gezählten Läufe
  ([hashes.txt](step1-counted/hashes.txt)) gelten für rustc 1.97.0 auf x86_64 Linux.
  Anderswo kann eine Gleitkomma-Funktion ihr letztes Bit anders runden; dann weichen die
  Läufe ab.
- **Kein Mutator.** Die Pilotläufe und die gezählten Läufe haben keine LLM-Runden
  (`llm_every` 0). Was Claude als Mutator auf Schaltungen ändert, haben wir nicht gemessen.
- **Erfindungen.** Kein Pilotlauf und kein gezählter Lauf fand einen Erfindungskandidaten.
  Ein Kandidat ist keine Erfindung, bis ein Mensch die Literatur geprüft hat. Da `--verify`
  eine veränderte `rediscovery.json` ablehnt, bleibt `literature_checked` im Lauf `false`;
  das Ergebnis einer solchen Prüfung gehört in einen Bericht.

## 9. Quellen

Die Quellen der Zielbibliothek und ihre Fundstellen hat die Recherche zu Schritt 1
zusammengetragen; je Ziel stehen sie in `src/cascade/circuit.rs`. Zhang et al. beschreiben
die Elternwahl von Arm A (Abschnitt 3). Die letzten vier Einträge sind die Quellen, die
Abschnitt 7 für die Schwellen nennt; die übrigen Quellen der Regel nennt die
Vorab-Registrierung in Abschnitt 8.

- Amy, Maslov, Mosca, Roetteler: A meet-in-the-middle algorithm for fast synthesis of
  depth-optimal quantum circuits. IEEE TCAD 32(6), 2013, S. 818–830.
  [arXiv:1206.0758](https://arxiv.org/abs/1206.0758)
- Amy, Mosca: T-count optimization and Reed-Muller codes. IEEE Transactions on Information
  Theory 65(8), 2019.
- Vatan, Williams: Optimal quantum circuits for general two-qubit gates. Physical Review A
  69, 032315, 2004. [arXiv:quant-ph/0308006](https://arxiv.org/abs/quant-ph/0308006)
- Shende, Bullock, Markov: Recognizing small-circuit structure in two-qubit operators.
  Physical Review A 70, 012310, 2004.
  [arXiv:quant-ph/0308045](https://arxiv.org/abs/quant-ph/0308045)
- Shende, Markov: On the CNOT-cost of TOFFOLI gates. Quantum Information and Computation
  9(5–6), 2009, S. 461–486. [arXiv:0803.2316](https://arxiv.org/abs/0803.2316)
- Gosset, Kliuchnikov, Mosca, Russo: An algorithm for the T-count. Quantum Information and
  Computation 14(15–16), 2014, S. 1261–1276. [arXiv:1308.4134](https://arxiv.org/abs/1308.4134)
- Mosca, Mukhopadhyay: A polynomial time and space heuristic algorithm for T-count.
  Quantum Science and Technology 7, 015003, 2022.
  [arXiv:2006.12440](https://arxiv.org/abs/2006.12440)
- Di Matteo, Mosca: Parallelizing quantum circuit synthesis. Quantum Science and
  Technology 1, 015003, 2016. [arXiv:1606.07413](https://arxiv.org/abs/1606.07413)
- Gheorghiu, Mosca, Mukhopadhyay: T-count and T-depth of any multi-qubit unitary. npj
  Quantum Information 8, 141, 2022.
  [doi:10.1038/s41534-022-00651-y](https://doi.org/10.1038/s41534-022-00651-y)
- Selinger: Quantum circuits of T-depth one. Physical Review A 87, 042302, 2013.
  [arXiv:1210.0974](https://arxiv.org/abs/1210.0974)
- Maslov: Advantages of using relative-phase Toffoli gates with an application to multiple
  control Toffoli optimization. Physical Review A 93, 022311, 2016.
  [arXiv:1508.03273](https://arxiv.org/abs/1508.03273)
- Niroula, Nam: A quantum algorithm for string matching. npj Quantum Information 7, 37,
  2021, Supplementary Note 1.
  [doi:10.1038/s41534-021-00369-3](https://doi.org/10.1038/s41534-021-00369-3)
- Zhang, Hu, Lu, Lange, Clune: Darwin Gödel Machine: Open-Ended Evolution of
  Self-Improving Agents. ICLR 2026. [arXiv:2505.22954](https://arxiv.org/abs/2505.22954)
- International Conference on Harmonisation: ICH Harmonised Tripartite Guideline E9,
  Statistical Principles for Clinical Trials. Step-4-Fassung, 5. Februar 1998. Eine
  regulatorische Leitlinie, nicht begutachtet.
- Kass, Raftery: Bayes factors. Journal of the American Statistical Association 90(430),
  1995, S. 773–795.
  [doi:10.1080/01621459.1995.10476572](https://doi.org/10.1080/01621459.1995.10476572)
- Vovk, Wang: E-values: Calibration, combination and applications. Annals of Statistics
  49(3), 2021, S. 1736–1754. [doi:10.1214/20-AOS2020](https://doi.org/10.1214/20-AOS2020)
- Cohen: A power primer. Psychological Bulletin 112(1), 1992, S. 155–159.
  [doi:10.1037/0033-2909.112.1.155](https://doi.org/10.1037/0033-2909.112.1.155)
