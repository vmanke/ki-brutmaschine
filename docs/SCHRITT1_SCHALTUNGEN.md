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
  15 Beleg-Schaltungen der Bibliothek, je eine Schaltung aus der Literatur zu jedem
  Eintrag, liefern beide dieselbe Matrix bis auf 1e-12.
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
  oder abgeleiteten Schranke (Proven, Derived) hält den Lauf mit einem Integritätsfehler an,
  denn dann irrt der Prüfer oder die Referenz; die Generation wird nicht festgeschrieben.
  Eine exakte Antwort unter einem besten bekannten Wert (BestKnown: 7 CNOT für Fredkin,
  4 CNOT für CCiX) wird als Erfindungskandidat festgehalten, mit
  `literature_checked = false`. Neu nennen wir sie erst, wenn ein Mensch die Literatur
  geprüft hat (Q38).
- **Positivkontrolle** (`src/cascade/designer.rs`, `src/cascade/circuit_genes.rs`). Im
  Container schreibt kein Sprachmodell Schaltungen. Ein simulierter Entwerfer unter den
  Modellnamen `sim:small` und `sim:large` beantwortet die Schaltungs-Prompts (Abschnitt 5).
  Die Gene der Schaltungs-Zucht sind vier Startprogramme, 50 Satzbausteine, der Auftrag an
  den Mutator und die Aufnahmeregel `no_gate_lists`: Ein Abschnitt mit drei oder mehr
  Gatter-Nennungen wie „CNOT 0 1“ wird abgelehnt, damit kein Mutator Antworten in Prompts
  ablegt.
- **Programm** (`src/bin/brut-breed.rs`). `--domain circuits` startet einen Schaltungslauf.
  Ein fertiger Lauf schreibt vor seinem Bericht `rediscovery.json`, die vorab festgelegte
  Zählung (Abschnitt 4). `--verify` rechnet Bericht und Zählung ohne Modellaufruf aus dem
  Checkpoint nach. `--promote` lehnt Schaltungsläufe mit Exit-Code 1 ab, denn die Beförderung
  von Schaltungs-Kaskaden gehört nicht zu Schritt 1.
- **Auswertung** (`examples/circuit_positive_control.rs`). Sie liest die fertigen Läufe
  beider Arme, paarweise nach Seed, und wendet die vorab festgelegten Regeln an
  (Abschnitt 6).
- **Tests.** `tests/circuits.rs` hat 9 Tests: Schaltungsläufe sind über Prozesse hinweg
  wiederholbar und bestehen `--verify`; Stopp und Fortsetzen ergeben dieselbe Abstammung und
  dieselben `state`-Zeilen wie ein durchgehender Lauf; 1 und 4 Threads ergeben byte-gleiche
  Dateien; `--promote` wird abgelehnt; `rediscovery.json` gleicht einer Nachzählung aus den
  protokollierten Antworten, und jede Änderung an der Datei lässt `--verify` scheitern; ein
  Quelltext-Scan zeigt, dass `designer.rs` die Bibliothek nicht kennt; ein Test legt die
  Kalibrierung fest; dazu kommen die beiden Gegenproben mit `quantum.rs`.

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
| Peres | 3 | Toffoli, dann CNOT von a auf b | 7 (bewiesen) | 5 (bewiesen) | Shende, Markov 2009 |
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

# Eingefrorene Auswertung; für weniger als die 10 gezählten Seeds ohne Aussage
cargo run --example circuit_positive_control -- --a runs/c101-A --r runs/c101-R
```

Arm A wählt die Eltern wie die Darwin-Gödel-Maschine (Zhang et al., ICLR 2026,
Anhang C.2), Arm R gleichverteilt; sonst gleichen sich beide Arme. Die gezählten Seeds 1
bis 10 sind dem vorab festgelegten Experiment vorbehalten; die Beispiele nutzen darum den
Pilot-Seed 101. Ein Lauf druckt je Generation eine Zeile (beste
Kaskade, Fitness, gelöste Such-Aufgaben, Archivgröße, Aufrufe), dann eine Zeile zu
`rediscovery.json` und am Ende die besten Kaskaden und den Vergleich auf dem Auswahl-Panel.
Für Arm A auf Seed 101 lautet die Zeile zu `rediscovery.json`:

```
brut-breed: wrote runs/c101-A/rediscovery.json: the seeds solve 2 of 11 targets, bred Cascades 7 of the other 9; D > 1/2: yes; calibrated
```

`--verify` meldet bei diesem Lauf „brut-breed: 1362 events, hash chain intact, head
e93ed521…; checkpoint at entry 1362 matches; settings match; report recomputed and equal;
rediscovery recomputed and equal“. Wir haben Arm A auf Seed 101 erneut laufen lassen:
`events.jsonl` und `rediscovery.json` sind byte-gleich mit dem Pilotlauf, und der Lauf
dauerte 3.2 s (Release-Build, 2 Threads).

**Was `--verify` prüft.** Wie in Stufe 2c prüft es die Hash-Kette, den Anker des
Checkpoints und die Einstellungen. Bei einem fertigen Lauf baut es den Lauf aus dem
Checkpoint neu auf, rechnet den Bericht nach und leitet den Stoppgrund aus dem Checkpoint ab,
statt ihn dem Bericht zu glauben („all generations done“ oder „call budget reached“). Bei
einem fertigen Schaltungslauf verlangt es zudem, dass `rediscovery.json` Byte für Byte der
Neuberechnung gleicht. Fehlt die Datei oder wurde sie verändert, endet `--verify` mit
Exit-Code 1 und einer Meldung „integrity: …“. Bei einem unfertigen Lauf meldet es
„rediscovery.json not checked: the run is not finished“. `timings.jsonl` gehört nicht zum
geprüften Protokoll.

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

**`rediscovery.json`.** Für jedes Ziel (`targets`) stehen dort die Referenz mit ihrem Status,
die Zahl der Such-Aufgaben des Ziels (`tasks`), die Kosten der billigsten exakten Antwort
der Startprogramme und der gezüchteten Kaskaden (`best_seed`, `best_bred`: T, CNOT, H,
Gatter, Tiefe, T-Tiefe), ob Startprogramme und gezüchtete Kaskaden die Referenz erreicht
haben (`reached_by_seed`, `reached_by_bred`), die früheste Generation einer gezüchteten Kaskade, die das Ziel löste
(`first_generation`), der Text der billigsten exakten gezüchteten Antwort (`circuit`; er
setzt die verkleidete Aufgabe um, nicht das Ziel in seiner Grundform) und ein etwaiger
Erfindungskandidat (`below_best_known`, `candidate`). Darunter stehen |L| (`library`),
|R_seed| (`reached_by_seeds`), |R_bred \ R_seed| (`reached_by_bred_only`), der Anteil D
(`share`), die Schranke 3 · |R_seed| ≤ |L| (`calibrated`) und `passes` (kalibriert und
D > 1/2). Die Begriffe definiert die Vorab-Registrierung, Abschnitt 7.

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
Aufnahmeregel; die Zucht könnte ihn also bauen.

**Wege zu den Zielen.** Die Tests in `designer.rs` legen fest, welche Familien und wie viele
Denk-Tokens ein Ziel braucht; ein Denk-Token weniger verfehlt es.

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
`residual` (9), `style` (6), `check` (5), `role` (3) und `format` (2); das haben wir aus dem
Quelltext gezählt. Kein Baustein nennt zwei von Hadamard, Phase und Linear. Ein
Startprogramm kennt eine dieser drei Familien, ein zufälliger Schritt mit einem Baustein
lehrt es höchstens eine weitere. Von den 704 Kindern, welche ein solcher Schritt aus den
Startprogrammen macht, lösen 40 ein Ziel über SWAP und CS hinaus (20 CZ, 20 CCZ) und keines
zwei. QFT2, Toffoli, Peres, RTOF, CCiX und Fredkin brauchen alle drei Familien, also
mindestens zwei Schritte: Die 60 Kinder, deren letzter Worker zwei der drei kennt, haben
384 Kinder, die alle drei kennen, und jedes davon löst CZ, QFT2, Toffoli, Peres, RTOF und
CCiX. Diese Zahlen legen die Tests in `circuit_genes.rs` fest, je Ziel unverkleidet und in
drei Verkleidungen.

## 6. Pilotläufe

Die Seeds 101 und 102 sind Pilot-Seeds: Wir zählen sie nicht, und sie legen nur
`--generations` und `--max-calls` der gezählten Läufe fest. Wir haben beide Arme auf beiden
Seeds 60 Generationen lang laufen lassen, mit `--search 33 --select 22 --children 8`,
Release-Build auf Commit d4a650e, rustc 1.97.0, 2 Threads. Die voreingestellte Grenze von
20 000 Aufrufen beendete Arm A auf Seed 101 nach Generation 20 (18 451 Aufrufe); darum nutzen
alle vier Läufe `--max-calls 200000`. Jeder Lauf wurde danach mit 200 Generationen
wiederholt. Alle Zahlen stammen aus [pilot.json](pilot-step1/pilot.json); wir haben alle neun
Lauf-Verzeichnisse erneut mit `--verify` geprüft und die Auswertung erneut ausgeführt.

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
- **Erste Lösungen.** Jede erste Lösung kam bis Generation 90, keine in den Generationen 91
  bis 200. CH und Fredkin löste kein Lauf.
- **Engpass.** In Arm A auf Seed 101 fielen fünf Ziele in Generation 32, der ersten, in der
  ein Worker Hadamard, Phase und Linear zugleich nannte. In den drei anderen Läufen entstand
  in 200 Generationen kein solcher Worker. Diese Generationen (`first_worker_naming` in
  `pilot.json`) sind außerhalb des Programms durch eine Suche nach den Wortstämmen in den
  Abschnittstexten ermittelt.
- **Auswertung.** Für die vier Läufe mit 60 Generationen meldet die eingefrorene Auswertung
  „No claim: only 2 seed pair(s)“. D_A > 1/2 gilt auf einem von zwei Seeds (Median 1/2),
  D_A > D_R in einem von zwei Paaren (p = 3/4). In der Variante nur nach T lösen die
  Startprogramme zusätzlich CCZ (7 T, 10 CNOT); dann ist D_A 6/8 und 1/8, D_R 3/8 und 3/8.
- **Auswahl-Panel.** In Arm A auf Seed 101 löst der beste Finalist 16 der 22
  Auswahl-Aufgaben, das beste Startprogramm 2. In den drei anderen Läufen löst der beste
  Finalist 4 (`report.json` der Pilotläufe).
- **Wiederholbarkeit.** Wir haben Arm A auf Seed 101 erneut laufen lassen; `events.jsonl` und
  `rediscovery.json` sind byte-gleich mit dem Pilotlauf.

Aus dem Pilot folgen die Einstellungen der gezählten Läufe: `--generations 200`, da alle
ersten Lösungen bis Generation 90 kamen und ein Lauf höchstens 26 s dauert, und
`--max-calls 2600000`. Nach unserer Rechnung aus den Aufrufgrenzen in `arena.rs` braucht
ein Lauf mit 200 Generationen höchstens 2 547 412 Aufrufe, sodass die Grenze keinen
gezählten Lauf vorzeitig beendet.

## 7. Grenzen

- **Positivkontrolle.** Der Entwerfer ist kein Sprachmodell. Seine Zahlen zeigen, ob die
  Zucht Gene findet, von denen wir wissen, dass es sie gibt; über echte Modelle sagen sie
  nichts.
- **Sechs offene Entscheidungen.** Gezählte Läufe beginnen erst, wenn die Eigentümerin bzw.
  der Eigentümer sechs Punkte bestätigt hat: exakte Gleichheit als Kriterium (sie verschärft
  Q38), die Bewertung erst nach T und dann nach CNOT, die `state`-Zeile auch in
  Gitter-Läufen, keine Beförderung von Schaltungs-Kaskaden in Schritt 1, die Regel
  „9 von 10“ und die Panelgrößen 33 und 22. Mit derselben Freigabe friert die
  Vorab-Registrierung ein.
- **Was der Pilot zeigt.** Die Kalibrierung hält in allen Läufen. Arm A kann die Hauptregel
  auf einem Seed erfüllen (Seed 101: D = 7/9). Alle Läufe bestehen `--verify`. Nach
  Generation 90 kam kein Ziel mehr hinzu, und ein Lauf mit 200 Generationen dauert höchstens
  26 s.
- **Was der Pilot nicht zeigt.** Zwei Seeds schätzen keine Erfolgsquote. Arm A erfüllt die
  Hauptregel auf einem von zwei Seeds. Nach unserer Rechnung hält die Regel „9 von 10“ bei
  einer unabhängigen Quote von 0.5 je Seed nur mit Wahrscheinlichkeit 0.0107, bei 0.9 mit
  0.736; genau ein Erfolg in zwei Seeds hat bei 0.5 die Wahrscheinlichkeit 0.5, bei 0.9
  noch 0.18.
  Ob die Auswahl hilft, zeigt der Pilot ebenfalls nicht: D_A > D_R in einem Paar,
  Gleichstand im anderen. Mit den jetzigen Genen kann das gezählte Experiment also
  scheitern. Ob die Gene vor dem Einfrieren geändert werden, entscheidet die Eigentümerin
  bzw. der Eigentümer; jede Änderung verlangt neue Pilotläufe.
- **CH und Fredkin.** Kein Pilotlauf löste sie. CH braucht Search und Meet mit mindestens
  3 124 Denk-Tokens auf `sim:large`, Fredkin außer Hadamard, Phase und Linear auch Compose
  und für 7 CNOT Simplify.
- **Abgeleitete Schranken.** Bei vier Zahlen ist die Schranke abgeleitet: CS mit 2 CNOT, CH
  mit 2 T, RTOF und CCiX mit je 4 T. Die Zahl steht in einer Quelle; dass es nicht besser
  geht, folgert die Recherche aus veröffentlichten Ergebnissen. Unterbietet eine exakte
  Antwort eine solche Zahl, stoppt der Lauf mit einem Integritätsalarm.
- **Gemeinsame Fehlerquelle.** Entwerfer und Prüfer nutzen beide `qubit.rs`; ein Fehler dort
  könnte beide gleich täuschen. Dagegen stehen die Gegenprobe mit `quantum.rs`, die aus ihren
  Definitionen gebauten Ziele und die Beleg-Schaltungen aus der Literatur.
- **Plattform.** Die Prüfsummen der Pilotläufe gelten für rustc 1.97.0 auf x86_64 Linux.
  Anderswo kann eine Gleitkomma-Funktion ihr letztes Bit anders runden; dann weichen die
  Läufe ab.
- **Kein Mutator im Pilot.** Die Pilotläufe haben keine LLM-Runden (`llm_every` 0). Was
  Claude als Mutator auf Schaltungen ändert, haben wir nicht gemessen.
- **Erfindungen.** Kein Pilotlauf fand einen Erfindungskandidaten. Ein Kandidat ist keine
  Erfindung, bis ein Mensch die Literatur geprüft hat. Da `--verify` eine veränderte
  `rediscovery.json` ablehnt, bleibt `literature_checked` im Lauf `false`; das Ergebnis einer
  solchen Prüfung gehört in einen Bericht.

## 8. Quellen

Die Quellen der Zielbibliothek und ihre Fundstellen hat die Recherche zu Schritt 1
zusammengetragen; je Ziel stehen sie in `src/cascade/circuit.rs`.

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
