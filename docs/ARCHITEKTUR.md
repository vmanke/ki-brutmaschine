# Architektur der Brutmaschine

Wir beschreiben hier, wie die Teile zusammenarbeiten, und erklären den
Datenfluss, die Auswahlregel für Eltern und die verdeckte Bewertung. Die
ersten Abschnitte gelten für den MLP-Modus (kleine neuronale Netze). Der
Abschnitt „Lerner-Modus (Stufe 1)“ beschreibt den zweiten Modus, in dem
Kandidaten kleine Lernprogramme sind.

## Begriffe

- **Genom:** die vollständige Beschreibung eines Kandidaten als Daten.
  Dazu gehören die Breiten der verdeckten Schichten, die Aktivierungsfunktion
  (ReLU, Tanh, GELU, Sinus), das Optimierungsverfahren (SGD, Momentum, Adam),
  Lernrate, L2-Gewichtsabnahme, Batch-Größe, Epochen, Startskala der Gewichte
  und ein Seed (Startwert für Zufallszahlen).
- **Kandidat:** ein Genom plus das trainierte Netz und seine Messwerte.
  Die Kennung (ID) ist der Anfang des SHA-256-Hashes über das Genom.
- **Generation:** eine Runde der Evolution. Generation 0 ist die zufällige
  Startpopulation.
- **Fitness:** die Zahl, nach der ausgewählt wird. Höher ist besser.
- **FLOPs:** Gleitkomma-Operationen, ein Maß für Rechenarbeit.
- **Orchestrator:** das Programm `brut run`, das die Evolution steuert
  (`src/evolve.rs`). Es erzeugt Train und Valid, trainiert, bewertet und
  schreibt das Archiv. Die geheimen Daten sieht es nie.
- **Guard (Wächter):** prüft vor jeder Generation Not-Aus, Zeit und
  FLOP-Budget (`safety::Guard`).
- **Hash:** ein digitaler Fingerabdruck fester Länge (hier SHA-256). Ändert
  sich ein Byte der Daten, ändert sich der Hash vollständig.
- **Median:** der mittlere Wert einer sortierten Liste.

## Drei Datensätze

Jede Aufgabe erzeugt drei Datensätze aus derselben Verteilung, aber mit
getrennten Zufallszahlen:

| Datensatz | Wer erzeugt ihn? | Wofür? |
|---|---|---|
| Train | Orchestrator | Training der Netze |
| Valid | Orchestrator | Bewertung und Auswahl |
| Hidden | nur `brut-eval` | ehrliche Schlussbewertung |

Der Seed für Train und Valid kommt aus dem Master-Seed der Konfiguration.
Der Seed für Hidden steht nur in `secret/hidden.env`. Diese Datei liest
nur der Prozess `brut-eval`.

## Datenfluss

```
config.json ──► evolve::run
                  │
                  ├─ task::generate(Train, Valid)
                  │
                  ├─ Generation 0: Genome::random  ─┐
                  ├─ Generation g: select_parents   │
                  │      └─ RandomMutator.propose ──┤  (Mutation oder Kreuzung)
                  │                                 ▼
                  │                    Filter: Duplikate, FLOP-Budget
                  │                                 ▼
                  │      parallel (rayon, teuerste zuerst): train::train
                  │                                 ▼
                  │                     score::evaluate(Valid), fitness
                  │                                 ▼
                  │          models/<id>.json  +  archive.jsonl (Candidate)
                  │
                  ├─ Guard: STOP-Datei, Zeit, FLOPs (vor jeder Generation)
                  │
                  └─ Ende: beste k Kandidaten ──► brut-eval (eigener Prozess)
                                                    │ liest secret/hidden.env
                                                    │ erzeugt Hidden, bewertet
                                                    ▼
                                    ein abgerundeter Wert je Anfrage
                                                    ▼
                               archive.jsonl (HiddenEval, ggf. Quarantäne)
```

Schritte im Einzelnen (`src/evolve.rs`):

1. Konfiguration laden und prüfen. Archiv anlegen oder öffnen. Beim Öffnen
   prüft der Lauf die ganze Hash-Kette und den Anker `archive.head`. Hat das
   Archiv schon eine verdeckte Bewertung und verlangt die Konfiguration neue
   Generationen, bricht der Lauf ab (siehe „Verdeckte Bewertung“). Sonst hängt
   er einen Eintrag `RunStarted` an.
2. Wenn das Archiv schon Kandidaten hat, macht der Lauf nach der höchsten
   Generation weiter. Die bereits verbrauchten FLOPs zählen zum Budget.
   Vorher vergleicht der Lauf `config.json` mit der Konfiguration des ersten
   Laufs im Archiv (`Config::resume_conflicts`). Nur `generations`, `threads`
   und die Laufzeit-Grenzen (`max_candidate_seconds`, `max_total_seconds`,
   `max_total_flops`) dürfen sich ändern. Sonst bricht `brut run` ab, ohne
   etwas ins Archiv zu schreiben.
3. Je Generation:
   - Der Guard prüft Not-Aus, Zeit und FLOP-Budget.
   - Für jedes gewünschte Kind wählt die Archiv-Auswahl Eltern aus. Mit der
     Wahrscheinlichkeit `crossover_rate` entsteht ein Kind durch Kreuzung
     zweier Eltern, sonst durch Mutation eines Elters.
   - Kinder mit schon bekannter ID werden übersprungen. Kinder, die nicht mehr
     ins FLOP-Budget passen, auch.
   - Die Kinder werden parallel trainiert und auf Valid bewertet. Die
     teuersten Kinder (nach geschätzten FLOPs) starten zuerst. Jeder freie
     Thread nimmt sich das nächste noch nicht begonnene Kind. So warten am
     Ende einer Generation weniger Threads auf ein einzelnes großes Kind.
   - Die Ergebnisse werden nach ID sortiert und dann ins Archiv geschrieben.
     So ist die Reihenfolge unabhängig vom Zufall der Thread-Planung.
4. Am Ende werden die besten `hidden_eval_top_k` Kandidaten verdeckt bewertet
   (siehe „Verdeckte Bewertung“). Danach ist der Lauf abgeschlossen.
5. Ein Eintrag `RunFinished` schließt den Lauf ab.

Alle Zufallszahlen der Suche kommen aus ChaCha8-Generatoren. Deren Seeds
werden mit `util::derive_seed` aus dem Master-Seed und einem Text abgeleitet,
zum Beispiel `"gen/3"` oder `"data"`. Darum liefert der gleiche Seed die
gleichen Kandidaten und die gleichen Fitness-Werte, auch bei anderer
Thread-Zahl. Ausnahmen: Abbrüche durch Zeitgrenzen oder den Not-Aus hängen
von der Rechnergeschwindigkeit ab, und der Seed der Hidden-Daten wird bei
`brut init` zufällig gezogen (siehe „Verdeckte Bewertung“).

## Training

- Mini-Batch-Training, die Reihenfolge wird in jeder Epoche neu gemischt.
- Die Lernrate fällt nach einer Kosinus-Kurve vom Startwert auf 5 % ab.
- Das Netz rechnet mit `f32`. Vorwärts- und Rückwärtsrechnung nutzen einen
  wiederverwendbaren Arbeitsspeicher (`nn::Workspace`). In der inneren
  Schleife wird kein Speicher angefordert.
- Für Tanh und GELU nutzt das Netz statt der Bibliotheksfunktion `tanh` eine
  Näherung durch einen Bruch zweier Polynome (Koeffizienten aus der
  Bibliothek Eigen, größter Fehler etwa 5 · 10⁻⁷). Sie ist etwa dreimal so
  schnell, weil der Compiler sie vektorisieren kann.
- Die Rückwärtsrechnung fasst je vier Beispiele (bei den Gradienten) und je
  vier Gewichtszeilen (beim Rückpropagieren) zusammen. Das spart
  Speicherzugriffe. Die Reihenfolge der Summen ist fest, die Ergebnisse
  bleiben also für gleiche Eingaben gleich.
- Werte unter 10⁻³⁰ in den Puffern von Momentum und Adam werden auf 0
  gesetzt. Sonst würden sie zu sogenannten subnormalen Zahlen, mit denen
  der Prozessor sehr langsam rechnet. Das Ergebnis ändert sich dadurch
  praktisch nicht.
- Wird der Verlust oder ein Gewicht unendlich oder NaN („keine Zahl“), bricht
  das Training ab. Der Kandidat bekommt den Status `Diverged`.
- Überschreitet ein Kandidat seine Zeit- oder FLOP-Grenze, bekommt er den
  Status `OverBudget`. Er wird bewertet, ist aber kein Elter.
- Das Training fragt alle 32 Schritte und zu Beginn jeder Epoche den Not-Aus
  ab (`train::train_with_stop`, `safety::StopSignal`; die Datei wird
  höchstens alle 0.1 Sekunden angesehen). Ist die Datei `STOP` da, hört es
  sauber auf. Der Kandidat bekommt den Status `Stopped`: Er wird bewertet, ist
  aber kein Elter. Kinder, die noch nicht begonnen haben, werden gar nicht
  erst trainiert, und der Lauf endet nach dieser Generation.

## Fitness

```
fitness = primary − 0.005 · log10(Parameter) − 0.002 · log10(Trainings-FLOPs)
```

`primary` ist die Genauigkeit (Klassifikation) oder max(0, R²) (Regression).
Die kleinen Strafterme bevorzugen bei gleicher Leistung kleinere und
sparsamere Modelle. Alle gespeicherten Zahlen sind endlich. Nicht bewertbare
Modelle bekommen `WORST_FITNESS = −10⁹`.

## Auswahlregel für Eltern

Die Auswahl ist an die Darwin-Gödel-Maschine angelehnt (Zhang et al. 2026,
siehe Quellen). Wir wählen nur unter Kandidaten mit Status `Ok`.

```
Gewicht(i) = sigmoid(λ · (fitness_i − m)) / (1 + Kinder_i)
λ = 10
m = Median der Fitness der besten min(n, 10) wählbaren Kandidaten
```

Die Sigmoid-Funktion 1 / (1 + e^(−x)) bevorzugt gute Kandidaten. Der Nenner
`1 + Kinder` senkt das Gewicht von Kandidaten, die schon oft Eltern waren.
So bleiben auch schwächere, aber andere Linien im Spiel. Das ist
„offene“ Suche statt reiner Bergsteiger-Suche. Wir ziehen mit Zurücklegen.

Aus der Darwin-Gödel-Maschine übernehmen wir nur λ = 10 und den Faktor
1 / (1 + Kinder). Zwei Dinge sind anders: Die Mitte der Sigmoid-Kurve ist
dort fest 0.5, hier der Median der besten zehn. Und dort zählen nur Kinder,
die funktionieren und ins Archiv aufgenommen werden, hier jedes Kind (auch
solche mit Status `Diverged`, `Invalid`, `OverBudget` oder `Stopped`).

Ein Beispiel aus dem Demo-Lauf (Spiralen, Seed 1): Der beste Kandidat
(Generation 20, Validierung 0.9925) stammt von einem Elter aus Generation 5
mit nur 0.7525 ab. Eine rein gierige Suche hätte diesen Elter nicht
gewählt.

## Verdeckte Bewertung

### Das Problem

Die Evolution wählt viele Male nach dem Validierungswert aus. Dadurch passt
sie sich nach und nach auch an Zufälligkeiten der Validierungsdaten an,
weshalb der Validierungswert des Siegers zu optimistisch ist. Wir nennen das
Überanpassung an die Validierung.

### Die Lösung im Prototyp

1. **Eigener Datensatz, nur einmal am Ende.** Die Hidden-Daten nutzen wir
   nie für Auswahl oder Training. Nur am Ende testen wir wenige Kandidaten
   darauf. Danach verweigert `brut run` weitere Generationen
   (`evolve::check_not_final`). Sonst würden Eltern nur noch unter Kandidaten
   gewählt, die die verdeckte Prüfung bestanden haben, und die geheimen Daten
   würden die Suche lenken. Wer in Etappen rechnen will, nutzt dafür
   `brut run --no-hidden`.
2. **Feste Liste.** Bewertet werden die besten `hidden_eval_top_k`
   Kandidaten nach Validierungs-Fitness (`evolve::hidden_set`). Ein Lauf,
   welcher CNNs züchtet oder mit beförderten Modellen beginnt, bewertet dazu
   sein bestes Basismodell, falls es nicht darunter ist, damit `brut promote`
   die Gewinner mit ihm vergleichen kann (ADR 0008). Kandidaten in
   Quarantäne behalten ihren Platz in dieser Liste. Ein erneuter Aufruf von
   `brut run` bewertet nur fehlende Kandidaten derselben Liste nach (zum
   Beispiel nach einem Not-Aus). Er rückt nicht in der Rangliste nach unten.
   Sonst würde so lange der nächste Kandidat geprüft, bis einer besteht, und
   dessen Hidden-Wert wäre kein ehrlicher Schätzwert mehr.
3. **Eigener Prozess.** `brut-eval` läuft als Kindprozess mit leerer
   Umgebung (`env_clear`). Er bekommt die Anfrage als JSON auf der
   Standardeingabe und antwortet als JSON auf der Standardausgabe.
   `brut run` liest die geheime Datei nie; `brut init` erzeugt sie einmalig
   (`hidden_eval::create_secret`). Ein Test prüft, dass `src/evolve.rs` und
   `src/main.rs` weder `HiddenEvaluator` noch `parse_env_file` noch
   `HIDDEN_SEED_KEY` verwenden. Beide Dateien nutzen aber Teile des Moduls
   `hidden_eval`: `evolve.rs` die Anfrage an `brut-eval`, `main.rs` das
   Anlegen der geheimen Datei.
4. **Nur ein gerundeter Wert.** Je Anfrage gibt es genau eine Zahl, auf
   `hidden_score_step` (Standard und Mindestwert 0.01) abgerundet. Den
   Mindestwert erzwingt `brut-eval` selbst. Keine Einzelergebnisse,
   keine Verluste. Das begrenzt, wie viel Information über die geheimen
   Daten je Anfrage nach außen dringt.
5. **Festes Abfrage-Budget.** `brut-eval` zählt die Abfragen in
   `secret/eval_state.json` und verweigert weitere, wenn das Budget
   (`hidden_query_budget`, Standard 30) verbraucht ist. Die Abfrage wird vor
   der Bewertung gezählt, unter einer Dateisperre (`secret/eval.lock`).
   Eine fehlende oder unpassende Modelldatei wird vorher abgewiesen und
   kostet keine Abfrage. Ein begrenztes Budget begrenzt die adaptive
   Überanpassung an die Testdaten, verhindert sie aber nicht. In der geheimen
   Datei kann zusätzlich `BRUT_HIDDEN_BUDGET` eine Obergrenze setzen, die der
   Orchestrator nicht erhöhen kann.
6. **Quarantäne.** Ist `Validierung − Hidden > hidden_gap_threshold`
   (Standard 0.05), wird der Kandidat gesperrt (`Quarantined`). Er kann dann
   nicht freigegeben werden. Da nach der verdeckten Bewertung keine
   Generation mehr folgt, beeinflusst die Quarantäne die Elternauswahl nicht.

### Grenzen dieser Lösung

Runden und ein festes Budget sind einfache Faustregeln. Genau diese
Faustregeln nennen Blum & Hardt (2015) als bisher üblich und schlecht
verstanden; ihr Verfahren „The Ladder“ ist eine Verbesserung davon. Wir
setzen „The Ladder“ nicht um und haben deshalb auch nicht deren
Garantien. „The Ladder“ gibt einen neuen Wert nur heraus, wenn er den
bisher besten deutlich übertrifft, und wiederholt sonst den alten Wert. Das
passt nicht zur Quarantäne, die für jeden Kandidaten seinen eigenen Wert
braucht. Auch das Verfahren „Thresholdout“ von Dwork et al. (2015) setzen
wir nicht um; es fügt Rauschen hinzu und vergleicht mit den
Trainingsdaten. Die Garantien beider Arbeiten gelten hier also nicht.

Wählt ein Mensch unter den k bewerteten Kandidaten den mit dem besten
Hidden-Wert aus, ist auch dieser Wert wieder leicht zu optimistisch. Mit
kleinem k bleibt der Effekt klein.

Der Seed der Hidden-Daten wird bei `brut init` zufällig gezogen, nicht aus
dem Master-Seed abgeleitet. Zwei Läufe mit gleichem Seed haben also gleiche
Kandidaten, aber verschiedene Hidden-Daten. Ihre Hidden-Werte und damit auch
ihre Quarantäne-Entscheidungen können sich unterscheiden.

Durch das Abrunden ist der gemeldete Hidden-Wert im Mittel etwa
eine halbe Rundungsstufe zu niedrig, höchstens eine ganze Stufe. Kleine
Abstände bis 0.01 können also allein vom Runden kommen. Darum verlangt
`Config::validate`, dass `hidden_score_step` höchstens halb so groß ist wie
`hidden_gap_threshold` (Standard: 0.01 und 0.05). Ein Modell, das auf den
Hidden-Daten genauso gut ist wie auf Valid, kommt so nie allein wegen des
Rundens in Quarantäne. Wer in der geheimen Datei mit `BRUT_HIDDEN_STEP` eine
gröbere Stufe festlegt, sollte `hidden_gap_threshold` entsprechend
mindestens doppelt so groß wählen; der Orchestrator kennt diese Stufe nicht.

## Abstammungsarchiv

`archive.jsonl` hat eine JSON-Zeile pro Eintrag. Jeder Eintrag enthält:
laufende Nummer `seq`, Zeitstempel, `prev_hash` (Hash des vorigen Eintrags),
den Inhalt und `hash`. `hash` ist der SHA-256-Wert über genau diese Zeile,
in der `hash` leer ist. Beim Prüfen rechnen wir den Hash über die
gespeicherten Bytes, womit die Prüfung nicht davon abhängt, wie
Gleitkommazahlen eingelesen werden.

Einträge: `RunStarted`, `Candidate`, `HiddenEval`, `Approved`,
`SafetyEvent`, `RunFinished`. Der Lerner-Modus fügt eigene Einträge hinzu
(siehe „Speicherung im Lerner-Modus“). Der MLP-Modus überliest sie.

Die Kette erkennt keine Einträge, die am Ende fehlen. Darum schreibt das
Archiv nach jedem Eintrag die Datei `archive.head` neu (Zahl der Einträge und
Hash des letzten Eintrags, atomar ersetzt). Beim Öffnen und bei `brut verify`
muss der Anker zum Archiv passen. Er darf höchstens einen Eintrag
zurückliegen; das passiert nur, wenn das Programm zwischen den beiden
Schreibvorgängen abstürzt.

Vor dem Schreiben prüft das Archiv, dass der Eintrag nach JSON und zurück
unverändert bleibt. So gelangen keine NaN-Werte ins Archiv (JSON kann sie
nicht darstellen).

## Aufgaben

| Name | Eingaben | Art | Train / Valid / Hidden | Besonderheit |
|---|---|---|---|---|
| `spirals` | 2 | 2 Klassen | 400 / 400 / 4000 | zwei Spiralen mit je 3 Windungen, Rauschen 0.03 |
| `checkerboard` | 2 | 2 Klassen | 1000 / 1000 / 4000 | 4 × 4 Schachbrett auf [−1, 1]², Eingaberauschen 0.02 (etwa 5 % der Punkte im Feld der anderen Klasse; erreichbar ≈ 0.95) |
| `parity` | 8 | 2 Klassen | 256 / 256 / 1024 | Parität von 8 Bits; nur 256 mögliche Eingaben |
| `sine` | 1 | Regression | 500 / 500 / 2000 | y = sin(3x) + 0.3 sin(9x) + Rauschen 0.1 |

Die Spiralen-Aufgabe hat Alexis P. Wieland vorgeschlagen; bekannt wurde sie
durch Lang & Witbrock (1988). Die Version hier ist eine stetige Abwandlung
mit zufälligen Punkten, nicht der Originaldatensatz mit 194 Punkten.

Beim Schachbrett wird die Klasse aus dem Punkt ohne Rauschen bestimmt, das
Rauschen kommt erst danach auf die Eingabe. Dadurch liegen etwa 5 % der
Punkte im Feld der anderen Klasse. Mehr als etwa 0.95 Genauigkeit ist dort
also nicht erreichbar; ein Plateau bei 0.95 ist keine Unteranpassung.

## Lerner-Modus (Stufe 1)

Der Lerner-Modus ist ein zweiter, eigenständiger Modus. Er teilt mit dem
MLP-Modus das Archiv, den Not-Aus, die Grenzen und den getrennten Auswerter
`brut-eval`. Verbindlich für den Code ist der Bauvertrag
`docs/STAGE1_CONTRACT.md` mit den Korrekturen in
`docs/STAGE1_CONTRACT_ERRATA.md`. Die Absicht dahinter erklärt
`docs/PLAN_GENERALISIERUNG.md`.

Datenfluss im Überblick (`src/meta_evolve.rs`):

```
meta.json ──► meta_evolve::meta_run
                │
                ├─ Vergleichsverfahren abstimmen (einmal je Lauf)
                ├─ Startpopulation: 4 Startprogramme, leicht mutiert
                │
                ├─ je Runde, parallel, Ergebnisse in fester Reihenfolge:
                │     Turnier ─► Mutation ─► check() ─► Fingerabdruck
                │       ─► Stufe A (5 Aufgaben, Hürde) ─► Stufe B (20 Aufgaben)
                │       ─► Fitness ─► Population, Ruhmeshalle, Abstammungsblock
                │
                ├─ alle 50 Runden: 5 Panel-Aufgaben austauschen
                ├─ alle 200 Runden: Stufe C auf frischen SELECT-Aufgaben
                ├─ alle 1000 Runden: Schnappschuss und BatchDigest
                ├─ Guard: STOP-Datei, Zeit, FLOPs (vor jeder Runde)
                │
                └─ Endphase: Finalisten, Nachstimmen, Δ_public
                      ─► brut-eval (verdeckte Stufen) ─► report.json
```

### Begriffe im Lerner-Modus

- **Lernprogramm (Lerner):** ein Kandidat des Lerner-Modus. Es besteht aus
  drei geraden Befehlsfolgen `Setup`, `Predict` und `Learn`
  (`LearnerGenome`). „Gerade“ heißt: ohne Schleifen und ohne Sprünge. Jeder
  Befehl läuft je Aufruf genau einmal.
- **DSL:** die kleine, geschlossene Sprache der Lernprogramme (englisch
  „domain-specific language“, `src/learner/dsl.rs`).
- **Register:** feste Speicherplätze eines Lernprogramms: 10 Zahlen s0 bis
  s9, 8 Vektoren v0 bis v7 der Länge F und 4 Matrizen m0 bis m3 der Größe
  F × F.
- **Breite F:** die Zahl der Eingabemerkmale einer Aufgabe: 8, 16 oder 32.
- **Familie und Instanz:** Eine Aufgabenfamilie ist eine Art von Aufgabe.
  Eine Instanz ist eine konkrete Aufgabe daraus, mit eigenen Zufallszahlen.
- **Stufe:** eine Menge von Aufgaben mit einer festen Rolle. SEARCH dient der
  Suche, SELECT der Auswahl der Finalisten. Die verdeckten Stufen
  HIDDEN-INSTANCE, HIDDEN-NEAR, HIDDEN-FAR und HIDDEN-GRAMMAR gibt es nur in
  `brut-eval`.
- **Panel:** die 20 SEARCH-Aufgaben, auf denen die Suche die Fitness misst.
- **AULC:** Fläche unter der Lernkurve (Formel unten).
- **Vergleichsverfahren (Baseline):** ein fest gewähltes Lernverfahren, mit
  dem jedes Programm verglichen wird.
- **Δ (Delta):** Vorsprung eines Programms vor dem besten gleich teuren
  Vergleichsverfahren.
- **Fingerabdruck:** ein Hash über die Ausgaben eines Programms auf zwei
  kleinen Probeaufgaben. Gleiche Fingerabdrücke heißen: gleiches Verhalten.
- **Hürde:** eine Schwelle nach den ersten 5 Panel-Aufgaben. Nur wer sie
  nimmt, wird auf allen 20 Aufgaben bewertet.
- **Ruhmeshalle (Hall of Fame):** die besten Programme nach Fitness, ohne
  doppelte Fingerabdrücke (Standard: 32).
- **Finalist:** eines der Programme (Standard: 3), die nach der Suche
  verdeckt bewertet werden.
- **Protokoll-Hash:** SHA-256 über alles, was die Messung festlegt:
  Familien, Aufteilung, Grammatik-Töpfe, Messregeln, Startprogramme und
  Befehlssatz.
- **Abfragebuch (Ledger):** die Datei `ledger.jsonl` neben der geheimen
  Datei. Sie hält jede verdeckte Abfrage als Hash-Kette fest, über alle
  Läufe.
- **Gemeinsame Zufallszahlen:** Zufallsbefehle eines Programms ziehen aus
  dem Seed der Aufgabe, nicht aus dem Programm. Alle Programme sehen auf
  derselben Aufgabe dieselben Zufallszahlen. Niemand kann sich einen
  glücklichen Seed „erkaufen“.

### Ein Lernprogramm

Feste Regeln für die Register:

- Vor jedem `Predict` steht die Eingabe x in v0. Das Label (die richtige
  Antwort) in v2 und s0 wird vorher auf null gesetzt.
- Nach `Predict` steht die Vorhersage in v1. Der Auswerter wendet die
  passende Ausgabefunktion an: bei Regression keine, bei zwei Klassen die
  Sigmoid-Funktion auf v1[0], bei k Klassen Softmax über v1[0..k].
- Direkt vor `Learn` steht das Label in v2 und s0. Bei k Klassen ist v2 ein
  „One-hot“-Vektor: 1 an der Stelle der richtigen Klasse, sonst 0.
- `Setup` läuft einmal je Aufgabe auf genulltem Speicher.

Der Befehlssatz hat 46 Befehle in fünf Gruppen: 15 mit einer Zahl als
Ergebnis (zum Beispiel Addieren, Sinus, Zufallszahl), 17 mit einem Vektor
als Ergebnis (zum Beispiel Matrix mal Vektor), 5 mit einer Zahl aus einem
Vektor (zum Beispiel Skalarprodukt), 8 mit einer Matrix als Ergebnis (zum
Beispiel äußeres Produkt) und 1 mit einer Zahl aus einer Matrix (Norm).
Absichtlich fehlen: das Produkt zweier Matrizen, Verzweigungen, Schleifen
und jeder Befehl mit einem Begriff des maschinellen Lernens (etwa
„Gradient“). So hat es auch AutoML-Zero gemacht.

Jeder Befehl hat feste Kosten in FLOPs, abhängig von F (zum Beispiel Matrix
mal Vektor: 2·F²). Weil es keine Schleifen gibt, stehen die Kosten eines
Programms vor dem Start fest. Die **Kostengrenze**: `Predict` plus `Learn`
dürfen je Schritt höchstens das Vierfache des Referenznetzes kosten
(10·F² + 5·F), für jede Breite F. Das prüft `LearnerGenome::check_cost`,
bevor ein Programm überhaupt läuft. Dazu kommen Längengrenzen (16, 24 und
32 Befehle für `Setup`, `Predict` und `Learn`) und eine Obergrenze von 1000
für den Betrag jeder Konstanten.

`LearnerGenome::simplify` entfernt toten Code: Befehle, deren Ergebnis nie
gelesen wird. Die Vorhersagen bleiben dabei bitgenau gleich. Die Länge des
vereinfachten Programms heißt **effektive Länge**; sie geht in die Fitness
ein.

### Ablauf auf einer Aufgabe

`protocol::run_learner` (Vertrag Abschnitt 6.2):

1. Speicher auf null. Der Zufallsgenerator des Programms bekommt den Seed
   `derive_seed(task.seed, "interp/<Versuch>")`. `Setup` läuft.
2. 384 Lernschritte. Jeder Schritt nimmt das nächste Trainingsbeispiel aus
   einer gemischten Reihenfolge (jede Runde durch die Daten neu gemischt):
   Eingabe nach v0, Label-Register leeren, `Predict`, Ausgabefunktion.
   Abbruch bei einer nicht endlichen Ausgabe oder, bei Regression, bei einem
   Fehler über 100. Dann Label setzen und `Learn`.
3. Nach 24, 96 und 384 Schritten: Der Speicher wird kopiert. Auf der Kopie
   läuft `Predict` für jedes Validierungsbeispiel. Daraus entsteht die
   Qualität q. Die Validierung kann den trainierten Lerner also nicht
   verändern.
4. Not-Aus und Zeitgrenze werden bei Schritt 0, danach alle 64 Schritte und
   vor jeder 64. Validierungsvorhersage geprüft (Errata E11 und V5).
5. Ein Programm über der Kostengrenze läuft gar nicht und bekommt 0.

Qualität an einem Prüfpunkt (0 = nicht besser als Raten, 1 = perfekt):

```
Regression:      q = clip(1 − MSE / Var(y_valid), 0, 1)
Klassifikation:  q = clip((Genauigkeit − a0) / (1 − a0), 0, 1)
                 a0 = Anteil der häufigsten Klasse in den Validierungsdaten
AULC:            A = 0.25 · q(24) + 0.25 · q(96) + 0.5 · q(384)
```

MSE ist der mittlere quadratische Fehler, Var die Varianz. `clip` begrenzt
den Wert auf das Intervall. Eine abgebrochene Aufgabe zählt A = 0.
Ist an einem Prüfpunkt auch nur eine Validierungsausgabe nicht endlich
(NaN oder unendlich), gilt q = 0 für diesen Prüfpunkt (Erratum V2). Sonst
würde eine NaN-Zeile als Vorhersage der Klasse 0 zählen.

Zwei Rechenregeln des Interpreters weichen von reinem IEEE-Rechnen ab
(Errata V3 und V4): min und max liefern bei einem NaN-Operanden den anderen
Operanden, unabhängig von der Reihenfolge. Beträge unter 2⁻³⁴ (in Matrizen
unter 2⁻⁶⁸) werden als 0 gespeichert, damit keine langsamen subnormalen
Zahlen entstehen.

### Aufgaben

Jede Instanz zieht eine verborgene Struktur in 2 bis 4 Dimensionen. Diese
wird mit einer zufälligen Drehung in F Eingabedimensionen eingebettet; die
restlichen Dimensionen sind Rauschen. Danach werden die Merkmale mit
Mittelwert und Standardabweichung der Trainingsdaten standardisiert. Jede
Instanz hat 16, 64 oder 256 Trainingsbeispiele, einen Strom von 384
Lernschritten und 64 (Suche) oder 256 (verdeckt) Validierungsbeispiele.

24 handgeschriebene Familien in 6 Gruppen (`src/learner/families.rs`):

| Gruppe | Familien |
|---|---|
| linear | lin-reg, lin-cls, sparse-lin-reg, multiclass-lin |
| Wechselwirkung | quad-reg, xor-cls, sign-parity-cls, bilinear-reg |
| radial | sphere-cls, blobs-multiclass, rbf-reg, prototype-cls |
| periodisch | sine-reg, checkerboard-cls, spirals-cls, sawtooth-reg |
| stückweise | tree-cls, relu-teacher-reg, step-reg, dnf-cls |
| Robustheit | drift-cls, noisy-label-cls, heavy-tail-reg, tanh-teacher-cls |

Die Aufteilung „split-v1“ ist mit SHA-256 ausgelost, nicht von Hand gewählt.
Ein Test hält sie fest:

| Stufe | Familien |
|---|---|
| SEARCH | lin-cls, multiclass-lin, quad-reg, bilinear-reg, blobs-multiclass, sphere-cls, dnf-cls, relu-teacher-reg, drift-cls, noisy-label-cls |
| SELECT | lin-reg, sign-parity-cls, prototype-cls, step-reg, tanh-teacher-cls |
| HIDDEN-NEAR | sparse-lin-reg, xor-cls, rbf-reg, tree-cls, heavy-tail-reg |
| HIDDEN-FAR | die ganze Gruppe „periodisch“ |

Fünf Familien hat der Entwickler schon im Vorversuch oder im bestehenden
Code gesehen (xor-cls, sine-reg, checkerboard-cls, spirals-cls,
sign-parity-cls). Sie tragen die Markierung `developer_seen`. Darum sind
HIDDEN-NEAR und HIDDEN-FAR nur beschreibend.

**Formel-Grammatik** (`src/learner/grammar.rs`, nach Legg & Hutter 2007):
Eine Familie ist eine zufällige Formel über drei Projektionen der
verborgenen Struktur. Bausteine: Addition, Multiplikation, Maximum, Sinus,
Betrag, Quadrat und Stufe. Die Tiefe ist höchstens 3. Die Länge L (Zahl der
Knoten) liegt zwischen 2 und 9; kürzere Formeln sind wahrscheinlicher
(Gewicht 2^−L). Klassifikation teilt die Formelwerte in 2 bis 4 gleich große
Klassen. Regression nutzt die standardisierte Formel plus Rauschen. Die
**Signatur** ist die Gestalt der Formel ohne Konstanten. Drei Töpfe:

| Topf | Familien | Seed | Bedingung |
|---|---|---|---|
| SEARCH-Grammatik | 64 | öffentlich, im Code | — |
| SELECT-Grammatik | 32 | öffentlich, im Code | Signaturen nicht in SEARCH |
| HIDDEN-GRAMMAR | 40 | geheim | Signaturen in keinem öffentlichen Topf; gleiche Längenverteilung wie SEARCH |

Die verdeckten Stufen (nur in `brut-eval`, je 256 Validierungsbeispiele):

| Stufe | Familien | Instanzen | F |
|---|---|---|---|
| HIDDEN-INSTANCE | die 10 SEARCH-Familien | 10 je Familie (100) | 8, 16 |
| HIDDEN-NEAR | die 5 HIDDEN-NEAR-Familien | 20 je Familie (100) | 8, 16, 32 |
| HIDDEN-FAR | die 4 periodischen Familien | 25 je Familie (100) | 8, 16, 32 |
| HIDDEN-GRAMMAR | 40 geheime Grammatik-Familien | 5 je Familie (200) | 8, 16, 32 |

Für alle Mengen, in denen F und die Zahl der Trainingsbeispiele „reihum“
gehen, gilt eine Regel (Erratum E7): Die k-te Aufgabe der Menge hat
`F = dims[k mod |dims|]` und `n_train = [16, 64, 256][(k / |dims|) mod 3]`.
So kommen alle Paare aus F und n_train etwa gleich oft vor. Sonst hätte jede
Aufgabe mit F = 32 auch 256 Beispiele, und der Vergleich zwischen den
Breiten würde eigentlich die Datenmenge messen.

### Suche

Die Suche ist eine **regularisierte Evolution** („Aging Evolution“, Real et
al. 2019): Eine Population von 256 Programmen; jedes neue Programm verdrängt
das älteste. Je Runde („Batch“) entstehen 64 Kinder parallel.

- **Elternwahl:** Turnier. 10 verschiedene Mitglieder werden zufällig
  gezogen; das mit der besten Fitness wird Elter, bei Gleichstand das
  jüngere.
- **Mutation** wie bei AutoML-Zero: Befehl einfügen (10 %), Befehl entfernen
  (20 %), einen Programmteil neu würfeln (10 %), ein Argument ändern (30 %),
  eine Konstante mit einem Faktor zwischen 0.5 und 2 multiplizieren (30 %,
  dabei 10 % Vorzeichenwechsel). Jedes Kind besteht `check()`.
- **Startpopulation:** Mitglied i ist Startprogramm i mod 4, höchstens
  dreimal mutiert.
- **Suchmodi:** `full` (alles), `constants-only` (nur Konstanten),
  `random-search` (kein Turnier: ein Startprogramm mit 1 bis 8 Mutationen,
  Erratum E10).

**Gestufte Bewertung** je Kind:

1. Fingerabdruck auf zwei Probeaufgaben. Ein Speicher hält Ergebnisse je
   (Fingerabdruck, Aufgabe). Bekannte Ergebnisse werden wiederverwendet.
2. Stufe A: 5 der 20 Panel-Aufgaben. Liegt ihr Mittelwert unter dem
   75. Perzentil der letzten 1000 Stufe-A-Werte, ist das Kind an der Hürde
   gescheitert. Es kommt nicht in die Population und wird nur gezählt. In den
   ersten 100 Werten gibt es noch keine Hürde.
3. Stufe B: die übrigen Panel-Aufgaben, daraus die Fitness. Das Kind kommt in
   die Population.

Während einer Runde wird der Ergebnis-Speicher nur gelesen. Neue Einträge
und die Hürdenwerte kommen erst nach der Runde dazu, in der festen
Reihenfolge der Kinder. Darum hängt das Ergebnis nicht von der Thread-Zahl
ab.

**Panel:** Positionen 0 bis 9 sind je eine Instanz der 10 SEARCH-Familien.
Positionen 10 bis 19 kommen aus der SEARCH-Grammatik. Alle 50 Runden werden
5 Positionen durch frische Instanzen ersetzt. Ein festes Panel führte im
Vorversuch zu Überanpassung. Nur die ersetzten Aufgaben werden für die
Population neu gerechnet.

**Fitness:**

```
A_t(g)   = AULC von g auf Panel-Aufgabe t
ref_t    = AULC des unveränderten Startprogramms nn2-sgd auf t
M(g)     = Mittelwert der A_t über die 20 Aufgaben
B(g)     = Mittelwert der 5 kleinsten Werte von (A_t(g) − ref_t)
Fitness  = 0.75 · M(g) + 0.25 · B(g)
           − 0.005 · log10(1 + effektive Länge)
           − 0.002 · log10(max(Kosten je Schritt bei F = 16; 1))
```

B(g) ist der **Breitenterm**: Wer nur wenige Familien meistert, gewinnt
nicht. `max(…; 1)` verhindert log10(0).

**Stufe C (Auswahl):** Alle 200 Runden und einmal am Ende werden die
Programme der Ruhmeshalle auf frischen SELECT-Aufgaben gemessen (8 je
SELECT-Familie und 32 SELECT-Grammatik-Familien), zusammen mit den
gleich teuren Vergleichsverfahren. Daraus entsteht Δ_select. Stufe C wirkt
nie auf Fitness oder Elternwahl zurück.

**Vergleichsverfahren:** die vier Startprogramme, deren Konstanten mit
gleichem Aufwand abgestimmt werden (200 Versuche auf 36 SELECT-artigen
Aufgaben, wie bei AutoML-Zero), und ein Adam-MLP aus `nn.rs`. Sein Budget
an jedem Prüfpunkt sind die Trainingskosten des Programms bis dahin (Setup
plus Schritte mal Kosten je Schritt, ohne die Validierung; Erratum V1). Es
trainiert aber mindestens eine Epoche über alle bisher gesehenen Beispiele.
Bei billigen, linearen Programmen sind das ein bis wenige volle Schritte,
und das Netz rechnet dann oft mehr als das Programm. Es ist dort ein
schwacher Vergleich; das stärkste Vergleichsverfahren war in Plan S das
abgestimmte lineare Startprogramm. In den Vergleich
B(g) eines Programms kommen nur Startprogramme, die je Schritt höchstens so
viel kosten wie es. `tuned_baselines()` ist deterministisch. Orchestrator
und `brut-eval` rechnen es beide und müssen denselben Hash erhalten
(Erratum E8).

**Endphase** (nach dem Ende der Suche, auch nach einer Zeit- oder
FLOP-Grenze; der Not-Aus bricht sie ab):

1. Letzte Stufe C. Die 3 Programme mit dem größten Δ_select werden
   Finalisten.
2. Ihre Konstanten werden wie die der Vergleiche abgestimmt (200 Versuche).
   Das Ergebnis bekommt eine neue Kennung, den Generator „retune“ und eine
   Datei `programs/<id>.json`.
3. Δ_public: Vorsprung auf 100 frischen öffentlichen Aufgaben der
   SEARCH-Familien.
4. Verdeckte Bewertung (siehe unten), außer mit `--no-hidden`.
5. `report.json` und der Eintrag `RunFinished`.

`max_total_flops` begrenzt nur die Suche (Startpopulation, Runden,
Panel-Austausch, Stufe C). Das Abstimmen der Vergleiche (gemessen 5.6 · 10¹⁰
FLOPs) und die Endphase (in Plan S 0.6 bis 5.2 · 10¹⁰ FLOPs) zählen getrennt und stehen
getrennt im Bericht (Erratum E6, vom Projektleiter noch zu bestätigen).

### Verdeckte Bewertung im Lerner-Modus

```
meta_evolve (Orchestrator)                    brut-eval (eigener Prozess)
  Finalist + Stufe
    └─ JSON-Anfrage mit dem Programm ───────► liest ~/.brut/learner/hidden.env
       (höchstens 1 MiB)                      prüft Programm und Protokoll-Hash
                                              bucht die Abfrage (Charge, unter eval.lock)
                                              baut die Aufgaben der Stufe (HiddenToken)
                                              rechnet Programm und Vergleiche
                                              rundet Δ ab, beim Hauptwert + Entscheidung
                                              hält die Ausgabe fest (Release)
  archive.jsonl (LearnerHiddenEval) ◄──────── eine Zahl, höchstens ein Bit
```

Für eine Stufe T mit den Familien f gilt:

```
S_T(g) = Mittelwert über die Familien von (Mittelwert der A_t in der Familie)
b*     = das Vergleichsverfahren in B(g) mit dem größten S_T
Δ_T(g) = S_T(g) − S_T(b*)          herausgegeben: auf 0.01 abgerundet
```

**Hauptwert und Entscheidung** (nur HIDDEN-GRAMMAR, vorab festgelegt):

- Für jede der 40 Familien ist d_f die mittlere Differenz zwischen Programm
  und b* über ihre 5 Aufgaben.
- Ein einseitiger **Vorzeichen-Permutationstest** prüft, ob der Mittelwert
  der d_f über 0 liegt. Die Idee: Ist das Programm in Wahrheit nicht besser,
  sind positive und negative d_f gleich wahrscheinlich. Der Test dreht die
  Vorzeichen 100 000-mal zufällig um. Der p-Wert ist der Anteil der
  Umdrehungen, deren Mittelwert mindestens so groß ist wie der echte.
- „Bestanden“ heißt: p < 0.05 **und** Δ ≥ 0.01 **und** Δ bei F = 32 ist
  höchstens 0.05 kleiner als Δ bei F = 8 und 16.
- Nur Δ und dieses eine Bit verlassen `brut-eval`. Der p-Wert, Intervalle
  und Werte je Familie bleiben dort.

**Budget:** In der geheimen Datei stehen die Obergrenzen: 48 Abfragen je
Protokollversion über alle Läufe, 12 je Lauf. Das Flag `--budget` kann sie
nur senken. Jede Abfrage wird gebucht, bevor bewertet wird. Abgelehnte
Abfragen (zum Beispiel Budget erschöpft oder falscher Protokoll-Hash)
stehen im Abfragebuch, kosten aber nichts. Ein ungültiges Programm wird
ebenfalls vor der Buchung abgewiesen. Ist das Budget erschöpft, hängt
der Orchestrator einen `SafetyEvent` an, und der Lauf endet trotzdem
normal.

**Quarantäne:** Lücke = Δ_public − Δ_HIDDEN-INSTANCE. Beide Werte entstehen
mit denselben Einstellungen und denselben Vergleichen. Ist die Lücke größer
als `gap_threshold` (Standard 0.05), kommt der Finalist in Quarantäne und
kann nicht freigegeben werden. Die Rundung (höchstens 0.01) muss kleiner als
die Hälfte der Schwelle sein; das prüft `MetaConfig::validate`.

**Feste Liste und Endgültigkeit:** Die Finalisten stehen vor der ersten
verdeckten Abfrage fest. Nach dem ersten `LearnerHiddenEval` ist der Lauf
abgeschlossen. Ein weiterer `brut run` füllt nur fehlende Paare aus
Finalist und Stufe derselben Liste auf und lehnt neue Runden ab.

**Fähigkeits-Schutz:** Die Funktionen, die verdeckte Aufgaben bauen,
verlangen ein Objekt `HiddenToken`. Nur `src/learner_hidden.rs` kann es
erzeugen. Das ist Ordnung im Code, keine Sicherheitsgrenze: Wer den Code
ändert, kann es umgehen.

### ARC-Außentest

ARC-AGI besteht aus Rätseln mit farbigen Rastern (bis 30 × 30 Zellen, 10
Farben). Aus wenigen Beispielpaaren muss man die Regel erkennen und auf eine
Testeingabe anwenden.

- **phi32:** Jede Zelle bekommt 32 handgemachte Merkmale: ihre Farbe (10),
  ob die 8 Nachbarn dieselbe Farbe haben (8), ob die 8 Nachbarn nicht leer
  sind (8) und ihre Lage (6). Zielgröße ist die Farbe der Ausgabezelle.
- Aus jedem Testpaar wird eine Lernaufgabe mit F = 32 und 10 Klassen: Die
  Zellen aller Trainingspaare sind die Trainingsbeispiele, die Zellen der
  Testeingabe die Validierung. Nur Aufgaben, deren Raster ihre Größe
  behalten, sind lösbar.
- Zwei Versuche (Interpreter-Versuche 0 und 1), je Zelle die Farbe mit der
  höchsten Ausgabe. Gezählt wird nur ein exakt gleiches Raster („pass@2“).
- Ergebnisse: **offizieller Wert** (Anteil richtiger Testausgaben je
  Aufgabe, gemittelt über alle Aufgaben), **Kaggle-Wert** (richtige
  Testausgaben durch alle Testausgaben) und **ausgeglichene
  Zellgenauigkeit** (Mittel der Genauigkeit auf veränderten und auf
  unveränderten Zellen; die reine Kopie der Eingabe erreicht damit nur 0.5).

Ablauf in `brut-eval` vor jeder Bewertung: Programm prüfen, Protokoll-Hash
prüfen, Kontaminationsregeln, Hashes aller Dateien im Manifest, Vergleich
normierter Hashes. Erst dann wird die Abfrage gebucht. **Kontamination**
heißt: Daten aus dem Bewertungssatz stecken schon im Training. Diese Quellen
sind als Trainingsquellen verboten:

| Bewertungssatz | verbotene Trainingsquellen |
|---|---|
| ARC-AGI-1 Bewertung | ARC-AGI-1 Bewertung, ARC-AGI-2 Training, ARC-GEN, ARC-TGI, BARC |
| ARC-AGI-2 Bewertung | ARC-AGI-2 Bewertung, ARC-AGI-1 Bewertung, ARC-TGI, BARC |
| ConceptARC | ConceptARC, BARC |

BARC ist immer verboten, weil es keine Lizenzdatei hat. Der **normierte
Hash** eines Beispielpaars ist über alle 8 Drehungen und Spiegelungen und
über neu nummerierte Farben gleich. Findet `brut-eval` ein Trainingspaar mit
demselben normierten Hash wie ein Bewertungspaar, lehnt es die Prüfung ab.
Je Bewertungssatz und Protokollversion sind 3 Prüfungen erlaubt.

Feste Vergleichslinien, auf dem Entwicklungsrechner nachgezählt:

| Bezug | ARC-AGI-1 Bewertung | ARC-AGI-2 Bewertung | ConceptARC |
|---|---|---|---|
| Aufgaben mit gleicher Rastergröße | 270 von 400 | 81 von 120 | 119 von 160 |
| Obergrenze jedes phi32-Zell-Lerners | 9 von 400 | 0 von 120 | 17 von 160 |
| 1-nächster-Nachbar mit phi32 | 2 von 400 | 0 von 120 | 0 von 160 |
| icecuber-Programmsuche (2020), pass@2 | 28.9 % der Testausgaben | 1 von 120 | — |

### Reproduzierbarkeit im Lerner-Modus

Alle Zufallszahlen kommen aus ChaCha8-Generatoren mit Seeds aus
`util::derive_seed(master, tag)`:

| Zweck | master | tag |
|---|---|---|
| Startmitglied i | Master-Seed des Laufs | `init/<i>` |
| Runde b, Platz s (Turnier und Mutation) | Master-Seed | `batch/<b>/slot/<s>` |
| Panel-Position p, Generation g | Master-Seed | `panel/<p>/<g>` |
| Stufe C, Runde r | Master-Seed | `select/<r>/<Familie>/<i>` |
| Nachstimmen eines Finalisten | Master-Seed | `retune/<id>` |
| öffentliche Vergleichsaufgaben | Master-Seed | `pubref/<Familie>/<i>` |
| Zufallsbefehle eines Programms | Seed der Aufgabe | `interp/<Versuch>` |
| Reihenfolge der Beispiele, Epoche e | Seed der Aufgabe | `stream/<e>` |
| Probeaufgaben des Fingerabdrucks | 0 | `fec/probe/<0\|1>` |
| öffentliche Grammatik-Töpfe | 0 | `grammar/search/learner-v1`, `grammar/select/learner-v1` |
| Abstimmen der Vergleiche | 0 | `baselines/tune` |
| ARC-Zellaufgabe j der Aufgabe id | 0 | `arc-cell/<id>/<j>` (Erratum E9) |
| verdeckte Instanzen | geheimer Seed | `tier/<Stufe>/<Familie>/<i>` |
| verdeckter Grammatik-Topf | geheimer Seed | `grammar/hidden` |
| Vorzeichen-Permutationstest | geheimer Seed | `test/<id>/<Stufe>` |

Zahlen stehen ohne führende Nullen, Stufen in Kleinbuchstaben mit
Unterstrich (zum Beispiel `hidden_grammar`, Erratum E13).

Gleiche `meta.json` ergibt dasselbe Archiv (bis auf Zeitstempel) und
dieselben Hashes der Abstammungsblöcke, bei jeder Thread-Zahl. Ausnahmen
sind, wie im MLP-Modus, Abbrüche durch Zeitgrenzen und den Not-Aus.

**Fortsetzen:** Der Lauf nimmt den letzten `BatchDigest`, lädt dessen
Schnappschuss und Ergebnis-Speicher, löscht neuere Abstammungsblöcke und
macht mit der nächsten Runde weiter. Ein fortgesetzter Lauf erzeugt danach
dieselben Blöcke wie ein ununterbrochener.

### Speicherung im Lerner-Modus

Neue Einträge im Archiv (alte Archive bleiben lesbar):

| Eintrag | Inhalt |
|---|---|
| `MetaRunStarted` | Konfiguration, Protokoll-Hash, Version |
| `ProtocolRegistered` | das ganze Messprotokoll (Vorab-Registrierung) |
| `LearnerCandidate` | ein Programm mit Status: Startprogramm, Ruhmeshalle, Finalist, Vergleich oder Quarantäne |
| `BatchDigest` | Runden-Bereich, Hashes von Abstammungsblock, Schnappschuss und Ergebnis-Speicher, Zähler |
| `PanelRotated` | Austausch von Panel-Aufgaben |
| `SelectEval` | Ergebnis einer Stufe-C-Messung |
| `LearnerHiddenEval` | ein verdecktes Ergebnis: Stufe, Δ, Entscheidung, Δ_public, Quarantäne |
| `ExternalEval` | ein ARC-Ergebnis |

Kinder, die in die Population kommen, stehen nicht einzeln im Archiv. Sie
stehen in **Abstammungsblöcken** (`lineage/chunk-*.jsonl`, etwa 110 Byte je
Kind). Alle 1000 Runden (`snapshot_every`) schreibt der Lauf einen Block,
einen **Schnappschuss** der Population (`snapshots/pop-*.json`) und den
Ergebnis-Speicher (`snapshots/cache-latest.json`). Der Eintrag `BatchDigest`
hält ihre SHA-256-Hashes fest. `Archive::open` und `brut verify` prüfen
diese Hashes; beim Ergebnis-Speicher nur den letzten.

`report.json` enthält je Finalist: Δ für jede verdeckte Stufe, die
Entscheidung beim Hauptwert, Δ_public, die Quarantäne, das
Transferverhältnis Δ_HIDDEN-GRAMMAR / Δ_HIDDEN-INSTANCE, die Kopplung an die
Breite (Δ bei F = 32 gegen F = 8 und 16 auf SELECT), den „Fluch des
Gewinners“ (Panel-Wert minus Wert auf frischen Aufgaben) und die Lernkurve.

### Rechenaufwand im Lerner-Modus

Das Referenznetz kostet etwa 10·F² je Lernschritt und 4·F² je Vorhersage.
Eine Aufgabe kostet damit 384 · 10·F² + 3 · 64 · 4·F² = 4608·F² FLOPs, also
etwa 3 · 10⁵ bei F = 8 und 1.2 · 10⁶ bei F = 16. Ein neues Kind kostet mit
Hürde und Ergebnis-Speicher im Mittel etwa 1.5 · 10⁶ FLOPs (Schätzung des
Plans). Ein Lauf mit 1.5 · 10¹² FLOPs bewertet damit grob eine Million
Kinder.

## Erweiterung

Die Schnittstelle `variation::VariantGenerator` (in Rust ein „Trait“)
erlaubt andere Generatoren. Jeder Generator muss ein Genom liefern, das
`Genome::validate` besteht. Ein Genom bleibt dabei immer reine Daten.

Im Lerner-Modus spielt `mutate::ProgramProposer` dieselbe Rolle. Jeder
Vorschlag muss `LearnerGenome::check` bestehen. Das ist die vorbereitete
Stelle für Stufe 2, in der ein Sprachmodell Programme vorschlagen könnte.
Ebenfalls vorbereitet ist ein festes Nachrichtenformat (`src/learner/wire.rs`,
eine JSON-Zeile je Nachricht). Über dieses Format könnte später ein Lerner in
einem abgeschotteten Prozess mit dem Auswerter sprechen. Die eingebaute
DSL-Fassung (`DslWireLearner`) dient dabei als Referenz: Sie muss bitgenau
dieselben Werte liefern wie `run_learner`. Ein Unterschied ist bekannt: Ein
fremder Lerner sähe die Eingaben der Validierung und könnte sie nutzen; die
DSL kann das nicht.

## Quellen

- Lang, K. J., & Witbrock, M. J. (1988). *Learning to tell two spirals
  apart.* Proceedings of the 1988 Connectionist Models Summer School.
- Blum, A., & Hardt, M. (2015). *The Ladder: A Reliable Leaderboard for
  Machine Learning Competitions.* Proceedings of the 32nd International
  Conference on Machine Learning (ICML), PMLR 37, 1006–1014.
- Dwork, C., Feldman, V., Hardt, M., Pitassi, T., Reingold, O., & Roth, A.
  (2015). *The reusable holdout: Preserving validity in adaptive data
  analysis.* Science, 349(6248), 636–638.
- Real, E., Liang, C., So, D., & Le, Q. (2020). *AutoML-Zero: Evolving
  Machine Learning Algorithms From Scratch.* Proceedings of the 37th
  International Conference on Machine Learning (ICML), PMLR 119, 8007–8019.
  Vorbild des Lerner-Modus: drei Programmteile, Befehlssatz ohne Begriffe des
  maschinellen Lernens, Mutationen, Abbruchregeln, Hürden und Fingerabdrücke.
- Real, E., Aggarwal, A., Huang, Y., & Le, Q. V. (2019). *Regularized
  Evolution for Image Classifier Architecture Search.* Proceedings of the
  AAAI Conference on Artificial Intelligence 33(01), 4780–4789.
- Legg, S., & Hutter, M. (2007). *Universal Intelligence: A Definition of
  Machine Intelligence.* Minds and Machines 17(4), 391–444. Vorbild der
  Formel-Grammatik (kürzere Aufgaben häufiger).
- Chollet, F. (2019). *On the Measure of Intelligence.* arXiv:1911.01547
  (Vorabdruck, nicht begutachtet). Herkunft von ARC-AGI.
- Chollet, F. et al. (2025). *ARC-AGI-2: A New Challenge for Frontier AI
  Reasoning Systems.* arXiv:2505.11831 (Vorabdruck).
- Moskvichev, A., Odouard, V. V., & Mitchell, M. (2023). *The ConceptARC
  Benchmark: Evaluating Understanding and Generalization in the ARC Domain.*
  Transactions on Machine Learning Research (TMLR, 08/2023, begutachtet).
  arXiv:2305.07141.
- Zhang, J., Hu, S., Lu, C., Lange, R., & Clune, J. (2026). *Darwin Gödel
  Machine: Open-Ended Evolution of Self-Improving Agents.* International
  Conference on Learning Representations (ICLR 2026). arXiv:2505.22954. Die
  Brutmaschine übernimmt nur einen Teil der Idee der Elternauswahl (siehe
  oben), nicht die Selbstveränderung von Code.
