Bezogen auf die Branch-Rules schreibt Claude:

```
Wichtig — nichts anderes eintragen. Fügen Sie keine App/keinen Bot in die Bypass-Liste ein. Nur so bleibt das Tor für den Agenten wirksam: Ein agenten-erzeugter PR, der etwas außerhalb candidates/ anfasst, braucht weiter Ihre Code-Owner-Freigabe (die Sie geben können, da Sie dort nicht der Autor sind); ein PR, der nur candidates/ berührt, braucht ohnehin keine.
```


# Brutmaschine

<p align="center">
  <img src="assets/cover.jpg" alt="Brutmaschine" width="480">
</p>

Die Brutmaschine ist ein kleiner, sicherer und reproduzierbarer Prototyp eines
„KI-Modell-Brutkastens“. Sie erzeugt viele kleine neuronale Netze, trainiert
sie, bewertet sie und behält die besten. Aus den besten entstehen neue
Varianten. So verbessern sich die Modelle über mehrere Generationen.

Die Netze sind kleine MLPs (mehrschichtige Perzeptrons, also einfache
vorwärts gerichtete Netze). Sie lösen künstliche Lernaufgaben, zum Beispiel
zwei ineinander verschlungene Spiralen auseinanderhalten. Diesen Modus nennt
dieses Dokument den **MLP-Modus**.

Im MLP-Modus kommen **CNNs** als zweite Modellart hinzu, faltende neuronale Netze (englisch
„convolutional neural networks“). Teil 1 und ein Zwischenstand von Teil 2 sind
zusammengeführt. Jetzt bauen wir die Mutationen und die Beförderung der Gewinner.

Dazu kommt der **Lerner-Modus** (Stufe 1). Dort ist ein Kandidat kein Netz,
sondern ein kleiner Lernalgorithmus: ein Programm, das selbst aus Beispielen
lernt. Die Brutmaschine misst, wie gut so ein Programm *neue Arten* von
Aufgaben lernt, die die Suche nie gesehen hat. Der MLP-Modus bleibt dabei
unverändert. Einzelheiten stehen unten im Abschnitt „Lerner-Modus“.

Seit Stufe 2c züchtet die Brutmaschine außerdem **Kaskaden** aus Sprachmodell-Workern,
die einander blind bewerten. Schritt 1 öffnet diese Zucht für eine zweite Aufgabenart:
Quantenschaltungen, deren Richtigkeit ein exakter Simulator prüft.
Beides beschreiben die Abschnitte „Kaskaden-Zucht (Stufe 2c)“ und „Universen, die
erfinden (Schritt 1)“.

**Schritt 2** soll jedem Worker eine verborgene, vererbte **Seele (Phasenzustand)** geben,
die sich nur in seinem Charakter zeigt. Ein Urteil wird dann zur Beobachtung: Es rückt die
Seelen beider Seiten aufeinander zu, so stark, wie die **Kopplungsstärke des Beobachters** κ
es sagt. Der Entwurf ist zusammengeführt (PR #28 und #29) und seit dem 2026-10-31 im Bau.

Einen bebilderten Überblick gibt der [Atlas](docs/atlas/index.html), eine Seite auf
Deutsch und Englisch mit Diagrammen, einem Speckle-Simulator und dem Ergebnis von Schritt 1.
Man öffnet die Datei im Browser.

## Stand und Fahrplan

Stand: 2026-10-31.

**Gebaut**

- **Bis Schritt 1.** MLP-Modus, Lerner-Modus (Stufe 1), Kaskaden-Zucht mit dem simulierten
  Modell (Stufe 2c, PR #8, #10 und #11) und Schritt 1 mit dem Ergebnis der gezählten Läufe
  (PR #21). Die Abschnitte unten beschreiben sie.
- **CNNs, Teil 1 (PR #24).** Die Bibliothek candle, das CNN-Genom und Bildaufgaben (Tickets
  01 und 02; [Spezifikation](.scratch/cnn-basismodelle/spec.md),
  [ADR 0007](docs/adr/0007-cnns-train-through-candle-at-a-pinned-commit.md)).
- **CNNs, Teil 2, Zwischenstand (PR #27 und #32).** Ticket 03 ist gebaut, Ticket 04 ist
  begonnen. PR #32 zog die CNN-Tests auf Ticket 04 nach und korrigierte GoogLeNet und
  ConvNeXt.
- **Entwurf von Schritt 2 (PR #28 und #29).** Er beschreibt Seele und Kopplungsstärke κ
  ([Spezifikation](.scratch/seele-und-kappa/spec.md)). PR #28 war der erste Entwurf, PR #29
  ist die Endfassung. Die Fragen Q67 bis Q74 wurden am 2026-10-31 mit der jeweils
  empfohlenen Antwort entschieden: Q67 a, Q68 a, Q69 a, Q70 a, Q71 a, Q72 a, Q73 a und d,
  Q74 a ([Fragen](.scratch/seele-und-kappa/owner-questions.md)). Die Eigentümerin bzw. der
  Eigentümer kann jede dieser Entscheidungen aufheben.
- **Echte Sprachmodelle (PR #33).** Das eigene Paket `backend-ollama` mit dem Programm
  `brut-breed-ollama`, siehe „Echte Modelle über Ollama“. Die ersten drei Läufe auf der H200
  fasst [docs/ERGEBNISSE_STUFE2C_ERSTE_LAEUFE.md](docs/ERGEBNISSE_STUFE2C_ERSTE_LAEUFE.md)
  zusammen (PR #34).
- **Eigene VM (PR #26, #30 und #31).** `tools/vm-setup.sh` richtet eine Ubuntu-24.04-VM mit
  einem Befehl ein, siehe „Eigene GPU-VM einrichten (freiwillig)“.

**Im Bau**

- **CNNs, Teil 2.** Ticket 05 (Mutationen, Basismodelle in der Population, Kommandozeile,
  README, ADR 0008) und Ticket 07 (Beförderung der Gewinner).
- **Schritt 2.** Die Tickets 01 bis 09 und 12, begonnen am 2026-10-31.

**Geplant**

- **CNNs, Teil 3.** Ticket 06, ARC-Raster als Aufgaben.
- **Schritt 2, Vergleich.** Vorab-Registrierung (Ticket 10), Pilot (Ticket 11) und gezählte
  Läufe (Ticket 13).
- **Sprachmodelle nutzen CNNs (Entwurf geschrieben, Pull-Request folgt).** F1: ein CNN als
  Worker in einer Kaskade. F2: ein Sprachmodell als Architekt, der Baupläne für CNNs
  vorschlägt. F3: ein CNN als Werkzeug, welches das Sprachmodell aufruft, mit höchstens 3
  Aufrufen je Antwort.
- **Chemische Netze (Recherche fertig, PR #37).** Chemische Reaktionsnetze als verallgemeinerte
  neuronale Netze, eine spätere Modellart. Die Notiz mit 74 Quellen und vier Tickets für
  erste Prototypen liegen in [.scratch/chemische-netze/](.scratch/chemische-netze/map.md).

**Geparkt**

- **Quantenneuronale Netze.** Die Idee ruht, bis echte Quantenhardware in Reichweite ist
  ([Idee 12](.scratch/universum-erfinden/issues/12-quantum-neural-networks-later.md)).

## Was die Brutmaschine nicht ist

- **Kein selbstverändernder Code.** Ein Kandidat ist reine Daten (ein
  „Genom“: Schichtbreiten, Lernrate und so weiter). Die Brutmaschine führt
  niemals Code aus, den ein Kandidat erzeugt hat. Im Lerner-Modus gilt:
  Kandidaten sind Programme in einer geschlossenen Sprache; sie werden nur
  interpretiert und können nur rechnen. „Geschlossen“ heißt: Es gibt eine
  feste, kleine Liste von Rechenbefehlen. Ein Interpreter ist ein Teil der
  Brutmaschine, der diese Befehle einzeln ausführt. Die Programme können
  keine Dateien öffnen, kein Netzwerk nutzen und keinen anderen Code starten.
- **Kein Produktionssystem.** Es ist Forschungssoftware zum Lernen und
  Ausprobieren. Es gibt keine Garantie für Sicherheit, Korrektheit oder
  Eignung für einen bestimmten Zweck.
- **Kein großes Sprachmodell.** Die Modelle haben standardmäßig höchstens
  20 000 Parameter. Das lässt sich in `config.json` einstellen (`max_params`),
  aber nie über die feste Obergrenze von 2 000 000 hinaus. Gerechnet wird nur
  auf der CPU.
- **Keine allgemeine KI.** Auch der Lerner-Modus baut keine allgemeine
  künstliche Intelligenz. Er ist ein ehrlicher Messstab für eine kleinere
  Frage: Lernt ein Programm neue Aufgabenfamilien besser als abgestimmte,
  höchstens gleich teure Vergleichsverfahren? Ein negatives Ergebnis ist
  wahrscheinlich (siehe „Was Sie erwarten dürfen“).

## Schnellstart (MLP-Modus)

Voraussetzung: Rust 1.89 oder neuer (Edition 2024).

```sh
cargo build --release

# 1. Lauf-Verzeichnis anlegen (Konfiguration + geheime Datei)
./target/release/brut init runs/spirals --task spirals --seed 1

# 2. Evolution starten (mit verdeckter Bewertung am Ende)
./target/release/brut run runs/spirals

# 3. Ergebnis ansehen
./target/release/brut status runs/spirals
./target/release/brut lineage runs/spirals <kandidaten-id>

# 4. Archiv auf Manipulation prüfen
./target/release/brut verify runs/spirals

# 5. Menschliche Freigabe eines Kandidaten
./target/release/brut approve runs/spirals <kandidaten-id> --by "Ihr Name" --note "geprüft"

# Not-Aus: der Lauf hält nach spätestens etwa einer Sekunde an
./target/release/brut stop runs/spirals
```

Aufgaben (`--task`): `spirals`, `checkerboard`, `parity`, `sine`.
Mit `brut help` sehen Sie alle Befehle. Exit-Codes: 0 = in Ordnung,
1 = Fehler bei der Ausführung, 2 = falsche Bedienung.

Ein abgebrochener Lauf lässt sich fortsetzen: Rufen Sie `brut run` einfach
erneut auf. Die Brutmaschine macht nach der letzten gespeicherten Generation
weiter. Wenn Sie mehr Generationen möchten, erhöhen Sie vorher `generations`
in `config.json`. Beim Fortsetzen dürfen sich nur `generations`, `threads`,
`max_candidate_seconds`, `max_total_seconds` und `max_total_flops` ändern.
Alle anderen Einstellungen (zum Beispiel Aufgabe, Seed oder Fitness-Gewichte)
legen das Experiment fest. Weichen sie vom ersten Lauf ab, verweigert
`brut run` die Arbeit und ändert nichts am Archiv. Für ein anderes Experiment
legen Sie mit `brut init` ein neues Lauf-Verzeichnis an.

Nach der verdeckten Bewertung ist ein Lauf abgeschlossen. Das gilt auch,
wenn er wegen der Zeit- oder FLOP-Grenze früher endete. Weitere Generationen
lehnt `brut run` dann ab. Ein erneuter Aufruf ohne neue Generationen bewertet
nur noch Kandidaten nach, die in derselben festen Liste der besten
`hidden_eval_top_k` fehlen, zum Beispiel nach einem Not-Aus. Wenn Sie einen
Lauf in Etappen rechnen möchten, rufen Sie für die Etappen
`brut run <run_dir> --no-hidden` auf und erst zum Schluss `brut run <run_dir>`.

## Beispielergebnis (MLP-Modus)

Ein echter Lauf mit den Standardwerten (`brut init <dir> --task spirals`,
also Seed 1; 4 CPU-Kerne, die nebenbei mit anderen Programmen ausgelastet
waren; Release-Build):

| Größe | Wert |
|---|---|
| Generationen | 30 (plus Startpopulation) |
| Kandidaten | 496 |
| Bester Validierungswert (Genauigkeit) | 0.9925 |
| Verdeckter Testwert (abgerundet auf 0.01) | 0.98 |
| Abstand Validierung – verdeckt | +0.0125 |
| Laufzeit | etwa 17 Sekunden |
| Geschätzte Rechenarbeit | 2.7 · 10¹¹ FLOPs |

Der Median der Validierungsgenauigkeit je Generation lag in der
Startpopulation bei 0.56 und in den Generationen 1 und 2 bei 0.57 und 0.58.
Ab Generation 3 lag er zwischen 0.71 und 0.95. Der beste Wert stieg von
0.935 auf 0.9925.

Der verdeckte Testwert ist abgerundet: Der wahre Wert liegt zwischen 0.98
und 0.99. Er lässt sich auch nicht genau nachrechnen. Den Seed der geheimen
Testdaten zieht `brut init` zufällig, darum kann der verdeckte Wert bei Ihnen
anders ausfallen. Alle anderen Zahlen außer der Laufzeit sind bei gleichem
Seed gleich.

Zum Vergleich: Vor den Geschwindigkeitsverbesserungen brauchte derselbe Lauf
auf demselben Rechner 51 bis 66 Sekunden (jetzt 17 bis 26 Sekunden, jeweils
drei Läufe abwechselnd gemessen). Die Zahlen der Kandidaten haben sich dabei
etwas verschoben, weil Tanh und GELU jetzt eine Näherung nutzen und die
Rückwärtsrechnung anders summiert.

## Lerner-Modus (Stufe 1)

Stand: Der Lerner-Modus folgt dem Bauvertrag
[docs/STAGE1_CONTRACT.md](docs/STAGE1_CONTRACT.md) (englisch, mit den
Korrekturen in [docs/STAGE1_CONTRACT_ERRATA.md](docs/STAGE1_CONTRACT_ERRATA.md))
und dem Plan [docs/PLAN_GENERALISIERUNG.md](docs/PLAN_GENERALISIERUNG.md).
Die Ergebnisse des Versuchsplans S stehen in
[docs/ERGEBNISSE_STUFE1.md](docs/ERGEBNISSE_STUFE1.md). Kurz: Der Hauptwert
ist nicht bestanden (0 von 3 Seeds, in beiden Läufen von Plan S), und die
Suche fand nur Varianten des linearen Lernens; in einem Seed des zweiten
Laufs verlor sie sogar alle lernenden Programme. Die Erwartungen weiter unten sind die Schätzungen des Plans
von vor den Läufen.

### Worum es geht

- Ein Kandidat ist ein **Lernprogramm**. Es besteht aus drei kurzen
  Programmteilen. `Setup` läuft einmal je Aufgabe. `Predict` macht eine
  Vorhersage für ein Beispiel. `Learn` passt danach den Speicher an. Das
  Vorbild ist AutoML-Zero (Real et al., ICML 2020).
- Die Programme rechnen nur mit Zahlen, Vektoren (Zahlenlisten) und Matrizen
  (Zahlentabellen). Diese liegen in festen Speicherplätzen, den
  **Registern**.
- Die Suche startet mit vier handgeschriebenen Lernprogrammen: lineares
  Lernen, ein kleines Netz mit zwei Schichten, ein Prototyp-Lerner und ein
  Perzeptron. Aus ihnen entstehen durch kleine zufällige Änderungen
  (**Mutationen**) neue Programme.
- Jedes Programm lernt viele kleine künstliche Aufgaben. Eine
  **Aufgabenfamilie** ist eine Art von Aufgabe, zum Beispiel „Punkte innerhalb
  einer Kugel erkennen“. Jede Familie liefert beliebig viele verschiedene
  Aufgaben.
- Gemessen wird, wie schnell und wie gut ein Programm lernt. Die Kennzahl
  heißt **AULC** (englisch „area under the learning curve“, Fläche unter der
  Lernkurve). Die Qualität wird nach 1/16, nach 1/4 und am Ende des Lernens
  gemessen und gewichtet gemittelt.
- Verglichen wird immer mit **Vergleichsverfahren** („Baselines“), die
  höchstens gleich viel rechnen. Das sind die vier Startprogramme mit
  abgestimmten Konstanten und ein klassisches Netz (MLP mit dem
  Lernverfahren Adam). Das Netz bekommt als Budget die Trainingskosten des
  Programms, aber mindestens eine Epoche über alle bisher gesehenen
  Beispiele. Bei billigen Programmen sind das nur wenige Lernschritte, oft
  mehr als das Budget. Gegen lineare Programme ist es darum schwach; dort
  entscheidet meist das abgestimmte lineare Startprogramm. Der Vorsprung vor
  dem besten dieser Vergleiche heißt **Δ** (Delta).
- Der **Hauptwert** ist Δ auf 40 geheimen Aufgabenfamilien
  („HIDDEN-GRAMMAR“). Sie entstehen aus einer Formel-Grammatik mit einem
  geheimen Seed (Startwert für Zufallszahlen). Niemand hat sie gesehen, auch
  wir nicht. Nur der Auswerter `brut-eval` erzeugt sie. Ihre
  Formeln kommen in den öffentlichen Familien nicht vor, auch nicht in den
  zwei bekannten gleichwertigen Umformungen (Erratum V6). Die Grammatik ist
  aber dieselbe, darum sind es neue Familien derselben Bauart, kein ganz
  fremdes Gebiet.
- **ARC-AGI** ist ein öffentlicher Test mit Rasterrätseln. Er dient als
  harter Außentest. Er fließt nie in die Suche ein.

### Schnellstart

```sh
cargo build --release

# 1. Geheime Datei anlegen (einmal je Rechner). Gibt nur den Pfad aus.
./target/release/brut learner secret-init

# 2. Lauf-Verzeichnis anlegen: Rauchtest mit 10^11 FLOPs
./target/release/brut init runs/e0 --mode learner --seed 1 --flops 1e11

# 3. Suche, Endphase und verdeckte Bewertung
./target/release/brut run runs/e0

# 4. Ergebnis ansehen
./target/release/brut status runs/e0
./target/release/brut learner report runs/e0
./target/release/brut learner show runs/e0 <programm-id>

# 5. Archiv prüfen und ein Programm freigeben
./target/release/brut verify runs/e0
./target/release/brut approve runs/e0 <programm-id> --by "Ihr Name" --note "geprüft"
```

Ein **Rauchtest** ist ein kurzer erster Lauf, der nur zeigt, ob alles
zusammen funktioniert. Nach dem Bauvertrag soll er auf 4 Kernen im
Release-Build in höchstens 5 Minuten fertig sein.

`brut learner secret-init` legt die Datei `~/.brut/learner/hidden.env` an.
Diesen Ordner gibt es einmal je Rechner, nicht je Lauf. So setzt ein neues
Lauf-Verzeichnis das Abfrage-Budget nicht zurück. Fehlt die Datei, bricht
`brut init --mode learner` mit einem Hinweis ab. Mit `--secret <datei>`
wählen Sie eine andere Datei.

`--flops` begrenzt die Rechenarbeit der Suche. Zwei einmalige Posten zählen
getrennt davon (Erratum E6): das Abstimmen der Vergleichsverfahren am Anfang
und die Endphase. Beide stehen seit `learner-v2` als `LearnerCost` im
Archiv und unter `one_time` im Bericht (Erratum V11).

Ein abgebrochener Lauf lässt sich mit `brut run` fortsetzen. Er macht beim
letzten Schnappschuss weiter und rechnet die fehlenden Runden genau gleich
nach. Beim Fortsetzen dürfen sich in `meta.json` nur `max_batches`,
`threads`, `max_candidate_seconds`, `max_total_seconds` und
`max_total_flops` ändern. Nach der ersten verdeckten Bewertung ist der Lauf
abgeschlossen. Wenn Sie in Etappen rechnen möchten, nutzen Sie bis zur
letzten Etappe `brut run <dir> --no-hidden`.

### Befehle

| Befehl | Wirkung |
|---|---|
| `brut learner secret-init [--dir D] [--arc-root R] [--manifest M]` | legt die geheime Datei an; zeigt nur den Pfad, nie den Seed |
| `brut init <dir> --mode learner [--seed N] [--flops X] [--variation full\|constants-only\|random-search] [--secret <datei>] [--arc-cells]` | legt einen Lerner-Lauf an; ohne `--mode` entsteht wie bisher ein MLP-Lauf |
| `brut run <dir> [--no-hidden]` | erkennt den Modus selbst |
| `brut status <dir>` | Runden, bestes Programm, Finalisten, verdeckte Ergebnisse |
| `brut lineage <dir> <id>` | Vorfahren eines Programms |
| `brut learner show <dir> <id>` | das vereinfachte Programm und seine Kosten |
| `brut learner report <dir>` | schreibt `report.json` und zeigt eine Tabelle |
| `brut learner arc-check <dir> <id> --set a1-eval\|a2-eval\|concept-arc` | ARC-Außentest für ein Programm |
| `brut verify <dir>` | prüft Archiv, Abstammungsblöcke und Schnappschüsse |
| `brut approve <dir> <id> --by B --note N` | Freigabe nur nach HIDDEN-GRAMMAR und ohne Quarantäne |
| `brut stop <dir>` | Not-Aus |

Die Exit-Codes bleiben 0, 1 und 2. Die drei Suchmodi (`--variation`):

- `full`: alle Arten von Mutationen.
- `constants-only`: Nur die Konstanten der Startprogramme ändern sich. Das
  ist die wichtigste **Ablation** (ein Kontrollversuch, bei dem man einen
  Teil weglässt). Sie zeigt, ob die Suche mehr findet als eine bessere
  Feinabstimmung.
- `random-search`: keine Auswahl, nur Zufall (Kontrolle).

Der Zusatzversuch E2 (`--arc-cells`, ARC-Zellen in der Suche) ist
abgeschaltet. Der Schalter ist nur vorbereitet; `brut init` darf ihn
ablehnen.

### Was Sie erwarten dürfen

Ehrliche Erwartungen (Schätzungen aus dem Plan, Abschnitt 9):

| Messung | Erwartung |
|---|---|
| Hauptwert HIDDEN-GRAMMAR bestanden in mindestens 2 von 3 Seeds | Wahrscheinlichkeit unter 25 % |
| HIDDEN-INSTANCE (neue Aufgaben bekannter Familien) | kleiner Vorsprung von +0.01 bis +0.05 in 1 bis 2 von 3 Seeds denkbar |
| voller Suchraum besser als „nur Konstanten“ | unsicher, eher klein |
| HIDDEN-FAR (ganz neue Gruppe von Familien) | etwa 0, wahrscheinlich „nicht aussagekräftig“ |
| ARC-AGI-1 Bewertung, exakt gelöst | 0 bis 2 von 400 Aufgaben |
| ARC-AGI-2 Bewertung | 0 von 120 Aufgaben |
| ConceptARC | 0 bis 2 von 160 Aufgaben |

Ein negatives Ergebnis ist also wahrscheinlich. Es ist trotzdem wertvoll,
weil der Messstab es zuverlässig erkennt. Im Vorversuch schlug kein Lauf die
abgestimmten Vergleiche auf zurückgehaltenen Familien (0 von 7 Läufen).

Zu den ARC-Zahlen: Ein Lernprogramm sieht ein ARC-Raster nur über 32
handgemachte Merkmale je Zelle („phi32“: Farbe der Zelle, Nachbarn, Lage).
Mit diesen Merkmalen sind höchstens 9 von 400 (ARC-AGI-1), 0 von 120
(ARC-AGI-2) und 17 von 160 (ConceptARC) Aufgaben überhaupt lösbar. Diese
Obergrenzen haben wir auf dem Entwicklungsrechner nachgezählt.

**Bestanden** heißt beim Hauptwert: Der Vorsprung ist im einseitigen
Vorzeichen-Permutationstest über die 40 Familien signifikant (p < 0.05),
er beträgt mindestens 0.01, und er fällt bei Breite F = 32 um höchstens
0.05 gegenüber F = 8 und 16. Der **p-Wert** ist die Wahrscheinlichkeit, einen
so großen Vorsprung allein durch Zufall zu sehen, wenn es in Wahrheit keinen
gibt. Die Einzelheiten stehen in [docs/ARCHITEKTUR.md](docs/ARCHITEKTUR.md).

**Erfolg der Stufe 1** heißt (vorab festgelegt): Der beste Kandidat besteht
in mindestens 2 von 3 Seeds, und der volle Suchraum schlägt „nur Konstanten“.
Das wäre ein kleines, echtes Ergebnis im Meta-Lernen (Lernen, wie man lernt),
aber kein Sprung zu allgemeiner KI.

**Abbruchkriterium** (vorab festgelegt): Ist nach Plan S der Vorsprung auf
den SELECT-Aufgaben in allen 3 Seeds höchstens 0 und ist der volle Suchraum
nicht besser als „nur Konstanten“, wird die Programm-Evolution nicht weiter
ausgebaut. Das Ergebnis wird dann als negatives Ergebnis berichtet.

### Versuchsplan S

Für Stufe 1 gelten diese Entscheidungen (Standardwerte des Projektleiters):

- Budgetplan S: zusammen etwa 9.5 · 10¹² FLOPs, etwa 0.5 bis 1.7 Stunden
  Rechenzeit.
- Versuch E2 (ARC-Zellen in der Suche) ist aus.
- Der geheime Ordner ist `~/.brut/learner` (Protokoll `learner-v1`). Für
  `learner-v2` gilt ein neuer geheimer Ordner, damit dieselben geheimen
  Familien nicht erneut abgefragt werden (Korrektur G2). Welcher Ordner das
  war, steht in [docs/ERGEBNISSE_STUFE1.md](docs/ERGEBNISSE_STUFE1.md).
- Einziger Hauptwert ist HIDDEN-GRAMMAR. Die anderen verdeckten Stufen sind
  nur beschreibend.

| Versuch | Befehl (Beispiel für Seed 1) | FLOPs |
|---|---|---|
| E0 Rauchtest | `brut init runs/e0 --mode learner --seed 1 --flops 1e11` | 10¹¹ |
| E1 voller Suchraum, Seeds 1, 2, 3 | `brut init runs/e1-s1 --mode learner --seed 1 --flops 1.5e12` | 3 × 1.5 · 10¹² |
| Ablation „nur Konstanten“, Seeds 1, 2, 3 | `brut init runs/ab-s1 --mode learner --seed 1 --flops 1.5e12 --variation constants-only` | 3 × 1.5 · 10¹² |
| verdeckte Bewertung, ARC, Endphasen | automatisch oder `brut learner arc-check` | etwa 4 · 10¹¹ |

Jeder Lauf stellt 6 verdeckte Abfragen (3 Finalisten × 2 Stufen). E0, E1 und
die Ablation brauchen also 42 der 48 erlaubten Abfragen. Die ARC-Prüfung ist
für das beste Programm jedes E1-Seeds gedacht: 3 Prüfungen je
Bewertungssatz, genau das erlaubte Budget.

### ARC-Daten einrichten (freiwillig)

1. Laden Sie die Daten selbst herunter. Legen Sie sie unter
   `data/external/` ab. Der Ordner `data/` steht in `.gitignore`;
   ARC-Daten kommen nie ins Git.
   - ARC-AGI-1: github.com/fchollet/ARC-AGI (Lizenz Apache-2.0)
   - ARC-AGI-2: github.com/arcprize/ARC-AGI-2 (Lizenz Apache-2.0)
   - ConceptARC: github.com/victorvikram/ConceptARC (Lizenz MIT)
2. Schreiben Sie eine **Manifest-Datei** `data/external/manifest.json`. Sie
   nennt jede Datei mit ihrem SHA-256-Hash, ihrer Lizenz, ihrer Quelle und
   ihrer Rolle. Das Format zeigt `data/manifest.example.json`. `brut-eval`
   prüft die Hashes vor jeder ARC-Prüfung.
3. Geben Sie beide Pfade beim Anlegen der geheimen Datei an:

   ```sh
   brut learner secret-init --arc-root /abs/pfad/data/external \
       --manifest /abs/pfad/data/external/manifest.json
   ```

   Gibt es die geheime Datei schon, tragen Sie `BRUT_ARC_DATA_ROOT` und
   `BRUT_ARC_MANIFEST` von Hand ein (Format: `.env.example`).
4. Starten Sie die Prüfung, zum Beispiel
   `brut learner arc-check runs/e1-s1 <programm-id> --set a1-eval`.

Die Pfade stehen nur in der geheimen Datei. Der Orchestrator kennt sie nicht.
Gezählt wird nach ARC-Regeln: Je Testeingabe gibt es zwei Versuche
(„pass@2“), und nur ein exakt gleiches Raster zählt.

### Aufbau eines Lerner-Lauf-Verzeichnisses

```
runs/e1-s1/
├── meta.json                     Einstellungen dieses Laufs
├── protocol.json                 Messprotokoll zum Lesen (verbindlich ist der Eintrag im Archiv)
├── archive.jsonl                 Abstammungsarchiv mit Hash-Kette
├── archive.head                  Anker: Zahl der Einträge und Hash des letzten Eintrags
├── programs/<id>.json            Programme der Finalisten
├── programs/baseline-<name>.json abgestimmte Vergleichsprogramme
├── lineage/chunk-000001.jsonl    Abstammungsblock: Kinder, die in die Population kamen
├── snapshots/pop-000001000.json  Schnappschuss der Population
├── snapshots/cache-latest.json   letzter Stand des Ergebnis-Speichers
├── report.json                   Bericht des Laufs
└── STOP                          Not-Aus (nur wenn gesetzt)

~/.brut/learner/                  nur für brut-eval (Ordner 0700)
├── hidden.env                    geheimer Seed und Budgets (Datei 0600)
├── ledger.jsonl                  Abfragebuch mit Hash-Kette, gilt für alle Läufe
├── ledger.head                   Zahl der Einträge und letzter Hash des Abfragebuchs
├── eval.lock                     Sperrdatei, damit jede Abfrage genau einmal zählt
└── baseline_cache.json           Ergebnisse der Vergleichsverfahren auf den verdeckten Stufen
```

Das **Abfragebuch** (`ledger.jsonl`) hält jede verdeckte Abfrage fest: erst
die Belastung des Budgets, dann genau das, was herausgegeben wurde. Wie das
Archiv ist es eine Hash-Kette.

## Kaskaden-Zucht (Stufe 2c)

Stand: gebaut mit dem simulierten Modell (PR #8); die Funde zweier Gegenprüfungen sind
behoben (PR #10 und #11); echte Modelle bindet seit PR #33 das Paket `backend-ollama` an
(unten). Englische Fassung: [README.en.md](README.en.md). Beschrieben in
[docs/STUFE2C_KASKADEN_ZUCHT.md](docs/STUFE2C_KASKADEN_ZUCHT.md) und
[ADR 0005](docs/adr/0005-cascades-are-bred-by-darwin-goedel-selection-and-every-cascade-is-also-a-judge.md).

Hier züchtet die Brutmaschine **Kaskaden** aus Sprachmodell-**Workern**. Ein Worker
ist ein Modell mit einem Systemprompt und mit **Traits** (Denkbudget, Temperatur,
Antwortlänge, Selbstprüfung). Eine Kaskade hat ein bis vier Stufen mit je ein bis drei
Workern. Jede Kaskade ist zugleich Auswerter: Sie urteilt blind über die Antworten
anderer, nicht eng verwandter Kaskaden (weder Elter noch Kind noch Geschwister). Die
Eltern wählt die Brutmaschine wie die Darwin-Gödel-Maschine. Claude kann als
**Mutator** neue Kaskaden schreiben.

Auch hier gilt: Kandidaten sind Daten. Eine Kaskade ist JSON; ihr Prompt wird an ein
Modell geschickt, nie ausgeführt. Im Container gibt es kein Sprachmodell; die Worker
nutzen ein simuliertes Modell als Positivkontrolle. Echte Läufe brauchen einen
GPU-Server mit Ollama (ADR 0003).

```sh
cargo build --release

# Zuchtlauf mit dem simulierten Modell
./target/release/brut-breed --out runs/k1 --seed 1 --generations 10

# Fortsetzen, auch mit mehr Generationen oder mehr Aufrufen
./target/release/brut-breed --out runs/k1 --resume --generations 20

# Protokoll prüfen und die zwei besten Kaskaden für einen PR vorbereiten
./target/release/brut-breed --out runs/k1 --verify
./target/release/brut-breed --out runs/k1 --promote
scripts/propose-pr.sh runs/k1/promote

# Not-Aus: der Lauf hält vor dem nächsten Rätsel an
touch runs/k1/STOP

# Weiter nach dem Not-Aus: erst STOP löschen, dann fortsetzen
rm runs/k1/STOP
./target/release/brut-breed --out runs/k1 --resume
```

### Echte Modelle über Ollama (`backend-ollama`)

Das eigene Paket `backend-ollama` bindet ein lokales Modell über Ollama an
(`docs/STAGE2_BACKEND_CONTRACT.md`). Das Kernpaket bekommt dadurch keine neue
Abhängigkeit (ADR 0001). Sein Programm `brut-breed-ollama` ist `brut-breed` mit einem
Zusatz: Für jeden Tag `ollama:<modell>` in der Erlaubnisliste baut es einen Backend.
Dieser spricht nur mit einer Loopback-Adresse, kennt eine eigene Modellliste, eine
Zeitgrenze und eine Bytegrenze. Jeder Worker-Aufruf landet samt voller Antwort in
`<out>/llm/worker-calls.jsonl`, und ein Fortsetzen spielt ihn ab, statt das Modell neu
zu fragen (Vertrag I7). Zugelassen sind vorerst nur Modelle ohne Denkmodus
(`qwen2.5:7b`, `qwen2.5:14b`, `qwen2.5:32b`, `llama3.1:8b`).

```sh
ollama pull qwen2.5:7b
cd backend-ollama && cargo build --release && cd ..
# config.json: eine vollständige BreedConfig (etwa runs/k1/config.json eines simulierten
# Laufs) mit "limits": {"models": [{"tag": "ollama:qwen2.5:7b", "cost_per_call": 0.002}], ...}
backend-ollama/target/release/brut-breed-ollama --out runs/q1 --config config.json \
  --seed 1 --generations 2 --children 4 --search 10 --select 10 --max-calls 800
```

Tags der Form `vllm:<name>` gehen statt an Ollama an einen vLLM-Server mit
OpenAI-kompatibler Schnittstelle, welcher die parallel geschickten Rätsel eines Panels
gebündelt rechnet (auf der H200 Faktor 40 mehr Durchsatz). Der Server muss das Modell
unter diesem Namen anbieten:

```sh
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen2.5-32B-Instruct --host 127.0.0.1 \
  --port 8000 --served-model-name qwen2.5-32b --max-model-len 8192
# in config.json: "models": [{"tag": "vllm:qwen2.5-32b", "cost_per_call": 0.002}]
```

`VLLM_URL` (Vorgabe `http://127.0.0.1:8000`), `OLLAMA_URL` (Vorgabe `http://127.0.0.1:11434`) und `OLLAMA_TIMEOUT_SECS` (Vorgabe 300, für beide Server)
stellen den Server und die Zeitgrenze ein. Der Live-Test:
`cd backend-ollama && OLLAMA_LIVE=1 cargo test --release -- --ignored live`.

Die ersten drei Läufe auf einer H200 (`qwen2.5:7b` und `qwen2.5:32b`, 2704
Worker-Aufrufe) fasst [docs/ERGEBNISSE_STUFE2C_ERSTE_LAEUFE.md](docs/ERGEBNISSE_STUFE2C_ERSTE_LAEUFE.md)
zusammen. Kurz: Die Maschine läuft mit echten Modellen, ein Zuchtgewinn ist bei
Panels von 12 bis 24 Rätseln nicht messbar, und Ollama arbeitet die parallel
geschickten Anfragen nacheinander ab (0.35 s je Aufruf bei 7B, 1.7 s bei 32B).

Mit `--llm-every K` ist jede K-te Generation eine LLM-Runde. Fehlt eine Antwort des
Mutators, hält der Lauf an und endet mit Exit-Code 3. Die Anfragen liegen dann unter
`runs/k1/llm/requests/`. Schreiben Sie (oder eine Claude-Sitzung) jede Antwort als
`runs/k1/llm/answers/<schlüssel>.txt` und setzen Sie den Lauf mit `--resume` fort.
Liegen die Antworten mit `--llm-answers <ordner>` woanders, geben Sie die Option auch
beim Fortsetzen an. Exit-Codes: 0 = fertig oder sauber angehalten, 1 = Fehler,
2 = falsche Bedienung (auch: ein anderer Prozess hält das Lauf-Verzeichnis),
3 = wartet auf LLM-Antworten.

`--resume` darf nur `--generations`, `--max-calls`, `--max-seconds` und
`--llm-answers` ändern; alles andere legt das Experiment fest. Vor jeder Generation
legt der Checkpoint ihren schlimmsten Verbrauch an Aufrufen zurück. Endet der Prozess
hart (Kill, Absturz), zählt das Fortsetzen diese Rücklage als verbraucht, sodass
`--max-calls` auch dann hält. Einzelheiten stehen in
[docs/STUFE2C_KASKADEN_ZUCHT.md](docs/STUFE2C_KASKADEN_ZUCHT.md), Abschnitt 5.

Seit Schritt 1 schreibt jeder Lauf nach jeder festgeschriebenen Generation eine
`state`-Zeile ins Protokoll. Sie enthält sechs Kennzahlen des „Universums“: Leistung,
Vielfalt (mit der Shannon-Entropie der Bausteine), Abstammung, Ehrlichkeit, Richter und
Kosten. Die Zeile hängt nur vom festgeschriebenen Zustand ab. Darum ist sie nach Not-Aus
und Fortsetzen dieselbe wie bei einem durchgehenden Lauf. Die Laufzeit je Generation
steht in `timings.jsonl`; diese Datei gehört nicht zum geprüften Protokoll.

```sh
# Kontrollarm: Eltern gleichverteilt statt nach Darwin-Gödel wählen
./target/release/brut-breed --out runs/k2 --seed 1 --generations 10 --selection uniform
```

`--domain` und `--selection` legen das Experiment fest; ein Fortsetzen lehnt sie ab.
`--domain grid` ist die Voreinstellung; `--domain circuits` startet einen Schaltungslauf
(Schritt 1, unten). `--verify` rechnet den Bericht eines fertigen Laufs ohne Modellaufruf
aus dem Checkpoint nach, auch seinen Stoppgrund.

### Kontrollarm Mehrheitsvotum bei gleichem Aufwand (`vote-control`)

Ein Vorsprung des besten gezüchteten Finalisten mischt Zucht mit Rechenaufwand, da er
oft mehr Worker-Aufrufe je Rätsel macht als die Startprogramme
([docs/STUFE2C_KASKADEN_ZUCHT.md](docs/STUFE2C_KASKADEN_ZUCHT.md), „Nicht rechenmäßig
abgeglichen“). Das Programm `vote-control` aus `backend-ollama` rechnet dafür den
Kontrollarm mit gleich vielen Aufrufen: Self-Consistency nach Wang et al. (2022), also
ein Startprogramm N-mal fragen und das häufigste Gitter nehmen, kostengleich wie in
arXiv 2606.13003 (Vorabdruck).

- **Startprogramm.** Vorgabe ist der Start-Finalist mit der besten Such-Fitness. Die
  Wahl liest kein Ergebnis des Auswahl-Panels.
- **N.** Die mittleren Worker-Aufrufe je Such-Rätsel des Champions (des gezüchteten
  Finalisten mit der besten Such-Fitness), geteilt durch die des Startprogramms,
  gerundet, mindestens 1. Bei einem Champion mit 3.18 Aufrufen je Rätsel ist N = 3.
- **Panel.** Das Programm erzeugt das Auswahl-Panel genau wie die Arena neu und prüft
  es gegen den Checkpoint und die erste Zeile des Protokolls.
- **Stimmen.** Jede Stichprobe läuft mit eigenem Seed durch `run_cascade` des Kerns, also
  mit denselben Prompts und demselben Parser wie im Lauf. Das häufigste Gitter gewinnt,
  bei Gleichstand das zuerst gesehene. Bewertet wird exakt gegen die verdeckte
  Testausgabe.

```sh
# Plan ansehen, ohne Modellaufruf (geht auch bei einem unfertigen Lauf)
backend-ollama/target/release/vote-control --run runs/q1 --dry-run
# Kontrollarm rechnen; Ergebnis auf stdout und in runs/q1/vote-control.json
backend-ollama/target/release/vote-control --run runs/q1
```

Das Ergebnis nennt je Rätsel die Stimmen, ob das Mehrheitsgitter stimmt und ob
irgendeine Stichprobe stimmte. Zum Vergleich stehen daneben die Antworten von Champion und
Startprogramm aus der Auswahlphase des Laufs. Die Summen enthalten die diskordanten Paare
gegen den Champion mit exaktem Vorzeichentest sowie Aufrufe und Tokens. `--samples N`
setzt N fest, `--cascade ID` wählt ein anderes Archivmitglied, `--temperature T` setzt
die Temperatur aller Worker (eine andere Kaskade mit eigener Id), `--puzzles K` rechnet
als Rauchtest nur die ersten K Rätsel und schreibt keine Datei, `--jobs J` (Vorgabe 16)
legt die gleichzeitigen Stichproben fest. Echte Modelle laufen über `Recorded` in
`<run>/vote-control/worker-calls.jsonl`, getrennt vom Protokoll des Laufs. Eine
Wiederholung spielt sie ab, und ein größeres N nutzt die Stichproben eines kleineren mit.
Ein gescheiterter Aufruf zählt als Enthaltung, steht in `worker-errors.jsonl` daneben und
setzt den Exit-Code 1; ein erneuter Aufruf holt ihn nach. Exit-Code 2 heißt falsche
Bedienung, auch bei einem unfertigen Lauf.

## Universen, die erfinden (Schritt 1)

Stand: gebaut; das gezählte Experiment ist gelaufen. Die sechs Entscheidungen unten wurden
am 2026-10-30 freigegeben, Entscheidung 5 zuerst mit der Schwelle 20 und höchstens 60
Paaren. Ihre wissenschaftlich tragende Fassung (Nachweisschwelle 100, höchstens 120
Seed-Paare), deren Zahlen Claude wählte, wurde mit dem Zusammenführen von Pull-Request #19
bestätigt, das die Vorab-Registrierung einfror. Danach lief das gezählte Experiment, und
die Regel hielt nach 27 Seed-Paaren an. Der Hauptnachweis ist bei Paar 15 erbracht, der
Nachweis zur Auswahl bei Paar 27
([docs/SCHRITT1_SCHALTUNGEN.md](docs/SCHRITT1_SCHALTUNGEN.md), Abschnitt 7; Daten in
[docs/step1-counted/](docs/step1-counted/README.md)). Das Experiment ist eine
Positivkontrolle mit dem simulierten Entwerfer und sagt nichts über echte Sprachmodelle.
Beschrieben in [docs/SCHRITT1_SCHALTUNGEN.md](docs/SCHRITT1_SCHALTUNGEN.md) und
[ADR 0006](docs/adr/0006-cascades-breed-on-domains-and-circuits-are-checked-exactly.md);
das Experiment ist in [docs/PREREG_STEP1_CIRCUITS.md](docs/PREREG_STEP1_CIRCUITS.md)
vorab festgelegt. Spezifikation:
[.scratch/universum-erfinden/spec.md](.scratch/universum-erfinden/spec.md), mit den
Tickets in `.scratch/universum-erfinden/issues/`. Schritt 1 umfasst die Tickets 01 bis 03.

Die Zucht soll vor allem **erfinden**: Artefakte, die ein exakter Prüfer als richtig
bestätigt, die eine vorab festgelegte Referenz übertreffen und die neu sind. Das erste
Feld sind Quantenschaltungen aus den Gattern H, S, S†, T, T†, X, Z und CNOT. Schritt 1
misst zunächst als Positivkontrolle, ob die Zucht die bekannten besten T- und
CNOT-Zahlen wiederentdeckt, die ihre Startprogramme verfehlen.

Gebaut sind:

- **Universums-Kennzahlen** (Ticket 01): die `state`-Zeile je Generation, siehe oben.
- **Eine Zucht für jede Aufgabenart:** Die Bewerterseite einer Aufgabenart (`Domain`)
  sieht nie eine Kaskade; die Züchterseite (`GenePool`) liefert Startprogramme,
  Satzbausteine, den Auftrag an den Mutator und eine Aufnahmeregel. Gitter-Rätsel und
  Schaltungen nutzen dieselbe Zuchtschleife. Golden-Tests, aufgenommen auf dem
  Gitter-Code vor dem Umbau, halten einen Gitter-Lauf und einen Lauf mit LLM-Runden (ohne
  Gleitkommazahlen), die Schlüssel der Mutator-Anfragen und die Prüfsummen der
  Einstellungen fest; einzig neu im Protokoll eines Gitter-Laufs ist die `state`-Zeile.
- **Exakter Simulator** (Ticket 02, `src/cascade/qubit.rs`): Er speichert Amplituden
  exakt als x/√2^k mit x in Z[ω], ω = e^(iπ/4), in 64-Bit-Ganzzahlen. „Richtig“ heißt
  exakt gleich bis auf eine globale Phase; die Fidelity wird nur berichtet. Grenzen:
  12 Qubits für Zustände, 6 für ganze Unitaries, 256 Gatter und 120 H-Gatter.
- **Schaltungs-Aufgaben** (Ticket 03, `src/cascade/circuit.rs`): 11 Ziele mit T- und
  CNOT-Zahlen aus der Literatur: CZ, SWAP, CS, CH, QFT2, CCZ, Toffoli, Peres, Fredkin,
  RTOF und CCiX. Jede Zahl trägt ihren Status: bewiesen, abgeleitet oder bester
  bekannter Wert.
  - Eine Aufgabe ist ein verkleidetes Ziel: Die Qubits sind umbenannt, an Ein- und
    Ausgängen stehen X-Gatter. Ein Prompt nennt nie den Namen eines Ziels.
  - Bewertet wird zuerst die Richtigkeit, dann die T-Anzahl, dann die CNOT-Anzahl.
  - Eine exakte Antwort unter einer bewiesenen oder abgeleiteten Schranke stoppt den Lauf
    als Integritätsalarm.
  - Eine exakte Antwort unter einem besten bekannten Wert wird als Erfindungskandidat
    festgehalten, mit `literature_checked = false`, bis ein Mensch die Literatur geprüft
    hat.
- **Positivkontrolle** (`src/cascade/designer.rs`, `src/cascade/circuit_genes.rs`): ein
  simulierter Schaltungs-Entwerfer unter den Modellnamen `sim:small` und `sim:large`.
  Die Methoden, die ein Systemprompt nennt, schalten seine Syntheseverfahren frei; er
  kennt keine Schaltung und keine Zielliste. Die vier Startprogramme erreichen nur SWAP
  und CS. Ein starker Prompt erreicht alle 11 Referenzwerte. Seine Zahlen sagen nichts
  über echte Modelle.
- **Zweiter Simulator** (`src/quantum.rs`, PR #12): ein Gleitkomma-Simulator mit
  Fidelity-Schwelle, der keinen Code mit dem exakten Simulator teilt. Er dient als
  unabhängige Gegenprobe: Auf 900 Paaren zufälliger Schaltungen und den 15
  Beleg-Schaltungen der Bibliothek entscheidet er wie der exakte Simulator
  (`tests/circuits.rs`).
- **Schaltungs-Läufe** (`src/bin/brut-breed.rs`): `brut-breed --domain circuits`. Ein
  fertiger Lauf schreibt `rediscovery.json`, die vorab festgelegte Zählung, und `--verify`
  rechnet sie aus dem Checkpoint nach. Die eingefrorene Auswertung ist
  `examples/circuit_positive_control.rs`.

Die Pilotläufe liegen in [docs/pilot-step1/](docs/pilot-step1/README.md): der erste Pilot
auf den Seeds 101 und 102 und vier spätere Reihen. Pilotläufe zählen nie; die Daten der
gezählten Läufe liegen in [docs/step1-counted/](docs/step1-counted/README.md). Die
Eigentümerin bzw. der Eigentümer hat am 2026-10-30 sechs Entscheidungen bestätigt. Noch am
selben Tag verlangte sie bzw. er für Entscheidung 5 die wissenschaftlich tragende Fassung.
Deren Zahlen wählte Claude, der KI-Assistent, nach einer Prüfung von 16 Entwürfen; die
Eigentümerin bzw. der Eigentümer bestätigte sie mit dem Zusammenführen von
Pull-Request #19, das die Vorab-Registrierung einfror. Entscheidung 5 steht hier in dieser
Fassung:

1. Exakte Gleichheit entscheidet über „richtig“.
2. Bewertet wird erst nach T-Anzahl, dann nach CNOT-Anzahl.
3. Die Kennzahlen-Zeile steht auch in Gitter-Läufen.
4. Schaltungs-Kaskaden werden in Schritt 1 nicht befördert.
5. Das Experiment entscheidet selbst, wie viele Seeds es braucht (sequenzielle Regel). Ein
   Lauf besteht, wenn die Zucht bis Generation 1000 mehr als die Hälfte der Ziele findet,
   welche die Startprogramme verfehlen. Erfolg heißt: Mehr als die Hälfte aller Seeds
   besteht. Gezählt werden die Seeds 1 bis 100, dann 111 bis 130, in dieser Reihenfolge.
   Nach jedem Seed-Paar prüft die Auswertung, ob die Belege für ein Urteil reichen; das
   Experiment endet spätestens nach 120 Paaren. Gleich wann es endet, meldet es einen
   falschen Erfolg höchstens mit Wahrscheinlichkeit 1/100; der ungünstigste Fall liegt bei
   rund 0.0079. Das ist etwas strenger als die alte Regel „9 von 10“, deren Schranke bei
   11/1024 = 0.0107 lag; die sequenzielle Regel ersetzt sie.
6. Die Panels haben 33 Such- und 22 Auswahl-Aufgaben.

## Die sechs Komponenten

| Komponente | Aufgabe | Dateien |
|---|---|---|
| 1. Variantengenerator | erzeugt neue Genome durch Mutation und Kreuzung | `src/genome.rs`, `src/variation.rs` |
| 2. Aufgabenumgebung | erzeugt die künstlichen Datensätze (Train, Valid, Hidden) | `src/task.rs` |
| 3. Bewertung | misst Genauigkeit oder R², berechnet die Fitness | `src/score.rs` |
| 4. Training | trainiert ein Netz mit festen Grenzen | `src/nn.rs`, `src/train.rs` |
| 5. Verdeckte Bewertung | testet die besten Modelle auf geheimen Daten | `src/hidden_eval.rs`, `src/bin/brut-eval.rs` |
| 6. Abstammungsarchiv | speichert alles in einer Hash-Kette, mit Anker | `src/archive.rs` |

Dazu kommen: `src/config.rs` (Einstellungen und Grenzen), `src/safety.rs`
(Not-Aus, Budgets), `src/evolve.rs` (die Hauptschleife), `src/util.rs`
(Hilfsfunktionen) und `src/main.rs` (Kommandozeile `brut`).

Bausteine des Lerner-Modus:

| Baustein | Aufgabe | Dateien |
|---|---|---|
| Befehlssatz und Interpreter | 46 Rechenbefehle, Prüfung, Kosten, Ausführung | `src/learner/dsl.rs`, `src/learner/interp.rs` |
| Startprogramme und Mutation | vier Startprogramme, zufällige Änderungen | `src/learner/seeds.rs`, `src/learner/mutate.rs` |
| Protokoll | Lernen auf einer Aufgabe, Qualität, AULC, Fingerabdruck | `src/learner/protocol.rs`, `src/learner/fec.rs` |
| Aufgaben | 24 handgeschriebene Familien, Formel-Grammatik, Panels | `src/learner/families.rs`, `src/learner/grammar.rs` |
| Vergleich und Statistik | Vergleichsverfahren, Tests | `src/learner/baselines.rs`, `src/learner/stats.rs` |
| ARC | Raster als Lernaufgabe; Laden mit Prüfungen | `src/learner/arc_cells.rs`, `src/arc_data.rs` |
| Verdeckte Bewertung | geheime Stufen, Budget, Abfragebuch | `src/learner_hidden.rs`, `src/ledger.rs`, `src/bin/brut-eval.rs` |
| Steuerung | Suche, Endphase, Bericht, Abstammung | `src/meta_evolve.rs`, `src/meta_report.rs`, `src/lineage_store.rs` |
| Nachrichtenformat | Brücke zu Stufe 2 | `src/learner/wire.rs` |

Alles unter `src/learner/` ist reine Rechnung ohne Datei- oder Netzzugriff.

Bausteine der Kaskaden-Zucht (Stufe 2c und Schritt 1):

| Baustein | Aufgabe | Dateien |
|---|---|---|
| Zucht | Darwin-Gödel-Auswahl, Richterrunden, Aufrufbudget, Checkpoints | `src/cascade/arena.rs` |
| Kaskaden und Varianten | Genom und Grenzen, Mutation und Kreuzung, Aufträge an den Mutator | `src/cascade/genome.rs`, `src/cascade/variation.rs` |
| Gitter-Rätsel | Rätselgenerator, Prompt-Protokoll, simuliertes Modell | `src/cascade/puzzle.rs`, `src/cascade/sim.rs` |
| Naht | Aufgabenart (`Domain`) und Gene (`GenePool`) | `src/cascade/domain.rs` |
| Kennzahlen | die `state`-Zeile je Generation | `src/cascade/observe.rs` |
| Qubits | exakter Simulator, Textprotokoll für Schaltungen | `src/cascade/qubit.rs`, `src/cascade/circuit_text.rs` |
| Schaltungen | Zielbibliothek, Aufgaben, Bewertung, Wiederentdeckung | `src/cascade/circuit.rs` |
| Positivkontrolle | simulierter Entwerfer, Startprogramme, Satzbausteine | `src/cascade/designer.rs`, `src/cascade/circuit_genes.rs` |
| Programm | Lauf-Verzeichnis, Hash-Kette, Fortsetzen, Prüfen, Beförderung | `src/bin/brut-breed.rs` |

Alles unter `src/cascade/` ist reine Rechnung ohne Datei-, Netz- oder Uhrzugriff; das
prüft `tests/cascade.rs`.

Die Einzelheiten stehen in [docs/ARCHITEKTUR.md](docs/ARCHITEKTUR.md).

## Sicherheitsfunktionen

- **Genome sind Daten.** Es wird nie Code aus Kandidaten ausgeführt.
- **Harte Grenzen** für Netzgröße, Epochen, Rechenarbeit (FLOPs) und Zeit,
  je Kandidat und für den ganzen Lauf. Auch die einstellbaren Grenzen haben
  feste Obergrenzen im Code.
- **Not-Aus:** Eine Datei `STOP` im Lauf-Verzeichnis beendet den Lauf sauber.
  Das Training prüft sie alle paar Schritte (höchstens alle 0.1 Sekunden),
  die verdeckte Bewertung vor jeder Anfrage und während sie wartet.
- **Verdeckte Bewertung** in einem eigenen Prozess (`brut-eval`). Nur dieser
  Prozess liest das Geheimnis. Er gibt je Anfrage nur einen abgerundeten Wert
  zurück (Stufe mindestens 0.01), hat ein festes Abfrage-Budget und liest nur
  Modelldateien aus dem Ordner `models/`.
- **Quarantäne:** Ist ein Modell auf den geheimen Daten deutlich schlechter
  als auf den Validierungsdaten, wird es gesperrt. Es kann dann nicht
  freigegeben werden.
- **Verdeckte Bewertung nur einmal am Ende:** Danach ist der Lauf
  abgeschlossen. `brut run` verweigert weitere Generationen, damit die
  geheimen Daten die Suche nicht lenken.
- **Manipulationsgeschütztes Archiv:** Jeder Eintrag enthält den SHA-256-Hash
  des vorigen Eintrags (einen digitalen Fingerabdruck: Ändert sich ein Byte,
  ändert sich der Hash vollständig). Die Datei `archive.head` merkt sich Zahl
  und Hash des letzten Eintrags. `brut verify` erkennt so geänderte, vertauschte und
  gelöschte Einträge, auch am Ende. Wer beide Dateien gezielt neu schreibt,
  bleibt unentdeckt (siehe Bekannte Lücken).
- **Menschliche Freigabe:** `brut approve` gibt einen Kandidaten nur frei,
  wenn er gültig ist, verdeckt bewertet wurde und nicht in Quarantäne ist.
- **Reproduzierbar:** Gleicher Seed, gleiche Kandidaten und gleiche
  Validierungswerte, auch bei anderer Thread-Zahl. Ausnahmen: Abbrüche durch
  Zeitgrenzen oder den Not-Aus, und die verdeckte Bewertung. Deren Seed zieht
  `brut init` zufällig, darum können verdeckte Werte und Quarantäne bei
  gleichem Seed verschieden ausfallen.

Zusätzlich im Lerner-Modus:

- **Geschlossene Sprache:** 46 Rechenbefehle, höchstens 72 Befehle je
  Programm, keine Schleifen, keine Sprünge. Die Rechenkosten stehen deshalb
  vor dem Start fest. Ein Programm, das mehr als das Vierfache des
  Referenznetzes je Schritt kosten würde, läuft gar nicht erst.
- **Kein Schummeln mit dem Label:** Das Label (die richtige Antwort) steht
  nur direkt vor `Learn` im Speicher. Die Validierung läuft auf einer Kopie
  des Speichers und kann den Lerner nicht verändern.
- **Globales Abfrage-Budget:** höchstens 48 verdeckte Abfragen je
  Protokollversion über alle Läufe und 12 je Lauf. Ein neues
  Lauf-Verzeichnis setzt das Budget nicht zurück.
- **Wenig Information je Abfrage:** eine Zahl (Δ, auf 0.01 abgerundet), beim
  Hauptwert dazu ein Bit „bestanden / nicht bestanden“.
- **Vorab-Registrierung:** Der erste Eintrag im Archiv hält das Messprotokoll
  als Hash fest. Passt der Hash nicht, verweigert `brut-eval` die Arbeit.
- **ARC nur im Auswerter:** Hashes aller Dateien werden geprüft,
  verbotene Trainingsquellen und Kopien von Bewertungsaufgaben führen zur
  Ablehnung.
- **Tests als Wächter:** Sie prüfen, dass `src/learner/` keine Dateien und
  kein Netzwerk anfasst, dass der Orchestrator keine Namen der geheimen Seite
  enthält und dass keine neuen Bibliotheken dazukommen.

Mehr dazu, und welche Lücken es gibt, steht in
[docs/SICHERHEIT_UND_RECHT.md](docs/SICHERHEIT_UND_RECHT.md).

## Grenzen

- Nur kleine MLPs, nur CPU, nur künstliche Aufgaben.
- Die Prozesstrennung ist keine echte Sicherheitsgrenze. Wer unter demselben
  Benutzerkonto arbeitet, kann die geheime Datei lesen.
- Die Aufgabe `parity` hat nur 256 verschiedene Eingaben. Trainings-,
  Validierungs- und Testdaten überschneiden sich daher. Sie misst eher
  Auswendiglernen als Verallgemeinern.
- Die FLOP-Zahlen sind Schätzungen (6 × Parameter × Beispiele), keine
  Messungen.
- Die Gesamtzeit (`max_total_seconds`) wird nur zwischen Generationen
  geprüft. Ein Lauf kann sie daher um bis zu eine Generation überschreiten.
  Die verdeckte Bewertung läuft danach trotzdem noch. Der Not-Aus wirkt
  dagegen schon während des Trainings.
- Lerner-Modus: Einige verdeckte Familien (HIDDEN-NEAR, HIDDEN-FAR) kennen wir
  aus dem Quelltext und dem Vorversuch. Sie zählen darum nur als
  beschreibende Werte. Aussagekräftig ist nur HIDDEN-GRAMMAR.
- Lerner-Modus: Die 32 ARC-Merkmale je Zelle sind handgemachtes Vorwissen.
  Sie begrenzen, was überhaupt lösbar ist (siehe oben).
- Lerner-Modus: Auch hier sind die FLOP-Zahlen aus einer festen Kostentabelle
  je Befehl gerechnet, nicht gemessen.
- Kaskaden-Zucht: Die simulierten Modelle (`sim:small`, `sim:large`) sind
  Positivkontrollen. Ihre Zahlen zeigen, dass die Zucht funktioniert, nicht, wie gut
  echte Modelle wären.
- Schritt 1: Das gezählte Experiment ist eine Positivkontrolle mit dem simulierten
  Entwerfer. Sein Hauptnachweis besagt, dass mehr als die Hälfte aller Seeds bis
  Generation 1000 besteht; wie hoch die Erfolgsquote ist, schätzt es nicht, und über echte
  Sprachmodelle sagt es nichts. Was das Experiment und der Pilot zeigen und was nicht,
  steht in [docs/SCHRITT1_SCHALTUNGEN.md](docs/SCHRITT1_SCHALTUNGEN.md), Abschnitt 8.

## Aufbau eines Lauf-Verzeichnisses (MLP-Modus)

```
runs/spirals/
├── config.json          Einstellungen dieses Laufs
├── archive.jsonl        Abstammungsarchiv (eine JSON-Zeile je Eintrag, Hash-Kette)
├── archive.head         Anker: Zahl der Einträge und Hash des letzten Eintrags
├── models/<id>.json     trainierte Modelle mit Status Ok (oder später Quarantined)
├── STOP                 Not-Aus (nur wenn gesetzt)
└── secret/              nur für brut-eval (Ordner 0700)
    ├── hidden.env       geheimer Seed der Testdaten (Datei 0600)
    ├── eval_state.json  verbrauchte Abfragen der verdeckten Bewertung
    └── eval.lock        Sperrdatei, damit jede Abfrage genau einmal zählt
```

Die Datei `.env.example` zeigt das Format beider geheimer Dateien (MLP- und
Lerner-Modus), nur mit Platzhaltern. Echte Geheimnisse erzeugen `brut init`
und `brut learner secret-init`. Bitte legen Sie nie echte Geheimnisse ins
Git-Repository.

## Eigene GPU-VM einrichten (freiwillig)

Echte Läufe mit Sprachmodellen brauchen einen GPU-Server mit Ollama (ADR 0003). Das Skript
`tools/vm-setup.sh` richtet eine frische Ubuntu-24.04-VM mit einem Befehl ein (PR #26,
erweitert in PR #30 und #31). Es erledigt Folgendes:

- **Warten.** Es wartet, bis die Einrichtung beim ersten Start (cloud-init) und die
  automatischen Updates fertig sind.
- **Pakete.** Build-Werkzeuge, git, curl, python3, tmux und ufw, dazu die
  GitHub-Kommandozeile `gh` aus der eigenen Paketquelle von GitHub.
- **Firewall.** Von außen ist nur SSH erreichbar (ufw). Ist Docker installiert,
  veröffentlicht Docker Ports danach standardmäßig nur auf 127.0.0.1, da ufw diese Ports
  nicht filtert. Ausgenommen sind Netze von docker compose und Ports mit ausdrücklich
  `0.0.0.0`.
- **Werkzeuge.** Rust 1.97.0 mit clippy und rustfmt, cargo-audit und Claude Code.
- **Repository.** Ein Klon in `~/ki-brutmaschine` und signierte Commits, da `main` nur
  signierte Commits annimmt. Den SSH-Schlüssel dafür legt das Skript an und meldet ihn bei
  GitHub an. Die Anmeldung bei GitHub braucht einen einzigen Code, den Sie auf
  github.com/login/device eingeben.
- **Dienst.** Der systemd-Dienst `brut-remote-control` startet Claude Remote Control nach
  jedem Boot in tmux.

Ollama und GPU-Treiber installiert das Skript nicht. Es legt kein Geheimnis im Repository
ab. API-Schlüssel gehören in die Datei `~/ki-brutmaschine/.env`, die in `.gitignore` steht.
Das Skript darf jederzeit erneut laufen und überspringt, was schon erledigt ist.

Starten Sie es mit Ihrem normalen Benutzerkonto, nicht als root. Auf einer frischen VM laden
Sie es an einem festen Commit herunter und prüfen seine SHA-256-Summe. Commit und Summe
übernehmen Sie aus dem Pull-Request oder Release, in dem beide stehen:

```sh
COMMIT=<commit>
SHA256=<sha256>
curl -fsSL -o vm-setup.sh \
  "https://raw.githubusercontent.com/vmanke/ki-brutmaschine/$COMMIT/tools/vm-setup.sh"
echo "$SHA256  vm-setup.sh" | sha256sum -c - && bash vm-setup.sh
```

Liegt der Klon schon in `~/ki-brutmaschine`, etwa nach einem früheren Lauf des Skripts,
holen Sie den neuen Stand und starten das Skript von dort:

```sh
cd ~/ki-brutmaschine && git pull && bash tools/vm-setup.sh
```

Am Ende nennt das Skript drei Schritte, die Sie einmal von Hand erledigen:

1. **Neu anmelden.** Melden Sie sich mit `exit` ab und verbinden Sie sich erneut per SSH.
   Erst dann findet die Shell `claude` und `cargo`.
2. **Bei Claude anmelden.** Nutzen Sie Ihr Konto von claude.ai: `claude auth login`.
3. **Remote Control starten.** Starten Sie den Dienst und verbinden Sie sich mit seiner
   tmux-Sitzung. Beantworten Sie dort die ersten Fragen von Remote Control mit `y`:

   ```sh
   sudo systemctl start brut-remote-control
   tmux -L brut attach -t brut
   ```

   Mit Strg+b und danach d verlassen Sie tmux. Remote Control läuft weiter und startet nach
   einem Ende von selbst neu.

Danach startet der Dienst Remote Control nach jedem Boot. Ein späterer Lauf des Skripts
schaltet einen abgeschalteten Dienst nicht wieder ein.

```sh
# Zustand des Dienstes
systemctl status brut-remote-control

# Zuschauen; verlassen mit Strg+b, dann d
tmux -L brut attach -t brut

# Abschalten; ein Beenden in tmux genügt nicht, da systemd den Dienst neu startet
sudo systemctl disable --now brut-remote-control
```

Die VM der Eigentümerin bzw. des Eigentümers ist eine unterbrechbare Nebius-VM zum
Spot-Preis mit einer NVIDIA H200. Wir messen dort 143 771 MiB GPU-Speicher, 16 vCPU und
196 GB Arbeitsspeicher.

**Kosten.** Bei Nebius endet die Abrechnung der GPU erst, wenn Sie die VM in der Konsole
oder über die API anhalten. Ein Herunterfahren aus der VM heraus beendet sie nicht (laut der
Dokumentation von Nebius). Die Datenträger kosten auch, solange die VM angehalten ist.

## Tests

```sh
cargo test                            # Unit- und End-to-End-Tests
cargo clippy --all-targets -- -D warnings
cargo audit                           # bekannte Schwachstellen (Werkzeug cargo-audit)

# Ende-zu-Ende-Tests des Lerner-Modus, schneller im Release-Build
cargo test --release --test learner_e2e

# Messungen der Rechengeschwindigkeit (nur im Release-Build sinnvoll)
cargo test --release --test bench -- --ignored --nocapture --test-threads 1
```

Die Tests des Lerner-Modus:

- `tests/learner_security.rs` prüft die Sicherheitsregeln: keine verbotenen
  Namen im Orchestrator, kein Datei- oder Netzzugriff unter `src/learner/`,
  `#![forbid(unsafe_code)]` in jeder neuen Datei, keine neuen Bibliotheken,
  keine unfertigen Stellen (`todo!()`, `unwrap()`) im Bibliothekscode, und
  eine Anfrage über 1 MiB an `brut-eval` wird ohne Belastung des Budgets
  abgelehnt.
- `tests/learner_e2e.rs` macht winzige echte Läufe: verdeckte Bewertung,
  gleiche Ergebnisse bei 1 und 4 Threads, Not-Aus und Fortsetzen,
  Manipulation von Abstammungsblöcken, die Kommandozeile und eine ARC-Prüfung
  auf fünf handgemachten Aufgaben (`tests/fixtures/arc/`). Jeder dieser
  Läufe stimmt einmal die Vergleichsverfahren ab (gemessen 5.6 · 10¹⁰ FLOPs). Ohne
  Optimierung (`cargo test` ohne `--release`) dauert das sehr lange.
- `bench_learner_interpreter` in `tests/bench.rs` misst den Interpreter in
  GFLOP/s (Milliarden gezählte Rechenoperationen je Sekunde). Ziel nach dem
  Bauvertrag (Release-Build, F = 16): mindestens 2.5 GFLOP/s je Thread und
  mindestens 3 000 ganze Aufgaben je Sekunde auf 4 Threads.
- Ein Test zu den ARC-Obergrenzen läuft nur, wenn die Umgebungsvariable
  `BRUT_ARC_TEST_ROOT` auf die ARC-Daten zeigt.

Die Tests der Kaskaden-Zucht:

- `tests/cascade.rs` prüft die Reinheit von `src/cascade/` und macht kleine echte Läufe
  mit `brut-breed`: gleiche Ergebnisse über Prozessgrenzen, `--verify` mit
  manipulierten Protokollen und Berichten, Not-Aus und Fortsetzen, harte Abbrüche, die
  Sperre und LLM-Runden.
- `tests/universe.rs` zeigt mit Golden-Fixtures aus der Zeit vor Schritt 1, dass die neue
  Naht Gitter-Läufe nicht verändert. Dazu prüft es die `state`-Zeilen (je Generation genau
  eine, gleich nach Abbruch und Fortsetzen, gleich bei 1 und 4 Threads) und die neuen
  Optionen.
- Die Schaltungs-Bausteine haben Unit-Tests in ihren eigenen Dateien. Dazu gehören die 11
  Beleg-Schaltungen der geprüften Ziele, die meisten aus der Literatur, die bei genau
  ihren T- und CNOT-Zahlen exakt sein müssen, und ein unabhängiger Vergleichssimulator auf
  500 Zufallsschaltungen.
- `tests/circuits.rs` macht echte Schaltungs-Läufe mit `brut-breed` (gleiche Ergebnisse
  über Prozesse und bei 1 und 4 Threads, Stopp und Fortsetzen, `--verify` gegen eine
  veränderte `rediscovery.json`, abgelehnte Beförderung), legt die Kalibrierung der
  Positivkontrolle fest und vergleicht den exakten Simulator mit `src/quantum.rs`.

## Schneller rechnen (freiwillig)

Der normale Build läuft auf jedem x86-64-Prozessor. Wenn Sie die Programme nur
auf dem Rechner nutzen, auf dem Sie sie bauen, können Sie dessen
Befehlssatz freischalten:

```sh
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

Auf dem Testrechner (mit AVX2) war die Vorwärts- und Rückwärtsrechnung damit
etwa 1.5- bis 1.8-mal so schnell. Die Ergebnisse blieben gleich, denn Rust
fasst Multiplikation und Addition nicht von selbst zu einem Befehl zusammen
und ändert die Reihenfolge der Summen nicht. So gebaute Programme laufen aber
nicht auf Prozessoren ohne diese Befehle.
