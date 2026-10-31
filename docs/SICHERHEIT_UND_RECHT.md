# Sicherheit und Recht

Dieses Dokument beschreibt, gegen welche Gefahren die Brutmaschine geschützt
ist, wie der Schutz im Code aussieht und wo es Lücken gibt. Am Ende steht eine
Orientierung zur europäischen KI-Verordnung. Die Abschnitte 1 bis 6 gelten
vor allem für den MLP-Modus (kleine neuronale Netze). Die Abschnitte 7 bis 11
beschreiben, was der Lerner-Modus (Stufe 1) zusätzlich braucht. Dort sind
Kandidaten kleine Lernprogramme (siehe docs/ARCHITEKTUR.md).

## Bedrohungsmodell

Wir nehmen an: Die Person, die den Lauf startet, ist ehrlich. Gefahren kommen
vor allem aus dem Suchprozess selbst (er optimiert, was gemessen wird), aus
Fehlern und aus unbeabsichtigtem oder böswilligem Ändern von Dateien.

### 1. Ziel-Hacking (Goodharts Gesetz)

**Gefahr:** Goodharts Gesetz, in der bekannten Fassung von Marilyn
Strathern (1997): „Wenn eine Kennzahl zum Ziel wird, hört sie auf, eine gute
Kennzahl zu sein.“ Eine Suche findet Wege, die Bewertung zu überlisten,
statt die Aufgabe zu lösen.

**Gegenmaßnahmen im Code:**

- Kandidaten sind reine Daten (`src/genome.rs`). Sie enthalten keinen Code
  und können die Bewertung nicht verändern.
- Bewertung und Fitness rechnet der Orchestrator selbst (`src/score.rs`),
  nicht der Kandidat. (Orchestrator heißt das Programm `brut run`, das die
  Evolution steuert, siehe `src/evolve.rs`.)
- `Genome::validate` und `variation::repair` erzwingen erlaubte Wertebereiche
  für jedes Gen.
- Strafterme für Parameter und FLOPs in `score::fitness` verhindern, dass
  Größe allein belohnt wird.
- Die verdeckte Bewertung prüft das Ergebnis unabhängig (siehe 2).

### 2. Überanpassung an die Validierung

**Gefahr:** Viele Auswahlrunden auf denselben Validierungsdaten machen den
Validierungswert zu optimistisch.

**Gegenmaßnahmen im Code:**

- Geheimer Testdatensatz `Split::Hidden`, erzeugt nur in `brut-eval`
  (`src/hidden_eval.rs`).
- Nur ein abgerundeter Wert je Anfrage (`hidden_eval::quantize`). Die
  Rundungsstufe ist mindestens 0.01. Diese Untergrenze erzwingt `brut-eval`
  selbst (`MIN_HIDDEN_SCORE_STEP`), auch wenn der Aufrufer mit `--step` einen
  kleineren Wert verlangt. `BRUT_HIDDEN_STEP` in der geheimen Datei kann die
  Stufe weiter erhöhen.
- Festes, dauerhaft gezähltes Abfrage-Budget (`EvalState`,
  `secret/eval_state.json`); Zählung vor der Bewertung. Gezählt wird unter
  einer exklusiven Dateisperre (`secret/eval.lock`), so dass auch mehrere
  gleichzeitige `brut-eval`-Prozesse jede Abfrage genau einmal zählen.
  Anfragen mit fehlender oder unpassender Modelldatei werden vorher
  abgewiesen und kosten keine Abfrage, weil sie die geheimen Daten nicht
  berühren.
- Quarantäne bei zu großem Abstand (`safety::gap_exceeded`,
  `CandidateStatus::Quarantined`). Gesperrte Kandidaten sind nicht
  freigabefähig (`brut approve`).
- Die verdeckte Bewertung ist endgültig: Danach lehnt `brut run` weitere
  Generationen ab (`evolve::check_not_final`). Bewertet wird eine feste Liste
  der besten Kandidaten nach Validierungs-Fitness (`evolve::hidden_set`).
  Gesperrte Kandidaten bleiben darin, damit ein erneuter Aufruf nicht so
  lange weitere Kandidaten prüft, bis einer besteht.

### 3. Abfluss des Geheimnisses

**Gefahr:** Kennt die Suche den Seed der Testdaten, ist die verdeckte
Bewertung wertlos.

**Gegenmaßnahmen im Code:**

- Der Seed steht nur in `secret/hidden.env`. Die Datei hat die Rechte 0600,
  der Ordner 0700 (`hidden_eval::create_secret`).
- `HiddenEvaluator::open` verweigert eine Datei, die für Gruppe oder andere
  lesbar ist.
- `brut init` erzeugt die Datei einmalig (`hidden_eval::create_secret`).
  Danach liest `brut run` sie nie: Der Orchestrator startet `brut-eval` mit
  leerer Umgebung und übergibt nur den Pfad
  (`hidden_eval::request_hidden_eval`).
- Ein Integrationstest prüft, dass `src/evolve.rs` und `src/main.rs` weder
  `HiddenEvaluator` noch `parse_env_file` noch `HIDDEN_SEED_KEY` verwenden.
  Andere Teile des Moduls `hidden_eval` (Anfrage, Anlegen der Datei) nutzen
  sie dagegen schon. Für den Lerner-Modus prüft er auch die neuen Dateien des
  Orchestrators (`src/meta_evolve.rs`, `src/meta_report.rs`,
  `src/lineage_store.rs`) und weitere Namen der geheimen Seite (zum Beispiel
  `LearnerEvaluator`, `HiddenToken`, `LEARNER_SEED_KEY`) sowie Pfade der
  ARC-Bewertungssätze.
- Im Lerner-Modus liegt das Geheimnis einmal je Rechner in
  `~/.brut/learner/hidden.env` (Datei 0600, Ordner 0700). Es entsteht mit
  `brut learner secret-init`. Der Befehl zeigt nur den Pfad, nie den Seed.
- `.gitignore` schließt `.env`, `*.env`, `secret/`, `runs/` und den Ordner
  `data/` aus (bis auf die Beispieldatei `data/manifest.example.json`).
  `.env.example` enthält nur Platzhalter; ein Test prüft das.

### 4. Manipulation des Archivs

**Gefahr:** Jemand ändert nachträglich Ergebnisse, löscht Einträge oder
vertauscht sie.

**Gegenmaßnahmen im Code:**

- SHA-256-Hash-Kette in `src/archive.rs`. Jeder Eintrag enthält den Hash des
  vorigen. Jede Änderung, Löschung oder Vertauschung innerhalb der Datei
  bricht die Kette.
- Werden Einträge am Ende entfernt, bleibt die Kette gültig. Darum steht
  neben dem Archiv die Datei `archive.head` (Anker). Sie enthält die Zahl der
  Einträge und den Hash des letzten Eintrags und wird nach jedem Schreiben
  erneuert. Fehlt der Anker oder passt er nicht zum Archiv, meldet die
  Prüfung einen Fehler. Nur nach einem Absturz darf der Anker zurückliegen:
  um genau einen Eintrag, oder um so viele Einträge, wie er vor einem
  Gruppen-Schreiben angekündigt hat (Feld `pending`, siehe unten).
- `brut verify` und `Archive::open` prüfen die ganze Kette und den Anker.
  `brut verify` zeigt außerdem den Hash des letzten Eintrags an. Diesen Wert
  können Sie an einem anderen Ort aufbewahren.
- Im MLP-Modus wird jeder Eintrag sofort fest auf die Platte geschrieben
  (`fsync`), bevor der nächste folgt. Die Suche im Lerner-Modus sammelt ihre
  Einträge und schreibt sie in Gruppen mit einem `fsync`: bei jeder Stufe-C-
  Runde, bei jedem Zwischenstand (`BatchDigest`), spätestens nach 4 096
  Einträgen und am Ende der Suche. Das spart je nach Platte 13 bis 23 % der
  Suchzeit (Erratum V7). Ein Absturz verliert höchstens die Einträge seit der letzten
  Gruppe; sie liegen alle nach dem letzten Zwischenstand, und das Fortsetzen
  erzeugt sie neu. Alles nach der Suche (Finalisten, verdeckte Ergebnisse,
  ARC-Prüfungen) wird wieder Eintrag für Eintrag geschrieben.
- Einträge, die eine JSON-Rundreise nicht unverändert überstehen (zum
  Beispiel NaN), werden abgelehnt.

### 5. Ausufernde Rechenlast

**Gefahr:** Die Suche erzeugt immer größere Modelle und verbraucht
unbegrenzt Rechenzeit.

**Gegenmaßnahmen im Code:**

- Harte Grenzen in `config::Limits`: Schichten, Breite, Parameter, Epochen,
  FLOPs und Sekunden je Kandidat, Sekunden und FLOPs je Lauf.
- Diese Grenzen sind einstellbar, haben aber feste Obergrenzen im Code
  (`config::HARD_MAX_*`): höchstens 64 verdeckte Schichten, 4096 Neuronen je
  Schicht, 2 000 000 Parameter, 100 000 Epochen und 10 000 Kandidaten je
  Startpopulation oder Generation. `Config::validate` lehnt größere Werte ab.
  So kann auch eine gültige, aber unsinnige Konfiguration den Lauf nicht
  hängen lassen oder den Speicher füllen.
- `variation::repair` verkleinert jedes Kind, bis es die Grenzen einhält.
- `safety::Guard` prüft vor jeder Generation Zeit und FLOP-Budget.
  Kinder, die das Restbudget sprengen würden, werden vorher aussortiert
  (`Guard::can_afford`).
- `train::train` prüft Zeit und FLOPs während des Trainings und hört sauber
  auf (`stopped_by_budget`).
- Not-Aus: die Datei `STOP` (`brut stop`). Das Training prüft sie alle 32
  Schritte, aber höchstens alle 0.1 Sekunden (`safety::StopSignal`), und
  hört dann sauber auf (Status `Stopped`). Noch nicht begonnene Kinder werden
  nicht mehr trainiert. Die verdeckte Bewertung prüft den Not-Aus vor jeder
  Anfrage und beendet einen laufenden `brut-eval`-Prozess, sobald die Datei
  erscheint. Gemessen (Debug-Build, ein Thread, 16 große Kandidaten): Der
  Lauf endete 0.2 Sekunden nach dem Not-Aus statt nach 78 Sekunden.
- Die Zahl der Threads ist begrenzt (`threads`, höchstens 1024).

### 6. Bösartige Modelldateien

**Gefahr:** Eine manipulierte Modelldatei bringt den Auswerter zum Absturz,
belegt sehr viel Speicher oder führt Code aus.

**Gegenmaßnahmen im Code:**

- Modelle sind reines JSON (`nn::Mlp`, für Faltungsnetze
  `cnn::model::CnnModel`): Zahlenlisten, kein ausführbares Format.
- `brut-eval` liest nur Modelldateien im Ordner `models/` des
  Lauf-Verzeichnisses (oder in dem Ordner, den `BRUT_MODELS_DIR` in der
  geheimen Datei nennt). Symbolische Links und `..` werden vorher aufgelöst.
  Ist eine Datei kein gültiges Modell, meldet es nur das, ohne Inhalte der
  Datei zu zitieren. So kann ein Aufrufer `brut-eval` nicht als Leser für
  fremde Dateien missbrauchen.
- `brut-eval` liest nur gewöhnliche Dateien (keine Geräte wie `/dev/zero`,
  keine Pipes, keine Ordner) und davon höchstens 64 MiB je Modell, auch wenn
  die Datei während des Lesens wächst. Je Anfrage liest es höchstens 1 MiB.
  Es prüft die Form des Netzes (`Mlp::check_shape`), die Parameterzahl
  (höchstens 2 000 000), die Tiefe (höchstens 64 verdeckte Schichten), die
  Breite (höchstens 4096 Neuronen je Schicht), endliche Gewichte und die
  passende Ein- und Ausgabegröße. Gemessen: Ein Modell mit einer Schicht aus
  399 999 Neuronen belegte vorher 3.1 GB Speicher und brauchte 34 Sekunden;
  jetzt wird es sofort abgewiesen.
- Eine Datei mit `"kind": "cnn"` ist ein Faltungsnetz. Sie darf keine
  unbekannten Felder enthalten, auch nicht im Genom. Der Leser bricht ab,
  sobald die Listen der trainierten Werte zusammen mehr als 2 000 000 Zahlen
  halten oder die Listen der laufenden Statistiken mehr als 86 528
  (`MAX_RUNNING_VALUES`), denn mehr Werte hat kein gültiges Modell. Bevor
  `brut-eval` einen Tensor anlegt, prüft es die Architektur: die Formprüfung
  des Genoms gegen die Aufgabe, die Parameterzahl (höchstens 2 000 000), Tiefe
  und Breite des dichten Kopfes wie beim MLP und die Rechenkosten. Der
  Vorwärtslauf über den verborgenen Teil darf höchstens 1e11 FLOPs kosten
  (`MAX_EVAL_FLOPS`), da ein Faltungsnetz jedes Gewicht an jeder Bildstelle
  wiederverwendet und die Parameterzahl seine Kosten nicht begrenzt. Die FLOPs
  allein begrenzen die Zeit jedoch nicht: Eine Datei aus Faltungen je Kanal
  mit Schichtnorm und GELU kostete 9.2e10 FLOPs und hätte etwa 179 Sekunden
  gebraucht (ein Thread; 200 der 2000 Zeilen brauchten 17.9 Sekunden), länger
  als die 120 Sekunden, nach denen der Orchestrator `brut-eval` beendet.
  Deshalb schätzt `brut-eval` die Zeit der Bewertung mit dem Zeitmodell aus
  `cnn::shape`, welches auch die kopierten Fensterwerte, die Zwischenwerte und
  die Aufrufe je Gruppe zählt, und weist eine Datei ab, deren Schätzung 60
  Sekunden übersteigt (`MAX_EVAL_SECONDS`). In unseren Messungen blieb jede
  Bewertung unter 0.91 ihrer Schätzung; zwei Dateien knapp unter der Grenze
  brauchten 17 und 42 Sekunden. Nach unserer Rechnung braucht ein Faltungsnetz
  innerhalb der voreingestellten Grenzen auf den Bild- und Sequenzaufgaben
  weniger als 3 % der FLOP-Grenze. Danach prüft `brut-eval` jede Werteliste
  gegen ihre Tensorform, auf endliche Werte und auf laufende Varianzen nicht
  unter 0. Eine Fehlermeldung nennt nur Anzahlen, nicht die Listen selbst,
  sodass sie kurz bleibt. Eine abgewiesene Datei kostet keine Abfrage.
- Beim Bewerten rechnet `score::evaluate` in Blöcken, deren Größe von der
  Netzbreite abhängt. Bei einem MLP braucht ein Block höchstens etwa 8 MiB für
  Zwischenwerte; Puffer für die Rückwärtsrechnung legt die reine Bewertung
  nicht an. Bei einem Faltungsnetz legt candle beim Bewerten keinen
  Rechengraphen an (`Param::value`), sodass jede Zwischenschicht frei wird,
  sobald die nächste sie gelesen hat. Hinzu kommen die Puffer, in welche
  candle die Fenster einer Faltung kopiert: einer je Thread und je 512
  Bildstellen. Die Zeitgrenze hält einen solchen Puffer unter etwa 16 MB.
  Gemessen (opt-level 2): Eine Datei knapp unter der Zeitgrenze belegte mit
  einem Thread höchstens 25 MiB, mit vier Threads 111 MiB; ein kleines
  Faltungsnetz belegte 11 MiB.
- Der Orchestrator wartet höchstens 120 Sekunden auf eine Antwort von
  `brut-eval` (`hidden_eval::EVAL_TIMEOUT_SECONDS`). Danach beendet er den
  Prozess und vermerkt den Fehler im Archiv.
- Bibliothekscode stürzt bei falschen Eingaben nicht ab, sondern gibt einen
  Fehler zurück.
- `config.json` mit unbekannten Feldern wird abgelehnt.

### 7. Lerner-Modus: Programme als Kandidaten

**Gefahr:** Im Lerner-Modus sind Kandidaten Programme. Ein Programm könnte
versuchen, Dateien zu lesen, das Netzwerk zu nutzen, endlos zu rechnen oder
den Auswerter zum Absturz zu bringen.

**Gegenmaßnahmen im Code:**

- Die Programme sind in einer geschlossenen Sprache geschrieben: 46
  Rechenbefehle auf Registern fester Größe, höchstens 72 Befehle, keine
  Schleifen, keine Sprünge, keine Zeiger. Programme sehen nur ihre Register.
- Der Interpreter (`src/learner/interp.rs`) ist der Sandkasten. Ein
  **Sandkasten** ist eine Umgebung, aus der ein Programm nicht ausbrechen
  kann. Der Interpreter ist ein `match` über eine feste Liste von Befehlen.
  Es gibt keine Erzeugung von Maschinencode und kein Nachladen von Code.
- Jede neue Datei beginnt mit `#![forbid(unsafe_code)]`. Damit verbietet der
  Rust-Compiler „unsicheren“ Code, der die Speicherprüfungen umgehen könnte.
- Ein Test (`tests/learner_security.rs`) prüft, dass `src/learner/` weder
  `std::fs`, `std::net`, `std::process`, `File::`, `Command`, `TcpStream`,
  `env::var`, `include_bytes`, `include_str` noch `unsafe` erwähnt.
  Dateizugriff für ARC liegt deshalb außerhalb, in `src/arc_data.rs`.
- Die Rechenkosten stehen vor dem Start fest. Ein Programm über der
  Kostengrenze (das Vierfache des Referenznetzes je Schritt) läuft gar nicht
  erst (`LearnerGenome::check_cost`).
- Sehr kleine Zahlen werden zu 0 (Erratum V4). Rechnungen mit
  „subnormalen“ Zahlen (unter etwa 1.2 · 10⁻³⁸) sind auf x86-Prozessoren bis
  zu 100-mal langsamer. Ein gültiges Programm konnte so die Kostengrenze
  unterlaufen. Jetzt hält ein Register keine Beträge unter 2⁻³⁴ (Matrizen:
  2⁻⁶⁸), und kein Rechenbefehl erzeugt einen subnormalen Wert.
- Je Aufgabe gibt es eine Zeitgrenze. Not-Aus und Zeitgrenze werden bei
  Schritt 0, danach alle 64 Schritte und alle 64 Vorhersagen der Validierung
  geprüft (Erratum V5; vorher nur alle 1024 Schritte, also bei Aufgaben mit
  384 Schritten nur vor dem ersten).
- Eine nicht endliche Ausgabe oder ein Trainingsfehler über 100 beendet die
  Aufgabe mit dem Wert 0 (wie bei AutoML-Zero).
- `brut-eval` liest keine Programmdateien. Das Programm reist in der Anfrage
  mit (höchstens 1 MiB, unbekannte Felder werden abgelehnt) und wird vor der
  Buchung vollständig geprüft. Ein Test schickt 2 MiB und prüft, dass nichts
  gebucht wird.

### 8. Lerner-Modus: Schummeln beim Lernen

**Gefahr:** Die Suche findet einen Weg, die Messung zu überlisten, statt
besser zu lernen.

**Gegenmaßnahmen:**

| Angriffsfläche | Gegenmittel |
|---|---|
| Label vor der Vorhersage lesen | Label-Register werden vor jedem `Predict` gelöscht; ein Test mit einem „Schummel-Programm“ verlangt, dass es höchstens Zufallsniveau erreicht |
| Validierung verändert den Lerner | Validierung auf einer Kopie des Speichers |
| den Auswerter verändern | unmöglich: Programme sehen nur ihre Register |
| Reihenfolge oder häufigste Klasse ausnutzen | gemischte Reihenfolge der Beispiele; Zufallsniveau wird abgezogen; Panel-Austausch |
| Lernrate an die Breite F koppeln | F wechselt in der Suche; F = 32 kommt nur verdeckt vor; die Kopplung wird gemessen |
| Glücksfund durch Rauschen („Fluch des Gewinners“) | gestufte Bewertung, Test über 40 Familien, 3 Seeds |
| Vorsprung mit mehr Rechenzeit kaufen | Vergleich nur mit gleich teuren Verfahren |
| glücklicher Zufalls-Seed | gemeinsame Zufallszahlen: der Seed kommt von der Aufgabe, nicht vom Programm |
| Anpassung durch uns Entwickler | ausgeloste Aufteilung, Vorab-Registrierung, geheime Grammatik-Familien, globales Abfragebuch, ARC als Außentest |

### 9. Lerner-Modus: Überanpassung an die verdeckten Daten über viele Läufe

**Gefahr:** Im MLP-Modus gilt das Abfrage-Budget je Lauf-Verzeichnis. Wer
viele Läufe macht, bekommt immer neues Budget und kann sich Schritt für
Schritt an die geheimen Daten anpassen.

**Gegenmaßnahmen im Code:**

- Ein globales Geheimnis je Rechner und daneben ein **Abfragebuch**
  (`ledger.jsonl`, Hash-Kette, `src/ledger.rs`). Es zählt über alle Läufe:
  höchstens 48 Abfragen je Protokollversion und 12 je Lauf. Die Grenzen
  stehen in der geheimen Datei; `--budget` kann sie nur senken.
- Jede Abfrage wird gebucht, bevor bewertet wird, unter einer exklusiven
  Dateisperre (`eval.lock`). Abgelehnte Abfragen stehen im Buch, kosten aber
  nichts.
- Je Abfrage kommt nur eine Zahl heraus (Δ, auf 0.01 abgerundet), beim
  Hauptwert dazu ein Bit „bestanden / nicht bestanden“. Das sind etwa 8 Bit
  Information je Abfrage.
- **Vorab-Registrierung:** Der erste Eintrag im Archiv ist der
  Protokoll-Hash. Passt der Hash einer Anfrage nicht zu dem von `brut-eval`,
  lehnt es ab, ohne zu buchen.
- Die 40 geheimen Grammatik-Familien entstehen aus dem geheimen Seed. Ihre
  Signaturen kommen in den öffentlichen Töpfen nicht vor.
- Die Finalisten stehen vor der ersten verdeckten Abfrage fest. Danach ist
  der Lauf abgeschlossen. Quarantäne gilt, wenn Δ auf frischen öffentlichen
  Aufgaben um mehr als 0.05 über Δ auf den verdeckten Aufgaben derselben
  Familien liegt.
- Nur `src/learner_hidden.rs` kann ein `HiddenToken` erzeugen; nur damit
  lassen sich verdeckte Aufgaben bauen.
- Die Pfade zu den ARC-Daten stehen nur in der geheimen Datei. Ein Test
  prüft, dass der Code des Orchestrators keine Pfade der Bewertungssätze
  enthält.

### 10. Lerner-Modus: fremde Daten (ARC)

**Gefahr:** Heruntergeladene Dateien könnten verändert, riesig oder kaputt
sein. Trainingsdaten könnten Bewertungsaufgaben schon enthalten
(**Kontamination**). Fremder Code könnte mitlaufen.

**Gegenmaßnahmen im Code:**

- Eine Manifest-Datei nennt jede Datei mit SHA-256-Hash, Lizenz, Quelle und
  Rolle. `brut-eval` prüft alle Hashes vor jeder ARC-Prüfung.
- Der Lader (`src/arc_data.rs`) liest nur gewöhnliche Dateien, höchstens
  1 MiB je Aufgabe. Er akzeptiert nur rechteckige Raster von 1 × 1 bis
  30 × 30, nur Farben 0 bis 9, 1 bis 10 Trainingspaare und 1 bis 4
  Testpaare. Fehlermeldungen zitieren keine Inhalte.
- Feste Kontaminationsregeln (Tabelle in docs/ARCHITEKTUR.md). Ein Verstoß
  ist ein harter Fehler. Zum Beispiel ist ARC-AGI-2-Training bei der Prüfung
  von ARC-AGI-1 verboten, weil es 376 der 400 ARC-AGI-1-Bewertungsaufgaben
  enthält.
- Vor jeder Prüfung vergleicht `brut-eval` normierte Hashes aller
  Trainingspaare mit den Bewertungspaaren (über alle 8 Drehungen und
  Spiegelungen und mit neu nummerierten Farben). Jede Überschneidung führt
  zur Ablehnung, ohne Buchung. Ein Test pflanzt ein gedrehtes und
  umgefärbtes Duplikat und prüft, dass es erkannt wird.
- Während `brut` und `brut-eval` läuft kein fremder Code. Nur der
  abgeschaltete Zusatzversuch E2 bräuchte eine einmalige Umwandlung der
  re-arc-Daten; sie liefe nur im getesteten Sandkasten-Rezept des Plans.

### 11. Lieferkette

**Gefahr:** Eine neue oder manipulierte Bibliothek bringt Schadcode oder
Fehler mit.

**Gegenmaßnahmen:**

- Stufe 1 fügt keine Bibliothek hinzu. Erst die CNN-Modelle bringen zwei
  hinzu: `candle-core` und `candle-nn` von GitHub, an einem festen Commit
  (ADR 0007). Ein Test prüft, dass `[dependencies]` in `Cargo.toml` genau
  diese beiden und `hex`, `rand`, `rand_chacha`, `rayon`, `serde`,
  `serde_json` und `sha2` enthält, die beiden nur von diesem Commit, und dass
  es keine weiteren Abhängigkeits-Tabellen und keine Ersetzungen (`[patch]`,
  `[replace]`) gibt.
- Vor jedem Build für Versuche läuft `cargo audit` mit der
  RustSec-Datenbank. Gefordert sind 0 bekannte Schwachstellen. Mit candle
  messen wir 0 Schwachstellen über 164 Pakete und eine Warnung: `paste`
  1.0.15 wird nicht mehr gepflegt (RUSTSEC-2024-0436).
- Stufe 1 braucht keinen API-Schlüssel und kein Netzwerk.

## Bekannte Lücken

Diese Punkte sind **nicht** gelöst. Bitte beachten Sie sie, bevor Sie Ideen
aus dem Prototyp übernehmen.

1. **Prozesstrennung ist keine Sicherheitsgrenze.** Alles läuft unter
   demselben Benutzerkonto. Wer dieses Konto nutzt, kann die geheime Datei
   lesen. Für echten Schutz braucht der Auswerter einen eigenen Benutzer,
   einen Container oder einen eigenen Rechner.
2. **Budget und Rundung kommen vom Aufrufer.** Der Orchestrator übergibt
   `--budget` und `--step`. Ein veränderter Orchestrator könnte ein höheres
   Budget angeben. Abhilfe: `BRUT_HIDDEN_BUDGET` in der geheimen Datei
   setzen. Die Rundungsstufe kann er nicht unter 0.01 drücken; eine gröbere
   Stufe legt `BRUT_HIDDEN_STEP` in der geheimen Datei fest. Diese gröbere
   Stufe kennt der Orchestrator nicht. Wählen Sie `hidden_gap_threshold`
   dann mindestens doppelt so groß, sonst kann allein das Abrunden zur
   Quarantäne führen.
3. **Modelldateien sind nicht im Archiv verankert.** Das Archiv speichert
   keinen Hash der Modelldatei. Eine ausgetauschte Datei würde so bewertet,
   wie sie ist.
4. **Die Hash-Kette hat keinen externen Anker.** Wer Schreibrechte hat, kann
   das ganze Archiv neu berechnen oder das Ende abschneiden und `archive.head`
   passend neu schreiben. Erkannt werden nur Änderungen, bei denen Archiv und
   Anker nicht zusammenpassen. Abhilfe: den Hash, den `brut verify` anzeigt,
   regelmäßig an einem anderen Ort ablegen oder signieren. Archive aus
   älteren Versionen ohne `archive.head` lehnt die Brutmaschine ab.
5. **Keine Dateisperre für das Archiv.** Zwei gleichzeitige Prozesse im
   selben Lauf-Verzeichnis (z. B. `brut run` und `brut approve`) könnten
   durcheinander schreiben. `brut verify` erkennt das danach, verhindert es
   aber nicht. (Den Abfragezähler von `brut-eval` schützt dagegen eine
   Sperre.)
6. **Festes Zeitlimit für `brut-eval`.** Der Orchestrator beendet den
   Auswerter nach 120 Sekunden. Das Limit ist eine Konstante im Code, keine
   Einstellung. Hatte der Auswerter die Abfrage schon gezählt, bleibt sie
   verbraucht.
7. **Grobe Kontrolle der Gesamtzeit.** `max_total_seconds` wird nur
   zwischen Generationen geprüft; eine Generation kann die Grenze also
   überziehen (je Kind höchstens `max_candidate_seconds`). Die verdeckte
   Bewertung läuft nach einer Zeitgrenze trotzdem noch, nach einem Not-Aus
   nicht. Der Not-Aus selbst wirkt nach höchstens etwa 0.1 Sekunden plus 32
   Trainingsschritten.
8. **FLOPs sind geschätzt**, nicht gemessen. Im Lerner-Modus zählt die
   Kostentabelle der Befehle. Nach dem Nullsetzen sehr kleiner Zahlen
   (Erratum V4) brauchen die Befehle mit Normalverteilung (VGaussian,
   MGaussian) noch etwa zehnmal so viel Zeit je gezähltem FLOP wie MatVec,
   SSin/SCos mit sehr großen Argumenten etwa neunmal und SExp/SLog etwa
   fünfmal. Ein Programm aus solchen Befehlen rechnet also länger, als seine
   FLOPs sagen. Die Zeit begrenzen dann nur `max_total_seconds`, die
   Zeitgrenze je Aufgabe, das Zeitlimit von `brut-eval` und der Not-Aus.
   Nicht umgesetzt: neue Kosten für diese Befehle und eine Messung der
   echten Zeit im Guard (Erratum K1).
9. **Gleiche Verteilung.** Alle drei Datensätze kommen aus derselben
   Verteilung. Robustheit gegen veränderte Daten wird nicht geprüft.
10. **Parity misst Auswendiglernen.** Es gibt nur 256 mögliche Eingaben.
11. **Die Freigabe prüft nicht, wer freigibt.** `--by` ist ein freier Text,
    keine Anmeldung.
12. **Kleine Lücke zwischen Prüfung und Öffnen.** `brut-eval` prüft den Pfad
    der Modelldatei und öffnet sie danach. Wer im Ordner `models/` schreiben
    darf, könnte die Datei genau dazwischen gegen einen Link tauschen. Gelesen
    würde dann trotzdem höchstens eine gewöhnliche Datei bis 64 MiB, und
    Fehlermeldungen zitieren keine Inhalte.
13. **Keine Garantie gegen adaptive Überanpassung an die verdeckten Daten.**
    Rundung und Abfrage-Budget begrenzen, wie viel über die geheimen Daten
    nach außen dringt, verhindern Überanpassung aber nicht. Verfahren mit
    Garantien („The Ladder“, „Thresholdout“) sind nicht umgesetzt (siehe
    docs/ARCHITEKTUR.md). Das gilt auch für das Abfragebuch des
    Lerner-Modus.

Zusätzlich im Lerner-Modus:

14. **Der Entwickler kennt einige verdeckte Familien.** Die Familien von
    HIDDEN-NEAR und HIDDEN-FAR stehen als Quelltext im Repository, und
    ähnliche Familien liefen im Vorversuch. Diese Stufen sind darum nur
    beschreibend. Aussagekräftig ist nur HIDDEN-GRAMMAR.
15. **Auch das Abfragebuch hat keinen externen Anker.** Die Datei
    `ledger.head` daneben hält die Zahl der Einträge und den letzten Hash.
    Damit fällt auf, wenn jemand nur `ledger.jsonl` löscht, kürzt oder
    ändert. Wer aber beide Dateien löscht oder beide neu schreibt, setzt das
    Budget zurück. Abhilfe: den Hash des letzten Eintrags an einem anderen
    Ort aufbewahren.
16. **`HiddenToken` ist keine Sicherheitsgrenze.** Es hält nur den Code
    ordentlich. Wer den Code ändert oder unter demselben Benutzerkonto
    arbeitet, kommt an das Geheimnis.
17. **Die Textsuch-Tests sind keine Beweise.** Sie fangen Versehen, keine
    Absicht. Wer Namen geschickt umschreibt, umgeht sie.
18. **ARC-Daten sind öffentlich.** Die Bewertungsdateien liegen auf der
    Platte. Sie stehen sehr wahrscheinlich auch in den Trainingsdaten großer
    Sprachmodelle. Für Stufe 2 mit einem Sprachmodell sind sie deshalb kein
    „entwicklerfremder“ Test.
19. **Einmalige Kosten zählen getrennt.** `max_total_flops` begrenzt nur die
    Suche. Das Abstimmen der Vergleiche und die Endphase kommen dazu
    (Erratum E6). Seit `learner-v2` stehen beide als `LearnerCost` im Archiv
    und unter `one_time` im Bericht (Erratum V11). Die verdeckte Bewertung
    und die ARC-Prüfungen rechnet `brut-eval`; ihre FLOPs zählt niemand.
20. **Zufallstreffer beim Hauptwert.** Hat ein Programm in Wahrheit keinen
    Vorsprung, besteht es den Test trotzdem in höchstens etwa 5 % der Fälle.
    Darum verlangt der Erfolg 2 von 3 Seeds und einen Sieg über die Ablation
    „nur Konstanten“.
21. **Zeitgrenzen hängen vom Rechner ab.** Bricht eine Aufgabe wegen der
    Zeitgrenze ab, ist das Ergebnis nicht mehr reproduzierbar. Das kommt nur
    bei sehr langsamen Rechnungen vor.
22. **Stufe C zählt zum Suchbudget.** Die Zwischenprüfung auf SELECT-Aufgaben
    wirkt nicht auf die Suche zurück, verbraucht aber Such-FLOPs. In
    `learner-v1` war das fast die Hälfte des Budgets im vollen Suchraum und
    nur 12 bis 25 % in der Ablation „nur Konstanten“. Die Ablation bekam so
    mehr echte Suchrechnung. Seit `learner-v2` ist Stufe C billiger, und der
    Bericht nennt ihren Anteil getrennt (`search.stage_c_flops`). Gleich
    viel Suchrechnung für beide Arme gibt es aber weiterhin nicht (Erratum
    K2).
23. **Fehlermeldungen beim Laden der ARC-Daten** nennen Dateipfade und
    Aufgabennamen. Sie bleiben seit `learner-v2` im Abfragebuch von
    `brut-eval` (im geschützten Ordner). Der Orchestrator bekommt nur den
    festen Text „ARC data failed …“ (Erratum V12).

## Orientierung: EU-KI-Verordnung

**Wichtig:** Dies ist eine Orientierung für das Design. Es ist **keine
Rechtsberatung** und **keine Konformitätsbewertung**. Die Brutmaschine ist
Forschungssoftware. Ob und welche Pflichten der Verordnung (EU) 2024/1689
(„KI-Verordnung“, englisch „AI Act“), geändert durch die Verordnung
(EU) 2026/1744 („Digital-Omnibus zur KI“, in Kraft seit 27. Juli 2026), für
ein konkretes System gelten, hängt von dessen Zweck und Einsatz ab. Das muss
im Einzelfall fachkundig geprüft werden. Stand dieser Übersicht: Oktober 2026.

Für Forschung gibt es zwei Ausnahmen, die hier wohl am wichtigsten sind:

- **Art. 2 Abs. 6:** Die Verordnung gilt nicht für KI-Systeme oder
  KI-Modelle, einschließlich ihrer Ausgabe, die eigens für den alleinigen
  Zweck der wissenschaftlichen Forschung und Entwicklung entwickelt und in
  Betrieb genommen werden.
- **Art. 2 Abs. 8:** Sie gilt auch nicht für Forschungs-, Test- und
  Entwicklungstätigkeiten zu KI-Systemen oder KI-Modellen, bevor diese in
  Verkehr gebracht oder in Betrieb genommen werden (Tests unter realen
  Bedingungen ausgenommen).

Die Artikel in der Tabelle (Kapitel III Abschnitt 2) gelten nur für
Hochrisiko-KI-Systeme (Art. 8). Nach der Änderung durch (EU) 2026/1744
gelten diese Pflichten erst ab dem 2. Dezember 2027 für Hochrisiko-Systeme
nach Anhang III und ab dem 2. August 2028 für solche nach Anhang I
(ursprünglich 2. August 2026 und 2. August 2027). Die Tabelle beschreibt
also keine Pflichten der Brutmaschine. Sie zeigt nur, welche Funktionen des
Prototyps zu welchen Themen der Verordnung passen.

| Artikel | Thema | Passende Funktion im Prototyp |
|---|---|---|
| Art. 9 | Risikomanagementsystem | Bedrohungsmodell und Lückenliste oben; harte Grenzen (`config::Limits`); Guard und Not-Aus (`src/safety.rs`); Kostengrenze für Lernprogramme |
| Art. 10 | Daten und Daten-Governance | künstliche, dokumentierte, reproduzierbare Datensätze (`src/task.rs`, `src/learner/families.rs`); strikte Trennung Train / Valid / Hidden; ausgeloste Aufteilung der Familien; ARC-Daten nur mit Manifest (Hash, Lizenz, Quelle) und Kontaminationsregeln |
| Art. 12 | Aufzeichnungspflichten | automatisches, verkettetes Ereignisprotokoll (`archive.jsonl`) mit Konfiguration, Abstammung, Messwerten, Sicherheitsereignissen und Freigaben; Abstammungsblöcke mit Hashes im Archiv; Abfragebuch `ledger.jsonl` |
| Art. 14 | Menschliche Aufsicht | `brut approve` nur nach verdeckter Bewertung (im Lerner-Modus: HIDDEN-GRAMMAR) und ohne Quarantäne; `brut status`, `brut lineage`, `brut learner show`; Not-Aus `brut stop` |
| Art. 15 | Genauigkeit, Robustheit und Cybersicherheit | verdeckte Testbewertung mit Budget; Quarantäne; Vergleich nur mit gleich teuren Verfahren; Vorab-Registrierung; Integritätsprüfung `brut verify`; Eingabeprüfung von Modell-, Programm-, ARC- und Konfigurationsdateien |

Zum Lerner-Modus (Stufe 1), ebenfalls ohne Gewähr:

- Er ist Forschung vor jedem Inverkehrbringen. Ein Versuch nach Plan S
  braucht zusammen etwa 10¹³ FLOPs, Stufe 1 insgesamt höchstens etwa
  10¹⁴. Art. 51 Abs. 2 vermutet „Fähigkeiten mit hoher Wirkung“ bei einem
  KI-Modell mit allgemeinem Verwendungszweck erst ab mehr als 10²⁵ FLOPs
  Training. Das ist etwa elf Zehnerpotenzen mehr. Ob ein Lernprogramm
  überhaupt ein solches Modell wäre, ist hier nicht geprüft; die Zahl zeigt
  nur die Größenordnung.
- Datenlizenzen: ARC-AGI-1 und ARC-AGI-2 stehen unter Apache-2.0, ConceptARC
  und re-arc unter MIT. Werden Daten weitergegeben, gehört die Namensnennung
  in eine NOTICE-Datei. BARC hat keine Lizenzdatei und wird nicht genutzt;
  `brut-eval` lehnt es als Trainingsquelle immer ab. Die ARC-Daten liegen
  nur lokal unter `data/` und kommen nicht ins Git.
- Es werden keine personenbezogenen Daten verarbeitet. Alle Aufgaben sind
  künstlich oder öffentliche Rätsel.

Was fehlt, um ernsthaft in diese Richtung zu gehen (Auswahl): eine
technische Dokumentation des Einsatzzwecks, Tests auf veränderte
Datenverteilungen, echte Zugriffskontrolle und Authentifizierung für
Freigaben, ein externer Anker für das Protokoll und ein geregelter Prozess für
Vorfälle.
