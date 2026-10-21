# Positiv-Kontrolle: hat der Messstab Spielraum?

## 1. Frage

Stufe 1 ergab auf dem Hauptwert HIDDEN-GRAMMAR 0 von 3 (docs/ERGEBNISSE_STUFE1.md).
Der beste Kandidat lag mit einem Vorsprung Δ von −0.01 bis −0.06 unter der
rechengleichen Vergleichsmethode. Ein solcher Null-Befund hat zwei mögliche
Ursachen, welche er selbst nicht trennt: Entweder fand die Suche keinen Gewinner,
obwohl einer im Suchraum liegt, oder es gibt keinen Gewinner, da die
Vergleichsmethode jedes im DSL ausdrückbare Programm bei gleicher Rechenlast
schlägt. Das Opus-Review benannte diese Mehrdeutigkeit als Hauptrisiko, weshalb
wir sie vor jedem Ausgeben von Stufe-2-Rechenzeit klären.

Die Positiv-Kontrolle fragt daher: Gibt es überhaupt einen DSL-Lerner, welchen der
vorregistrierte Hauptwert als „bestanden“ zertifiziert?

## 2. Aufbau

Wir fahren die echte, unveränderte Hauptwert-Pipeline (`LearnerEvaluator`,
`src/learner_hidden.rs`) auf einem **Entwickler-Panel**. Der Auswerter liest seine
Saat aus einer Datei, deren Pfad wir übergeben; wir zeigen ihn auf eine selbst
geschriebene `hidden.env` mit einer Entwickler-Saat im Scratchpad, womit die
Messung bis auf die Saat Bit für Bit die des verdeckten Auswerters ist. Die echte
`~/.brut/learner/hidden.env` fassen wir nicht an, und wir geben kein Geheimnis aus.

- **Panel.** Je Entwickler-Saat zieht `grammar::disjoint_pool_semantic` 40
  Grammatik-Familien, genau wie im verdeckten Lauf, nur aus einer nicht-geheimen
  Saat. Wir messen über 6 Saaten (0x00C0FFEE00000001 bis …06).
- **Regel.** Bestehen heißt p < 0.05 und Δ ≥ 0.01 und Δ bei F = 32 fällt um
  höchstens 0.05 gegenüber F = 8 und 16 (`protocol::primary_decision`).
- **Werkzeug.** `examples/positive_control.rs`; die geprüften Programme liegen in
  `examples/data/positive_control_candidates.json`.

## 3. Ergebnis

**Der Messstab hat Spielraum.** Mehrere von Hand geschriebene DSL-Lerner bestehen
den Hauptwert, stabil über alle 6 Panels und mit klarem Abstand zur Schwelle.

| Lerner | Schrittkosten (F = 16) | Δ (6 Saaten) | Ergebnis |
|---|---|---|---|
| nlms-glm-4 | 330 | 0.05 bis 0.10 | 6 von 6 bestanden |
| nlms-glm-1 | 102 | 0.03 bis 0.09 | 6 von 6 bestanden |
| starve-lms-03 | 83 | 0.03 bis 0.08 | 6 von 6 bestanden |
| rf-lin-relu | 330 | 0.03 bis 0.05 | 3 von 3 bestanden (erster Lauf) |

Zum Vergleich kostet der Referenz-Lerner nn2-sgd je Schritt 2640 bei F = 16. Die
vier Saat-Lerner und unsere sieben ersten eigenen Entwürfe bestanden alle nicht;
das lineare Modell lag mit Δ ≈ −0.02 am nächsten, jeder teurere Entwurf fiel
weiter zurück.

Die bestehenden Lerner fanden vier unabhängige Entwurfs-Agenten, welche wir mit
verschiedenen Blickwinkeln angesetzt haben. Der tragende Kniff ist ein billiges
Online-Modell: Statt einer Matrix-Vektor-Multiplikation mit Kosten 2 F² nutzen die
Gewinner das Skalarprodukt `VDot` mit Kosten 2 F und bauen die Ausgabe Einheit für
Einheit, womit die Schrittkosten auf 83 bis 330 statt 2640 fallen.

## 4. Mechanismus, ehrlich geprüft

Der Durchgang hängt an zwei Dingen, welche wir getrennt nachgewiesen haben.

- **Der Rechengleich-Hebel.** Die rechengleiche Vergleichsmethode bekommt ein
  Budget in Höhe der Rechenlast des Kandidaten. Ein billiger Kandidat schließt
  damit alle Programm-Vergleiche aus seiner Menge aus (linear-sgd kostet 1056,
  perceptron 1136, prototype 1280) und hungert das abgestimmte Adam-MLP auf etwa
  eine Epoche aus. Nehmen wir denselben Lerner starve-lms-03 und blähen ihn mit
  reiner Füll-Rechenlast auf Kosten 1619 auf, bei Bit für Bit gleichen Vorhersagen,
  so kippt er auf allen 6 Saaten von bestanden auf nicht bestanden (Δ −0.02 bis
  −0.05). Die Rechenlast allein entscheidet hier über das Ergebnis.
- **Echtes Lernen ist nötig.** Billigkeit allein genügt nicht. Ein Null-Lerner,
  welcher konstant 0 ausgibt (Kosten 16), und ein Lerner mit der GLM-Form, aber
  Lernrate 0 (Kosten 83), fallen beide auf allen 6 Saaten durch (Δ −0.05 bis
  −0.09). Der bestehende Lerner muss die lineare und schwach nichtlineare Struktur
  der Ziele tatsächlich online anpassen.

Aus unserer Sicht misst der Hauptwert damit eine Mischung aus Lernqualität und
Billigkeit, nicht Lernqualität allein.

## 5. Was das für Stufe 1 bedeutet

Der Null-Befund von Stufe 1 war ein **Suchversagen, kein unmögliches Ziel**. Es
gibt DSL-Lerner, welche der Hauptwert zertifiziert, doch die Suche fand sie nicht.
Nach unserer Rechnung liegt der Grund in der Ausrichtung der Suche: Sie startet aus
den teuren Saat-Netzen mit Kosten rund 2640 und blieb in deren Umgebung, während
die Gewinner im billigen Bereich mit Kosten 83 bis 330 liegen. Der dauerhafte
Bestenspeicher (K3) und der LlmProposer, welche wir gebaut haben, können die Suche
dorthin führen, doch belegt ist das noch nicht.

## 6. Vorbehalt zum Messstab

Der Rechengleich-Hebel ist ein Schlupfloch: Der leichteste Weg zum Bestehen führt
derzeit über möglichst geringe Rechenlast, nicht über das beste Lernverfahren. Wer
Stufe 2 an diesem Hauptwert misst, misst teils, wie billig ein Lerner ist. Zwei
mögliche Gegenmittel, beide noch zu prüfen:

- **Mindestbudget.** Wir geben der Vergleichsmethode ein Budget nicht unter einer
  festen Untergrenze, sodass ein sehr billiger Kandidat sie nicht aushungern kann.
- **Budget-Streuung.** Wir vergleichen gegen die beste Vergleichsmethode über eine
  Spanne von Budgets statt nur beim Budget des Kandidaten, womit der reine
  Billigkeits-Vorteil entfällt.

## 7. Empfehlung

Die Kontrolle ist bestanden, das Tor zu Stufe 2 ist offen. Wir sehen zwei Wege,
über welche die Admin entscheidet.

- **Weg 1: Stufe-2a-Pilot jetzt.** Wir verdrahten den LlmProposer in die Suche und
  messen, ob sie die billigen, bestehenden Lerner selbst findet. Der Vorbehalt aus
  Abschnitt 6 bleibt als bekannte Grenze vermerkt.
- **Weg 2: Messstab zuerst schärfen.** Wir schließen das Billigkeits-Schlupfloch
  nach Abschnitt 6, prüfen die Positiv-Kontrolle erneut und starten Stufe 2a danach.

Aus unserer Sicht trägt Weg 1, da der Pilot auch mit dem bekannten Vorbehalt zeigt,
ob die Suche überhaupt Signal erzeugt, und das Schärfen des Messstabs eine eigene
Änderung mit eigener Prüfung ist.

## 8. Reproduktion

```
cargo build --release --example positive_control
POSCTRL_CANDIDATES=examples/data/positive_control_candidates.json \
  ./target/release/examples/positive_control <scratch_dir> \
  54324593195745281 54324593195745282 54324593195745283
```

Die Läufe nutzen nur Entwickler-Saaten und schreiben allein in `<scratch_dir>`.
Kein Geheimnis und kein verdeckter Lauf sind beteiligt.
