# Stufe-2a-Pilot: der LlmProposer in der Suche

## 1. Frage

Die Positiv-Kontrolle (docs/POSITIVKONTROLLE.md) zeigte, dass der Messstab
Spielraum hat. Der 2a-Pilot prüft die nächste Stufe: Lässt sich der LlmProposer
in die Lauf-Suche einhängen, läuft die Suche damit sauber und deterministisch, und
nimmt sie einen guten Vorschlag an und trägt ihn bis in die Finalisten?

Ob ein echtes offenes Modell gute Lerner vorschlägt, klärt der Pilot hier nicht;
das braucht ein echtes Modell auf einem Rechner (ADR 0003). Hier prüfen wir die
Mechanik, ohne Netz, ohne Geheimnis und ohne verdeckte Auswertung.

## 2. Austauschpunkt

Die Suche band bisher den Zufallsmutator fest ein. Wir haben einen Austauschpunkt
ergänzt, womit ein Lauf einen beliebigen Proposer bekommen kann.

- **`meta_evolve::meta_run_with_proposer(run_dir, eval, no_hidden, proposer)`** nimmt
  einen `Arc<dyn ProgramProposer>` entgegen.
- **`meta_run`** bleibt unverändert und reicht den Zufallsmutator durch, weshalb jeder
  bestehende Lauf sich genau wie vorher verhält.
- Der Proposer wird über die Arbeiter-Fäden geteilt, weshalb er `Send + Sync` ist; der
  `LlmProposer` erfüllt das über einen Mutex und einen Zähler.

Die Änderung ist rein additiv. Nach ihr bestehen alle 372 Bibliothekstests und alle
10 End-zu-End-Tests, darunter die Tests für Determinismus und Fortsetzung, und clippy
meldet keine Warnung.

## 3. Aufbau

Wir fahren mit `examples/stage2a_pilot.rs` eine kurze Suche (24 Mitglieder, 3 Runden,
Breite F = 8, keine verdeckte Phase) über zwei Modell-Rückseiten.

- **Platzhalter (`StubModel`).** Gibt deterministisch kein gültiges Programm zurück,
  weshalb der Proposer zweimal wiederholt und dann auf den Zufallsmutator zurückfällt.
- **Orakel (`OracleModel`).** Gibt genau einen bekannten bestehenden Lerner (nlms-glm-4
  aus der Positiv-Kontrolle) als gültiges JSON zurück, sodass der Proposer ihn annimmt.

## 4. Ergebnis

| Rückseite | Versuche | angenommen | Rückfall | deterministisch | Orakel in Finalisten |
|---|---|---|---|---|---|
| StubModel | 72 | 0 (0 %) | 24 | ja (zwei Läufe gleich) | — |
| OracleModel | 24 | 24 (100 %) | 0 | — | ja (strukturgleich) |

- **Plumbing steht.** Der Platzhalter-Lauf fällt in allen Fällen sauber auf den
  Zufallsmutator zurück, protokolliert jeden Versuch, und zwei Läufe mit gleicher Saat
  liefern Bit für Bit dieselbe Zusammenfassung.
- **Annahme- und Weitergabe-Pfad stehen.** Der Orakel-Lauf nimmt den Lerner in jedem
  Versuch an. Sein in der Endphase konstanten-nachjustierter Nachkomme wird zum besten
  Finalisten mit einem Panel-Mittel von 0.2385, gegenüber 0.2297 im Zufallslauf. Wir
  prüfen die Weitergabe über Struktur-Gleichheit, da die Nachjustierung die Kennung
  ändert, nicht aber die Struktur.

## 5. Grenze

Mit dem Platzhalter misst der Pilot nur die Mechanik, nicht die Lernkraft eines echten
Modells. Das Orakel beweist, dass ein gültiger Vorschlag angenommen wird und sich
durchsetzt, doch liefert es immer dasselbe Programm. Ob ein echtes offenes Modell
vielfältige, gute Lerner vorschlägt, zeigt erst der Rechnerlauf mit der Ollama-Rückseite
(docs/STAGE2_BACKEND_CONTRACT.md, ADR 0003).

Der Vorbehalt der Positiv-Kontrolle bleibt: Der Hauptwert belohnt teils Billigkeit. Wer
einen echten 2a-Lauf bewertet, prüfe daher nicht nur, ob die Suche besteht, sondern ob
sie Lerner findet, welche über den Billigkeits-Vorteil hinaus besser lernen.

## 6. Reproduktion

```
cargo run --release --example stage2a_pilot -- <scratch_dir>
```

Der Lauf nutzt keine verdeckte Auswertung, kein Geheimnis und kein Netz; er schreibt
allein in `<scratch_dir>`. Je Lauf legt er die Versuchs-Datensätze (Modell,
Prompt-Hash, Token-Zahl, Ergebnis, Genom-Kennung) nach `<scratch_dir>/<lauf>/attempts.json`
ab, womit sich die Protokollierung aus den Ausgabedateien prüfen lässt. Das ist der
verfügbare Umfang, nicht die vollständige Wiedergabe von Anfrage und Antwort, welche
erst die echte Modell-Rückseite auf einem Rechner leistet.
