# Stufe-2b: Mutation über einen menschlich geprüften Pull Request

## 1. Zweck

Die AI entwirft Kandidaten-Mutationen als Pull Requests; ein Mensch prüft und mergt,
bevor eine Mutation in einen Lauf geht. Der Austauschpunkt aus dem 2a-Pilot bleibt für
schnelle, versiegelte Experimente; dieser Pfad ist der Aufsichts-Pfad und bewusst langsam.
Die Entscheidung und ihre Grenzen stehen in docs/adr/0004. Wir setzen hier zunächst nur
die Genom-Variante um (ein DSL-Programm als JSON); die Config-plus-Prompt- und die
Rust-Variante sind spätere Arbeit.

## 2. Ablauf

- **`brut-propose`** liest einen Eltern-Topf (oder fällt auf die Startprogramme zurück),
  mutiert zwei Kandidaten unter einem Haupt-Seed, prüft jeden und schreibt die zwei
  Kandidaten-Dateien samt Herkunfts-Datei. Das Werkzeug ruft weder Git noch Netz; es
  rechnet und schreibt nur in sein Ausgabe-Verzeichnis (ADR 0001).
- **`scripts/propose-pr.sh`** kopiert die zwei Dateien nach `candidates/`, committet auf
  einem `mutate/`-Zweig, pusht und öffnet einen Entwurfs-Pull-Request mit der Herkunft im
  Text. Es mergt nie.
- **`brut-ingest --candidates candidates`** prüft den Topf nach dem Merge: Jede Datei muss
  sich einlesen lassen, `check` bestehen (Struktur und Kostendeckel) und einen Dateinamen
  tragen, der ihrer Inhalts-Kennung gleicht. Das ist das Tor, das gemergte Kandidaten in
  einen Lauf lässt.

Je Pull Request genau zwei mutierte Kandidaten, gemeinsam geprüft und gemergt. Die
Herkunft (Eltern-Kennung, Operator, Seed, Adapter, Schrittkosten je Breite) steht im
Pull-Request-Text, womit sich die Mutation Bit für Bit wiedergeben lässt (ADR 0002).

## 3. Umfangs-Zaun

Ein Mutations-Pull-Request darf nur Dateien unter `candidates/` ändern. Die eine Regel
liegt in der Bibliothek (`hitl::path_in_scope`); Git liegt in `scripts/scope-fence.sh`,
welches die geänderten Pfade an `brut-ingest --scope-check` reicht. Die CI
(`.github/workflows/scope-fence.yml`) läuft nur für `mutate/`-Zweige und weist jeden
solchen Pull Request ab, der einen Pfad außerhalb von `candidates/` berührt. So kann ein
Mutant nie den Auswerter, das Kostenmodell (`dsl.rs`), die verdeckte Grammatik, die
Abnahme oder ein Geheimnis ändern. Das ist der direkte Schutz gegen den Fall, den die
Darwin-Gödel-Maschine berichtet (angenommen bei der ICLR 2026): ein Agent schaltete seinen
Prüfer ab, statt die Aufgabe zu lösen.

## 4. Blindheit und Freigabe

Vorschlager und Prüfer sehen die verdeckten Familien nie. Die Review prüft Gültigkeit,
Umfang, Sicherheit und Herkunft sowie, ob die Änderung eine plausible Lernänderung ist,
nie einen Leistungswert auf den verdeckten Familien, da zur Prüfzeit kein solcher Wert
vorliegt. Sähe der Prüfer die Leistung, würde die Review ein Abstimmkanal und bräche die
entwickler-bewusste Trennung. Nichts wird selbsttätig gemergt; der Merge ist die
ausdrückliche menschliche Freigabe.

## 5. Reproduktion

```
cargo run --release --bin brut-propose -- --out <scratch_dir> --seed 42
cargo run --release --bin brut-ingest  -- --candidates <scratch_dir>/candidates
```

Der erste Befehl schreibt zwei Kandidaten-Dateien und `proposal.json` nach
`<scratch_dir>`; der zweite prüft sie. Beide nutzen keine verdeckte Auswertung, kein
Geheimnis und kein Netz.
