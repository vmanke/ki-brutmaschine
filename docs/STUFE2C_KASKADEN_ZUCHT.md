# Stufe 2c: Zucht von Kaskaden aus Sprachmodell-Workern

Englische Fassung: [STAGE2C_CASCADE_BREEDING.md](STAGE2C_CASCADE_BREEDING.md).

## 1. Zweck

Wir züchten nicht mehr einzelne Lernprogramme, sondern **Kaskaden**. Eine Kaskade
besteht aus ein bis vier Stufen, welche nacheinander laufen. In jeder Stufe arbeiten ein
bis drei **Worker** parallel; ein Worker ist ein Sprachmodell mit einem Systemprompt aus
benannten Abschnitten und mit **Traits**. Jede Stufe sieht die ursprüngliche Aufgabe und
die gewählten Antworten aller früheren Stufen (der residuale Pfad). Die Kaskaden lösen
Rasterrätsel im Stil von ARC. Jede Kaskade ist zugleich Auswerter: Sie bewertet blind
die Antworten anderer Kaskaden. Die Entscheidungen stehen in
[ADR 0005](adr/0005-cascades-are-bred-by-darwin-goedel-selection-and-every-cascade-is-also-a-judge.md),
die Begriffe in `CONTEXT.md`, die offenen Arbeiten in `.scratch/kaskaden-zucht/`.

## 2. Ablauf einer Generation

- **Eltern wählen.** Wir ziehen Eltern aus dem ganzen Archiv, wie es die
  Darwin-Gödel-Maschine tut (Zhang et al., ICLR 2026, Anhang C.2): Das Gewicht ist
  sigmoid(10 · (Fitness − 0.5)) / (1 + Zahl der Kinder). Gute Kaskaden mit wenigen
  Kindern kommen also öfter dran; jede Kaskade behält eine Chance.
- **Kinder bilden.** Zufällige Operatoren ändern Traits, Prompt-Abschnitte, Worker oder
  Stufen. Die Stufen-Kreuzung nimmt den Anfang von Elter A und das Ende von Elter B; die
  Abschnitts-Kreuzung erbt jeden benannten Abschnitt und jeden Trait von A oder B. Den
  Kreuzungspartner wählen die Peers: Von drei gezogenen Kandidaten gewinnt der mit der
  besten Peer-Wertung.
- **LLM-Runden.** In festen Abständen schreibt der **Mutator** (normalerweise Claude)
  Kinder: Er verbessert eine Kaskade oder verbindet zwei. Seine Antworten werden
  gespeichert und wiedergegeben (ADR 0002).
- **Bewerten.** Jedes Kind löst alle Rätsel des Such-Panels. Danach vergleicht es sich
  mit bis zu drei nicht verwandten Kaskaden, und eine dritte, mit keiner der beiden
  verwandte Kaskade urteilt blind.
- **Fitness.** Fitness = 0.8 · Lösungsquote + 0.2 · Urteilskraft − Kosten. Die Kosten
  zählen Antwort- und Denk-Tokens (0.01 Fitness je 1000 Tokens je Rätsel) und einen
  festen Preis je Modellaufruf.

## 3. Jedes Netz ist Auswerter

- **In der Stufe.** Sind sich die Worker einer Stufe uneinig, beurteilt jeder Worker jedes
  Paar verschiedener Antworten, ohne zu wissen, von wem sie stammen; die Reihenfolge der
  beiden Antworten zieht ein Zufallsgenerator. Die Antwort mit den meisten Siegen geht
  weiter. Hier zahlt sich Urteilskraft im Einsatz aus, da dort keine Lösung bekannt ist.
- **Im Archiv.** Ein Urteil zählt nur, wenn der Richter weder Elter noch Kind noch
  Geschwister der beiden Beurteilten ist. Entferntere Verwandte (Großeltern, Enkel,
  Vettern) dürfen urteilen. Als Richter kommen zuerst die neuen Kaskaden der Generation
  dran, und unter ihnen die, die bisher am seltensten geurteilt haben.
- **Urteilskraft.** Ist genau eine der beiden Antworten richtig, sehen wir, ob der Richter
  sie gewählt hat. Antwortet er dann ohne gültiges Urteil, zählt das als falsch. Die
  Urteilskraft ist (richtig + 1) / (geurteilt + 2). Wurde eine Kaskade nie so gewertet,
  behält sie den Startwert 1/2.
- **Peer-Wertung.** Aus allen Urteilen schätzen wir eine Bradley-Terry-Wertung nach dem
  MM-Verfahren von Hunter (2004). Eine Stimme wiegt ln(u / (1 − u)) bei Urteilskraft
  u > 0.5, sonst 0; das ist das optimale Gewicht unabhängiger Wähler bekannter
  Treffsicherheit (Nitzan und Paroush 1982). Die Peer-Wertung entscheidet bei
  Gleichstand und bei der Partnerwahl. Sie ersetzt nie die Lösungsquote.

Warum nicht nur Peer-Urteile? Bewerten sich Kandidaten nur gegenseitig, können sie sich
loben, ohne etwas zu lösen. Die Darwin-Gödel-Maschine berichtet einen solchen Fall: Ein
Agent entfernte die Marken, an denen sein Prüfer Halluzinationen erkannte, und erzielte
so die volle Punktzahl, ohne das Problem zu lösen (Zhang et al., Anhang H).

## 4. Traits und harte Grenzen

| Trait | Bereich | harte Grenze im Code |
|---|---|---|
| Denkbudget | 0 bis 4096 Tokens | 4096, je Stufe enger einstellbar |
| Temperatur | 0.0 bis 1.5 | 1.5 |
| Antwortlänge | 64 bis 2048 Tokens | 2048 |
| Selbstprüfung | an oder aus | fester Prüftext des Orchestrators |

Die Grenzen sind Konstanten im Code und keine Gene. Eine Laufkonfiguration darf die
Grenzen für Stufen, Worker je Stufe und Denkbudget je Stufe verengen, aber keine Grenze
lockern. Temperatur und Antwortlänge haben nur feste Grenzen. Unbekannte Schlüssel in
der Konfiguration lehnt der Lauf ab. Nur ein vom Code-Owner geprüfter Pull Request
ändert die Grenzen. Wir nehmen an, dass die Auswahl das Denkbudget ohne Grenze und ohne
Preis stetig erhöhen würde, da tieferes Nachdenken mehr Rätsel löst. Darum kostet jedes
Token Fitness.

## 5. Claude als Mutator, ohne Netz im Kern

Fehlt in einer LLM-Runde eine Antwort, hält der Lauf an. `brut-breed` schreibt die
Anfragen nach `<lauf>/llm/requests/<schlüssel>.json` und endet mit Exit-Code 3. Eine
Claude-Sitzung oder ein Mensch schreibt je Anfrage die Antwort nach
`<lauf>/llm/answers/<schlüssel>.txt`. Danach setzt `brut-breed --out <lauf> --resume`
den Lauf an derselben Stelle fort und spielt die Antworten ein. Der Kern bekommt so
keinen Netzzugang (ADR 0001), und jeder Lauf bleibt wiederholbar (ADR 0002). Liegen die
Antworten mit `--llm-answers <ordner>` woanders, gehört die Option auch zum Fortsetzen;
`brut-breed` druckt den passenden Befehl.

**Fortsetzen, Abstürze und Sperre.**

- `--resume` darf nur `--generations`, `--max-calls`, `--max-seconds` und
  `--llm-answers` ändern. Alles andere legt das Experiment fest; der Lauf lehnt es mit
  Exit-Code 2 ab.
- Vor dem ersten Aufruf einer Generation oder der Auswahlphase schreibt `brut-breed`
  einen Checkpoint, der deren schlimmsten Verbrauch an Aufrufen zurücklegt. Endet der
  Prozess hart (Kill, Absturz, Stromausfall), zählt das Fortsetzen diese Rücklage als
  verbraucht, denn die wahre Zahl ging mit dem Prozess verloren. So hält `max_calls`
  auch dann. Im schlimmsten Fall zählt eine Generation doppelt.
- Nur ein Prozess zugleich hält `<lauf>/LOCK`. Ein zweiter endet mit Exit-Code 2.
- Ein abgelehntes Fortsetzen ändert keine Datei. Ereignisse nach dem letzten Checkpoint
  verwirft erst das nächste Schreiben ins Protokoll; eine halb geschriebene letzte Zeile
  ebenso.
- Beim Fortsetzen entfallen `report.json` und `promote/`, bis die Auswahlphase wieder
  fertig ist. Frühere Finalisten bleiben im Protokoll als Geschichte. Ein schon fertiger
  Lauf liefert beim Fortsetzen denselben Bericht, ohne neue Aufrufe und ohne neue
  Ereignisse.
- Eine gestoppte Auswahlphase behält ihre Antworten. Das Fortsetzen bewertet nur die
  fehlenden Rätsel; ein Stopp kostet also nie eine zweite Auswahlrunde.
- Nach dem Not-Aus bleibt `<lauf>/STOP` liegen. Löschen Sie die Datei vor dem
  Fortsetzen; `brut-breed` sagt das beim Anhalten auch.

## 6. Beförderung über einen Pull Request

`brut-breed --out <lauf> --promote` schreibt die zwei besten gezüchteten Finalisten nach
`<lauf>/promote/candidates/cascades/` und dazu die Herkunft. `scripts/propose-pr.sh
<lauf>/promote` öffnet daraus einen Entwurfs-Pull-Request. Die CI lässt nur Dateien unter
`candidates/` zu und prüft jede Kaskade mit `brut-ingest --cascades`. Ein Mensch mergt;
erst dann gehört die Kaskade zum Topf (ADR 0004). `--promote` verlangt einen fertigen
Lauf, dessen `report.json` zu den letzten Finalisten im geprüften Protokoll passt.
`--verify` prüft die Hash-Kette, den Anker des Checkpoints, die Einstellungen und den
Bericht.

## 7. Messung im Container

Wir messen drei Läufe mit den Standardwerten (`brut-breed --out <dir> --seed N
--generations 10`: 8 Kinder je Generation, 40 Rätsel im Such-Panel, 40 im Auswahl-Panel,
höchstens 20 000 Aufrufe), Release-Build, 4 Kerne, simuliertes Modell. Die Messung lief
nach den Korrekturen aus der Gegenprüfung (Richterwahl, Budget, Fortsetzen) und ersetzt
die frühere. Die Laufzeit ist die Wanduhrzeit des Prozesses samt Checkpoints.

| Seed | bestes Kind, Such-Panel | Fitness | bester Finalist, Auswahl-Panel | bestes Startprogramm, Auswahl-Panel | Vorsprung | Worker-Aufrufe | Peer-Urteile | Laufzeit |
|---|---|---|---|---|---|---|---|---|
| 1 | 20/40 | 0.566 | 0.400 | 0.300 | +0.100 | 8 645 | 477 | 0.33 s |
| 2 | 30/40 | 0.745 | 0.825 | 0.300 | +0.525 | 8 496 | 488 | 0.26 s |
| 3 | 25/40 | 0.657 | 0.525 | 0.325 | +0.200 | 6 980 | 492 | 0.24 s |

- **Vorsprung.** In allen drei Seeds liegt der beste gezüchtete Finalist auf dem
  Auswahl-Panel, welches die Suche nie gesehen hat, vor dem besten Startprogramm, um
  0.100 bis 0.525 (4 bis 21 der 40 Rätsel).
- **Nicht rechenmäßig abgeglichen.** Die Startprogramme sind einzelne sim:small-Worker
  mit einem Aufruf je Rätsel. Der beste Finalist von Seed 1 nutzt sim:small und sim:large
  mit 2 Aufrufen je Rätsel, der von Seed 2 drei sim:small-Worker mit 3 Aufrufen. In
  Seed 3 ist der beste Finalist ein einzelner sim:small-Worker mit einem Aufruf je Rätsel
  wie die Startprogramme; er denkt aber länger (300 statt 80 Tokens je Rätsel). Der
  Vorsprung mischt also Zucht mit Modell und Rechenaufwand. Für eine Aussage über die
  Zucht verlangt ADR 0005 gleiche Modelle und gleiches Budget (Q22); das leistet erst der
  vorab festgelegte Hauptwert (Ticket 02).
- **Bauform.** Die beste Kaskade hat 2, 3 und 1 Stufe(n). In Seed 1 entstand sie durch
  eine Stufen-Kreuzung, in Seed 3 durch eine Abschnitts-Kreuzung, in Seed 2 durch das
  Ersetzen eines Abschnitts. Unter den fünf besten je Lauf stammen 3, 2 und 4 aus einer
  Kreuzung.
- **Stillstand.** Das beste Ergebnis steigt nach Generation 5, 6 und 3 nicht mehr. Der
  Simulator lässt mehr zu: Ein Prompt mit allen Strategien, Selbstprüfung und
  Denkbudget 4096 löst nach der Messung beim Bau des Simulators alle 60 Rätsel eines
  anderen Panels. Zehn Generationen sind also kurz.
- **Urteilskraft verschiebt den Rang.** In Seed 1 löst die zweitbeste Kaskade mehr Rätsel
  (21 statt 20), liegt aber wegen ihrer Urteilskraft von 0.71 (4 von 5 gewerteten
  Urteilen richtig) hinter der besten mit 0.86 (5 von 5). In Seed 2 löst die drittbeste
  Kaskade 32 statt 30 Rätsel und ist auf dem Auswahl-Panel der beste Finalist (0.825);
  sie liegt zurück, weil sie nur eines von zwei gewerteten Urteilen richtig hatte (0.50).
  So wenige Urteile messen die Urteilskraft nur grob. Das folgt aus dem Gewicht 0.2
  (Q13); wir beobachten es weiter.
- **Wer urteilt.** Seit die am seltensten genutzten neuen Kaskaden zuerst urteilen, haben
  8, 6 und 3 der 84 Kaskaden kein gewertetes Urteil (vor der Korrektur 13, 15 und 3). Die
  beste Kaskade jedes Laufs ist gemessen (5, 2 und 3 gewertete Urteile, alle richtig).
  44, 68 und 63 Urteile stammen von entfernteren Vorfahren oder Nachkommen (Großeltern,
  Enkel und weiter) eines der Beurteilten; die Regel erlaubt das (Abschnitt 3).
- **Grenzen der Aussage.** Drei Seeds ergeben keinen Signifikanztest. Der Simulator ist
  eine Positivkontrolle; die Zahlen sagen nichts über echte Sprachmodelle.

## 8. Was der Simulator ist und was nicht

Im Container läuft kein Sprachmodell. Die Worker rufen ein **simuliertes Modell** auf.
Es liest Strategie-Wörter im Systemprompt, testet so viele Hypothesen, wie sein
Denkbudget erlaubt, und prüft sie bei eingeschalteter Selbstprüfung an allen Beispielen
statt nur am ersten. Damit ist es eine Positivkontrolle: Es zeigt, dass Zucht, Urteile,
Kosten und Grenzen zusammen bessere Kaskaden finden, wenn es bessere gibt. Über echte
Sprachmodelle sagt es nichts. Echte Läufe brauchen den Ollama-Adapter auf einem
gemieteten GPU-Server (ADR 0003, Ticket 01 und 03).

## 9. Grenzen

- Die Peer-Urteile sind im Container nicht unabhängig von der Wahrheit, da der Simulator
  dieselben Hypothesen zum Urteilen nutzt wie zum Lösen.
- Das Auswahl-Panel ist öffentlich erzeugt; es ist nur beschreibend. Der vorab festgelegte
  Hauptwert läuft später verdeckt auf ARC-AGI-1 in `brut-eval` (Ticket 02).
- Die Preise je Modellaufruf sind Platzhalter, bis wir sie auf dem Server messen.

## 10. Quellen

- Zhang, Hu, Lu, Lange, Clune: Darwin Gödel Machine: Open-Ended Evolution of
  Self-Improving Agents. ICLR 2026. [arXiv:2505.22954](https://arxiv.org/abs/2505.22954)
- Hunter: MM algorithms for generalized Bradley-Terry models. Annals of Statistics 32(1),
  2004, S. 384–406.
- Nitzan, Paroush: Optimal decision rules in uncertain dichotomous choice situations.
  International Economic Review 23(2), 1982, S. 289–297.
- Doya: Metalearning and neuromodulation. Neural Networks 15, 2002, S. 495–506.
