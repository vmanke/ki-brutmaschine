# Ergebnisse der Stufe 1 (Lerner-Modus, Versuchsplan S)

Stand: 3. Oktober 2026. Wir messen auf dem Entwicklungsrechner (4 virtuelle
Kerne, Xeon mit 2.8 GHz, 15 GB Arbeitsspeicher) mit dem Release-Build von
Commit `50a7863` (Branch `stage1-fixes-wip`), wobei spätere Commits nur
Dokumente ändern. Das Messprotokoll trägt den Hash
`c4b67a4310ef980d41bb54feb7dacd6bc611098f3978d8daa719febe0e78463e`
(`learner-v2`).

Dies ist der **zweite Lauf** von Plan S. Den ersten Lauf (Protokoll
`learner-v1`, Hash `d430e049…`) fassen wir weiter unten im Abschnitt „Was sich
gegenüber dem ersten Lauf geändert hat“ zusammen. Seine vollständige
Fassung steht in der Git-Geschichte dieser Datei (Commit `a268ea3`).

Die Regeln stehen im Bauvertrag
[STAGE1_CONTRACT.md](STAGE1_CONTRACT.md) und in den Korrekturen
[STAGE1_CONTRACT_ERRATA.md](STAGE1_CONTRACT_ERRATA.md) (für diesen Lauf
besonders V1 bis V13). Die Absicht steht im Plan
[PLAN_GENERALISIERUNG.md](PLAN_GENERALISIERUNG.md).

## Kurzfassung

- **Der Hauptwert ist nicht bestanden.** Auf den 40 geheimen
  Aufgabenfamilien (HIDDEN-GRAMMAR) bestand der beste Kandidat in **0 von 3**
  Seeds. Sein Vorsprung Δ vor dem besten höchstens gleich teuren
  Vergleichsverfahren war −0.01, −0.01 und −0.06, wobei Bestehen mindestens
  +0.01 verlangt hätte. Das Ergebnis des ersten Laufs (0 von 3) bestätigt sich.
- **Die Stufe 1 ist damit nicht erfolgreich.** Das ist ein negatives
  Ergebnis. Der Plan hielt das für wahrscheinlich (Erfolg unter 25 %).
- **Der volle Suchraum schlägt die Ablation „nur Konstanten“ nicht.** Im
  Mittel über die drei Seeds lag Δ auf HIDDEN-GRAMMAR bei −0.027 (voller
  Suchraum) gegenüber −0.010 („nur Konstanten“).
- **Das vorab festgelegte Abbruchkriterium greift.** Der Vorsprung auf den
  SELECT-Aufgaben war in allen 3 Seeds kleiner als 0, und der volle
  Suchraum war nicht besser als „nur Konstanten“, weshalb wir die
  Programm-Evolution nicht weiter ausbauen. Über Stufe 2a entscheiden Sie.
- **Was die Suche fand:** In Seed 1 und 2 ist der beste Kandidat wieder im
  Kern das Startprogramm „lineares Lernen“ mit anderer Lernrate. In Seed 3
  ist die Suche **zusammengebrochen**, da am Ende Population und
  Ruhmeshalle nur noch Programme enthielten, welche gar nicht lernen, sondern
  Zufallszahlen ausgeben. Einen neuen Lernmechanismus hat die Suche nicht
  gefunden.
- **ARC (Außentest):** ARC-AGI-1: 1, 1 und 0 von 419 Testausgaben gelöst.
  ARC-AGI-2: 0 von 167. ConceptARC: 6, 6 und 0 von 480. Das ist nicht besser
  als ein einfacher Nächster-Nachbar-Vergleich auf denselben Zellmerkmalen.
- **Technik:** Alle Tests laufen durch. Die Läufe brauchten nur noch etwa
  halb so viel Zeit wie im ersten Lauf (11 statt 23 Minuten für die sechs
  Hauptläufe). AC-11 (Platz und Speicher bei 10¹³ FLOPs) ist weiter nicht
  erfüllt, und wir haben es nicht neu gemessen.

## Was sich gegenüber dem ersten Lauf geändert hat

Eine unabhängige Prüfung des ersten Laufs fand Fehler im Code und
ungenaue Aussagen in den Dokumenten. Diese Fehler haben wir behoben (Errata
V1 bis V13). Da sich dadurch Messwerte ändern, trägt das Protokoll eine neue
Fassung (`learner-v2`), und wir haben Plan S mit denselben vorab festgelegten
Regeln (P1 bis P6) neu gerechnet.

**Die wichtigsten Änderungen am Messverfahren:**

- **Das Vergleichsnetz bekommt das richtige Budget (V1).** Das Adam-MLP
  bekam bisher auch die Kosten der Validierung angerechnet, womit es auf den
  verdeckten Stufen etwa doppelt so viel Rechenarbeit wie vereinbart erhielt.
  Jetzt sind es genau die Trainingskosten des Kandidaten (aber weiter
  mindestens eine volle Epoche, siehe unten).
- **Nicht endliche Ausgaben zählen 0 (V2).** Eine NaN-Ausgabe zählte bisher
  als Vorhersage der Klasse 0 und brachte so manchmal Punkte.
- **min und max sind symmetrisch (V3)**, und sehr kleine Zahlen werden als
  0 gespeichert (V4), sodass ein gültiges Programm den Rechner nicht
  mehr durch langsame „subnormale“ Zahlen bremsen kann.
- **Not-Aus und Zeitgrenze greifen auch innerhalb einer Aufgabe (V5).**
- **Die geheimen Familien haben keine „Zwillinge“ mehr unter den
  öffentlichen (V6).** Zwei Formeln können verschieden aussehen und doch
  genau dieselben Aufgaben erzeugen, weshalb wir solche Familien jetzt
  aussortieren.
- **Stufe C ist billiger (V8, V9)**, das Archiv schreibt gebündelt (V7),
  sodass die Suche bei gleichem Budget mehr Kinder bekommt und die Läufe
  schneller sind.
- **Die einmaligen Kosten messen wir (V11)** und nennen sie im Bericht.

**Neuer geheimer Ordner.** Der Auswerter hat für diesen Lauf einen neuen
geheimen Seed bekommen (Korrektur G2). Ein erster Versuch dieses zweiten
Laufs mit dem Ordner `~/.brut/learner-v2` brach nach einem Neustart des
Rechners ab, nachdem ein Lauf (E1, Seed 1) den Auswerter schon befragt
hatte, wobei damals noch nicht alle Korrekturen fertig waren. Diesen Ordner
werten wir als verbraucht; seine Ergebnisse haben wir verworfen und nicht
ausgewertet. Die Zahlen hier stammen aus einem dritten, frischen geheimen
Ordner `/tmp/claude-0/stage1-runs-v2/secret`, welchen wir nach allen
Korrekturen angelegt haben.

**Vergleich der Hauptzahlen:**

| Wert | erster Lauf (`learner-v1`) | dieser Lauf (`learner-v2`) |
|---|---|---|
| Prüfbit HIDDEN-GRAMMAR, Seeds 1 / 2 / 3 | nicht bestanden ×3 | nicht bestanden ×3 |
| Δ HIDDEN-GRAMMAR, bester Kandidat, Seeds 1 / 2 / 3 | −0.01 / −0.02 / −0.01 | −0.01 / −0.01 / −0.06 |
| Mittel Δ HIDDEN-GRAMMAR: voll / nur Konstanten | −0.013 / −0.010 | −0.027 / −0.010 |
| Δ HIDDEN-INSTANCE, Seeds 1 / 2 / 3 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / −0.20 |
| ARC-AGI-1 gelöst (von 419), Seeds 1 / 2 / 3 | 0 / 1 / 0 | 1 / 1 / 0 |
| ARC-AGI-2 gelöst (von 167) | 0 / 0 / 0 | 0 / 0 / 0 |
| ConceptARC gelöst (von 480) | 5 / 6 / 5 | 6 / 6 / 0 |
| Wandzeit der sechs Hauptläufe | 22.6 min | 11.1 min |
| Kinder je E1-Lauf | 1.43 bis 1.56 Mio. | 1.57 bis 3.07 Mio. |

Die Entscheidung bleibt gleich. Der einzige Unterschied ist der
Zusammenbruch der Suche in Seed 3 (Abschnitt „Was die Suche gefunden hat“).

## Was gerechnet wurde

| Lauf | Suchraum | Seed | Such-FLOPs | davon Stufe C | Kinder | neue Aufgabenläufe | Wandzeit |
|---|---|---|---|---|---|---|---|
| E0 Rauchtest | voll | 1 | 1.00 · 10¹¹ | 34 % | 107 008 | 216 080 | 12 s |
| E1, Seed 1 | voll | 1 | 1.50 · 10¹² | 38 % | 1 702 656 | 2 944 675 | 128 s |
| E1, Seed 2 | voll | 2 | 1.50 · 10¹² | 36 % | 1 574 272 | 2 972 547 | 118 s |
| E1, Seed 3 | voll | 3 | 1.50 · 10¹² | 53 % | 3 071 872 | 3 442 348 | 177 s |
| Ablation, Seed 1 | nur Konstanten | 1 | 1.50 · 10¹² | 5 % | 220 544 | 1 991 254 | 58 s |
| Ablation, Seed 2 | nur Konstanten | 2 | 1.50 · 10¹² | 12 % | 919 296 | 4 372 060 | 91 s |
| Ablation, Seed 3 | nur Konstanten | 3 | 1.50 · 10¹² | 12 % | 929 024 | 4 234 199 | 91 s |

- Ein **Kind** ist ein neu erzeugtes Programm. Ein **Aufgabenlauf** ist ein
  Programm, welches eine Aufgabe einmal ganz lernt. Bekannte Ergebnisse kommen
  aus einem Speicher und zählen nicht als neuer Aufgabenlauf.
- **Stufe C** bewertet die Ruhmeshalle regelmäßig auf den SELECT-Aufgaben.
  Sie wirkt nicht auf die Suche zurück, zählt aber zum Such-Budget. Im
  vollen Suchraum nahm sie 36 bis 53 % des Budgets, in der Ablation nur 5
  bis 12 %, weshalb die Ablation 1.4- bis 1.9-mal so viel Rechenarbeit
  für die eigentliche Suche wie der volle Suchraum hatte (Lücke K2 in den
  Korrekturen). Das ist eine Schwäche des Versuchsaufbaus, welche aber das
  negative Ergebnis nicht erklärt, da auch der beste Kandidat des vollen
  Suchraums vor dem Nachstimmen auf SELECT nur um höchstens +0.0008 besser
  als das Vergleichsverfahren war.
- Die Wandzeit enthält die Abstimmung der Vergleichsverfahren, die Endphase
  und die verdeckte Bewertung.
- **Rechenarbeit insgesamt (ohne E0):** Suche 9.00 · 10¹² FLOPs. Dazu kommen,
  jetzt gemessen (V11): die Abstimmung der Vergleichsverfahren mit
  5.6 · 10¹⁰ FLOPs je Lauf (zusammen 3.4 · 10¹¹; früher auf 3 · 10¹⁰
  geschätzt) und die sechs Endphasen mit zusammen 1.5 · 10¹¹ FLOPs. Die
  Summe liegt bei etwa 9.5 · 10¹² FLOPs, wie im Plan geschätzt. Nicht
  gezählt sind die Rechnungen im Auswerter `brut-eval` und die ARC-Prüfungen.
- **Zeit:** Die sechs Hauptläufe brauchten zusammen 11.1 Minuten, die neun
  ARC-Prüfungen zusammen 1.7 Minuten. Die Suche lief mit 9 bis 13 (voll)
  und 17 bis 26 (nur Konstanten) Milliarden gezählten FLOPs pro Sekunde auf
  4 Kernen.
- **Platz und Speicher:** Ein Lauf schrieb 68 bis 320 MB und brauchte
  höchstens 635 MB Arbeitsspeicher (E1, Seed 3, welcher wegen der billigen
  Programme doppelt so viele Kinder hatte).
- E0 nutzte einen eigenen, getrennten geheimen Ordner. Seine verdeckten
  Ergebnisse prüfen nur die Technik und gehören nicht zur Auswertung.

## Vorab festgelegte Regeln

Der Plan sagt: „In mindestens 2 von 3 Seeds besteht der beste Kandidat.“
Welcher Finalist der beste ist, sagt er nicht. Darum haben wir diese Regeln
vor der ersten Anfrage an den echten geheimen Auswerter des ersten Laufs
festgelegt und in die Korrekturen eingetragen (P1 bis P6, Commit
`164de79`). Für diesen Lauf gelten sie unverändert:

- **Bester Kandidat** eines Laufs ist der Finalist mit dem höchsten Δ auf
  den öffentlichen SELECT-Aufgaben (letzte Runde, nach dem Nachstimmen).
  Nur öffentliche Zahlen entscheiden das. Bei Gleichstand gilt der erste in
  der Reihenfolge des Berichts.
- **Hauptwert** ist das Prüfbit „bestanden/nicht bestanden“ dieses
  Kandidaten auf HIDDEN-GRAMMAR. Bestanden heißt: p < 0.05 und Δ ≥ 0.01 und
  der Vorsprung fällt bei Breite F = 32 um höchstens 0.05 gegenüber F = 8
  und 16.
- **Zusatzbedingung:** Das mittlere Δ auf HIDDEN-GRAMMAR der besten
  Kandidaten aus E1 muss echt größer sein als das der Ablation.
- **ARC:** Den besten Kandidaten jedes E1-Seeds prüfen wir je einmal auf
  ARC-AGI-1, ARC-AGI-2 und ConceptARC.

## Hauptwert: HIDDEN-GRAMMAR

| Seed | bester Kandidat | Δ auf SELECT (öffentlich) | Δ auf HIDDEN-GRAMMAR | Prüfbit |
|---|---|---|---|---|
| 1 | `8f16170d9885ff5f` | −0.0008 | −0.01 | nicht bestanden |
| 2 | `a373471aed796dcd` | −0.0029 | −0.01 | nicht bestanden |
| 3 | `12115ecbe6c05b7e` | −0.0677 | −0.06 | nicht bestanden |

So lesen Sie die Tabelle:

- **Δ** ist der Vorsprung vor dem besten Vergleichsverfahren, welches je
  Lernschritt höchstens so viel rechnet wie der Kandidat. Positiv heißt
  besser, negativ heißt schlechter.
- Zu den Vergleichsverfahren gehört immer ein klassisches Netz (Adam-MLP).
  Sein Budget sind die Trainingskosten des Kandidaten, aber es trainiert
  mindestens eine volle Epoche über alle bisher gesehenen Beispiele. Bei
  billigen Programmen sind das nur ein bis wenige Lernschritte, sodass es
  dann oft **mehr** rechnet als der Kandidat. Gegen lineare Programme ist es
  darum ein schwacher Gegner. In Seed 1 und 2 war das beste
  Vergleichsverfahren auf SELECT und auf den öffentlichen Referenzaufgaben
  das abgestimmte lineare Startprogramm. In Seed 3 war es das MLP, da der
  Kandidat billiger als jedes Startprogramm ist und diese darum nicht zum
  Vergleich zählen. Für die verdeckten Stufen gibt der Auswerter das beste
  Vergleichsverfahren nicht heraus.
- Der Auswerter gibt Δ nur **abgerundet** auf 0.01 heraus. −0.01 heißt also:
  Der wahre Wert liegt zwischen −0.01 und knapp unter 0. In allen drei Seeds
  ist der wahre Vorsprung kleiner als 0.
- Den **p-Wert** gibt der Auswerter absichtlich nicht heraus, sodass die
  Suche nichts über die geheimen Familien lernen kann. Hier entscheidet er
  nichts, da schon die Bedingung Δ ≥ 0.01 in allen drei Seeds verletzt ist.
- Die anderen zwei Finalisten jedes Laufs lagen gleich (Seed 1 und 2: −0.01;
  Seed 3: −0.06). Auch sie bestanden nicht und sind nur beschreibend.

## Ablation „nur Konstanten“

| Seed | voller Suchraum: Δ auf HIDDEN-GRAMMAR | nur Konstanten: Δ auf HIDDEN-GRAMMAR |
|---|---|---|
| 1 | −0.01 | −0.01 |
| 2 | −0.01 | −0.01 |
| 3 | −0.06 | −0.01 |
| Mittel | −0.027 | −0.010 |

Der volle Suchraum ist nicht besser als die bloße Feinabstimmung der
Konstanten der Startprogramme, womit die Zusatzbedingung nicht erfüllt ist.
In Seed 1 und 2 sind beide Spalten gleich, sodass der Unterschied im Mittel
allein vom Zusammenbruch der Suche in Seed 3 kommt. Dabei begünstigt die
ungleiche Rechenarbeit (Stufe C, siehe oben) die Ablation.

## Beschreibende Werte (kein Test)

Für die drei besten Kandidaten aus E1:

| Seed | Länge | Kosten je Schritt (F = 16) | Δ_public | Δ HIDDEN-INSTANCE | Quarantäne | Δ SELECT bei F = 32 | Δ SELECT bei F = 8/16 |
|---|---|---|---|---|---|---|---|
| 1 | 7 Befehle | 1 056 FLOPs | +0.0034 | 0.00 | nein | −0.0061 | −0.0003 |
| 2 | 7 Befehle | 1 057 FLOPs | −0.0011 | 0.00 | nein | −0.0043 | −0.0045 |
| 3 | 4 Befehle | 320 FLOPs | −0.1660 | −0.20 | nein | −0.0707 | −0.0842 |

- **HIDDEN-INSTANCE** (neue Aufgaben aus bekannten Familien): 0.00 in Seed 1
  und 2, also ein wahrer Vorsprung zwischen 0 und knapp unter 0.01; −0.20 in
  Seed 3. Der Plan hielt +0.01 bis +0.05 in 1 bis 2 Seeds für denkbar, was
  nicht eintrat.
- **Δ_public** ist derselbe Vergleich auf öffentlichen Aufgaben gleicher
  Bauart. Die Lücke zu HIDDEN-INSTANCE ist überall kleiner als die
  Quarantäne-Grenze von 0.05.
- **Transferverhältnis** Δ_GH / Δ_HI: nicht berechenbar, da Δ_HI nicht
  über 0.01 liegt.
- **Breitenkopplung:** In Seed 1 fällt der Vorsprung bei F = 32 stärker als
  bei F = 8 und 16 (−0.006 gegenüber −0.0003), wobei die Suche nur F = 8
  und 16 sieht.
- **Lerngeschwindigkeit** (Qualität q auf SELECT nach 1/16, 1/4 und dem
  ganzen Strom): Seed 1: 0.19 / 0.28 / 0.31; Seed 2: 0.16 / 0.23 / 0.25;
  Seed 3: 0.01 / 0.01 / 0.00. q nach Zahl der Trainingsbeispiele (16, 64,
  256): Seed 1: 0.22 / 0.30 / 0.40; Seed 2: 0.18 / 0.26 / 0.30; Seed 3:
  0.00 / 0.01 / 0.00.
- **„Fluch des Gewinners“** (Panel-Mittel minus Mittel auf öffentlichen
  Referenzaufgaben): −0.083, −0.093 und +0.018. Die Zahl ist hier nicht als
  Fluch des Gewinners deutbar, da Panel und Referenzaufgaben Familien,
  Breiten und Datenmengen verschieden mischen.
- **HIDDEN-NEAR und HIDDEN-FAR** haben wir nicht gemessen, da sie nicht in
  der Standardeinstellung der Läufe liegen.

## Was die Suche gefunden hat

Die besten Programme aus E1, nach dem Vereinfachen:

```
Seed 1 (8f16170d9885ff5f)            Seed 2 (a373471aed796dcd)        Seed 3 (12115ecbe6c05b7e)
setup:                               setup:                           setup:
  VUniform v2 <- uniform(±0,0143)      MUniform m0 <- uniform(±0,0683)    SUniform s0 <- uniform(±0,0144)
  VNorm s3 <- norm(v2)               predict:                           VGaussian v7 <- gaussian(0, 1,27²)
predict:                               MatVec v1 <- m0 . v0             predict:
  MatVec v1 <- m0 . v0                 SConst s3 <- 0,0386                VUniform v1 <- uniform(±0,363)
learn:                               learn:                           learn:
  VSub v3 <- v2 - v1                   VSub v3 <- v2 - v1                 VUniform v1 <- uniform(±0,0007)
  VScale v3 <- s3 * v3                 VScale v3 <- s3 * v3
  MOuter m1 <- v3 (x) v0               MOuter m1 <- v3 (x) v0
  MAdd m0 <- m0 + m1                   MAdd m0 <- m0 + m1
```

- **Seed 1 und 2** sind das Startprogramm „lineares Lernen“: Vorhersage =
  Gewichtsmatrix mal Eingabe; Lernschritt = Fehler mal Lernrate mal Eingabe.
  Seed 2 startet mit kleinen Zufallsgewichten. In Seed 1 ist die Lernrate
  die Länge eines kleinen Zufallsvektors (etwa 0.03 bei F = 16).
- **Seed 3** lernt nichts. Seine Vorhersage ist eine Zufallszahl, welche mit
  den Daten nichts zu tun hat. Die Ruhmeshalle wird bei jedem Wechsel des
  Aufgaben-Panels neu bewertet und behält nur die 32 aktuell besten
  Programme, ist also kein dauerhaftes Gedächtnis. In Seed 3 verdrängten
  bis etwa Runde 30 000 von 48 000 zunächst seltsame, schlecht lernende
  Varianten und dann sehr billige Zufallsprogramme alle linearen Lerner aus
  Population und Ruhmeshalle. Das Startprogramm „lineares Lernen“ selbst
  erreicht auf denselben späten Panels weiter 0.19 bis 0.27 (geprüft mit
  einem eigenen Messprogramm), weshalb es nicht an den Aufgaben liegt. Im
  ersten Lauf trat das nicht auf: Bis etwa Runde 2 970 verliefen beide Läufe von
  Seed 3 gleich, danach führten die geänderten Rechenregeln (V2 bis V4) zu
  einem anderen Weg, und die billigere Stufe C ließ die Suche doppelt so
  viele Runden machen. Dass die Suche gute Programme verlieren kann, ist
  eine Schwäche des Suchverfahrens, kein Messfehler.
- **Nachstimmen der Konstanten.** Die Konstanten der Finalisten stammen
  nicht direkt aus der Suche, sondern werden nach der Suche in der Endphase
  nachgestimmt (200 Versuche auf 36 SELECT-artigen Aufgaben, Bauvertrag
  6.7.2). Die Vergleichsverfahren werden genauso abgestimmt, weshalb der
  Vergleich auf den verdeckten Stufen in Seed 1 und 2 vor allem zwei
  gleich aufwendige Abstimmungen derselben linearen Bauart gegeneinander
  misst. Das Nachstimmen kann Δ_select in beide Richtungen verschieben:

| Lauf, bester Kandidat | Konstante vor → nach dem Nachstimmen | Δ_select vor → nach |
|---|---|---|
| E1, Seed 1 | 0.0109 → 0.0143 (Breite des Zufallsvektors) | +0.0003 → −0.0008 |
| E1, Seed 2 | Lernrate 0.0247 → 0.0386 | +0.0008 → −0.0029 |
| E1, Seed 3 | (Zufallsprogramm) | −0.0678 → −0.0677 |
| nur Konstanten, Seed 1 | 2-Schicht-Netz, Lernrate 0.0825 → 0.0847 | −0.0313 → −0.0074 |
| nur Konstanten, Seed 2 | Lernrate 0.0354 → 0.0389 | −0.0095 → −0.0123 |
| nur Konstanten, Seed 3 | Lernrate 0.0331 → 0.0241 | +0.0033 → +0.0042 |

  „Vor“ ist das Programm in der Ruhmeshalle, „nach“ der Finalist; beide auf
  denselben SELECT-Aufgaben der letzten Runde. Das abgestimmte lineare
  Vergleichsverfahren nutzt die Lernrate 0.0168. In Seed 1 und 2 des vollen
  Suchraums lag der Kandidat **vor** dem Nachstimmen knapp vorn und
  **danach** knapp hinten, wobei beide Unterschiede sehr klein sind.
- In der Ablation stammen die besten Programme von Seed 2 und 3 vom
  linearen Lernen ab, das von Seed 1 vom kleinen Netz mit zwei Schichten.
- Die Vielfalt der Population war im vollen Suchraum gering, da am Ende
  18 bis 82 der 256 Mitglieder ein verschiedenes Verhalten (verschiedener
  Fingerabdruck) hatten. In der Ablation waren es 246 bis 251.

Das deckt sich mit dem ersten Lauf und dem Vorversuch: Die Suche findet
Varianten der Startprogramme, aber nichts, was auf neuen Familien besser
lernt.

## ARC-Außentest

Je ein Versuch mit zwei Antworten pro Testausgabe („pass@2“). Die Programme
sehen ein Raster nur über 32 handgemachte Merkmale je Zelle („phi32“).

| Satz | Seed 1 | Seed 2 | Seed 3 | Nächster Nachbar auf phi32 | Obergrenze mit phi32 |
|---|---|---|---|---|---|
| ARC-AGI-1 Bewertung (400 Aufgaben, 419 Testausgaben) | 1 gelöst (0.25 %) | 1 gelöst (0.25 %) | 0 gelöst | 2 gelöst | 9 Aufgaben |
| ARC-AGI-2 Bewertung (120 Aufgaben, 167 Testausgaben) | 0 | 0 | 0 | 0 | 0 Aufgaben |
| ConceptARC (160 Aufgaben, 480 Testausgaben) | 6 (1.25 %) | 6 (1.25 %) | 0 | 6 | 17 Aufgaben |

Ausgeglichene Zellgenauigkeit. Hier erreicht schon das bloße **Kopieren
der Eingabe** genau 0.5; gleichmäßiges Raten einer Farbe erreicht etwa 0.1:

| Satz | Seed 1 | Seed 2 | Seed 3 | Nächster Nachbar |
|---|---|---|---|---|
| ARC-AGI-1 | 0.57 | 0.57 | 0.09 | 0.578 |
| ARC-AGI-2 | 0.46 | 0.46 | 0.09 | 0.428 |
| ConceptARC | 0.58 | 0.59 | 0.10 | 0.573 |

- Die Prozentzahlen sind der offizielle Wert (Anteil gelöster Testausgaben
  je Aufgabe, gemittelt über alle Aufgaben). Der Kaggle-Wert ist gleich oder
  fast gleich (ARC-AGI-1: 0.24 %).
- Der Auswerter gibt die Zellgenauigkeit nur auf 0.01 abgerundet heraus.
- Seed 1 und 2 liegen auf ARC-AGI-1 und ConceptARC nur knapp über der
  Kopie der Eingabe (0.57 bis 0.59 gegenüber 0.5), auf ARC-AGI-2 sogar
  darunter. Seed 3 rät nur (etwa 0.1), wie bei einem Zufallsprogramm zu
  erwarten.
- ARC-AGI-1 und ARC-AGI-2 liegen im Rahmen der Vorhersage des Plans (0 bis 2
  von 400; 0 von 120). Für ConceptARC sagte der Plan 0 bis 2 ganze Aufgaben
  voraus. Der Auswerter zählt aber gelöste Testausgaben (bis 6 von 480),
  nicht ganze Aufgaben. Ein Fortschritt gegenüber dem einfachen
  Nächster-Nachbar-Vergleich ist es nicht.
- Zur Einordnung: Menschen lösen im Mittel 64.2 % des öffentlichen
  ARC-AGI-1-Bewertungssatzes mit drei Versuchen, etwa 60 % mit zwei
  Versuchen (H-ARC, Scientific Data 2025, begutachtet). Die Zahl mit zwei
  Versuchen ist diejenige, welche zu pass@2 passt. Die Werte großer Systeme
  stehen im Plan, Abschnitt 9.
- Alle neun Prüfungen nutzten nur synthetische Trainingsdaten. Die
  Kontaminationsprüfung lief vor jeder Prüfung. Eine Prüfung dauerte samt
  Programmstart 7 bis 20 Sekunden.

## Rechenleistung des Interpreters

Release-Build, 4 virtuelle Kerne, gemessen am 3. Oktober 2026 mit dem Code
dieses Laufs. Eine **ganze Aufgabe** sind 384 Lernschritte und 3 × 64
Vorhersagen.

| Messung | Wert | Ziel (AC-2) |
|---|---|---|
| 2-Schicht-Netz, F = 16, ganze Aufgaben, 1 Kern | 7 980 bis 8 900 Aufgaben/s; 9.7 bis 11.7 Milliarden gezählte FLOPs/s | ≥ 2.5 Milliarden FLOPs/s |
| dasselbe, 4 Kerne | 24 900 bis 34 600 Aufgaben/s | ≥ 3 000 Aufgaben/s |
| 2-Schicht-Netz, F = 8 / F = 32, 1 Kern | 16 800 bis 19 300 / 2 630 bis 3 180 Aufgaben/s | – |
| 128 mutierte Programme, F = 16, 1 Kern | 6 090 Aufgaben/s; 3.5 Milliarden FLOPs/s | – |
| ganze Suche (E1), 4 Kerne | 14 000 bis 19 000 Kinder/s; 9 bis 13 Milliarden FLOPs/s | – |

Die Spannen stammen aus zwei Messprogrammen (Einheitstest
`bench_interpreter` und Integrationstest `bench_learner_interpreter`). Das
ist etwa 1.7-mal so schnell wie im ersten Lauf, vor allem da wir die
Reihenfolge der Trainingsbeispiele nur noch einmal je Aufgabe berechnen (V9).

**Langsame Sonderfälle** (Integrationstest `bench_learner_adversarial`,
F = 32): Programme, welche winzige oder subnormale Zahlen erzeugen, laufen
jetzt so schnell wie normale (Faktor 1.0; im ersten Lauf bis etwa 100-mal
langsamer je gezähltem FLOP). Weiter zu billig angesetzt sind die Befehle
mit Normalverteilung (etwa 10-mal langsamer je gezähltem FLOP als eine
Matrix-Vektor-Multiplikation), Sinus und Kosinus großer Zahlen und
Exponential- und Logarithmusfunktion (Lücke K1).

## Abnahmekriterien

| Nr. | Kriterium | Ergebnis |
|---|---|---|
| AC-1 | Tests, Clippy, `cargo audit`, Abhängigkeiten | erfüllt: 396 Tests bestanden, 0 fehlgeschlagen, 12 bewusst ausgelassen (Messprogramme und Tests mit echten Daten; alle 12 einzeln in Release bestanden); Clippy ohne Warnung; `cargo audit` 0 Schwachstellen (Datenbank f8dee89e); Abhängigkeiten unverändert |
| AC-2 | Interpreter-Leistung | erfüllt (Tabelle oben) |
| AC-3 | Rauchtest E0 ≤ 5 min, ≥ 10 000 Kinder, report.json | erfüllt: 12.5 s, 107 008 Kinder |
| AC-4 | gleiche Ergebnisse bei 1 und 4 Threads; Fortsetzen nach STOP | erfüllt (T-E2E-2, T-E2E-3) |
| AC-5 | kein Abschreiben der Lösung; Validierung ohne Nebenwirkung | erfüllt (T-PROT-1, T-PROT-2) |
| AC-6 | zu teure Programme laufen nie | erfüllt (Test mit Zähler: 0 Aufrufe) |
| AC-7 | verdeckte Phase ≤ 5 min mit erstmaliger Abstimmung; ein Δ je Antwort; Budget über Läufe | erfüllt: ganzer Lauf E0 12.5 s; 36 von 48 Anfragen über 6 Laufordner verbraucht, je Lauf 6; Abfragebuch bei jeder Anfrage geprüft |
| AC-8 | report.json mit allen Werten je Finalist | erfüllt; der Bericht nennt jetzt auch die einmaligen Kosten, den Anteil von Stufe C und das beste öffentliche Vergleichsverfahren |
| AC-9 | ARC-AGI-1 mit 400 Aufgaben ≤ 10 min; Obergrenzen 9 / 0 / 17 | erfüllt: 7 bis 20 s je Prüfung samt Programmstart |
| AC-10 | Kontamination wird vor dem Rechnen abgelehnt | erfüllt (T-E2E-9 und Einheitstests) |
| AC-11 | 10¹³-FLOP-Lauf ≤ 500 MB Platte, ≤ 2 GB Arbeitsspeicher | **nicht erfüllt** (Messung aus dem ersten Lauf, Abschnitt unten; nicht neu gemessen) |
| AC-12 | STOP beendet den Lauf in ≤ 2 s | erfüllt: 0.11 s im ersten Lauf; seit V5 wird STOP auch innerhalb einer Aufgabe geprüft (Test T-PROT-3) |
| AC-13 | MLP-Modus unverändert | erfüllt (alle alten Tests unverändert bestanden) |
| AC-14 | Fehlalarmrate des Haupttests 0.035 bis 0.065 | erfüllt: 0.0535 (Test T-STAT-1) |

### AC-11 im Einzelnen

Diese Messung stammt aus dem ersten Lauf (`learner-v1`). Für AC-11 lief
damals ein eigener Lauf mit 10¹³ Such-FLOPs (Seed 11, voller Suchraum, ohne
verdeckte Bewertung, 264 200 Runden, 54 Minuten). Er schrieb 1 593 MB statt
höchstens 500 MB:

| Teil | Größe | Inhalt |
|---|---|---|
| Abstammungsblöcke (`lineage/`) | 1 184 MB | 7.30 Mio. Einträge zu je etwa 170 Byte, einer je Kind, das die Hürde nahm |
| Archiv (`archive.jsonl`) | 311 MB | davon 310 MB für 331 021 Einträge neuer Mitglieder der Ruhmeshalle, je mit vollem Programm |
| Schnappschüsse | 99 MB | 265 Bevölkerungsstände und der letzte Ergebnis-Speicher (48 MB) |

Der Arbeitsspeicher erreichte 2 374 MB statt höchstens 2 GB. Das Archiv
liegt ganz im Speicher, da allein das Öffnen für `brut verify` 592 MB braucht.
Beim Schreiben des Berichts entstehen zusätzlich Kopien aller Einträge. Wo
genau die Spitze entsteht, haben wir nicht vermessen.

Der Plan hatte 6.5 Millionen Kinder je 10¹³ FLOPs und 1.6 Millionen
Abstammungseinträge zu je etwa 110 Byte geschätzt (etwa 180 MB). Gemessen
haben wir 16.9 Millionen Kinder (etwa 5.9 · 10⁵ FLOPs je Kind) und 7.30
Millionen Einträge zu je etwa 170 Byte, was zusammen die 1 184 MB erklärt.
Der häufige Wechsel in der Ruhmeshalle erklärt die Größe des Archivs, nicht
die der Abstammung. Seit `learner-v2` sind Kinder noch billiger (Stufe C
kostet weniger), weshalb ein 10¹³-Lauf eher mehr schreiben würde. Abhilfe
braucht eine Änderung des Bauvertrags: zum Beispiel Ruhmeshallen-Einträge
nur bei Stufe C oder bei Schnappschüssen schreiben, Abstammungsblöcke packen
und das Archiv beim Öffnen nicht ganz im Speicher halten.

## Bekannte Lücken und Grenzen

- **Der p-Wert und die Werte je Familie bleiben im Auswerter.** Das ist
  Absicht, weshalb dieser Bericht nur das Prüfbit und das abgerundete Δ
  nennen kann.
- **Nur zwei verdeckte Stufen.** HIDDEN-NEAR und HIDDEN-FAR haben wir nicht
  gemessen.
- **Die Suche kann gute Programme verlieren** (Seed 3, siehe oben). Eine
  dauerhafte Bestenliste über alle Panels gibt es nicht.
- **Ungleiche Rechenarbeit in der Ablation** durch Stufe C (Lücke K2).
  Stufe C aus dem Budget zu nehmen, würde den vorab festgelegten Aufbau
  ändern; das entscheiden Sie.
- **Das Vergleichsnetz (MLP) ist bei billigen Programmen schwach**, da es
  dann nur wenige volle Lernschritte macht (siehe Hauptwert).
- **Die geheimen Familien sind neue Familien derselben Grammatik.** Exakte
  „Zwillinge“ öffentlicher Familien sortieren wir jetzt aus (V6). Die
  Prüfung erkennt aber nur zwei bekannte Umformungen, sodass weitere
  gleichwertige Formeln möglich bleiben.
- **Plan L haben wir nicht gerechnet** (Zufallssuche, festes Panel, ohne
  Breitenterm, Start ab null). Ob mehr Rechenzeit etwas ändert, ist offen.
  Das Abbruchkriterium rät aber davon ab, das ohne neuen Ansatz zu tun.
- **Budget der geheimen Anfragen:** Im neuen geheimen Ordner sind 36 von 48
  Anfragen verbraucht, die ARC-Budgets je Satz 3 von 3. Der Ordner des
  abgebrochenen Versuchs (`~/.brut/learner-v2`) und der des ersten Laufs
  (`~/.brut/learner`) gelten als verbraucht.
- **Der Protokoll-Hash deckt den Code der 24 handgeschriebenen Familien
  nicht ab** (Korrektur G1).
- **Startprogramm „Perzeptron“:** Aus dem Nullzustand lernt es bei zwei
  Klassen nie, da sein Fehler genau an der Schwelle liegt. Die Abstimmung
  behebt das für das Vergleichsverfahren (Korrektur L8).
- **Lange Läufe brauchen zu viel Platz und Speicher** (AC-11, siehe oben).
- **Gezählte FLOPs begrenzen die Rechenzeit nicht genau** (Lücke K1). Die
  Zeit bleibt durch Zeitgrenzen und STOP begrenzt.
- **Rechner:** Alle Zahlen stammen von einem Rechner. Sinus, Exponential-
  und Logarithmusfunktion kommen aus der Mathematik-Bibliothek des Systems.
  Auf einem anderen System können einzelne Werte in den letzten Stellen
  abweichen.

## So prüfen Sie die Zahlen nach

Die Laufordner liegen unter `/tmp/claude-0/stage1-runs-v2/` (nicht im
Repository; die des ersten Laufs unter `/tmp/claude-0/stage1-runs/`):

```sh
./target/release/brut verify /tmp/claude-0/stage1-runs-v2/e1-s1
./target/release/brut learner report /tmp/claude-0/stage1-runs-v2/e1-s1
./target/release/brut learner show /tmp/claude-0/stage1-runs-v2/e1-s1 8f16170d9885ff5f
```

`brut verify` prüft die Hash-Kette des Archivs und die Hashes aller
Abstammungsblöcke und Schnappschüsse. Alle sieben Läufe bestehen diese
Prüfung. Seit V10 enthält `report.json` auch die ARC-Prüfungen; ein
erneutes `brut learner report` ergibt denselben Bericht. Der geheime Ordner
ist `/tmp/claude-0/stage1-runs-v2/secret`. Nur `brut-eval` liest ihn.
