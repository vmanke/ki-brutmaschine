# Brutmaschine: Entscheidungen zu Stufe 2a

Stand: 21. Oktober 2026. Ergebnis des Entscheidungs-Interviews zu Stufe 2a.

## Zusammenfassung

In Stufe 2a prüfen wir eine einzige Frage: Bringt Wissen aus einem Sprachmodell mehr als dieselbe Rechenzeit für Zufallssuche? Dazu schlägt Claude (Opus 5.5) neue Lernprogramme vor, deren Code wir nie ausführen; der bestehende Interpreter bleibt die Sandbox.

18 von 23 Punkten haben wir entschieden. Fünf Punkte (Q19 bis Q23) sind noch offen, wobei zu jedem eine Empfehlung vorliegt. Vor dem Start müssen wir zwei Lücken aus Stufe 1 schließen und den API-Schlüssel in der Cloud-Umgebung hinterlegen.

## Ausgangslage

- Stufe 1 ist fertig, und wir haben sie zweimal unabhängig gemessen. Das vorab festgelegte Hauptergebnis bestand in 0 von 3 Läufen, in beiden Messreihen.
- Die Suche fand nur bekanntes lineares Lernen. ARC-AGI-1: höchstens 1 von 419 Testausgaben gelöst, ARC-AGI-2: 0 von 167.
- Der Messstab funktioniert. In zwei Läufen schlug kein Kandidat den Vergleich; die Fälschungssicherheit ist damit aber nicht belegt, da die Suche nie versuchte, ihn zu täuschen (Probe-Angriff M7 steht aus). Ein Gutachter wies zudem darauf hin, dass der gleich teure Vergleich in der Hypothesenklasse der Suche liegt, weshalb ein Δ nahe null fast erzwungen ist; das 0-von-3-Ergebnis ist deshalb mehrdeutig. Vor Stufe 2 prüfen wir mit einer Positiv-Kontrolle, ob der Messstab Spielraum hat.
- Offene Lücken aus Stufe 1: K1 (einige Befehle werden zu billig gezählt) und K3 (die Ruhmeshalle verliert gute Programme).

## Entschiedene Punkte

| Nr. | Thema | Entscheidung |
|---|---|---|
| Q1 | Umfang | Nur Stufe 2a. Stufe 2b und 2c werden erst mit Messwerten aus 2a geplant. |
| Q2 | Rolle von Claude | Ein zusätzlicher Mutationsweg mit fester Zahl von Aufrufen pro Lauf (etwa 200). Die Zufallsmutation füllt den Rest. |
| Q3 | Hauptergebnis und Geheimschlüssel | Die Regel P1 bis P6 aus Stufe 1 gilt unverändert. Für Stufe 2 wird ein neuer Geheimschlüssel erzeugt. |
| Q4 | Modell | Claude Opus 5.5. |
| Q5 | Kostendeckel | 60 $ pro Monat in der Konsole, 5 $ für den Pilotlauf, 25 $ pro Lauf. |
| Q6 | Reproduzierbarkeit | Jede Anfrage und Antwort wird gespeichert. Neuläufe lesen die Antworten aus dem Speicher statt neu zu fragen. |
| Q7 | Was Claude sieht | Beschreibung der Befehlssprache, 1 bis 2 Eltern-Programme, deren öffentliche Werte pro Familie (gerundet) und die Namen der öffentlichen Familien. Nie: geheime Werte, Startwerte, Generator-Code. |
| Q8 | Ausweichregel bei Ablehnungen | Aus. Ablehnungen werden gezählt und protokolliert. |
| Q9 | Netzwerkzugang | Nur ein eigener Vermittler-Prozess (Broker) spricht mit der API. Der Kern bleibt ohne Netz. |
| Q10 | Freigaben | Persönliche Freigabe vor dem ersten bezahlten Aufruf, nach dem Pilotlauf und vor jeder Runde geheimer Abfragen. |
| Q11 | Sprache des Fachwörterbuchs | Englische Begriffe wie im Code, je mit kurzer Erklärung. Berichte bleiben deutsch. |
| Q12 | Lücken K1 und K3 | Beide werden vor Stufe 2a behoben. |
| Q13 | Vergleichsgruppe | Drei neue Kontrollläufe ohne Claude, mit derselben Protokoll-Version, demselben Schlüssel und denselben Startwerten. |
| Q14 | Wann Claude gefragt wird | Gleichmäßig verteilte Claude-Runden mit je 4 Vorschlägen aus den aktuell besten Programmen. |
| Q15 | Ungültige Vorschläge | Ein neuer Versuch mit der Fehlermeldung des Prüfers. Er zählt gegen das Budget. Der Anteil ungültiger Vorschläge wird berichtet. |
| Q16 | Batch-API | Nein. Die Wartezeit von bis zu 24 Stunden würde die Suche anhalten. |
| Q17 | Pilotlauf | Ein kurzer Lauf mit einem Startwert, ohne geheime Abfragen. Er misst Kosten pro Vorschlag, Anteil gültiger Vorschläge und ob Vorschläge in die Ruhmeshalle kommen. |
| Q18 | Probe-Angriff | Für 2a ein schmaler Test: Kaputte, zu teure und riesige Programme werden abgewiesen, und keine Anfrage an Claude enthält geheime Werte. Der volle Probe-Angriff folgt vor Stufe 2b. |

## Offene Punkte mit Empfehlung

| Nr. | Thema | Empfehlung |
|---|---|---|
| Q19 | Was fair heißt | Vergleich mit dem Kontrolllauf bei gleicher Suchrechenzeit. Die API-Kosten werden getrennt berichtet. |
| Q20 | Zweiter Vergleich (mit und ohne Claude) | Nur beschreibend, mit Zahlen pro Startwert. Drei Startwerte pro Gruppe reichen für keinen belastbaren Test. |
| Q21 | Abbruchkriterium für 2a | Kein Schritt zu 2b, wenn das Hauptergebnis in 0 von 3 Läufen besteht und die Läufe mit Claude in höchstens 1 von 3 Startwerten besser sind als ihr Kontrolllauf. |
| Q22 | Wie freigegeben wird | Budget: Die Ausgabengrenze in der Konsole ist die harte Freigabe, dazu eine Bestätigung im Chat. Geheime Abfragen: nur nach ausdrücklicher Nachricht im Chat. Beides wird mit Wortlaut im Archiv vermerkt. Ein Agent gibt nie selbst frei. |
| Q23 | Denkaufwand von Opus 5.5 | Start mit „medium“. Der Pilot misst die Kosten. Für die vollen Läufe wird der Aufwand vor dem Start festgelegt und dann nicht mehr geändert. |

## Aufgaben vor dem Start

1. In der Konsole einen eigenen Arbeitsbereich „Brutmaschine“ anlegen (platform.claude.com/settings/workspaces, Create workspace). Das darf nur ein Organisations-Admin.
2. Für diesen Arbeitsbereich die Monatsgrenze von 60 $ setzen (Arbeitsbereich auswählen, dann Spend limits).
3. Einen persönlichen Schlüssel für diesen Arbeitsbereich erzeugen (platform.claude.com/settings/keys, Create key, Linked account: Sie selbst, Ablauf 30 Tage).
4. Den Schlüssel in der Cloud-Umgebung der Sitzung eintragen, nicht im Chat: unter Network secrets, sonst als Umgebungsvariable ANTHROPIC_API_KEY. Erst eine neue Sitzung übernimmt ihn.
5. Lücken K1 und K3 beheben und neu messen.
6. Den Vermittler-Prozess (Broker) bauen und den schmalen Probe-Test bestehen.
7. Die offenen Punkte Q19 bis Q23 entscheiden.

## Festgehaltene Unterlagen im Projekt

- CONTEXT.md: das Fachwörterbuch des Projekts mit 29 Begriffen.
- docs/adr/0001-llm-access-only-through-a-separate-broker.md: Nur der Vermittler-Prozess spricht mit der API.
- docs/adr/0002-llm-proposals-are-recorded-and-replayed.md: Claudes Antworten werden gespeichert und bei Neuläufen wiederverwendet.
- docs/ENTSCHEIDUNGEN_STUFE2A.md: dieselben Entscheidungen als Datei im Projekt.
- docs/ERGEBNISSE_STUFE1.md und docs/PLAN_GENERALISIERUNG.md: Ergebnisse von Stufe 1 und Gesamtplan.

Das Projekt liegt bisher nur im Cloud-Container dieser Sitzung und in keinem entfernten Git-Repository.

## Nachtrag (21. Oktober 2026): Stufe 2 Infrastruktur

- Wir haben Weg B gewählt: Stufe 2 entwickeln wir hier als Code und testen sie gegen ein Platzhalter-Modell; der echte Lauf erfolgt auf einem eigenen Rechner des Nutzers (Cloud-VM mit Grafikkarte, Docker, Netz).
- Als Arbeiter nutzen wir lokale offene Modelle über Ollama in einer Docker-Sandbox, sodass keine API-Gebühren anfallen.
- Wir messen gegen ein vergleichbar großes offenes Modell bei gleicher Rechenzeit, nicht gegen Opus 5.5, wobei wir den fälschungssicheren Messstab aus Stufe 1 nutzen.
- Dieser Container kann Ollama und Docker nicht ausführen (nicht installiert, Registry gesperrt, 1.6 GB frei, kein Docker-Dienst, kein /dev/kvm, runc 1.3.6).

## Nachtrag (21. Oktober 2026): Backend und erster Bauschritt

Wir haben die letzten Punkte des Interviews zu Stufe 2a beschlossen (Q28 bis Q32):

- **Q28 (Aufgabenart).** Die Arbeiter lösen zuerst unseren bestehenden Lerner-Messstab, also neue Aufgabenfamilien lernen. Code-Aufgaben kommen später, da sie eine eigene fälschungssichere Prüfung brauchen.
- **Q29 (offenes Modell und Vergleich).** Als Platzhalter im Entwurf nennen wir `llama3.1:8b` und `qwen2.5:7b`. Wir vergleichen das nackte Modell gegen das Modell plus Hierarchie bei gleicher Rechenzeit. Die echte Modellwahl trifft der Nutzer nach dem ersten Lauf auf seinem Rechner.
- **Q30 (Hierarchie und Zahlen).** Es sind genau drei Ebenen: Arbeiter, Prompt-Mutator und Code-Mutator. Die Zahlen 2, 4, 8, 13, 103 und 361 lesen wir als Fächerung je Ebene. Wir bauen sie je Ebene einstellbar, mit diesen Werten als Standard; oben wenige Instanzen, unten bis zu 361 Arbeiter.
- **Q31 (Sprache des Agenten-Codes).** Python für den Code, den die oberste Ebene umschreibt. Die Standardbibliothek reicht, und er ist leicht zu prüfen.
- **Q32 (erster Bauschritt).** Wir bauen hier nur den Unterbau, ohne echtes Modell und ohne Netz: eine Modell-Schnittstelle als Rust-Trait (`ModelBackend`), ein deterministisches Platzhalter-Modell (`StubModel`) und einen Bauvertrag für den Ollama-Docker-Adapter.

Dieser erste Bauschritt ist erledigt: `src/backend.rs` enthält die Schnittstelle und das Platzhalter-Modell mit 6 Tests, und `docs/STAGE2_BACKEND_CONTRACT.md` ist der Bauvertrag für den Adapter. Der Adapter selbst und die Ebenen 1 und 2 entstehen erst, wenn dieser Unterbau steht und Stufe 2a Zahlen geliefert hat.

## Nachtrag (21. Oktober 2026): Lücken K1 und K3 behoben (Protokoll learner-v3)

Wir haben die beiden offenen Lücken aus Stufe 1 (Q12) im Code geschlossen und die Protokoll-Version auf `learner-v3` gesetzt. Die Neumessung gegen den Geheimschlüssel verschieben wir auf die Kontrollläufe der Stufe 2, da ein geändertes Protokoll einen frischen Schlüssel erhält; wir haben dafür keine geheime Abfrage ausgeführt.

- **K1 (Kosten).** Die sechs im Messbericht als zu billig gezählten Befehle kosten jetzt so viel, wie sie wirklich dauern: SSin und SCos um Faktor 9, SExp und SLog um Faktor 5, VGaussian und MGaussian um Faktor 10. MatVec bleibt der Bezug mit Faktor 1, weshalb die gezählten den gemessenen Kosten folgen.
- **K3 (Gedächtnis).** Die Suche hält nun ein dauerhaftes, begrenztes Gedächtnis der je besten Programme (höchstens 64), sodass ein gutes Programm nicht verloren geht, wenn eine Panel-Drehung es aus der Ruhmeshalle verdrängt. Das Gedächtnis ordnet nach dem je besten Select-Vorsprung auf den festen Select-Aufgaben, speist bei jeder Drehung wieder in die Bevölkerung ein und stellt in der Schlussphase Finalisten.
- **Beleg.** Die Einzelheiten stehen im Bauvertrags-Nachtrag (docs/STAGE1_CONTRACT_ERRATA.md, Einträge V14 und V15). Ein kurzer Prüflauf ohne geheime Abfragen zeigt alle drei Finalisten aus dem Gedächtnis stammend, nachdem die Ruhmeshalle sie verworfen hatte.
- **LlmProposer.** Auf dem Platzhalter-Modell aus Q32 steht nun ein erster Vorschlagsweg (`LlmProposer`) im Code, noch nicht in die Suche verdrahtet.
