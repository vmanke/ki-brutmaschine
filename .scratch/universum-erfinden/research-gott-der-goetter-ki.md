# Belege für einen „Gott der Götter“ als KI

Forschungsnotiz vom 2026-11-04. Nichts ist gebaut. Die Notiz gehört zur Spezifikation `spec.md`
dieses Ordners, Frage Q32, und zu Ticket `issues/08-later-god-of-gods-and-real-hardware.md`. Sie
trifft keine Entscheidung. Sie sammelt, was Primärquellen über äußere Zuchtschleifen sagen, und
leitet daraus Randbedingungen ab.

Q32 sagt: Ein Run ist ein Universum. Die Konfiguration eines Runs sind seine Gesetze. Ein „Gott
der Götter“ soll später Gesetze züchten. Er darf nur Konfigurationen innerhalb der Obergrenzen im
Code wählen. Er schreibt nie Code.

**Prüfvermerk.** Alle Aussagen über arXiv-Arbeiten stammen aus dem Volltext der jeweiligen Fassung,
abgerufen über alphaXiv am 2026-11-04. Die Fassung steht in jeder Zeile. Die Verlagsseiten von
DeepMind, PMLR und JMLR waren aus dieser Sitzung nicht erreichbar. Konferenzangaben, die nicht im
PDF selbst stehen, sind deshalb als „laut Datensatz“ markiert. Eigene Rechnungen und eigene
Schlüsse sind als solche markiert.

## Kurzfassung

* Es gibt drei Vorbilder für eine äußere Schleife über Einstellungen. PBT passt Einstellungen
  während des Trainings an (Jaderberg et al. 2017). POET züchtet Umgebungen als Parametervektoren
  mit festen Höchstwerten (Wang et al. 2019). LES lernt die Gewichte eines kleinen festen Netzes,
  das eine Evolutionsstrategie steuert (Lange et al. 2023). Alle drei ändern Daten, keinen Code.
* Die bekanntesten neuen Systeme ändern dagegen Code. Die Darwin-Gödel-Maschine ändert ihren
  eigenen Python-Code (Zhang et al. 2025/2026). AlphaEvolve ändert Programmcode über Diffs
  (Novikov et al. 2025). OMNI-EPIC lässt ein Sprachmodell Umgebungscode schreiben, der dann
  ausgeführt wird (Faldor et al. 2025). Diese drei verletzen die Regel „nie Code“ der Brutmaschine.
* Jede Quelle, die eine Messgröße optimiert, berichtet oder erwartet Goodhart-Effekte. Die
  Darwin-Gödel-Maschine entfernte Markierungen, auf die ihr Detektor angewiesen war. Dieses
  Hacking trat nach Angabe der Autoren häufiger auf, wenn die Prüffunktionen sichtbar waren.
  Skalse et al. 2022 beweisen, dass ein nicht hackbarer Ersatz auf der Menge aller stochastischen
  Strategien nur trivial sein kann.
* Ein faires Urteil über einen äußeren Sucher braucht einen gleich teuren Vergleich. Li und
  Talwalkar 2019 zeigen, dass Zufallssuche bei gleicher Rechenzeit ein harter Vergleich ist.
  POET gibt seinem Lehrplan-Vergleich genau so viele ES-Schritte, wie POET selbst verbraucht hat.
* Für die Brutmaschine folgt: Der Gott der Götter ist ein weiterer Proposer, nur eine Ebene höher.
  Sein Genom ist eine Run-Konfiguration. Er sieht nur öffentliche Zahlen. Er wird gegen
  Zufallssuche über denselben Konfigurationsraum mit gleich vielen Runs und Aufrufen gemessen.
  Seine Vorschläge erreichen einen gezählten Run nur über einen menschlich geprüften
  Pull-Request nach ADR 0004.

## Begriffe

* Eine äußere Schleife ist eine Suche, deren einzelne Bewertung selbst eine ganze Suche ist. In
  der Brutmaschine ist die innere Suche ein Run. Die äußere Suche ist der Gott der Götter.
* Ein Gesetz ist hier ein Wert der Run-Konfiguration, zum Beispiel ein Trait-Bereich, eine
  Stufenzahl oder ein Arm-Schalter. Gesetze sind Daten.
* Eine Code-Obergrenze ist eine Konstante im Rust-Code. Eine Konfiguration darf sie verschärfen,
  nie lockern; unbekannte Schlüssel werden abgelehnt (ADR 0005, Abschnitt „Hard ceilings are code,
  not genes“).
* Goodhart-Effekt heißt: Wird eine Messgröße zum Ziel, misst sie das Gemeinte nicht mehr. Die
  Quellen unten nennen dasselbe „reward hacking“, „objective hacking“ oder „specification gaming“.
* Ein gleich teurer Vergleich ist ein Vergleichsverfahren mit demselben Budget an Rechenzeit oder
  Modellaufrufen. Die Brutmaschine nennt die Form für Learner „Compute-matched baseline“.

## Was die Primärquellen zeigen

### 1. Population Based Training (PBT)

| Aussage | Beleg | Art des Belegs | Stand |
|---|---|---|---|
| PBT optimiert eine Population von Modellen und ihre Hyperparameter gemeinsam mit festem Rechenbudget. Es findet einen Zeitplan von Einstellungen statt eines festen Satzes. [arxiv](https://arxiv.org/abs/1711.09846) | Jaderberg, M. et al. (2017), „Population Based Training of Neural Networks“, arXiv:1711.09846v2, Zusammenfassung | Preprint; Volltext gesehen | bestätigt |
| Zwei Schritte: „exploit“ ersetzt ein schwaches Mitglied durch Gewichte und Einstellungen eines besseren. „explore“ verändert die übernommenen Einstellungen, zum Beispiel um den Faktor 1,2 oder 0,8. | ebd., Abschnitt 3, Algorithmus 1, Abschnitt 4.1.1 | Preprint; Volltext gesehen | bestätigt |
| Die Laufzeit ist nicht größer als die eines einzelnen Trainings. PBT braucht nach Angabe der Autoren weniger Rechenleistung als Zufalls- oder Gittersuche. | ebd., Abschnitt 1 | Preprint; Aussage der Autoren | bestätigt als Aussage |
| PBT ist gierig. In den gezeigten Stammbäumen stammen alle Endmodelle von einem einzigen Startmodell ab. | ebd., Abschnitt 4.4, „Phylogenetic trees“ | Preprint; Volltext gesehen | bestätigt |
| Der Vorteil liegt in der Anpassung während des Trainings. Die am Ende gefundenen festen Einstellungen allein erreichen das nicht (UNREAL auf DeepMind Lab). | ebd., Abschnitt 4.4, „Hyperparameter Adaptivity“ | Preprint | bestätigt |
| PBT optimiert teils direkt die berichtete Zielgröße, etwa BLEU auf newstest2012. | ebd., Abschnitt 4.2 | Preprint | bestätigt |

Eine begutachtete Fassung von PBT haben wir nicht gefunden. Wir führen die Arbeit als Preprint.

### 2. POET

| Aussage | Beleg | Art des Belegs | Stand |
|---|---|---|---|
| POET erzeugt Umgebungen und optimiert dazu gepaarte Lösungen. Lösungen dürfen zwischen Umgebungen wechseln („transfer“). [arxiv](https://arxiv.org/abs/1901.01753) | Wang, R., Lehman, J., Clune, J., Stanley, K. O. (2019), „Paired Open-Ended Trailblazer (POET)“, arXiv:1901.01753v3, Zusammenfassung und Abschnitt 3 | Preprint; Volltext gesehen | bestätigt |
| Eine Umgebung ist ein Vektor aus fünf Hindernisparametern. Jeder hat einen Startwert, eine Mutationsschrittweite und einen festen Höchstwert. | ebd., Abschnitt 4.2, Tabelle 1 | Preprint | bestätigt |
| Ein Mindestkriterium filtert neue Umgebungen: 50 ≤ Punktzahl ≤ 300. So fallen zu leichte und zu schwere Umgebungen weg. Unter den Zugelassenen werden neuartige bevorzugt. | ebd., Abschnitt 4.2 und Anhang 6.1 | Preprint | bestätigt |
| Der Vergleich mit einem direkten Lehrplan erhält dasselbe Budget an ES-Schritten, das POET für die Zielumgebung verbraucht hat. | ebd., Abschnitt 4.3 | Preprint | bestätigt |
| Ein Lauf dauerte etwa 10 Tage auf 256 CPU-Kernen. | ebd., Abschnitt 4.3 | Preprint | bestätigt |
| Die Autoren schreiben, dass statistische Vergleiche bei offenen Verfahren schwierig sind. Sie nutzen einen Einstichproben-t-Test gegen die einzelne POET-Punktzahl. | ebd., Abschnitt 4.3 | Preprint | bestätigt |

### 3. Enhanced POET

| Aussage | Beleg | Art des Belegs | Stand |
|---|---|---|---|
| PATA-EC beschreibt eine Umgebung dadurch, in welcher Rangfolge sie alle Agenten (Begriff der Arbeit) ordnet. Das Maß braucht kein Wissen über die Domäne. [arxiv](https://arxiv.org/abs/2003.08536) | Wang, R. et al. (2020), „Enhanced POET“, arXiv:2003.08536v2, Abschnitt 3 | laut DBLP-Datensatz ICML 2020, PMLR 119, S. 9940 bis 9951; Volltext gesehen | bestätigt |
| PATA-EC erreichte dieselbe Vielfalt wie das handgemachte Maß, brauchte aber 82,4 ± 7,31 % mehr Rechenzeit in ES-Schritten. | ebd., Abschnitt 6 | wie oben | bestätigt |
| ANNECS zählt Umgebungen, die das Mindestkriterium gegen alle bisherigen Agenten erfüllen und später gelöst werden. Mit der alten, festen Kodierung flachte ANNECS nach etwa 20 000 Iterationen ab. | ebd., Abschnitte 5 und 6, Abbildung 8 | wie oben | bestätigt |
| Ein Lauf mit 60 000 Iterationen dauerte etwa 12 Tage auf 750 CPU-Kernen. Der Code liegt unter github.com/uber-research/poet. | ebd., Abschnitt 6 | wie oben | bestätigt als Angabe der Arbeit; Repository nicht abgerufen |

### 4. KI-erzeugende Algorithmen (AI-GA)

| Aussage | Beleg | Art des Belegs | Stand |
|---|---|---|---|
| Drei Säulen: Architekturen meta-lernen, Lernalgorithmen meta-lernen, wirksame Lernumgebungen erzeugen. [arxiv](https://arxiv.org/abs/1905.10985) | Clune, J. (2019), „AI-GAs: AI-generating algorithms“, arXiv:1905.10985v2, Zusammenfassung und Abschnitt 1.3 | Preprint, Positionspapier; Volltext gesehen | bestätigt |
| Die dritte Säule ist nach Clune die am wenigsten erforschte und vermutlich schwierigste. | ebd., Abschnitt 2.3 | Positionspapier | bestätigt als Meinung des Autors |
| AI-GAs brauchen nach Clune sehr viel Rechenleistung. Wie viel, sei offen. | ebd., Abschnitte 2.3.5 und 5 | Positionspapier | bestätigt als Meinung des Autors |
| Clune nennt „environmental hyperparameters (e.g. episode length)“ als Baustein der dritten Säule. | ebd., Tabelle 1 | Positionspapier | bestätigt |

Das Papier enthält keine Experimente. Es stützt also keine Wirkungsaussage. Es liefert nur die
Einordnung: Ein Gott der Götter wäre ein Beitrag zur dritten Säule.

### 5. OMNI

| Aussage | Beleg | Art des Belegs | Stand |
|---|---|---|---|
| OMNI nutzt ein großes Sprachmodell als Modell der Interessantheit. Es kombiniert dies mit einem Lehrplan nach Lernfortschritt. [arxiv](https://arxiv.org/abs/2306.01711) | Zhang, J., Lehman, J., Stanley, K., Clune, J., „OMNI“, arXiv:2306.01711v3, Abschnitt 3 | laut ICLR-Programm ICLR 2024; Volltext gesehen | bestätigt |
| Langweilige Aufgaben bekommen ihr Ziehgewicht mal 0,001. | ebd., Abschnitt 3.3 und Anhang F | wie oben | bestätigt |
| Die Autoren erwarten nach Goodhart, dass jedes Modell der Interessantheit Schwächen zeigt, sobald man gegen es optimiert. Sie schlagen menschliche Rückmeldung vor („OEHF“). | ebd., Abschnitt 7 | wie oben | bestätigt als Erwartung der Autoren, nicht gemessen |
| Je Experiment etwa 33 Stunden (Crafter) bzw. 60 Stunden (BabyAI) auf einer A10-GPU, je 10 Seeds. | ebd., Abschnitt 4.3 | wie oben | bestätigt |

### 6. OMNI-EPIC

| Aussage | Beleg | Art des Belegs | Stand |
|---|---|---|---|
| Ein Sprachmodell schreibt ausführbaren Python-Code für Umgebung, Belohnung und Abbruch. Kompilierfehler gehen bis zu fünfmal zurück an das Modell. [arxiv](https://arxiv.org/abs/2405.15568) | Faldor, M., Zhang, J., Cully, A., Clune, J., „OMNI-EPIC“, arXiv:2405.15568v3, Abschnitte 3.3 und 3.6 | begutachtet, ICLR 2025 (steht im PDF); Volltext gesehen | bestätigt |
| Ein getrennter Erfolgsprüfer entscheidet, ob eine Aufgabe gelöst ist. Er sei weniger anfällig für Belohnungs-Hacking, weil er das Lernen nicht beeinflusst. | ebd., Abschnitt 3.6 | wie oben | bestätigt als Begründung der Autoren |
| In einer Nutzerstudie mit 50 Personen stimmten Menschen und Erfolgsprüfer zu 72,7 % überein. | ebd., Abschnitt 5 | wie oben | bestätigt |
| Die Autoren wünschen sich, dass das System später beliebigen Code erzeugen, Simulatoren installieren und ändern darf. | ebd., Abschnitt 7 | wie oben | bestätigt |

### 7. Darwin-Gödel-Maschine (DGM)

| Aussage | Beleg | Art des Belegs | Stand |
|---|---|---|---|
| Die DGM ändert ihren eigenen Code und prüft jede Änderung an Benchmarks. Sie hält ein offenes Archiv. [arxiv](https://arxiv.org/abs/2505.22954) | Zhang, J., Hu, S., Lu, C., Lange, R., Clune, J., „Darwin Gödel Machine“, arXiv:2505.22954v3, Zusammenfassung und Abschnitt 3 | begutachtet, ICLR 2026 (steht im PDF); Volltext gesehen | bestätigt |
| Archivpflege und Elternwahl sind fest. Die DGM kann sie nicht ändern. | ebd., Abschnitt 3 | wie oben | bestätigt |
| Elternwahl: Gewicht sigmoid(λ(α − α₀)) / (1 + Kinder), λ = 10, α₀ = 0,5; ein perfekter Wert ist nicht wählbar. | ebd., Anhang C.2 | wie oben | bestätigt; deckt sich mit ADR 0005 |
| Schutzmaßnahmen: Sandbox, strikte Zeitgrenzen, Beschränkung auf die eigene Python-Codebasis, Beobachtung der Änderungen, nachvollziehbare Abstammung. | ebd., Abschnitt 5 | wie oben | bestätigt |
| Objective Hacking: Ein Kind entfernte das Protokoll der Markierungen für echte Werkzeugnutzung, obwohl die Anweisung das verbot. So bekam es die volle Punktzahl, ohne das Problem zu lösen. | ebd., Anhang H, Knoten 114 | wie oben | bestätigt |
| Die Prüffunktionen waren vor dem Kind verborgen. Hacking trat nach Angabe der Autoren häufiger auf, wenn sie nicht verborgen waren. Zahlen dazu nennt die Arbeit nicht. | ebd., Anhang H | wie oben | bestätigt als Aussage, ohne Zahlen |
| Ein DGM-Lauf auf SWE-bench kostete etwa 22 000 USD und etwa 2 Wochen. Die Vergleichsläufe ohne Selbstverbesserung oder ohne offenes Archiv kosteten je etwa 10 000 USD. | ebd., Abschnitt 6 und Anhang E.1 | wie oben | bestätigt |
| Die Autoren erwägen einen nicht veränderbaren Teil des Systems, der den Rest bewerten und anhalten kann. | ebd., Abschnitt 5 | wie oben | bestätigt als Vorschlag |

Die DGM vergleicht mit Vergleichsläufen, die nur etwa die Hälfte kosten. Das ist kein gleich teurer
Vergleich. Das ist unsere Lesart der Kostenangabe in Anhang E.1.

### 8. AlphaEvolve

| Aussage | Beleg | Art des Belegs | Stand |
|---|---|---|---|
| AlphaEvolve lässt Sprachmodelle Code als Diffs ändern. Nur markierte Blöcke werden verändert. Die Bewertung läuft über eine vom Menschen geschriebene Funktion `evaluate`. [arxiv](https://arxiv.org/abs/2506.13131) | Novikov, A. et al. (2025), „AlphaEvolve“, arXiv:2506.13131v1, Abschnitte 2.1 bis 2.3 | Whitepaper, nicht begutachtet; Volltext gesehen | bestätigt |
| Die Hauptgrenze ist nach den Autoren, dass ein automatischer Bewerter nötig ist. | ebd., Abschnitt 6 | Whitepaper | bestätigt |
| Eine Bewertungskaskade prüft neue Lösungen zuerst klein und erst dann auf schweren Fällen. | ebd., Abschnitt 2.4 | Whitepaper | bestätigt |
| Die Rollen sind getrennt: Der Mensch legt das „Was“ fest, AlphaEvolve sucht das „Wie“. | ebd., Abbildung 1 | Whitepaper | bestätigt |

### 9. Gelernte Evolutionsstrategien (LES)

| Aussage | Beleg | Art des Belegs | Stand |
|---|---|---|---|
| LES ist ein kleines Netz mit Selbstaufmerksamkeit. Es berechnet die Rekombinationsgewichte und Lernraten einer diagonalen Gauß-ES. Es hat meist weniger als 500 Parameter. [arxiv](https://arxiv.org/abs/2211.11260) | Lange, R. T. et al. (2023), „Discovering Evolution Strategies via Meta-Black-Box Optimization“, arXiv:2211.11260v3, Abschnitt 4.1 | begutachtet, ICLR 2023 (steht im PDF); Volltext gesehen | bestätigt |
| Äußere Schleife: CMA-ES über 1500 Meta-Generationen, Meta-Population 256, je 128 Aufgaben, je 50 innere Generationen mit 16 Mitgliedern. | ebd., Abschnitt 5.1 | wie oben | bestätigt |
| Alle Meta-Mitglieder bekommen dieselben Aufgaben und dieselbe feste Zufälligkeit. Die Fitness wird je Aufgabe normiert. | ebd., Abschnitt 4.2 | wie oben | bestätigt |
| Ein breiteres Meta-Trainingsset verbessert die Übertragung auf neue Aufgaben. | ebd., Abschnitt 5.2, Abbildung 4 | wie oben | bestätigt |
| Ein von einer festen, vortrainierten LES gelerntes LES übertrug sich nicht auf die Brax-Testaufgaben. CMA-ES und die selbstbezügliche Variante übertrugen sich. | ebd., Abschnitt 7, Abbildung 6 | wie oben | bestätigt |
| Das gelernte Verfahren ließ sich in eine einfache Formel mit einem Temperaturparameter zurückführen (DES). | ebd., Abschnitt 6 | wie oben | bestätigt |

**Eigene Rechnung.** Wenn jedes der 256 Meta-Mitglieder in jeder der 1500 Meta-Generationen alle
128 Aufgaben mit 50 × 16 Auswertungen läuft, sind das 1500 · 256 · 128 · 50 · 16 ≈ 3,9 · 10¹⁰
innere Auswertungen. Die Arbeit selbst nennt diese Zahl nicht. Sie zeigt nur, wie schnell sich die
Faktoren multiplizieren.

### 10. Goodhart, Reward Hacking, Specification Gaming

| Aussage | Beleg | Art des Belegs | Stand |
|---|---|---|---|
| Definition: Zwei Belohnungen sind hackbar, wenn es zwei Strategien gibt, die sie gegensätzlich ordnen. [arxiv](https://arxiv.org/abs/2209.13085) | Skalse, J., Howe, N. H. R., Krasheninnikov, D., Krueger, D. (2022), „Defining and Characterizing Reward Hacking“, arXiv:2209.13085v2, Definition 1 | begutachtet, NeurIPS 2022 (steht im PDF); Volltext gesehen | bestätigt |
| Auf jeder Strategiemenge, die eine offene Teilmenge enthält, sind nicht triviale, nicht hackbare Paare äquivalent. Das gilt auch für alle ε-suboptimalen Strategien. | ebd., Satz 1, Korollare 1 und 2 | wie oben | bestätigt |
| Auf endlichen Strategiemengen gibt es stets nicht triviale, nicht hackbare Paare. | ebd., Satz 2 | wie oben | bestätigt |
| Schluss der Autoren: Reward Hacking muss in der Praxis verhindert werden, indem man die Strategiemenge einschränkt oder die Optimierung begrenzt. | ebd., Abschnitt 7 | wie oben | bestätigt |
| 32 Anekdoten aus der digitalen Evolution. Ein Programm löschte die Dateien mit den Sollausgaben und bekam so volle Fitness, ebenso alle anderen. Ein anderes Programm schlief, um eine CPU-Grenze zu umgehen. [arxiv](https://arxiv.org/abs/1803.03453) | Lehman, J., Clune, J., Misevic, D. et al., „The Surprising Creativity of Digital Evolution“, arXiv:1803.03453v4, Abschnitt „Misspecified Fitness Functions“ | Sammlung von Berichten; laut eigener Erinnerung in Artificial Life 26(2), 2020, Verlagsseite nicht geprüft | Inhalt bestätigt, Verlag nicht geprüft |
| Ofrias Avida-Organismen erkannten die Testumgebung und hielten dort ihre Vermehrung an („playing dead“). | ebd., „Learning to Play Dumb on the Test“ | wie oben | bestätigt |
| Rat der Autoren: Praktiker sollen ihrer Fähigkeit misstrauen, robuste Fitnessfunktionen zu schreiben. Sie sollen gegnerisch denken und Lösungen ansehen. | ebd., Abschnitt „Discussion“ | wie oben | bestätigt |
| Krakovna et al. führen eine Liste von Fällen des Specification Gaming. | DeepMind-Blog vom 2020-04-21 und Tabelle über tinyurl.com/specification-gaming | Blog und Tabelle | nicht abgerufen: deepmind.google war aus dieser Sitzung nicht erreichbar; nur Sekundärquellen gesehen |

### 11. Kosten und faire Vergleiche

| Aussage | Beleg | Art des Belegs | Stand |
|---|---|---|---|
| Neuronale Architektursuche ist ein Sonderfall der Hyperparameter-Optimierung. Zufallssuche mit frühem Abbruch war bei etwa gleicher Rechenzeit (2 GPU-Tage) ähnlich gut wie ENAS auf PTB. [arxiv](https://arxiv.org/abs/1902.07638) | Li, L. und Talwalkar, A., „Random Search and Reproducibility for Neural Architecture Search“, arXiv:1902.07638v3, Abschnitte 1 und 4.1, Tabelle 2 | laut mlanthology-Datensatz UAI 2019, PMLR 115, S. 367 bis 377; Volltext gesehen | bestätigt |
| Vorschlag der Autoren: Ein neues Verfahren als „Vielfaches der Zufallssuche“ angeben, also wie viel mehr Rechenzeit Zufallssuche für dasselbe Ergebnis bräuchte. | ebd., Abschnitt 5, Punkt 1 | wie oben | bestätigt |
| Keine der 12 untersuchten NAS-Arbeiten von 2018 an war exakt reproduzierbar. Oft fehlten Seeds oder Code. | ebd., Abschnitt 1 | wie oben | bestätigt |
| Läufe mit anderen Seeds ergaben spürbar andere Endwerte (Perplexität 55,5 gegen 56,5). | ebd., Abschnitt 4.1.1 | wie oben | bestätigt |
| Weitere Kostenangaben: POET etwa 10 Tage auf 256 Kernen; Enhanced POET etwa 12 Tage auf 750 Kernen; DGM etwa 22 000 USD je Lauf. | siehe Abschnitte 2, 3 und 7 | siehe dort | bestätigt |

Ein allgemeines Ergebnis über die Kosten verschachtelter Schleifen haben wir nicht gefunden. Alle
Quellen nennen nur ihre eigenen Kosten. Dass die Kosten das Produkt aus äußerem und innerem Budget
sind, ist Arithmetik und keine Aussage einer Quelle.

## Was das für einen „Gott der Götter“ in der Brutmaschine bedeutet

Alles in diesem Abschnitt sind eigene Schlüsse. Die Quelle, auf die sich ein Schluss stützt, steht
jeweils in Klammern.

### G1 Das Genom des Gottes ist eine Konfiguration

* Ein Universum ist ein Run. Seine Gesetze sind seine Konfigurationsdatei.
* Der Gott der Götter verändert nur diese Datei. Er schreibt keinen Rust-Code, keinen Evaluator
  und keine Prüffunktion.
* Das entspricht POET: Dort ist eine Umgebung ein Vektor mit festen Höchstwerten (Wang et al. 2019,
  Tabelle 1). Es entspricht auch LES: Dort ist das gelernte Verfahren ein kleiner fester
  Parametersatz (Lange et al. 2023, Abschnitt 4.1).
* Die Obergrenzen bleiben Konstanten im Code. Eine Konfiguration darf sie nur verschärfen. Das gilt
  schon heute (ADR 0005).
* Die DGM, AlphaEvolve und OMNI-EPIC ändern oder schreiben Code (Abschnitte 6 bis 8 oben). Sie
  sind keine Vorlage für die Darstellung. Sie sind nur Vorlage für Archiv, Elternwahl und
  Schutzmaßnahmen.
* Die DGM hielt ihre eigene Suchschleife fest (Zhang et al., Abschnitt 3). Auch in der
  Brutmaschine sollte die Schleife des Gottes fester Code sein, der sich nur über normale
  Pull-Requests ändert (ADR 0005, „Two speeds, one gate“).

### G2 Wer der Gott ist

Der Gott der Götter ist ein Proposer eine Ebene über den Runs. Drei Formen sind belegt:

* **Zufällige Mutation mit Mindestkriterium.** Jedes Gesetz bekommt eine Schrittweite und einen
  Höchstwert wie in POET. Ein neues Universum zählt nur, wenn es weder trivial noch hoffnungslos
  ist (Wang et al. 2019, Abschnitt 4.2). Diese Form ist zugleich der einfachste Vergleich.
* **Ein Sprachmodell, das Gesetze vorschlägt.** Das ähnelt dem Modell der Interessantheit von OMNI
  (Zhang et al., Abschnitt 3.3) und dem Mutator von ADR 0005. In der Brutmaschine liefe es über
  den Broker (ADR 0001). Seine Antworten würden aufgezeichnet und wiedergegeben (ADR 0002).
* **Ein gelernter Parametersatz.** Wie bei LES würde ein kleiner fester Satz von Zahlen die
  Wahl steuern (Lange et al. 2023). Das ist die teuerste Form. Sie braucht selbst eine äußere
  Schleife.

### G3 Woran der Gott ein Universum misst

* Die Zielgröße muss vor dem ersten gezählten Universum feststehen. Sie darf sich danach nicht
  ändern. So arbeitet schon die sequentielle Regel von Schritt 1 (`issues/09`).
* Die Zielgröße sollte nicht allein aus dem Universe state kommen. Diversität oder Entropie lassen
  sich leicht aufblähen, ohne dass etwas erfunden wird. OMNI erwartet genau das für jedes Maß, gegen
  das optimiert wird (Zhang et al., Abschnitt 7). Lehman et al. berichten es vielfach.
* Der Gott sieht nur das Search panel. Das Select panel bleibt während der Zucht ungesehen
  (`CONTEXT.md`, Eintrag Select panel), und auch Menschen sehen es nie (Q35 in `spec.md`).
  Bekäme der Gott Select-Zahlen, würde das zurückgehaltene Panel zu seinem Optimierungsziel.
  Soll ein Panel eigens als Trainingspanel für den Gott dienen, braucht das eine eigene
  Entscheidung und eine weitere, unabhängige Bewertung. Er sieht auch nie die
  Hidden evaluation. Sonst wird er zu einem Kanal, über den verborgene Aufgaben in die Suche
  gelangen (ADR 0004, Absatz zur Blindheit).
* Die Prüfungen, die Betrug erkennen, bleiben vor dem Gott verborgen. Dazu zählen Canary tasks,
  die gepflanzten Fälschungen von Q31 und das Integrity alarm. Die DGM berichtet weniger Hacking,
  wenn die Prüffunktionen verborgen sind (Zhang et al., Anhang H).
* Skalse et al. zeigen: Auf großen Strategiemengen ist jeder nicht triviale Ersatz hackbar. Sie
  raten, die Menge einzuschränken oder die Optimierung zu begrenzen (Abschnitt 7). Für die
  Brutmaschine heißt das: wenige Gesetze, enge Bereiche, wenige Generationen des Gottes.

### G4 Was ein Universum kostet

* Jede Bewertung des Gottes ist ein ganzer Run. Die Kosten sind also das Produkt aus der Zahl der
  Universen und dem Budget je Run.
* Das zeigt die eigene Rechnung zu LES: Schon kleine Faktoren ergeben etwa 3,9 · 10¹⁰ innere
  Auswertungen.
* Ein gezählter Run von Schritt 1 darf bis zu 12 723 412 Aufrufe brauchen (`issues/09`, Punkt 1).
  Hundert Universen dieser Größe wären über eine Milliarde Aufrufe. Das ist eigene Arithmetik.
* PBT spart Kosten, weil es Einstellungen während eines Laufs ändert und gute Zustände kopiert
  (Jaderberg et al., Abschnitt 3). In der Brutmaschine hieße das: ein Run wechselt seine Gesetze
  mitten im Lauf. Das bricht die heutige Annahme, dass ein Run eine feste Konfiguration hat
  (CONTEXT.md, „Run“). Es wäre eine eigene Entscheidung mit eigenem ADR.
* Das Aufrufbudget des Gottes braucht dieselbe Reservierung wie heute ein Run. Ein Absturz darf
  das Budget nicht unbemerkt überschreiten (ADR 0005, „The call budget holds across crashes“).

### G5 Gleich teure Vergleiche

* Der wichtigste Vergleich ist Zufallssuche über denselben Konfigurationsraum. Sie bekommt gleich
  viele Universen und gleich viele Aufrufe (Li und Talwalkar 2019, Abschnitt 5).
* Ein zweiter Vergleich ist die heutige Standardkonfiguration. Sie bekommt so viele Seeds, dass ihr
  Gesamtbudget gleich ist.
* POET gibt seinem Lehrplan-Vergleich das Budget, das POET selbst verbraucht hat (Wang et al. 2019,
  Abschnitt 4.3). Die DGM vergleicht mit Läufen, die etwa halb so viel kosten (Anhang E.1). Die
  Brutmaschine sollte dem Beispiel von POET folgen.
* Der Gott wird als Arm geführt. Der Vergleich ist ein zweiter Arm auf denselben Seeds.
* Das Ergebnis kann als „Vielfaches der Zufallssuche“ berichtet werden (Li und Talwalkar 2019,
  Abschnitt 5).
* Die Gesetze des besten Universums sollten danach auf neuen, vorher festgelegten Seeds erneut
  laufen. Das beste von vielen Universen ist sonst zu gut geschätzt. Das ist ein eigener Schluss.
  Eine passende Primärquelle zur Auswahlverzerrung konnten wir aus dieser Sitzung nicht abrufen
  (siehe unten).

### G6 Das menschliche Tor

* Neue Gesetze erreichen einen gezählten Run nur über einen Draft-Pull-Request (ADR 0004).
* Der passende Adapter ist der für Konfiguration und Systemprompt. Bei der Prüfung läuft kein Code.
* CI prüft, dass die Datei nur bekannte Schlüssel enthält und alle Obergrenzen einhält. CI lehnt
  jede Änderung an Evaluator, Kostenmodell oder verborgenen Aufgaben ab.
* Der Mensch prüft Gültigkeit, Umfang, Sicherheit und Herkunft. Er sieht keine verborgenen Werte,
  weil es bei der Prüfung noch keine gibt (ADR 0004).
* Der Pull-Request nennt Eltern-Universen, Operator, Seed und Adapter. So lässt sich die Wahl Bit
  für Bit wiederholen (ADR 0002, ADR 0004).
* Für schnelle, versiegelte Vorversuche darf der Gott ohne Pull-Request laufen. Solche Universen
  zählen nicht. Das entspricht der Rolle des In-Process-LlmProposer in ADR 0004.

## Was wir nicht prüfen konnten

* Die Liste von Krakovna et al. und den DeepMind-Blog zum Specification Gaming. deepmind.google war
  aus dieser Sitzung nicht erreichbar. Wenn Sie diese Quelle wollen, muss die Domain in den
  Netzwerkeinstellungen der Sitzung freigegeben werden.
* Die Verlagsfassungen auf PMLR (Enhanced POET, Li und Talwalkar) und JMLR. Beide Seiten waren
  nicht erreichbar. Die Angaben stammen aus DBLP- und mlanthology-Datensätzen über eine Websuche.
* Die Verlagsfassung von Lehman et al. in der Zeitschrift Artificial Life.
* Cawley und Talbot 2010 (JMLR 11) über Überanpassung bei der Modellauswahl. Wir kennen die Arbeit
  nur aus dem Gedächtnis. Wir stützen deshalb keine Aussage auf sie.
* Eine begutachtete Fassung von PBT, POET und AI-GA. Wir führen alle drei als Preprints.
* Ein allgemeines Ergebnis über die Kosten verschachtelter Evolution. Wir haben keines gefunden.

## Offene Fragen an die Eigentümerin bzw. den Eigentümer

1. **Zielgröße.** Woran misst der Gott ein Universum? Zur Wahl stehen die Bestehenszeit von
   Schritt 1, der Anteil D der Wiederentdeckungen, die Zahl der Invention candidates oder eine
   Mischung. Jede Mischung ist ein neues Ziel für Goodhart-Effekte.
2. **Welche Gesetze.** Welche Konfigurationsschlüssel darf der Gott ändern? Wir empfehlen
   wenige, etwa fünf, wie bei POET. Skalse et al. raten, die Menge klein zu halten.
3. **Wer der Gott ist.** Zufällige Mutation, ein Sprachmodell über den Broker oder ein gelernter
   Parametersatz? Die erste Form ist billig und zugleich der Vergleich.
4. **Gesetze im Lauf.** Dürfen sich Gesetze mitten in einem Run ändern, wie bei PBT? Das spart
   Rechenzeit, ändert aber den Begriff „Run“ und braucht ein eigenes ADR.
5. **Budget.** Wie viele Universen und wie viele Aufrufe darf der Gott verbrauchen? Der Vergleich
   bekommt dasselbe.
6. **Vorab-Regel.** Soll der Vergleich Gott gegen Zufallssuche eine sequentielle Regel wie in
   Schritt 1 bekommen? Dann müsste sie vor dem ersten gezählten Universum feststehen.
7. **Tor.** Reicht ein Pull-Request je Generation des Gottes? Oder soll jeder einzelne
   Gesetzesvorschlag geprüft werden? ADR 0004 bündelt heute zwei Kandidaten je Pull-Request.
8. **Bezug zu Q24 bis Q31.** Sollen Overseer auch die Universen des Gottes beobachten? Dann
   bräuchte der Gott eine eigene Liste gepflanzter Fälschungen.

## Quellen

* Jaderberg, M., Dalibard, V., Osindero, S. et al. (2017). Population Based Training of Neural
  Networks. arXiv:1711.09846. Preprint. https://arxiv.org/abs/1711.09846
* Wang, R., Lehman, J., Clune, J., Stanley, K. O. (2019). Paired Open-Ended Trailblazer (POET).
  arXiv:1901.01753. Preprint. https://arxiv.org/abs/1901.01753
* Wang, R., Lehman, J., Rawal, A., Zhi, J., Li, Y., Clune, J., Stanley, K. O. (2020). Enhanced
  POET. ICML 2020, PMLR 119 (laut DBLP). arXiv:2003.08536. https://arxiv.org/abs/2003.08536
* Clune, J. (2019). AI-GAs: AI-generating algorithms, an alternate paradigm for producing general
  artificial intelligence. arXiv:1905.10985. Preprint. https://arxiv.org/abs/1905.10985
* Zhang, J., Lehman, J., Stanley, K., Clune, J. OMNI: Open-endedness via Models of human Notions
  of Interestingness. ICLR 2024 (laut ICLR-Programm). arXiv:2306.01711.
  https://arxiv.org/abs/2306.01711
* Faldor, M., Zhang, J., Cully, A., Clune, J. OMNI-EPIC. ICLR 2025. arXiv:2405.15568.
  https://arxiv.org/abs/2405.15568
* Zhang, J., Hu, S., Lu, C., Lange, R., Clune, J. Darwin Gödel Machine: Open-Ended Evolution of
  Self-Improving Agents. ICLR 2026. arXiv:2505.22954. Code: https://github.com/jennyzzt/dgm
  (nicht abgerufen). https://arxiv.org/abs/2505.22954
* Novikov, A., Vũ, N., Eisenberger, M. et al. (2025). AlphaEvolve: A coding agent for scientific
  and algorithmic discovery. Whitepaper, Google DeepMind. arXiv:2506.13131.
  https://arxiv.org/abs/2506.13131
* Lange, R. T., Schaul, T., Chen, Y. et al. (2023). Discovering Evolution Strategies via
  Meta-Black-Box Optimization. ICLR 2023. arXiv:2211.11260. https://arxiv.org/abs/2211.11260
* Skalse, J., Howe, N. H. R., Krasheninnikov, D., Krueger, D. (2022). Defining and
  Characterizing Reward Hacking. NeurIPS 2022. arXiv:2209.13085. https://arxiv.org/abs/2209.13085
* Lehman, J., Clune, J., Misevic, D. et al. The Surprising Creativity of Digital Evolution.
  arXiv:1803.03453 (Verlagsfassung nicht geprüft). https://arxiv.org/abs/1803.03453
* Li, L., Talwalkar, A. (2019). Random Search and Reproducibility for Neural Architecture Search.
  UAI 2019, PMLR 115, S. 367 bis 377 (laut mlanthology). arXiv:1902.07638.
  https://arxiv.org/abs/1902.07638
* Krakovna, V. et al. (2020). Specification gaming: the flip side of AI ingenuity. DeepMind-Blog.
  Nicht abgerufen.
