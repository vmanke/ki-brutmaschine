# Grenzen rekursiver und hierarchischer Komposition von ML-Modellen (Ensembles, Kaskaden, MoE, Kaskaden aus Kaskaden, rekursive Selbstverbesserung)

Stand: 2026-11-04. Kennzeichnung: **[peer-reviewed]**, **[Preprint]**, **[nicht begutachtet]**. Einige klassische Quellen (DOIs) stammen aus Vorwissen und wurden in diesem Lauf nicht neu abgerufen. Sie sind mit „(Vorwissen, DOI nicht im Lauf geprüft)“ markiert. Hinweis: en.wikipedia.org und researchonline.lse.ac.uk waren aus dieser Sitzung nicht erreichbar (Netzwerk-Allowlist). Volltexte von Ladha 1992, Kaniovski & Zaigraev und Dietrich & Spiekermann wurden daher nicht gelesen, nur Abstracts und Zusammenfassungen.

## Frage 1: Condorcet-Jury-Theorem mit korrelierten Wählern – wie viel bringen mehr Wähler oder mehr Ebenen? Gibt es eine Decke?

### Takeaway
Bei unabhängigen Wählern mit Kompetenz p > 1/2 geht die Trefferquote der Mehrheit gegen 1. Sobald Fehler eine gemeinsame Ursache haben (gleiche Daten, gleiches Modell, gleicher Prompt), gilt das nicht mehr: Die Mehrheit wird mit wachsender Gruppe zwar meist besser, aber sie wird nicht unfehlbar. Die Grenze für große Gruppen hängt dann an der Verteilung der Kompetenz Θ über die Aufgaben. Sie liegt unter 1, sobald die Wähler auf einem Teil der Aufgaben im Mittel nicht besser als 1/2 sind. Korrelation allein erzwingt diese Decke nicht. Eine zweite Ebene (Mehrheit von Mehrheiten) hebt diese Grenze nicht an. Bei unabhängigen, gleich kompetenten Wählern ist sie höchstens so gut wie eine flache Mehrheit.

### Cited Findings
- Ladha (1992), „The Condorcet Jury Theorem, Free Speech, and Correlated Votes“, *American Journal of Political Science* [peer-reviewed]: Er gibt hinreichende Bedingungen für beide Teile des Theorems an. Die Bedingung ist eine obere Schranke für den Durchschnitt der positiven Korrelationskoeffizienten. Das asymptotische Theorem gilt weiter, solange die „durchschnittliche“ Korrelation klein genug ist. — [Kaniovski, condorcet_inv_corr.pdf](https://Serguei.Kaniovski.wifo.ac.at/fileadmin/pdf/condorcet_inv_corr.pdf); [MPRA 69546](https://mpra.ub.uni-muenchen.de/69546/1/MPRA_paper_69546.pdf)
- Ladha argumentiert: Geteilte Information und andere gemeinsame Ursachen machen Stimmen korreliert, auch ohne direkten Einfluss der Wähler aufeinander. — [MPRA 69546](https://mpra.ub.uni-muenchen.de/69546/1/MPRA_paper_69546.pdf)
- Ladha (1993) nutzt den Satz von de Finetti für austauschbare (exchangeable) Wähler und erhält stärkere Ergebnisse. — [MPRA 69546](https://mpra.ub.uni-muenchen.de/69546/1/MPRA_paper_69546.pdf)
- Kaniovski & Zaigraev [peer-reviewed, Volltext nicht gelesen]: Für eine homogene Jury mit korrelierten Stimmen geben sie Schranken für die Trefferquote der Mehrheit. Die beste obere Schranke kann unter 1 liegen. Die untere Schranke kann deutlich unter dem Condorcet-Wert bei Unabhängigkeit liegen. — [Kaniovski, condorcet_robust.pdf](https://Serguei.Kaniovski.wifo.ac.at/fileadmin/pdf/condorcet_robust.pdf); [MaRDI: Aggregation of correlated votes and Condorcet's jury theorem](https://portal.mardi4nfdi.de/wiki/Publication:708796)
- Kaniovski: Es gibt „second-order invariant“ Verteilungen. Bei ihnen hängt die Mehrheitsgenauigkeit nicht von den paarweisen Korrelationen ab. Also zählt auch die Struktur höherer Ordnung, nicht nur die paarweise Korrelation. Ladhas Beispiel mit 3 Juroren: paarweise unkorreliert, aber nicht unabhängig (Wert a = 0,6075). — [Kaniovski, condorcet_inv_corr.pdf](https://Serguei.Kaniovski.wifo.ac.at/fileadmin/pdf/condorcet_inv_corr.pdf)
- Peleg & Zamir (2012), „Extending the Condorcet Jury Theorem to a general dependent jury“, *Social Choice and Welfare* 39(1):91–125, DOI 10.1007/s00355-011-0546-1 [peer-reviewed]: Beliebige gemeinsame Verteilungen sind erlaubt. Sie geben hinreichende und zwei notwendige Bedingungen und eine vollständige Lösung für austauschbare Folgen. Arbeitspapier-Version: Das CJT gilt, wenn die mit √n skalierte Abweichung der mittleren Kompetenz von 1/2 divergiert und die Summe der paarweisen Kovarianzen für große n nicht positiv ist. — [EconPapers](https://econpapers.repec.org/RePEc:spr:sochwe:v:39:y:2012:i:1:p:91-125); [HUJI DP 477](https://ratio.huji.ac.il/files/dp477.pdf)
- Dietrich & Spiekermann (2013), „Epistemic Democracy with Defensible Premises“, *Economics and Philosophy* 29(1) [peer-reviewed]: Sie modellieren Abhängigkeit über gemeinsame Ursachen und problemspezifische Kompetenz. Ergebnis: Große Gruppen sind fehlbar, aber besser als kleine. Die Begleitarbeit „Jury Theorems“ sagt: Die verteidigbaren Versionen stützen „wachsende Zuverlässigkeit“, aber nicht „Unfehlbarkeit“. — [PhilArchive](https://philarchive.org/rec/DIEEDW); [EconPapers MPRA 40135](https://econpapers.repec.org/RePEc:pra:mprapa:40135)
- Eine spätere Arbeit argumentiert: Eine passend gewählte Abstimmungsregel kann trotz deutlicher Korrelation zum richtigen Ergebnis konvergieren, wenn die mittlere Korrelation im großen Limit klein wird. — [MPRA 79633](https://mpra.ub.uni-muenchen.de/79633/9/MPRA_paper_79633.pdf)
- Boland (1989), „Majority systems and the Condorcet jury theorem“, *The Statistician* 38:181–189 [peer-reviewed] (Vorwissen, DOI 10.2307/2348873 nicht im Lauf geprüft): Ergebnis für ungleiche Kompetenzen p_i. Bei Unabhängigkeit und mittlerer Kompetenz p̄ > 1/2 + 1/(2n) ist die Mehrheit mindestens so gut wie der Durchschnittswähler.
- Nitzan & Paroush (1982), *International Economic Review* 23:289–297 [peer-reviewed] (Vorwissen, nicht im Lauf geprüft): Bei unabhängigen Wählern ist die optimale Regel eine gewichtete Mehrheit mit Gewichten w_i = log(p_i/(1−p_i)).

### Inferences
- **Formel für die Decke (eigene Herleitung aus de Finetti, keine zitierte Formel):** Seien die Korrektheits-Indikatoren X_i austauschbar. Nach de Finetti gibt es eine latente „Aufgaben-Schwierigkeit“ Θ. Gegeben Θ sind die X_i unabhängig Bernoulli(Θ). Dann gilt für die einfache Mehrheit: P(Mehrheit richtig) → P(Θ > 1/2) + ½·P(Θ = 1/2) für n → ∞. Mit Mittel p und paarweiser Korrelation ρ ist Var(Θ) = ρ·p(1−p). Für ρ > 0 streut Θ also. Ob Θ dabei Masse bei oder unter 1/2 hat, folgt aus ρ allein nicht. Gegenbeispiel (eigene Rechnung): Θ ist mit je Wahrscheinlichkeit 1/2 gleich 0,6 oder 0,9. Dann ist p = 0,75, Var(Θ) = 0,0225 und ρ = 0,12, und die Trefferquote der Mehrheit geht trotzdem gegen 1. Echt kleiner als 1 ist die Grenze nur, wenn Θ Masse bei oder unter 1/2 hat. Bedeutung: Aufgaben, bei denen das Modell „im Mittel“ falsch liegt (Θ < 1/2), werden durch mehr Stimmen sicher falsch, nicht richtig.
- **Varianz des Stimmenmittels:** Var(X̄) = p(1−p)·[ρ + (1−ρ)/n]. Für n → ∞ bleibt der Anteil ρ·p(1−p). Die „effektive Zahl unabhängiger Wähler“ ist n_eff = n / (1 + (n−1)ρ) → 1/ρ. Bei ρ = 0,5 behält das Mittel aus beliebig vielen Wählern also so viel Varianz wie das Mittel aus etwa 2 unabhängigen. (Standardrechnung, keine einzelne Quelle.) n_eff beschreibt nur diese Varianz. Eine Grenze für die Trefferquote folgt daraus allein nicht (siehe das Gegenbeispiel oben).
- **Mehrheit von Mehrheiten:** Jede Hierarchie aus Mehrheitsentscheidungen ist eine Boolesche Funktion der Einzelstimmen. Bei gleich kompetenten, unabhängigen Wählern ist die flache (ungewichtete) Mehrheit nach Nitzan & Paroush optimal. Eine Hierarchie ist dann höchstens gleich gut, meist schlechter. Bei korrelierten Wählern teilt eine zweite Ebene dieselbe latente Ursache Θ. Darum hebt sie die Grenze P(Θ > 1/2) + ½·P(Θ = 1/2) nicht an. Nur neue, unabhängige Information (anderes Modell, anderes Werkzeug, Prüfung gegen die Realität) kann Θ verschieben.
- Für Brutmaschine: Die Grenze wird bestimmt durch die Verteilung der Kompetenz Θ der Worker über die Aufgaben, also durch den Anteil der Aufgaben, auf denen sie im Mittel nicht besser als 1/2 sind. Die Korrelation ρ misst nur, wie stark Θ von Aufgabe zu Aufgabe streut. Die Anzahl der Worker und die Zahl der Ebenen verschieben die Grenze nicht.

### Gaps
- Die exakte Grenzformel von Kaniovski & Zaigraev (Bahadur-Korrelationen) konnte nicht gelesen werden (Wikipedia/LSE nicht erreichbar).
- Ladha-1992-Bedingung im Wortlaut (genaue Schranke für die mittlere Korrelation) nicht verifiziert.
- Pivato (2017, „Epistemic democracy with correlated voters“, J. Math. Econ.) wurde nicht geprüft.

## Frage 2: Ensemble-Theorie – Bias-Varianz-Kovarianz, Boosting, Stacking. Bringt ein Ensemble aus Ensembles mehr als ein flaches Ensemble gleicher Kosten?

### Takeaway
Die Zerlegungen zeigen: Ein Ensemble senkt nur die Varianz und den „Uneinigkeits“-Teil. Bias und Kovarianz bleiben. Mit wachsender Größe geht der Fehler gegen Bias² + mittlere Kovarianz, nicht gegen 0. Boosting schafft „schwach → stark“, aber nur, weil jede Stufe neue, gezielt gewichtete Trainingsdaten (echte Labels) sieht. Für Mittelwert-Aggregation bringt bloßes Verschachteln nichts, was ein flaches, passend gewichtetes Mittel nicht auch kann, denn ein Mittel aus Mitteln ist wieder ein gewichtetes Mittel. Für Mehrheiten gilt das nicht allgemein: Eine Mehrheit aus Mehrheiten ist keine gewichtete Mehrheit. Bei unabhängigen, gleich kompetenten Wählern schlägt sie die flache Mehrheit aber nicht (Nitzan & Paroush). Ein bekannter Weg zu echtem Gewinn sind Zwischenebenen, die etwas *lernen* (Stacking, Gating, Boosting) und dafür eigene Daten haben.

### Cited Findings
- Krogh & Vedelsby (NIPS 7, 1995), „Neural Network Ensembles, Cross Validation, and Active Learning“, S. 231–238 [peer-reviewed]: Die „Ambiguity“ ist die Streuung der Ausgaben der Mitglieder, gemittelt über ungelabelte Daten. Mehr Ambiguity bei gleichem Einzelfehler verbessert die Generalisierung. Die Zerlegung lautet E = Ē − Ā (Ensemblefehler = mittlerer Einzelfehler minus mittlere Ambiguity). Sie zeigen auch, wie man optimale Gewichte mit ungelabelten Daten schätzt. — [NeurIPS Proceedings](https://proceedings.neurips.cc/paper/1994/hash/b8c37e33defde51cf91e1e03e51657da-Abstract.html)
- Ueda & Nakano (1996), „Generalization error of ensemble estimators“, Proc. IEEE ICNN [peer-reviewed] (Vorwissen, nicht im Lauf geprüft): Für das Mittel aus M Schätzern gilt E[(f̄ − y)²] = bias̄² + (1/M)·var̄ + (1 − 1/M)·covar̄. Für M → ∞ bleibt bias̄² + covar̄.
- Breiman (2001), „Random Forests“, *Machine Learning* 45:5–32, DOI 10.1023/A:1010933404324 [peer-reviewed] (Vorwissen, nicht im Lauf geprüft): Schranke für den Generalisierungsfehler PE* ≤ ρ̄·(1 − s²)/s². ρ̄ ist die mittlere Korrelation der Bäume, s die „Stärke“. Mehr Bäume senken den Fehler nur bis zu dieser Schranke.
- Schapire (1990), „The Strength of Weak Learnability“, *Machine Learning* 5(2):197–227, DOI 10.1007/BF00116037 [peer-reviewed] (Vorwissen, nicht im Lauf geprüft): Schwache PAC-Lernbarkeit ist gleich starker Lernbarkeit. Die Originalkonstruktion ist **rekursiv**: Mehrheit aus 3 Lernern, die auf gefilterten Verteilungen trainiert werden, und das rekursiv. Jede Ebene braucht dabei neue Beispiele aus der Verteilung.
- Freund & Schapire (1997), AdaBoost, *J. Comput. Syst. Sci.* 55:119–139, DOI 10.1006/jcss.1997.1504 [peer-reviewed] (Vorwissen, nicht im Lauf geprüft): Trainingsfehler ≤ exp(−2·Σ_t γ_t²), wobei γ_t der Vorsprung der Stufe t über 50 % ist. Voraussetzung: Jede Stufe hat auf der neu gewichteten Verteilung einen Vorsprung γ_t > 0.
- Wolpert (1992), „Stacked Generalization“, *Neural Networks* 5:241–259 [peer-reviewed] (Vorwissen, nicht im Lauf geprüft): Ein Meta-Lerner lernt die Kombination aus Vorhersagen auf zurückgehaltenen Daten.
- Jacobs, Jordan, Nowlan & Hinton (1991), „Adaptive Mixtures of Local Experts“, *Neural Computation* 3:79–87 [peer-reviewed] (Vorwissen, nicht im Lauf geprüft): Ein Gating-Netz verteilt Eingaben auf Experten. Gewinn kommt aus Spezialisierung, nicht aus Abstimmung.

### Inferences
- Ein Mittel aus Mitteln ist wieder ein gewichtetes Mittel. Bei fester linearer Mittelung ist eine Hierarchie also eine *Teilmenge* dessen, was eine flache gewichtete Kombination kann. Sie bringt dort keinen Gewinn bei gleicher Gesamtzahl an Workern.
- Für Mehrheiten gilt das nicht allgemein. Schon eine Mehrheit aus drei Dreiermehrheiten ist keine gewichtete Schwellwertfunktion der neun Einzelstimmen. Eigene Begründung: Ihre Symmetrie würde gleiche Gewichte erzwingen, sie weicht aber von der Mehrheit aus neun ab, etwa bei 3, 1 und 1 Ja-Stimmen in den drei Gruppen. Bei unabhängigen, gleich kompetenten Wählern ist die flache Mehrheit trotzdem optimal (Nitzan & Paroush). Dort bringt die Verschachtelung also keinen Gewinn.
- Ob feste Verschachtelung außerhalb dieser Annahmen hilft (ungleiche Kompetenz, abhängige Stimmen), ist offen. Einen Satz dazu fanden wir nicht. Ein bekannter Weg zu Gewinn sind gelernte Zwischenebenen (Stacking, Gating, Boosting). Dann gelten aber normale Lern-Grenzen: Überanpassung des Meta-Lerners, Bedarf an frischen Daten pro Ebene.
- Boosting zeigt: Die „schwach → stark“-Treppe funktioniert, wenn jede Stufe auf den Fehlern der vorigen Stufe *mit echten Labels* trainiert. Das passt zur Brutmaschine-Idee „spätere Stufe sieht frühere Antworten“, aber nur, wenn es ein externes Fehlersignal gibt. Ohne Labels ist es reines Voting und unterliegt der Decke aus Frage 1.
- Für Brutmaschine: Heterogene Worker (LLM, MLP, CNN, Programme) senken die Kovarianz. Das ist der einzige Hebel, der die Decke bias̄² + covar̄ bzw. ρ̄ verschiebt.

### Gaps
- Keinen rigorosen Satz gefunden, der „Ensemble aus Ensembles“ allgemein gegen „flaches Ensemble gleicher Kosten“ vergleicht. Die obige Aussage ist eine eigene Ableitung für feste Aggregation.
- Ueda/Nakano-, Breiman-, Schapire-Formeln wurden in diesem Lauf nicht im Original nachgelesen.

## Frage 3: LLM-spezifisch – Majority Voting, Self-Consistency, „More Agents“, Debatte, nicht-monotone Skalierung, compute-gematchte Vergleiche

### Takeaway
Mehr Samples plus Mehrheitsentscheid helfen meist, aber mit Sättigung. Chen et al. zeigen sogar: Der Nutzen kann wieder sinken, weil schwere Fragen mit mehr Stimmen schlechter werden. Ohne externen Prüfer flacht Auswahl bei etwa 100 Samples ab. Multi-Agent-Debatte schlägt bei gleichem Rechenbudget einfaches Self-Consistency in der Regel nicht. Fehler großer LLMs sind stark korreliert (60 % Übereinstimmung bei gemeinsamen Fehlern).

### Cited Findings
- Chen et al. (2024), „Are More LLM Calls All You Need? Towards the Scaling Properties of Compound AI Systems“, NeurIPS 2024 [peer-reviewed]: Bei einstufigen Voting-Systemen steigt die Leistung mit der Zahl der Aufrufe erst und **fällt dann**. Grund: Leichte Fragen werden besser, schwere Fragen werden schlechter. Ein Skalierungsgesetz mit Schwierigkeitsmodell sagt die optimale Zahl der Aufrufe für Vote und Filter-Vote voraus. Mehrstufige und heterogene Systeme sind offen. — [NeurIPS 2024](https://neurips.cc/virtual/2024/poster/93777); [arXiv 2403.02419](https://arxiv.org/html/2403.02419v1)
- Li et al. (2024), „More Agents Is All You Need“, TMLR 10/2024 [peer-reviewed]: Sampling-and-Voting. Leistung steigt mit der Zahl der Agenten. Der Gewinn hängt deutlich von der Aufgabenschwierigkeit ab. Methode ist mit CoT und Multi-Agent-Frameworks kombinierbar. Nachteil: hoher Ressourcenbedarf. — [arXiv 2402.05120](https://arxiv.org/html/2402.05120v2)
- Brown et al. (2024), „Large Language Monkeys“ [Preprint]: Coverage (irgendein Versuch richtig) wächst über 4 Größenordnungen der Sample-Zahl. SWE-bench Lite: 15,9 % mit 1 Sample → 56 % mit 250 Samples (mit Prüfer). Ohne automatischen Prüfer flachen Mehrheitsentscheid und Reward-Modelle ab etwa 100 Samples (Versionen nennen „einige hundert“ bzw. „~100“) ab. Mathe-Beispiel: Mehrheitsentscheid/Reward-Modell nur 40,50 % → 41,41 % über denselben Bereich. Schluss der Autoren: Skalierbare Verifikation ist nötig. — [arXiv 2407.21787](https://arxiv.org/abs/2407.21787)
- Kim, Garg, Peng & Garg (2025), „Correlated Errors in Large Language Models“, ICML 2025, PMLR 267:30038–30066 [peer-reviewed]: Über 350 LLMs. Wenn zwei Modelle beide falsch liegen, wählen sie in 60 % der Fälle dieselbe falsche Antwort (ein Leaderboard). Größere und genauere Modelle haben stark korrelierte Fehler, auch über Architekturen und Anbieter hinweg. Folgen für LLM-as-judge und „algorithmische Monokultur“. — [PMLR](https://proceedings.mlr.press/v267/kim25e.html); [arXiv 2506.07962](https://arxiv.org/abs/2506.07962v1)
- Huang et al., „Large Language Models Cannot Self-Correct Reasoning Yet“ (ICLR 2024; hier arXiv-Version) : GSM8K, bei gleicher Zahl an Antworten: Debatte 83,2 % vs. Self-Consistency 82,5 % (3 Antworten); 83,2 % vs. 85,3 % (6); 83,0 % vs. 88,2 % (9). Bei gleichen Kosten ist Debatte deutlich schlechter als einfacher Mehrheitsentscheid. Gewinn der Debatte kommt aus Aggregation, nicht aus Selbstkorrektur. — [arXiv 2310.01798](https://arxiv.org/pdf/2310.01798)
- Smit et al., „Should we be going MAD?“, ICML 2024 [peer-reviewed]: Multi-Agent-Debatte schlägt Self-Consistency und Ensembling nicht zuverlässig. Einige Systeme werden mit Hyperparameter-Tuning besser. — [PMLR v235](https://proceedings.mlr.press/v235/smit24a.html)
- Zhang et al. (2025), „Stop Overvaluing Multi-Agent Debate“ (früher „If Multi-Agent Debate is the Answer, What is the Question?“) [Preprint]: MAD schlägt CoT und Self-Consistency nicht zuverlässig, selbst mit mehr Rechenzeit. Die Autoren fordern Modell-Heterogenität. — [arXiv 2502.08788](https://arxiv.org/html/2502.08788v3)
- Gegenbeispiel RedDebate [Preprint]: Im Sicherheits-Red-Teaming senkt Debatte bei gleichem Tokenbudget Fehler zuverlässiger als Selbstkritik. Vergleich ist gegen Selbstkritik, nicht gegen Self-Consistency. — [arXiv 2506.11083](https://arxiv.org/pdf/2506.11083)
- Wang et al. (2023), Self-Consistency, ICLR 2023 [peer-reviewed] (Vorwissen, nicht im Lauf geprüft): Mehrere CoT-Pfade samplen, Mehrheit der Endantworten nehmen. — [arXiv 2203.11171](https://arxiv.org/abs/2203.11171)
- Snell et al., „Scaling LLM Test-Time Compute Optimally…“ (ICLR 2025) (Vorwissen, nicht im Lauf geprüft): Compute-optimales Test-Time-Scaling hängt von der Schwierigkeit ab. Bei leichten/mittleren Aufgaben kann ein kleines Modell mit mehr Test-Time-Compute ein deutlich größeres Modell schlagen, bei den schwersten nicht. — [arXiv 2408.03314](https://arxiv.org/abs/2408.03314)

### Inferences
- Chen et al. ist die LLM-Form der Condorcet-Decke: Für Fragen mit Θ < 1/2 macht mehr Voting das Ergebnis schlechter. Die Mischung aus leichten und schweren Fragen erzeugt das Maximum und den Abfall.
- Kim et al. heißt: Selbst „verschiedene“ LLMs als Worker haben hohe Fehlerkorrelation. Nach Frage 1 ist n_eff ≈ 1/ρ klein. Heterogenität über Modellfamilien hinweg (MLP, CNN, symbolische Programme, Werkzeuge) ist der wichtigere Hebel als mehr LLM-Worker.
- Brown et al.: Der Abstand zwischen Coverage und Mehrheits-Auswahl ist genau der Wert eines Prüfers. Für Brutmaschine bedeutet „blind gegenseitig bewerten“ ohne externe Prüfung: Auswahl wird früh gesättigt. Mit ausführbarem Test (Code laufen lassen, Ground-Truth) skaliert es weiter.
- Debatte/„spätere Stufe sieht frühere Antworten“ erhöht die Korrelation (Konformität). Bei gleichem Budget ist parallele Unabhängigkeit + Mehrheit oft besser.

### Gaps
- Keine compute-gematchte Studie für *mehrstufige, verschachtelte* LLM-Kaskaden (Kaskade aus Kaskaden) gefunden. Chen et al. nennen das selbst als offen.
- Exakte Zahlen von Snell et al. nicht im Lauf geprüft.

## Frage 4: Grenzen rekursiver Selbstverbesserung und Selbstreferenz (Rice, Gödel/Löb, Gödel-Maschine, Darwin-Gödel-Maschine, No-Free-Lunch)

### Takeaway
Es gibt harte formale Grenzen: Ein System kann nicht allgemein entscheiden, ob ein neues Programm „besser“ ist (Rice). Es kann nicht beweisen, dass ein gleich starker Nachfolger korrekt ist (Löb-Hindernis). Über alle möglichen Probleme gemittelt ist kein Lerner besser als ein anderer (NFL). Praktische Selbstverbesserung (Darwin-Gödel-Maschine) ersetzt Beweise durch empirische Tests. Sie bringt echte, aber begrenzte und teure Gewinne auf festen Benchmarks.

### Cited Findings
- Yudkowsky & Herreshoff (2013), „Tiling Agents for Self-Modifying AI, and the Löbian Obstacle“ [nicht begutachtet, MIRI-Entwurf]: Ein Agent, der einen Nachfolger baut und dessen Korrektheit in seinem eigenen formalen System beweisen will, stößt auf ein Gödel/Löb-Hindernis. Technische Umgehungen existieren, lösen das Problem aber nur teilweise. Ein grundlegendes Entscheidungskriterium für selbstmodifizierende probabilistische Agenten bleibt offen. Der Entwurf ist explizit nicht begutachtet und kann Fehler enthalten. — [MIRI-Ankündigung](https://intelligence.org/2013/06/06/new-research-page-and-two-new-articles/); [PDF-Spiegel](https://dcmpx.remotevs.com/org/intelligence/PL/files/TilingAgents.pdf)
- Kritik an Tiling Agents: Prädikatenlogik erster Stufe passe schlecht zu realen Umgebungen; der Inhalt sei weit weg von realistischer KI (Community-Kommentare, keine Fachbegutachtung). — [LessWrong/GreaterWrong](https://www.greaterwrong.com/posts/gnxDNEtkEo3sfeyPn/YmQ56kKbB9gcdYDsi)
- Zhang, Hu, Lu, Lange & Clune, „Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents“, ICLR 2026 [peer-reviewed; arXiv 2505.22954]: Ein Coding-Agent ändert seinen eigenen Code. Statt formaler Beweise (wie Schmidhubers Gödel-Maschine) nutzt er empirische Tests und ein wachsendes Archiv. SWE-bench: 20,0 % → 50,0 %. Polyglot: 14,2 % → 30,7 %. Gefunden wurden u. a. bessere Edit-Werkzeuge, Langkontext-Verwaltung, Peer-Review-Mechanismen. Deutlich besser als Varianten ohne Selbstverbesserung oder ohne offene Exploration. Bleibt hinter einigen geschlossenen SOTA-Lösungen. Ein SWE-bench-Lauf kostet geschätzt ca. 22.000 USD. Sandbox und menschliche Aufsicht. — [ICLR 2026](https://iclr.cc/virtual/2026/poster/10007327); [arXiv 2505.22954](https://arxiv.org/pdf/2505.22954v3); [Sakana AI](https://sakana.ai/dgm/)
- Schmidhuber (2003/2007), „Gödel Machines“ [Preprint/Buchkapitel] (Vorwissen, nicht im Lauf geprüft): Ein Selbst-Umschreiben ist nur erlaubt, wenn ein Beweis zeigt, dass es den Nutzen erhöht. Dann ist es „global optimal“ relativ zu Axiomen und Nutzenfunktion. Grenze: Wegen Gödel-Unvollständigkeit sind manche nützlichen Verbesserungen nicht beweisbar, und Beweissuche kann beliebig lange dauern. — [arXiv cs/0309048](https://arxiv.org/abs/cs/0309048)
- Rice (1953), *Trans. AMS* 74:358–366 [peer-reviewed] (Vorwissen, nicht im Lauf geprüft): Jede nicht-triviale semantische Eigenschaft von Programmen ist unentscheidbar. Folgerung: „Ist Programm B auf allen Eingaben besser als A?“ ist allgemein unentscheidbar.
- Wolpert & Macready (1997), „No Free Lunch Theorems for Optimization“, *IEEE Trans. Evol. Comput.* 1(1):67–82, DOI 10.1109/4235.585893 [peer-reviewed] (Vorwissen, nicht im Lauf geprüft): Über alle Zielfunktionen gemittelt haben alle Suchalgorithmen dieselbe Leistung. Gewinn auf einer Klasse wird durch Verlust auf einer anderen bezahlt.

### Inferences
- Für Brutmaschine: „Rekursive Intelligenz“ muss sich wie die DGM auf empirische Bewertung stützen. Formal garantierte Selbstverbesserung ist nicht erreichbar.
- NFL heißt nicht „keine Verbesserung möglich“. Es heißt: Jeder Gewinn kommt aus Annahmen über die Aufgabenverteilung. Eine Kaskade wird für die Verteilung besser, gegen die gezüchtet wird (Overfitting auf den Benchmark ist das Risiko).
- Die blinde gegenseitige Bewertung ist ein interner Prüfer. Sie erbt die Korrelation der Worker (Frage 1/3). Ein externer Prüfer (Tests, Ground Truth) ist das, was die DGM-Gewinne erst ermöglicht hat.

### Gaps
- Keine quantitative Theorie gefunden, wie schnell empirische Selbstverbesserung (DGM-Art) sättigt.
- Schmidhuber-, Rice-, NFL-Inhalte nur aus Vorwissen, nicht neu gelesen.

## Frage 5: Rekursive Reasoning-Modelle (TRM, HRM) – was bringt Rekursionstiefe?

### Takeaway
Kleine, rekursiv angewandte Netze erreichen auf ARC-AGI erstaunlich gute Werte. Der Gewinn kommt aus wiederholter Verfeinerung derselben Lösung (Tiefe in der Zeit), nicht aus einer Hierarchie. Mehr Verfeinerungsschritte beim Inferieren helfen nur bis zu einer kleinen Zahl (ca. 4 bei HRM).

### Cited Findings
- Jolicoeur-Martineau (2025), „Less is More: Recursive Reasoning with Tiny Networks“ (TRM) [Preprint, arXiv 2510.04871]: Ein einziges 2-Schicht-Netz, 7 Mio. Parameter. Es hält eine aktuelle Antwort und einen latenten Zustand und verfeinert beide in Zyklen. Volles Backprop statt HRMs Näherungsgradient. ARC-AGI-1: 45 %, ARC-AGI-2: 8 % (Ablationstabelle: 44,6 % / 7,8 %). HRM: 40 % / 5 %. Die Autorin sagt, das schlage viele LLMs (DeepSeek R1, o3-mini, Gemini 2.5 Pro) mit <0,01 % der Parameter. — [arXiv 2510.04871](https://arxiv.org/abs/2510.04871)
- Kritik (Community, nicht begutachtet): TRM nutzt starke Datenaugmentation im Training, die LLMs wurden zero-shot bewertet. Der Vergleich ist daher nicht fair. — [Hugging Face Paper-Seite (Spiegel)](https://huggingface-gw-kz.just-ai.com/papers/2510.04871)
- ARC Prize Team (15.08.2025), „The Hidden Drivers of HRM's Performance on ARC-AGI“ [nicht begutachtet, Blog mit Experimenten]: Die „hierarchische“ Architektur bringt wenig gegenüber einem gleich großen Transformer. Der äußere Verfeinerungs-Loop ist der Treiber: Training mit mehr Verfeinerung verdoppelt die Leistung (>15 Prozentpunkte), wenn beim Inferieren keine Verfeinerung erlaubt ist. Mehr als 4 Verfeinerungsschritte beim Inferieren haben wenig Effekt. HRM braucht trotz gleicher Parameterzahl mehr Rechenaufwand. Verifizierter Wert auf dem semi-privaten ARC-AGI-1-Set: 32 %, 1,48 USD pro Aufgabe. — [ARC Prize Blog](https://arcprize.org/blog/hrm-analysis)
- Eine spätere Arbeit deutet den äußeren Loop als Skalierung von „Rateversuchen“ für einen plausiblen latenten Zustand, nicht als schrittweise Verbesserung [Preprint]. — [arXiv 2601.10679](https://arxiv.org/html/2601.10679v1)

### Inferences
- Rekursion hilft hier, weil *dasselbe* gelernte Netz iterativ eine Lösung verbessert und *im Training* auf diese Iteration optimiert wurde. Das ist etwas anderes als „Kaskade aus fertigen Kaskaden“, die nie gemeinsam trainiert wurden.
- Die Sättigung bei wenigen Schritten (HRM ~4) passt zum allgemeinen Muster: Rekursionstiefe bringt abnehmenden Grenznutzen.
- Für Brutmaschine: Wenn MLP/CNN-Worker geplant sind, ist ein TRM-artiger „verfeinere deine eigene Antwort“-Worker plausibler als tiefe Verschachtelung von Kaskaden.

### Gaps
- Kein Ergebnis gefunden, das zeigt, wie TRM-Leistung über viele Rekursionsschritte skaliert (Kurve über Tiefe) jenseits der genannten Ablationen.
- HRM-Originalpaper (Wang et al. 2025, arXiv 2506.21734) nicht direkt gelesen.

## Frage 6: Gibt es einen rigorosen Satz, dass Rekursion/Verschachtelung unbegrenzte Intelligenz liefert – oder eine bekannte Decke?

### Takeaway
Nein, es gibt keinen rigorosen Satz für unbegrenzte Intelligenz durch Verschachtelung. Es gibt mehrere rigorose *Decken* für bestimmte Mechanismen: Voting mit korrelierten Fehlern (Grenze unter 1), Mittelung (Grenze Bias² + Kovarianz), Selbstverifikation (Löb/Rice), Suche ohne Annahmen (NFL). Die einzige positive „schwach → stark“-Garantie ist Boosting. Sie braucht aber frische, gelabelte Daten und einen Vorsprung γ > 0 auf jeder Stufe.

### Cited Findings
- Dietrich & Spiekermann: Verteidigbare Jury-Theoreme stützen „wachsende Zuverlässigkeit“, aber nicht „Unfehlbarkeit“. — [PhilArchive](https://philarchive.org/rec/DIEEDW)
- Kaniovski & Zaigraev: Bei korrelierten homogenen Juries kann schon die beste obere Schranke unter 1 liegen. — [Kaniovski, condorcet_robust.pdf](https://Serguei.Kaniovski.wifo.ac.at/fileadmin/pdf/condorcet_robust.pdf)
- Chen et al.: Mehr Aufrufe können die Leistung senken; es gibt ein Optimum. — [NeurIPS 2024](https://neurips.cc/virtual/2024/poster/93777)
- Brown et al.: Auswahl ohne Prüfer sättigt bei etwa 100 Samples. — [arXiv 2407.21787](https://arxiv.org/abs/2407.21787)
- Yudkowsky & Herreshoff: Löb-Hindernis für beweisbar korrekte Nachfolger [nicht begutachtet]. — [MIRI](https://intelligence.org/2013/06/06/new-research-page-and-two-new-articles/)
- DGM: Empirische Selbstverbesserung bringt Gewinne (20 % → 50 % SWE-bench), aber endlich, teuer, benchmarkgebunden. — [ICLR 2026](https://iclr.cc/virtual/2026/poster/10007327)

### Inferences
- **Zusammenfassende Faustformel (eigene Synthese):** Für eine Kaskade mit Aggregation ohne externes Signal gilt grob: Grenzgenauigkeit ≈ P(Θ > 1/2), wobei Θ die „gemeinsame Kompetenz“ aller Worker auf einer Aufgabe ist. Rekursion verschiebt Θ nicht. Θ verschiebt sich nur durch (a) neue unabhängige Information (heterogene Modelle, Werkzeuge, Daten), (b) externe Prüfung (Tests, Ground Truth), (c) echtes Lernen auf Fehlern (Boosting-artig mit Labels).
- Rechenkosten wachsen bei Verschachtelung multiplikativ (k Worker pro Stufe, d Ebenen → bis zu k^d Aufrufe). Der Nutzen wächst nach den obigen Ergebnissen höchstens sublinear und sättigt. Bei gleichem Budget ist meist eine flache, heterogene Kaskade mit Prüfer besser als eine tiefe homogene.
- Die Frage „welche Grenze?“ hat also keine einzelne Zahl. Die Grenze ist die Fehlerkorrelation der Bausteine plus die Qualität des Prüfsignals.

### Gaps
- Kein rigoroser Satz gefunden, der eine *universelle* Obergrenze für verschachtelte lernende Systeme angibt. Auch kein Satz für unbegrenztes Wachstum.
- Keine empirische Arbeit zu „Kaskaden aus Kaskaden“ (LLM-Systeme mit mehr als zwei verschachtelten Ebenen) mit Compute-Matching gefunden.
