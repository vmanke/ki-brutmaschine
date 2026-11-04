# Belege für das Seelenmodell „chemisches QNN-Resnet“

Forschungsnotiz vom 2026-11-04. Nichts ist gebaut. Die Notiz gehört zur Spezifikation
`spec.md` dieses Ordners. Sie wählt keinen Kandidaten aus.

**Prüfvermerk vom 2026-11-04.** Wir haben die eigenen Rechnungen dieser Notiz nachgerechnet.
Die Speichertabelle in Frage 4 stimmt. Zwei Stellen haben wir geändert:

- **Rate des Sprungoperators.** Die erste Fassung schrieb „Die Rate ist k · |⟨f|L|i⟩|²“ mit
  L = √k c†ab. Damit wäre k doppelt gezählt. Richtig ist |⟨f|L|i⟩|² = k · n_A · n_B · (n_C + 1)
  bei Bosonen bzw. k · n_A · n_B · (1 − n_C) bei Qubits. Der Abschnitt „Eigene Schlüsse“ ist
  entsprechend berichtigt.
- **Arm W.** Die erste Fassung schrieb „Das passt zum Scheitern von Arm W“. Im Repository ist
  Arm W noch nicht gebaut und nicht gelaufen: Ticket `issues/01` und `issues/02` stehen auf
  `needs-triage`. Ein Ergebnis von Arm W liegt hier also nicht vor. Die Stelle ist als offen
  markiert.

Die Quellenangaben selbst haben wir nicht erneut abgerufen. Der Stand je Zeile ist der des
ursprünglichen Rechercheberichts.

Die Einzelteile der Arbeitshypothese haben jeweils veröffentlichte Vorbilder. Die Verbindung zu
einem vererbten Qubit-Reaktionsnetz mit Restblöcken, die bei Kopplung 0 starten, haben wir in
keiner Quelle gefunden. Wir bewerten sie deshalb als eigene Konstruktion.

## TL;DR

* Die Brücke zwischen Reaktionsnetzen und Quantenformalismus ist gut belegt. Doi 1976 schreibt
  A + B → C ausdrücklich als Produkt von Erzeugungs- und Vernichtungsoperatoren, allerdings für
  Wahrscheinlichkeiten. Heinzen et al. 2000 nutzen den hermiteschen Drei-Wellen-Term
  χ(Φ†Φ†Φ + h.c.) für Atom-Molekül-Umwandlung und erhalten im Mittelfeldgrenzfall nichtlineare
  Gleichungen für komplexe Amplituden.
* Eine Lindblad-Gleichung, deren Sprungoperatoren Reaktionen wie √k c†ab sind und aus der
  Massenwirkung abgeleitet wird, haben wir nicht gefunden. Gefunden haben wir nur begutachtete
  Arbeiten zu reinen Vernichtungs- und Koagulationsreaktionen ohne Produktoperator (Perfetto,
  Carollo, Garrahan, Lesanovsky 2023, Phys. Rev. E 108, 064104, und Folgearbeiten derselben
  Gruppe). Die Zuordnung „starke Dephasierung ergibt Ratengleichung“ bleibt für
  Reaktionsoperatoren unbelegt.
* Restverbindungen in QNNs werden in der Literatur nicht als unitäres x + F(x) gerechnet. Wen et
  al. 2024 nutzen eine Linearkombination von Unitären mit Hilfsqubit und Post-Selektion, deren
  Erfolgswahrscheinlichkeit mit der Tiefe sinkt. Kashif und Al-Kuwari 2024 addieren klassische
  Messergebnisse. Der Nullstart ist bei Goyal et al. 2017 (γ = 0) und bei NEAT (Stanley und
  Miikkulainen 2002, minimaler Start ungefähr 7-mal schneller als zufälliger Start) belegt.
  [arxiv](https://ar5iv.labs.arxiv.org/html/1706.02677)

## Begriffe

* Ein Hamiltonian H ist ein hermitescher Operator. Er erzeugt über die Schrödinger-Gleichung
  eine norm- und wahrscheinlichkeitserhaltende (unitäre) Zeitentwicklung komplexer Amplituden.
* Ein stochastischer Generator ist eine Matrix, deren Nichtdiagonaleinträge nicht negativ sind
  und deren Spalten sich zu null summieren. Er erzeugt die Zeitentwicklung einer
  Wahrscheinlichkeitsverteilung, die chemische Master-Gleichung.
* Erzeugungsoperator a† und Vernichtungsoperator a erhöhen bzw. senken die Teilchenzahl einer
  Sorte um eins.
* Eine Lindblad-Gleichung beschreibt ein offenes Quantensystem über eine Dichtematrix ρ. Sie
  enthält einen Hamilton-Teil und Sprungoperatoren L_k mit Raten γ_k.
* Dephasierung heißt, dass die Nichtdiagonalelemente der Dichtematrix zerfallen, während die
  Besetzungen erhalten bleiben.
* Post-Selektion heißt, dass man nur die Läufe behält, in denen ein Hilfsqubit ein bestimmtes
  Messergebnis zeigt.
* LCU (linear combination of unitaries) ist ein Verfahren, das mit Hilfsqubit und
  Post-Selektion eine gewichtete Summe von Unitären auf einen Zustand anwendet.
* Ein Barren Plateau ist ein Bereich der Verlustlandschaft, in dem die Gradienten mit der
  Qubitzahl exponentiell klein werden.

## Frage 1 Formale Brücke zwischen Reaktionsnetzen und Quantenformalismus

| Aussage | Beleg | Art des Belegs | Stand |
|---|---|---|---|
| Doi schreibt A + B → C als Operator mit Gewinnterm ψ_C†ψ_Aψ_B und Verlustterm ψ_A†ψ_B†ψ_Aψ_B. Das Modell ist klassisch stochastisch, die Plancksche Konstante kommt nicht vor. [pku](https://math.pku.edu.cn/teachers/litj/notes/stoch_topics2015/JPhysA_9_1465_Doi.pdf) | Doi, M. (1976), J. Phys. A: Math. Gen. 9(9), S. 1465 bis 1477, DOI 10.1088/0305-4470/9/9/008 | begutachtet, Primärquelle; vom Unteragenten im Volltext gesehen (Abschnitt 5, OCR teilweise fehlerhaft) | bestätigt, Indexstellung am Druck prüfen |
| Peliti überführt Dois Fock-Raum-Formalismus in ein Pfadintegral für Geburts-Tod-Prozesse auf dem Gitter. [journaldephysique](https://jphys.journaldephysique.org/en/articles/jphys/abs/1985/09/jphys_1985__46_9_1469_0/jphys_1985__46_9_1469_0.html) Ein allgemeines Beispiel A + B → C wurde dort nicht gefunden. | Peliti, L. (1985), J. Physique 46(9), S. 1469 bis 1483, DOI 10.1051/jphys:019850046090146900 | begutachtet; Zusammenfassung und Volltext ohne Formeln gesehen | Datensatz bestätigt, Inhalt teilweise |
| Für jedes Reaktionsnetz gilt H = Σ_τ r(τ)(a†^t(τ) − a†^s(τ)) a^s(τ). Die Koeffizienten sind reelle, nicht negative Wahrscheinlichkeiten mit Summe 1. Im Quantenfall stehen dort komplexe Amplituden. [arxiv](https://arxiv.org/pdf/1306.3451) | Baez, J. C. (2013), „Quantum Techniques for Reaction Networks“, arXiv:1306.3451, Satz 5 | Preprint, nicht begutachtet in dieser Fassung; Volltext gesehen | bestätigt als Aussage des Preprints |
| Stochastische Mechanik hat Gleichungen wie die Quantenmechanik, aber mit Wahrscheinlichkeiten statt Amplituden. Manche Hamiltonians beschreiben beide Arten von Prozessen. [arxiv](https://arxiv.org/pdf/1209.3632) | Baez, J. C. und Biamonte, J. D., arXiv:1209.3632 (v1 2012, v5 2019); Buchfassung World Scientific 2018, DOI 10.1142/10623 laut Semantic-Scholar-Datensatz | Lehrbuch und Preprint; Datensatz und Textausschnitte gesehen | bestätigt als Ausschnitt |
| Hermitescher Drei-Wellen-Term für Atom-Molekül-Umwandlung: H = ½ Σ ∫ [Φ_i†Φ_j† χ_ijk Φ_k + H.c.]. Die Autoren nennen ausdrücklich die Fälle A + B → C. [arxiv](https://arxiv.org/pdf/cond-mat/0110565v1) | Heinzen, D. J., Wynar, R. H., Drummond, P. D., Kheruntsyan, K. V. (2000), Phys. Rev. Lett. 84(22), S. 5029 bis 5033, DOI 10.1103/PhysRevLett.84.5029, arXiv:cond-mat/0110565 | begutachtet; arXiv-Volltext gesehen | bestätigt |
| Anderson, Craciun und Kurtz zeigen Produktform-Gleichgewichtsverteilungen für Netze mit Defekt null. | Anderson, D. F., Craciun, G., Kurtz, T. G. (2010), Bull. Math. Biol. 72, S. 1947 bis 1970, arXiv:0803.3042 | nur als Literaturangabe in Baez 2013 und Baez und Fong gesehen | Datensatz nicht selbst geprüft |
| Baez und Fong beweisen den Satz von Anderson, Craciun und Kurtz mit Quantentechniken neu. [arxiv](https://ar5iv.arxiv.org/html/1305.4988) | Baez, J. C. und Fong, B., J. Complex Networks 3 (2014), S. 22 bis 34, arXiv:1305.4988 | Zusammenfassung gesehen | bestätigt als Zusammenfassung |
| „Quantum Petri Nets“ existieren als Begriff. Sie betreffen Nebenläufigkeit und Quantenprogrammierung, nicht chemische Kinetik. [arxiv](https://arxiv.org/pdf/2508.14531) [arxiv](https://arxiv.org/abs/2408.08369) | Schmidt, H. W., arXiv:2106.03539 (Tagungsband-DOI 10.1007/978-3-030-87568-8_1 laut Semantic Scholar); Joachim, de Visme, Haar, Winskel, arXiv:2508.14531; Shah und Oruç, arXiv:2408.08369 | Preprints bzw. Tagungsband; Zusammenfassungen gesehen | bestätigt, thematisch nicht passend |
| Quantenalgorithmen zur Analyse klassischer Reaktionsnetze über Quantenspaziergänge. Die Reaktionsdynamik selbst bleibt klassisch. [arxiv](https://arxiv.org/html/2509.07890v1) | „Quantum Walks for Chemical Reaction Networks“, arXiv:2509.07890 | Preprint; Zusammenfassung gesehen | bestätigt, thematisch nur am Rand |
| Ein Reaktionsnetz auf Qubits mit Hamilton-Term g(c†ab + h.c.) je umkehrbarer Reaktion | keine Quelle gefunden | ohne Beleg | nicht gefunden |

Der Drei-Wellen-Hamiltonian ist für bosonische Moden belegt, nämlich in der
Atom-Molekül-„Superchemie“ von Heinzen et al. 2000.
[arxiv](https://arxiv.org/pdf/cond-mat/0110565v1) Die Fassung mit je einem Qubit pro Familie
haben wir nicht gefunden. Der Unterschied zwischen beiden Welten steht bei Doi und Baez klar.
Der stochastische Generator ist nicht hermitesch, wirkt auf reelle Wahrscheinlichkeiten und
erhält deren Summe. [arxiv](https://arxiv.org/pdf/1306.3451) Der Hamiltonian ist hermitesch,
wirkt auf komplexe Amplituden und erhält die Summe ihrer Betragsquadrate.

## Frage 2 Klassische Grenzfälle

| Aussage | Beleg | Art des Belegs | Stand |
|---|---|---|---|
| Aus der Master-Gleichung folgt d⟨N⟩/dt = Σ r(τ)(s(τ) − t(τ))⟨N^{s(τ)} fallend⟩. Das ist die Ratengleichung nur dann, wenn ⟨N^m fallend⟩ = ⟨N⟩^m gilt. [arxiv](https://arxiv.org/pdf/1306.3451) | Baez 2013, arXiv:1306.3451, Satz 8 | Preprint; Volltext gesehen | bestätigt |
| Die Gleichheit gilt genau für kohärente Zustände, also Produkte unabhängiger Poisson-Verteilungen. Im Allgemeinen gilt sie nur zu einem Zeitpunkt. [arxiv](https://arxiv.org/pdf/1306.3451) | Baez 2013, Satz 9 und Folgetext | Preprint; Volltext gesehen | bestätigt |
| Die Rate einer Reaktion im Master-Gleichungsbild ist r(τ) mal fallende Potenz der Teilchenzahlen, für A + B → C also r · n_A · n_B. [arxiv](https://arxiv.org/pdf/1306.3451) | Baez 2013, Gleichung (4) und (5) | Preprint; Volltext gesehen | bestätigt |
| Im Mittelfeldgrenzfall des Drei-Wellen-Hamiltonians entstehen nichtlineare Gleichungen für komplexe Amplituden, zum Beispiel iħφ̇₂ = … + ½χ*φ₁². [arxiv](https://arxiv.org/pdf/cond-mat/0110565v1) | Heinzen et al. 2000, Gl. (3) und (4) | begutachtet; Volltext gesehen | bestätigt, aber nicht als formaler Grenzwert hergeleitet |
| Diese kohärente Kinetik unterscheidet sich von klassischer Kinetik. Die Rate hängt von der Besetzung des Produktmodus ab und skaliert wie χ n₁ √n₂. Verluste werden phänomenologisch als Dämpfung ergänzt, nicht über Lindblad. [arxiv](https://arxiv.org/pdf/cond-mat/0110565v1) | Heinzen et al. 2000 | begutachtet; Volltext gesehen | bestätigt |
| Lindblad-Gleichungen mit Sprungoperatoren für 2A → ∅, 3A → ∅, 4A → ∅ (√Γ b_j^k), laut Zusammenfassung „modelled as dissipative processes which involve losses of pairs 2A→∅, triplets 3A→∅, and quadruplets 4A→∅“. Die mikroskopische Herleitung dieser Sprungoperatoren wird dort als offenes Problem bezeichnet. Folgearbeiten behandeln bosonische binäre Annihilation A + A → ∅ und Koagulation A + A → A sowie eine Boltzmann-artige kinetische Gleichung für A + A → ∅. | Perfetto, G., Carollo, F., Garrahan, J. P., Lesanovsky, I. (2023), Phys. Rev. E 108, 064104, DOI 10.1103/PhysRevE.108.064104, arXiv:2305.06944; Rowlands, Lesanovsky, Perfetto, New J. Phys., DOI 10.1088/1367-2630/ad397a, arXiv:2311.04018; Gerbino, Lesanovsky, Perfetto, „Kinetics of Quantum Reaction-Diffusion systems“, SciPost Phys. Core 8, 014, DOI 10.21468/SciPostPhysCore.8.1.014, arXiv:2406.20028 | begutachtet laut Datensatz; Zusammenfassungen und Suchausschnitte gesehen | Inhalt teilweise bestätigt, nur Vernichtung und Koagulation ohne Produktoperator |
| Massenwirkung, abgeleitet aus einer Lindblad-Gleichung mit Reaktionssprungoperator der Form √k c†ab | keine Quelle gefunden | ohne Beleg | nicht gefunden |
| Haken und Strobl 1973 sowie Breuer und Petruccione 2002 zum Übergang Dephasierung zu Ratengleichung | nicht abgerufen | ohne Beleg | nicht geprüft |

Der Grenzfall große Teilchenzahl führt auf nichtlineare Amplitudengleichungen und ist mit
Heinzen et al. 2000 belegt. [arxiv](https://arxiv.org/pdf/cond-mat/0110565v1) Der Grenzfall
Wahrscheinlichkeiten führt auf Ratengleichungen nur für kohärente Zustände oder große Zahlen,
das zeigt Baez 2013. [arxiv](https://arxiv.org/pdf/1306.3451) Eine veröffentlichte Herleitung
der Massenwirkung aus Reaktionssprungoperatoren haben wir nicht gefunden. Heinzen et al.
sprechen indirekt dagegen, dass kohärente Reaktionsdynamik Massenwirkung reproduziert.
[arxiv](https://arxiv.org/pdf/cond-mat/0110565v1)

## Frage 3 Restblöcke

| Aussage | Beleg | Art des Belegs | Stand |
|---|---|---|---|
| γ = 0 in der letzten Batch-Normalisierung jedes Restblocks lässt das Signal anfangs nur über die Identitätsverbindung laufen. Das erleichtert die Optimierung zu Beginn und verbessert die Ergebnisse bei Minibatch 256 und 8k. [arxiv](https://ar5iv.labs.arxiv.org/html/1706.02677) | Goyal, P. et al. (2017), arXiv:1706.02677, Abschnitt 2.3 und Tabelle 2b | Preprint, nicht begutachtet; Zusammenfassung und Abschnittsausschnitte gesehen, Tabellenwerte nicht gesehen | bestätigt qualitativ |
| QResNet nach Wen et al. baut den Restkanal mit Hilfsqubits. Die Evolution entsteht als Linearkombination von Unitären mit einem Hilfsqubit, der Zielzustand durch Nachverarbeitung. [nature](https://www.nature.com/articles/s42005-024-01719-1) | Wen, J., Huang, Z., Cai, D., Qian, L. (2024), Communications Physics 7, Artikel 220, DOI 10.1038/s42005-024-01719-1 | begutachtet; Zusammenfassung, Einleitung und Ausschnitt aus dem Methodenteil gesehen | bestätigt |
| Für l Lagen braucht man bis zu l Hilfsqubits. Die Größe der Beobachtungsoperatoren wächst exponentiell, was mit der sinkenden Erfolgswahrscheinlichkeit der Post-Selektion zusammenhängt. Die Zahl der Frequenzformen steigt von einer auf O(l²). [nature](https://www.nature.com/articles/s42005-024-01719-1) | Wen et al. 2024 | begutachtet; Ausschnitt gesehen | bestätigt |
| ResQNets schreiben formal ψ_out = ψ + Uψ. Umgesetzt wird aber gemessen, das klassische Ergebnis y₁ zur Eingabe x addiert und neu kodiert. [springeropen](https://epjquantumtechnology.springeropen.com/articles/10.1140/epjqt/s40507-023-00216-8) | Kashif, M. und Al-Kuwari, S. (2024), EPJ Quantum Technology 11, Artikel 4, DOI 10.1140/epjqt/s40507-023-00216-8 | begutachtet; Abschnitte 1 bis 3.2 gesehen | bestätigt |
| LCU-Post-Selektion ergibt eine Restabbildung proportional zu I + β_l W_l. Ein Vorabdruck vermeidet die Post-Selektion, nennt die Wirkung nicht unitär und beschreibt eine Konzentration der Zustände. [arxiv](https://arxiv.org/pdf/2604.06866) | „A hardware efficient quantum residual neural network without post-selection“, arXiv:2604.06866 | Preprint; Ausschnitt gesehen | bestätigt als Ausschnitt |
| Restverbindungen zwischen Lagen können Barren Plateaus verhindern, aber nur mit Einbußen bei der Erfolgswahrscheinlichkeit („accompanied by a trade-off in success probability“). Die Arbeit nutzt dafür LCU. | Heredge, J., West, M., Hollenberg, L., Sevior, M. (2025), „Nonunitary quantum machine learning“, Phys. Rev. Applied 23, 044046, DOI 10.1103/PhysRevApplied.23.044046 | begutachtet; nur Zusammenfassung | bestätigt als Zusammenfassung |
| Ein Vorabdruck mischt Eingangs- und Ausgangszustand mit Gewicht α und erzwingt Nichtnegativität und Normierung. [arxiv](https://arxiv.org/html/2604.15626v1) | „Bridge the Gap between Classical and Quantum Neural Networks with Residual Connections“, arXiv:2604.15626 | Preprint; Ausschnitt gesehen | bestätigt als Ausschnitt |
| Neural ODEs werden mit Schrödinger- und Lindblad-Dynamik verbunden: das Netz lernt den Liouville-Operator, gelöst mit Euler- oder Runge-Kutta-Verfahren. [arxiv](https://arxiv.org/pdf/2207.09056) | „Learning quantum dissipation by the neural ordinary differential equation“, arXiv:2207.09056; „Learning quantum dynamics with latent neural ODEs“, arXiv:2110.10721 | Preprints; Ausschnitte gesehen, Zeitschriftendatensatz nicht geprüft | bestätigt als Ausschnitt |
| Ein Netz, das die Schrödinger-Gleichung als tiefes Netz auffasst [arxiv](https://arxiv.org/pdf/2506.21537) | Nakajima, M., Tanaka, K., Hashimoto, T., „Neural Schrödinger equation: Physical law as deep neural network“, IEEE TNNLS 33(6), S. 2686 bis 2700, 2022 | nur als Literaturangabe in arXiv:2506.21537 gesehen | Datensatz nicht selbst geprüft |
| He et al. 2016, E 2017, Haber und Ruthotto 2017, Chen et al. 2018 | nicht abgerufen | ohne Beleg | nicht geprüft |

Keine der geprüften QNN-Arbeiten rechnet x + F(x) direkt als einzelne Unitäre. Wen et al. nutzen
LCU mit Post-Selektion, Kashif und Al-Kuwari addieren klassische Messwerte.
[springeropen](https://epjquantumtechnology.springeropen.com/articles/10.1140/epjqt/s40507-023-00216-8)
[nature](https://www.nature.com/articles/s42005-024-01719-1) Ob die Normierung selbst eine
brauchbare Nichtlinearität liefert, behauptet nur der Vorabdruck arXiv:2604.06866 qualitativ.
[arxiv](https://arxiv.org/pdf/2604.06866) Eine begutachtete Quelle dafür haben wir nicht
gefunden.

## Frage 4 Nichtlinearität und Simulierbarkeit

| Aussage | Beleg | Art des Belegs | Stand |
|---|---|---|---|
| Repeat-until-success-Schaltungen (Wiederholen bis zum Erfolg) erzeugen eine nichtlineare Schwellenaktivierung. Dazu werden Drehungen um 2θ in Drehungen um 2q∘k(θ) überführt. [researchgate](https://www.researchgate.net/publication/321417576_Quantum_Neuron_an_elementary_building_block_for_machine_learning_on_quantum_computers) | Cao, Y., Guerreschi, G. G., Aspuru-Guzik, A. (2017), arXiv:1711.11240 | Preprint, nicht begutachtet; Datensatz und Ausschnitte gesehen | bestätigt als Preprint |
| Im CV-Modell (kontinuierliche Variablen) liefert ein nicht gaußsches Gatter wie das Kerr-Gatter K(κ) = exp(iκn̂²) Nichtlinearität und Universalität. Die Schaltung bleibt dabei unitär. Die Arbeit schlägt auch Restnetze vor. [mit](https://dspace.mit.edu/bitstreams/68178e06-4db9-4d56-86d4-71a25281f988/download) [semanticscholar](https://www.semanticscholar.org/paper/Continuous-variable-quantum-neural-networks-Killoran-Bromley/4e1a9a2a22913f2a5c79fba6c2e5716ee1ed20bd/figure/0) [inspirehep](https://inspirehep.net/files/b954cb45ee67a73ea17ef0d5794db4f6) | Killoran, N., Bromley, T. R., Arrazola, J. M., Schuld, M., Quesada, N., Lloyd, S. (2019), Phys. Rev. Research 1, 033063, DOI 10.1103/PhysRevResearch.1.033063 | begutachtet; Zusammenfassung und Textausschnitte gesehen | bestätigt |
| Viele Modelle mit beweisbar fehlenden Barren Plateaus lassen sich klassisch simulieren, weil die Rechnung in kleinen Unterräumen polynomialer Größe liegt. [nature](https://www.nature.com/articles/s41467-025-63099-6) [osti](https://www.osti.gov/pages/servlets/purl/3013787) | Cerezo, M., Larocca, M., García-Martín, D. et al. (2025), Nature Communications 16, 7907, DOI 10.1038/s41467-025-63099-6 | begutachtet, Perspektivartikel; Zusammenfassung und Ausschnitte gesehen | bestätigt |
| Zustandsvektor von n Qubits: 2^n komplexe Zahlen. Dichtematrix: 4^n komplexe Zahlen. | eigene Rechnung | berechnet | siehe Tabelle unten |

| n Familien | Zustandsvektor (16 Byte je Zahl) | Dichtematrix (16 Byte je Zahl) |
|---|---|---|
| 8 | 256 Zahlen, 4 KiB | 65 536 Zahlen, 1 MiB |
| 16 | 65 536 Zahlen, 1 MiB | etwa 4,3 · 10⁹ Zahlen, 64 GiB |
| 24 | etwa 1,7 · 10⁷ Zahlen, 256 MiB | etwa 2,8 · 10¹⁴ Zahlen, 4 PiB |

Alle Werte in dieser Tabelle sind berechnet, nicht gemessen; der Prüfvermerk hat sie
nachgerechnet. Ein Schritt ψ − i h H ψ mit R Reaktionstermen kostet ungefähr R · 2^n
Operationen, ein Lindblad-Schritt ungefähr R · 4^n (berechnet). Für eine einzelne Unitäre ist
Nichtlinearität nur über Messung, Post-Selektion, Hilfsmoden oder nicht gaußsche Gatter belegt.
Cerezo et al. 2025 stützen, dass ein klassisch simuliertes Modell keinen Quantenvorteil
beansprucht. Für das Repository ist das kein Problem, weil ohnehin klassisch simuliert wird.

## Frage 5 Vererbte Netze, die ihre eigene Variation steuern

| Aussage | Beleg | Art des Belegs | Stand |
|---|---|---|---|
| Modular wechselnde Ziele (MVG, Wechsel alle E = 20 Generationen) erzeugen spontan modulare Netze und Netzwerkmotive. Unter festen Zielen sinkt die Modularität innerhalb weniger Dutzend Generationen. Ein Beispiel wechselt zwischen „L UND R“ und „L ODER R“. [pnas](https://www.pnas.org/doi/pdf/10.1073/pnas.0503610102) [researchgate](https://www.researchgate.net/publication/7588653_Spontaneous_evolution_of_modularity_and_network_motifs) [ucl](http://gpbib.cs.ucl.ac.uk/gp-html/Kashtan_2005_PNAS.html) [pnas](https://www.pnas.org/doi/10.1073/pnas.0503610102) | Kashtan, N. und Alon, U. (2005), PNAS 102(39), S. 13773 bis 13778, DOI 10.1073/pnas.0503610102 | begutachtet; nur Suchausschnitte, Volltext blockiert | bestätigt qualitativ |
| Zahlen aus derselben Arbeit: perfekte Lösung im Mittel nach 1 400 ± 1 000 Generationen in einer Bedingung; unter MVG nahezu perfekte modulare Netze (Q_m = 0,35 ± 0,02) nach 2 800 (+9 500, −600) Generationen. [researchgate](https://www.researchgate.net/publication/7588653_Spontaneous_evolution_of_modularity_and_network_motifs) [pnas](https://www.pnas.org/doi/10.1073/pnas.0503610102) | Kashtan und Alon 2005 | begutachtet; Ausschnitte, Zuordnung zu den Bedingungen unsicher | gemessen laut Quelle, Kontext unvollständig |
| Wechselnde Umwelten beschleunigen die Evolution. Laut Zusammenfassung ist „the highest speedup … under modularly varying goals“, und „the harder the problem, the larger the speedup“. Einen Zahlenfaktor nennt die Zusammenfassung nicht. | Kashtan, N., Noor, E., Alon, U. (2007), PNAS 104(34), S. 13711 bis 13716, DOI 10.1073/pnas.0611630104 | begutachtet; Zusammenfassung gesehen | qualitativ bestätigt, Faktor nicht bestätigt |
| Evolvierbarkeit entsteht in Genregulationsnetzen als Empfindlichkeit eines „evolutionären Sensorgens“ gegenüber Duplikation und Deletion. Die Sensorgene werden zu Knoten mit vielen Verbindungen. [plos](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1000112) | Crombach, A. und Hogeweg, P. (2008), PLoS Comput. Biol. 4(7), e1000112, DOI 10.1371/journal.pcbi.1000112 | begutachtet; Zusammenfassung gesehen | bestätigt qualitativ, kein Faktor gesehen |
| Erleichterte Variation (facilitated variation) wächst unter MVG. Wenige Mutationen erzeugen große, nützliche Anpassungen. [plos](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1000206) | Parter, M., Kashtan, N., Alon, U. (2008), PLoS Comput. Biol. 4(11), e1000206, DOI 10.1371/journal.pcbi.1000206 | begutachtet; Zusammenfassung gesehen | bestätigt qualitativ, kein Faktor gesehen |
| NEAT startet mit Netzen ohne versteckte Knoten. Die Knoten-Mutation setzt Eingangsgewicht 1 und übernimmt das alte Gewicht, „um die anfängliche Wirkung der Mutation zu minimieren“. Die neue Nichtlinearität ändert die Funktion dennoch leicht. [gwern](https://gwern.net/doc/reinforcement-learning/exploration/2002-stanley.pdf) | Stanley, K. O. und Miikkulainen, R. (2002), Evolutionary Computation 10(2), S. 99 bis 127, DOI 10.1162/106365602320169811 | begutachtet; Abschnitte 1 bis 4.2 gesehen | bestätigt |
| Ablation Doppelpendel mit Geschwindigkeiten: zufälliger Start 23 033 Bewertungen, 5 % Fehlschläge, ungefähr 7-mal langsamer als volles NEAT. Ohne Artbildung 25 600 Bewertungen, 25 % Fehlschläge. [utexas](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) [utexas](https://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf) [gwern](https://gwern.net/doc/reinforcement-learning/exploration/2002-stanley.pdf) | Stanley und Miikkulainen 2002, Tabelle zur Ablation | begutachtet; Tabellenausschnitt gesehen | gemessen laut Quelle |
| XOR: 100 Läufe, im Mittel 32 Generationen (4 755 Netze, Standardabweichung 2 553), kein Fehlschlag. Auf dem Pendel 25-mal schneller als Cellular Encoding und 5-mal schneller als ESP. [gwern](https://gwern.net/doc/reinforcement-learning/exploration/2002-stanley.pdf) | Stanley und Miikkulainen 2002 | begutachtet; Volltext gesehen | gemessen laut Quelle |
| Neue Struktur senkt anfangs häufig die Fitness. Nicht funktionale Struktur wie bei GNARL wird eventuell nie angebunden. [gwern](https://gwern.net/doc/reinforcement-learning/exploration/2002-stanley.pdf) | Stanley und Miikkulainen 2002, Abschnitt 2.3 | begutachtet; Volltext gesehen | bestätigt |
| Wagner 2008 „Robustness and evolvability: a paradox resolved“, Proc. R. Soc. B 275(1630), S. 91 bis 100 | nur als Literaturangabe in arXiv:2211.02270 gesehen [arxiv](https://arxiv.org/pdf/2211.02270) | ohne Inhaltsprüfung | nicht geprüft |
| Wagner und Altenberg 1996 | nicht abgerufen | ohne Beleg | nicht geprüft |

Belegt ist, dass Evolvierbarkeit vor allem unter wechselnden, modular strukturierten Zielen
entsteht. [arxiv](https://arxiv.org/pdf/1204.5999)
[phys](https://phys.org/news/2008-11-evolution-environments.html) Die Beschleunigung ist bei
Kashtan und Alon angegeben, einen sauberen Faktor haben wir aber nicht gesehen. Für „minimal
starten“ gegen „zufällig starten“ gibt es mit NEAT einen gemessenen Faktor von ungefähr 7.
[gwern](https://gwern.net/doc/reinforcement-learning/exploration/2002-stanley.pdf) Das ist aber
nicht dasselbe wie „neutral starten“, weil NEAT-Knoten nicht exakt neutral eingefügt werden.
[gwern](https://gwern.net/doc/reinforcement-learning/exploration/2002-stanley.pdf) Der genau
neutrale Nullstart ist bei Goyal et al. 2017 nur qualitativ als Verbesserung belegt.
[arxiv](https://ar5iv.labs.arxiv.org/html/1706.02677)

## Eigene Schlüsse, getrennt von den Quellen

* Für den Block ψ' = normiert(ψ − i h H ψ) gilt mit hermiteschem H die Norm
  ‖ψ − i h H ψ‖² = 1 + h²⟨ψ|H²|ψ⟩ (berechnet). Die Normierung ist also zustandsabhängig und
  damit nichtlinear. Die Nichtlinearität tritt aber erst in zweiter Ordnung von h auf. In erster
  Ordnung ist der Block identisch mit exp(−i h H)ψ, also unitär.
* Eine exakt unitäre Alternative ist die Cayley-Form (1 + i h H/2)⁻¹(1 − i h H/2). Das ist eine
  Standardformel, keine Quelle wurde dafür abgerufen.
* Bei Kopplung g = 0 ist H = 0 und der Block ist exakt die Identität. Das entspricht dem
  Gedanken von γ = 0 bei Goyal et al. 2017.
* Ein hermitescher Term g(c†ab + h.c.) koppelt Hin- und Rückreaktion mit derselben Stärke g.
  Eine Gleichgewichtskonstante mit k₊ ≠ k₋ wie im stochastischen Generator ist so nicht
  darstellbar.
* Wendet man einen Sprungoperator L = √k c†ab auf Besetzungszustände an, bleiben diagonale
  Dichtematrizen diagonal, da L jeden Besetzungszustand auf höchstens einen anderen abbildet und
  L†L nur aus Zahloperatoren besteht. Ohne Hamilton-Teil ist die Lindblad-Gleichung auf den
  Besetzungen dann exakt eine klassische Master-Gleichung. Die Rate ist |⟨f|L|i⟩|² (im
  Prüfvermerk berichtigt). Bei Bosonen ist das k · n_A · n_B · (n_C + 1), bei Qubits
  k · n_A · n_B · (1 − n_C). Beides ist nicht Dois Rate k · n_A · n_B, weil der Produktfaktor
  fehlt bzw. sperrt. Das passt zu Heinzens Hinweis, dass kohärente Raten von der
  Produktbesetzung abhängen.
* Mit 0 und 1 als einzigen Besetzungen ist man weit weg vom Grenzfall großer Zahlen. Baez 2013
  zeigt, dass Master- und Ratengleichung nur dort oder für kohärente Zustände übereinstimmen.
* Eine einzelne UND-Bedingung über drei Familien entspricht einem Term in dritter Ordnung. Ein
  Drei-Wellen-Term c†ab erfasst direkt nur zwei Eingänge. Für drei Familien braucht man zwei
  Reaktionen hintereinander oder einen Vier-Wellen-Term.

## Bewertung der Arbeitshypothese

| Teil der Hypothese | Bewertung | Wichtigste Quelle |
|---|---|---|
| Reaktion als Produkt von Erzeugungs- und Vernichtungsoperatoren | belegt (klassisch stochastisch) | Doi 1976; Baez 2013 |
| Umkehrbare Reaktion als hermitescher Drei-Wellen-Term | belegt für bosonische Moden | Heinzen et al. 2000 |
| Familie als Qubit, besetzt oder leer, mit Reaktions-Hamiltonian | eigene Konstruktion, kein Vorbild gefunden | keine |
| Klassische Kontrolle als Master-Gleichung derselben Reaktionsliste | belegt als Formalismus | Doi 1976; Baez 2013 |
| Gleiche Reaktionsliste quantisch und klassisch lesen | teilweise belegt; Ratenkonstanten und Produktfaktoren unterscheiden sich | Baez und Biamonte; Heinzen et al. 2000; eigene Rechnung |
| Restblock als Identität plus Rest, normiert | eigene Konstruktion; QNN-Literatur nutzt LCU mit Post-Selektion oder klassische Addition | Wen et al. 2024; Kashif und Al-Kuwari 2024 |
| Start jeder Kopplung bei 0 | belegt als Prinzip in klassischen ResNets und NEAT | Goyal et al. 2017; Stanley und Miikkulainen 2002 |
| Vererbtes Netz steuert die eigene Variation | belegt als Phänomen unter wechselnden Zielen | Kashtan und Alon 2005; Parter et al. 2008; Crombach und Hogeweg 2008 |
| Massenwirkung aus Lindblad mit Reaktionssprüngen | nicht belegt | keine |

Folgende Quellen sprechen gegen Teile der Hypothese.

* Heinzen et al. 2000 zeigen, dass kohärente Reaktionsdynamik nicht der klassischen Kinetik
  folgt. [arxiv](https://arxiv.org/pdf/cond-mat/0110565v1) Die quantische und die klassische
  Lesart derselben Reaktionsliste werden sich deshalb im Verhalten unterscheiden.
* Baez 2013 zeigt, dass die Ratengleichung nur für kohärente Zustände oder große Zahlen aus der
  Master-Gleichung folgt. [arxiv](https://arxiv.org/pdf/1306.3451) Mit Besetzungen 0 und 1 ist
  diese Bedingung nicht erfüllt.
* Wen et al. 2024 zeigen, dass eine echte quantische Restverbindung eine Post-Selektion mit
  sinkender Erfolgswahrscheinlichkeit braucht.
  [nature](https://www.nature.com/articles/s42005-024-01719-1) Der Normierungsschritt der
  Hypothese ist in einer Simulation billig, auf Hardware wäre er es nicht.
* Kashtan und Alon 2005 zeigen, dass Modularität unter festen Zielen schnell verloren geht.
  [arxiv](https://arxiv.org/pdf/1204.5999) [pnas](https://www.pnas.org/doi/10.1073/pnas.0503610102)
  Bleibt die Aufgabe in den 57 bestandenen Pilotläufen fest, ist nicht zu erwarten, dass eine
  vererbte Variationssteuerung erhalten bleibt. Ein Ergebnis von Arm W liegt im Repository noch
  nicht vor (offen, siehe Prüfvermerk); ob es zu dieser Erwartung passt, ist ungeprüft.
* Stanley und Miikkulainen 2002 warnen, dass nicht funktionale Struktur eventuell nie angebunden
  wird. [gwern](https://gwern.net/doc/reinforcement-learning/exploration/2002-stanley.pdf) Eine
  Kopplung, die exakt bei 0 startet, ist für die Selektion unsichtbar, bis eine Mutation sie
  verändert.

## Empfehlungen

* Die Lesart „Qubit je Familie mit Drei-Wellen-Term“ sollte im Repository als eigene
  Konstruktion gekennzeichnet werden, mit Heinzen et al. 2000 und Doi 1976 als nächstliegenden
  Vorbildern.
* Die klassische Kontrolle sollte ausdrücklich Dois Rate k · n_A · n_B nutzen und die Abweichung
  zur Quantenseite dokumentieren. Sonst vergleicht man zwei Modelle mit verschiedenen
  Produktfaktoren.
* Für die UND-Bedingung über drei Familien sollte ein Block mindestens zwei Reaktionen
  hintereinander zulassen.
* Wenn die Seele ihre Variation selbst steuern soll, spricht die Literatur für wechselnde
  Teilziele statt eines festen Ziels. [arxiv](https://arxiv.org/pdf/1204.5999)

## Nicht bestätigt

* Eine Quelle für A + B ⇌ C als g(c†ab + h.c.) auf Qubits (Besetzung 0 oder 1).
* Eine Quelle, die Massenwirkung aus einer Lindblad-Gleichung mit Reaktionssprungoperatoren
  herleitet, samt Ratenausdruck.
* Eine begutachtete Quelle für „open quantum reaction network“ oder „Lindblad chemical master
  equation“ als feste Begriffe.
* Haken und Strobl 1973 sowie Breuer und Petruccione 2002, nicht abgerufen.
* He et al. 2016 (beide Arbeiten), E 2017, Haber und Ruthotto 2017, Chen et al. 2018, nicht
  abgerufen.
* Der Zeitschriftendatensatz von Baez 2013 (arXiv:1306.3451), nicht geprüft.
* Die Tabellenwerte zu γ = 0 bei Goyal et al. 2017, nicht gesehen.
* Ein Beschleunigungsfaktor bei Kashtan und Alon 2005 und bei Kashtan, Noor und Alon 2007.
* Inhalte von Wagner 2008 und Wagner und Altenberg 1996.
* Die Zeitschriftenfassungen der Vorabdrucke arXiv:2207.09056 und 2110.10721.
* Ein Ergebnis von Arm W im Repository.

## Vorbehalte

* Mehrere Belege stammen nur aus Zusammenfassungen oder Suchausschnitten. Das ist in den
  Tabellen je Zeile vermerkt.
* Die Formeln aus Dois Arbeit stammen aus einem OCR-Text mit Fehlern. Die Indexstellung sollte
  am gedruckten Artikel geprüft werden.
* Kashtan und Alon 2005 konnten wir nicht im Volltext lesen. Die Zuordnung der
  Generationenzahlen zu den Bedingungen ist deshalb unsicher.
* Die Vorabdrucke aus 2026 sind nicht begutachtet und können sich ändern.
