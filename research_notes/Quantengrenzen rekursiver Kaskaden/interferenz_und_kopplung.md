# Interferenz, Kopplung und Quantengrenzen bei komplexer Stimmen-Addition

Kontext: In der Brutmaschine hat jeder Worker eine Phase φ_f pro Hypothesen-Familie. Charakter χ = |Σ_f w_f · e^(iφ_f)|² (komplexe Ablesung). Die Protokolle nutzen den normierten Charakter χ / Σ_f w_f². Sein Mittel ist 1 bei unabhängigen, gleichverteilten Phasen. Mittel 1, Varianz und Maximum F in Abschnitt 1 beziehen sich auf den normierten Charakter. Stimmen einer Stufe addieren sich komplex. Eine Beobachtung zieht Phasen um κ·Δφ zum Beobachter. Stand der Recherche: 2026-11-04.

Hinweis zur Quellenlage: Viele Volltexte waren im Sandbox-Netz nicht abrufbar (z. B. ui.adsabs.harvard.edu, courses.physics.ucsd.edu blockiert). Mehrere Angaben stützen sich deshalb auf Abstracts und Suchauszüge. Das ist bei den Funden vermerkt. Eine spätere Faktenprüfung hat einzelne Angaben in den Quellen nachgelesen und berichtigt. Die Abschnitte "Gaps" geben den Stand der ersten Recherche wieder. Herleitungen mit "Herleitung" sind eigene, elementare Rechnungen. "Eigene Simulation" heißt: ein kurzer Monte-Carlo-Lauf (200 000 Ziehungen, numpy) in dieser Sitzung, keine Publikation.

## 1. Zeigersummen mit Zufallsphasen (Speckle-Statistik), Verschachtelung, kohärent vs. inkohärent

### Takeaway
Der Mittelwert von |Σ w_f e^(iφ_f)|² ist Σ w_f², das Maximum ist (Σ w_f)². Für viele unabhängige Phasen wird die Intensität exponentialverteilt (vollentwickeltes Speckle). Bei wenigen Familien ist sie es nicht: die Streuung ist kleiner und das Maximum hart begrenzt. Eine kohärente Kaskade aus Kaskaden mit unabhängigen Phasen ist mathematisch nur eine größere flache Zeigersumme. Sie bringt keine neue Statistik, sondern nähert sich nur schneller der Exponentialverteilung.

### Cited Findings
- Goodman modelliert Speckle als Summe vieler Zufallszeiger. Annahmen: Betrag und Phase jedes Zeigers sind unabhängig, die Phasen gleichverteilt über 2π. Für sehr viele Zeiger werden Real- und Imaginärteil asymptotisch gaußverteilt (zentraler Grenzwertsatz). Die Intensität folgt dann der negativen Exponentialverteilung, typisch für "vollentwickeltes Speckle" — [Review zu Speckle in OCT, arXiv 2108.13109 (Preprint, folgt Goodman)](https://arxiv.org/pdf/2108.13109)
- Rayleigh-Speckle: zirkular-gaußsches Feld → negative Exponentialverteilung der Intensität; Phase unabhängig vom Betrag und gleichverteilt über 2π — [Roadmap on structured waves, arXiv 2301.05349 (Preprint; erschienen in J. Opt.)](https://arxiv.org/pdf/2301.05349)
- Formel der Intensitätsdichte: p_I(I) = (1/I_s)·exp(−I/I_s), "the well-known negative exponential density for a fully developed speckle pattern" — [Soummer, Ferrari, Aime & Jolissaint, "Speckle noise and dynamic range in coronagraphic images", ApJ 669, 642–656 (2007), arXiv 0706.1739](https://arxiv.org/pdf/0706.1739)
- Goodmans Buch "Speckle Phenomena in Optics" (Roberts & Co., 2007) behandelt dies in Kap. 2 "Random Phasor Sums", §3.2.1 "Large Number of Random Phasors", §3.5 "Partially Developed Speckle" — [Inhaltsverzeichnis, Stanford](https://ee-www.stanford.edu/~goodman/Contents.pdf)
- Nicht-Rayleigh-Speckle gilt entweder als "unterentwickelt" ("under-developed": Summe weniger Streuwellen oder nicht voll randomisierter Wellen) oder als teilkohärent — [Roadmap on structured waves, Abschn. 7 (Bender & Cao), J. Opt. 25, 103001 (2023), arXiv 2301.05349](https://arxiv.org/pdf/2301.05349). Goodman (Appl. Opt. 47(4), A111–A118, 2008) zeigt: Bei endlich vielen Schritten überschätzt die Exponentialverteilung oft die Wahrscheinlichkeit, dass die Intensität eine Schwelle überschreitet — [Suchauszug zu "Speckle with a finite number of steps"](https://vufind-test.katalog.k.utb.cz/SummonRecord/FETCH-LOGICAL-c289t-ef05a8350026eeb58e653660511590d749584f7f3b7ae4c9af89dcf06bbbfb192) (nur Abstract/Snippet gesehen)

Herleitung (elementar, Phasen unabhängig und gleichverteilt):
- S = Σ_f w_f e^(iφ_f). Dann |S|² = Σ_f w_f² + Σ_{f≠g} w_f w_g cos(φ_f − φ_g). Der Erwartungswert von cos(φ_f − φ_g) ist 0. Also E|S|² = Σ w_f². Normierter Charakter hat Mittelwert 1.
- Maximum: Nach der Dreiecksungleichung |S| ≤ Σ |w_f|. Gleichheit nur, wenn alle Phasen gleich sind. Also max |S|² = (Σ w_f)². Normiert: max = (Σ w_f)²/Σ w_f². Bei F gleichen Gewichten ist das genau F.
- Streuung: E|S|⁴ = 2(Σ w_f²)² − Σ w_f⁴. Daraus Var(normierter Charakter) = 1 − Σ w_f⁴/(Σ w_f²)². Bei F gleichen Gewichten: Var = 1 − 1/F. Die Exponentialverteilung hat Var = 1. Sie wird erst für großes F erreicht.
- Kohärent vs. inkohärent: N gleich starke Zeiger mit gleicher Phase geben |S|² = N². Mit Zufallsphasen im Mittel N. Der kohärente Gewinn ist also höchstens Faktor N.

Eigene Simulation (gleiche Gewichte, normierter Charakter):
- F = 2: Var 0,50; F = 3: Var 0,67; F = 5: Var 0,79; F = 10: Var 0,90; F = 50: Var 0,98. Passt zu 1 − 1/F.
- P(Charakter > 3): für F = 2 und F = 3 exakt 0 (Maximum ist F). Für F = 10: 0,046; Exponential-Wert e^(−3) = 0,050.
- Verschachtelt kohärent (K Worker, je F Familien, Summen der Summen, alle Phasen unabhängig): F=3, K=3 → Var 0,89 (= 1 − 1/9); F=3, K=10 → Var 0,97 (= 1 − 1/30). Das ist genau eine flache Summe aus F·K Zeigern.
- Verschachtelt inkohärent (Mittel der K Intensitäten statt Summe der Amplituden), F=3, K=10: Mittel 1,0, Var 0,067. Die Streuung schrumpft um etwa 1/K.

### Inferences
- Eine Kaskade aus Kaskaden ändert die Statistik nicht grundsätzlich, solange die Phasen unabhängig sind. Kohärente Verschachtelung = eine große Zeigersumme → Grenzfall Exponentialverteilung (Mittel 1, Var 1). Mehr Ebenen machen das Ergebnis also zufälliger, nicht stabiler.
- Inkohärente Verschachtelung (Intensitäten mitteln) glättet dagegen. Das entspricht dem "Speckle-Mitteln" in der Optik. Will man stabile Endpunkte, ist das die passende Stellschraube.
- Mit wenigen Familien (F klein) ist der Charakter "unterentwickeltes Speckle": begrenzt (normiert ≤ F) und weniger streuend. Bei F gleichen Gewichten ist Var = 1 − 1/F und der Kontrast (Standardabweichung durch Mittel) √(1 − 1/F). Die Unterschiede zwischen Workern sind dann kleiner als bei vielen Familien, nicht größer. Ein frühes Wirken der Seele lässt sich damit nicht erklären.
- Wenn die Phasen nichts mit der Richtigkeit zu tun haben, erwartet man, dass ein Seelen-Arm nicht systematisch früher besteht als seine Kontrolle. Ein Vergleich nahe 1/2 passt dazu, beweist es aber nicht. Eine Trefferquote nahe 50 % folgt daraus nicht: Bei Gleichstand in Π, auch wenn alle Π null sind, entscheidet die Zahl der Urteile (`src/cascade/soul.rs:400–445`). Die Intensität misst Gleichphasigkeit, nicht Richtigkeit.

### Gaps
- Goodmans Buchtext selbst war nicht abrufbar. Die Formeln sind aus Sekundärquellen und eigener Herleitung bestätigt.
- Keine Quelle gefunden, die verschachtelte Zeigersummen ausdrücklich als "Kaskaden" in Entscheidungssystemen untersucht.

## 2. Phasensynchronisation unter Kopplung (Kuramoto, DeGroot) und Diversitätsverlust

### Takeaway
Das κ-Ziehen der Phasen ist eine Form von Phasenkopplung bzw. Konsensdynamik. Wiederholte Beobachtungen gleichen die Phasen derselben Familie zwischen Workern an. Den Charakter eines einzelnen Workers hebt das nicht, denn er hängt an der Lage seiner eigenen Familien zueinander. Sind die Seelen aller Richter gleich, ordnet die Abstimmung die Antworten wie die bloße Zählung. Ensemble-Theorie und Experimente zeigen: Ohne Vielfalt der Vorhersagen gibt es keinen Ensemble-Vorteil, und sozialer Einfluss kann die "Weisheit der Vielen" schwächen. Ob die Kopplung der Seelen die Vielfalt der Vorhersagen mindert, ist eine offene, prüfbare Frage.

### Cited Findings
- Acebrón, Bonilla, Pérez Vicente, Ritort, Spigler: "The Kuramoto model: A simple paradigm for synchronization phenomena", Rev. Mod. Phys. 77, 137–185 (2005). Review zu Theorie, Numerik, Varianten und Anwendungen — [DOI 10.1103/RevModPhys.77.137](https://doi.org/10.1103/REVMODPHYS.77.137); [offene Kopie Univ. Barcelona](https://diposit.ub.edu/dspace/handle/2445/12768)
- Strogatz: "From Kuramoto to Crawford", Physica D 143, 1–20 (2000), Review über 25 Jahre Kuramoto-Forschung — [ScienceDirect/ADS-Eintrag via Typeset-Zusammenfassung](https://typeset.io/papers/from-kuramoto-to-crawford-exploring-the-onset-of-3wpvjokr2x)
- Kuramotos exakte Formel für die kritische Kopplung: K_c = 2/(π g(0)), g(ω) = Verteilung der Eigenfrequenzen, gilt für gerade, unimodale g — [Typeset-Zusammenfassung von Strogatz 2000](https://typeset.io/papers/from-kuramoto-to-crawford-exploring-the-onset-of-3wpvjokr2x); [Dorogovtsev et al., Critical phenomena in complex networks, arXiv 0705.0010 (erschienen in Rev. Mod. Phys. 80, 2008)](https://arxiv.org/pdf/0705.0010); [Bedingung "even and unimodal", Kuramoto-Daido-Text](https://www.wpi-aimr.tohoku.ac.jp/chiba/paper/kuramoto_daido.pdf)
- Ordnungsparameter r e^(iψ) = (1/N) Σ_j e^(iθ_j). Im inkohärenten Zustand bei endlichem N fluktuiert r wie 1/√N; für große N gilt √N⟨r⟩ = √π/2 — [arXiv 2511.03700 (Preprint)](https://arxiv.org/pdf/2511.03700); [arXiv 2310.20048 (Preprint)](https://arxiv.org/pdf/2310.20048)
- Selbstkonsistente Lösung (für Lorentz-Frequenzverteilung) r = √(1 − K_c/K), also r ∝ √(K − K_c) nahe K_c — [Nordenfelt, arXiv 1410.2849 (Preprint)](https://ar5iv.labs.arxiv.org/html/1410.2849) (die Quelle nennt die Lorentz-Verteilung g(ω) = γ/(π(γ² + ω²)) selbst, S. 2, Gl. 5)
- DeGroot, "Reaching a Consensus", J. Am. Stat. Assoc. 69(345), 118–121 (1974): Agenten mitteln wiederholt gewichtet die Meinungen ihrer Nachbarn — [Wikipedia: DeGroot learning](https://en.wikipedia.org/wiki/DeGroot_learning)
- Golub & Jackson, "Naïve Learning in Social Networks and the Wisdom of Crowds", AEJ: Micro 2(1), 112–149 (2010): In großen Netzen nähert sich der Konsens der Wahrheit, wenn kein Einzelner dauerhaft großen Einfluss behält — [Wikipedia: DeGroot learning](https://en.wikipedia.org/wiki/DeGroot_learning); [Preprint-Kopie](https://webarchive.lib.cuhk.edu.hk/20210324140943mp_/http:/www.cse.cuhk.edu.hk/~cslui/CMSC5734/naivelearning.pdf)
- Ein einziger "sturer" Agent kann im DeGroot-Rahmen den Konsens auf jeden beliebigen Wert ziehen (Amir et al., J. Econ. Theory 2025) — [Wikipedia: DeGroot learning](https://en.wikipedia.org/wiki/DeGroot_learning) (nur Sekundärquelle gesehen)
- Lorenz, Rauhut, Schweitzer, Helbing, PNAS 108(22), 9020–9025 (2011): 144 Probanden; schon milder sozialer Einfluss schwächt den Wisdom-of-Crowd-Effekt. Wissen über andere Schätzungen verringert die Vielfalt, ohne den kollektiven Fehler zu senken — [PMC3107299](https://pmc.ncbi.nlm.nih.gov/articles/PMC3107299)
- Gegenbefund: Becker, Brackbill, Centola (PNAS 2017): In dezentralen Netzen kann Einfluss die Gruppengenauigkeit verbessern — [PMC5495222](https://pmc.ncbi.nlm.nih.gov/articles/PMC5495222)
- Krogh & Vedelsby (NIPS 7, 1995): Ensemble-Fehler = mittlerer Einzelfehler − "Ambiguity" (Streuung um das Ensemble-Mittel). Die Ambiguity ist ≥ 0, also ist das gemittelte Ensemble nie schlechter als der Durchschnitts-Member (Regression, quadratischer Verlust). Für Mehrheitswahl (0/1-Verlust) gilt diese Zerlegung nicht. Dort kann das Ensemble sogar schlechter sein als der Durchschnitts-Member (arXiv 1803.00314, Abschn. 2.1; Wood et al., Theorem 10). Für die Kreuzentropie gibt es eine analoge exakte Zerlegung, mit dem normierten geometrischen Mittel als Kombinierer (Wood et al., JMLR 24, 2023, Prop. 3 und Kor. 9) — [arXiv 1803.00314 (Preprint)](https://arxiv.org/pdf/1803.00314.pdf); [Wood et al., Unified Theory of Diversity, arXiv 2301.03962 (Preprint; JMLR 2023)](https://arxiv.org/pdf/2301.03962)

### Inferences
- κ entspricht grob einer diskreten Kopplungsstärke. Die Beobachtung wirkt in beide Richtungen: Der Beobachtete rückt um κ·Δ zum Beobachter, der Beobachter um κ·Δ/2 zum Beobachteten, beide vom Zustand vor der Beobachtung aus (`src/cascade/soul.rs:627–646`). Bei κ = 1 übernimmt der Beobachtete die alten Phasen des Beobachters, aber der Beobachter bewegt sich ebenfalls. Ohne Rundung und Phasenumbruch bleibt danach die Lücke Δ·(1 − 3κ/2), bei κ = 1 also −Δ/2 (eigene Rechnung). Ganz schließt sich die Lücke eines Paares in einem Schritt nur bei κ = 2/3. Bei identischen "Eigenfrequenzen" (hier: keine Eigendynamik der Phasen) und ohne Rauschen gibt es keine Schwelle K_c, die Kopplung zieht immer Richtung Gleichtakt. Eine Schwelle entsteht nur, wenn etwas die Phasen auseinandertreibt (Frequenzstreuung, Rauschen, Mutation). Kuramotos Formel K_c = 2/(π g(0)) gilt für Frequenzstreuung ohne Rauschen. Mit Rauschen der Stärke D gilt K_c = 2D bei gleichen Frequenzen und K_c = 2D + 2γ bei Lorentz-Breite γ (Acebrón et al. 2005, Abschn. III.C).
- Die Kopplung gleicht dieselbe Familie zwischen Workern an. Sie gleicht nicht die Familien innerhalb eines Workers an. Darum hebt sie den Charakter eines Workers nicht: Zwei Worker mit gleichen Phasen (0, π) und gleichen Stärken haben beide den Charakter 0. Gleiche Seelen geben auf derselben Aufgabe aber gleiche Stimmzeiger v_j (`vote_phasor`, `src/cascade/soul.rs:114`). Dann ist Π(x) proportional zum Quadrat der Urteile für x, und die Abstimmung ordnet die Antworten wie die bloße Zählung (eigene Folgerung aus `array_vote`, `src/cascade/soul.rs:400–445`). Synchrone Seelen nehmen der Seele also ihren Einfluss auf die Abstimmung.
- Nach Krogh & Vedelsby ist die Vielfalt der Vorhersagen der Hebel, mit dem ein gemitteltes Ensemble seinen Durchschnitts-Member schlägt. Gleiche Seelen erzwingen aber keine gleichen Vorhersagen, denn die Antworten hängen auch an Prompts und Traits. Ob die Kopplung den Ensemble-Vorteil der Brutmaschine mindert, ist darum eine Hypothese, die ein Versuch prüfen muss.
- Praktische Folge: Wenn Diversität erhalten bleiben soll, braucht das System einen Gegenspieler zur Kopplung (Phasenrauschen/Mutation, kleines κ, begrenzte Beobachtungsnetze) — analog zur Frequenzstreuung im Kuramoto-Modell.

### Gaps
- Strogatz 2000 und Acebrón 2005 im Volltext nicht gelesen (Hosts blockiert). Formel K_c und r-Skalierung aus Sekundärquellen.
- Kein Paper gefunden, das Kuramoto-Kopplung direkt mit Ensemble-Genauigkeit bei Abstimmungen verknüpft.

## 3. Quantenkognition / Quanten-Entscheidungstheorie

### Takeaway
Quantenkognition nutzt Hilbertraum-Wahrscheinlichkeiten, um Interferenz- und Reihenfolgeeffekte im menschlichen Urteil zu beschreiben. Es gibt eine bemerkenswerte parameterfreie Vorhersage (QQ-Gleichung) mit guter Bestätigung in Umfragen, aber auch ernste Kritik. Pothos & Busemeyer halten Quanten-Inferenz darüber hinaus für vielleicht das beste praktische rationale Schema. Kritiker in den Kommentaren zu ihrem Artikel halten das für unbelegt oder sehen nur nachträgliche Beschreibungen. Ein Mechanismus für bessere Entscheidungen ist nicht nachgewiesen.

### Cited Findings
- Pothos & Busemeyer, "Can quantum probability provide a new direction for cognitive modeling?", Behav. Brain Sci. 36(3), 255–274 (2013). Sie argumentieren, dass Quantenwahrscheinlichkeit manche Befunde besser erklärt als klassische (bayessche) Wahrscheinlichkeit — [City Research Online](https://openaccess.city.ac.uk/id/eprint/2428/); Antwort auf die Kommentare: "Quantum principles in psychology: The debate, the evidence, and the future", BBS 36(3), 310–327 (2013), doi 10.1017/S0140525X12003226 — [City Research Online](https://openaccess.city.ac.uk/id/eprint/2429/)
- Zwei der kritischen Kommentare im selben Heft: Behme (2013, BBS 36(3), 279–280, doi 10.1017/S0140525X12002889) sieht in den vereinfachten Szenarien höchstens den Nachweis, dass QP eine logische Möglichkeit mit anderen Vorhersagen ist. Navarro & Fuss (2013, BBS 36(3), 299–300, doi 10.1017/S0140525X12003056) fragen nach der Mechanik — [Autoren-PDF mit Artikel, Kommentaren und Antwort](https://jbusemey.pages.iu.edu/quantum/BBSQP2012.pdf)
- Busemeyer & Bruza, "Quantum Models of Cognition and Decision", Cambridge University Press 2012 (Standard-Lehrbuch; Bibliographie, Inhalt nicht neu geprüft) — Gap: kein eigener Link abgerufen.
- Wang & Busemeyer, Topics in Cognitive Science 5(4), 689–710 (2013): Quantenmodell für Fragereihenfolge-Effekte mit a priori, parameterfreier Vorhersage ("QQ-Gleichung"). Bayes- und Markov-Modelle erfüllen sie im Allgemeinen nicht — [Autoren-PDF](https://jbusemey.pages.iu.edu/quantum/QuestOrdEff.pdf)
- Wang, Solloway, Shiffrin, Busemeyer, PNAS 111(26), 9431–9436 (2014): QQ-Gleichung getestet an 70 repräsentativen Umfragen mit je 651–3 006 Teilnehmenden und an zwei Laborexperimenten — [PMC4084470, doi 10.1073/pnas.1407756111](https://pmc.ncbi.nlm.nih.gov/articles/PMC4084470/). Die Umfragen stammen von Gallup und Pew aus den Jahren 2001–2011 — [OSU-Pressemitteilung](https://news.osu.edu/quantum-theory-reveals-puzzling-pattern-in-how-people-respond-to-some-surveys/). Eine spätere Analyse zählt dieselben Daten als 73 Fragepaare, und die Gleichung hält in 72 davon — [Dzhafarov et al., arXiv 1504.07422 (Preprint)](https://arxiv.org/pdf/1504.07422). Die Zahlen passen zusammen. Die 70 Umfragen und die zwei Laborexperimente erfüllen die Gleichung. Dazu kommt das Rose-Jackson-Paar mit eingeschobener Information, das sie verletzt.
- Grenzen: Die Gleichung gilt nur, wenn zwei Fragen direkt hintereinander ohne neue Information kommen. Mit eingeschobener Information hält sie nicht — [OSU-Pressemitteilung](https://news.osu.edu/quantum-theory-reveals-puzzling-pattern-in-how-people-respond-to-some-surveys/)
- Dzhafarov et al. finden keine echte Kontextualität in den untersuchten Verhaltensdaten (Titel: "Is there contextuality in behavioral and social systems?") — [arXiv 1504.07422 (Preprint; erschienen in Phil. Trans. R. Soc. A 2016)](https://arxiv.org/pdf/1504.07422) (nur Abstract/Auszug gesehen)
- Boyer-Kassem, Duchêne, Guerci, Theory and Decision 81, 479–510 (2016): "Quantum-like models cannot account for the conjunction fallacy". Ihre experimentellen Ergebnisse widersprechen den Vorhersagen der getesteten Quantenmodelle — [arXiv 1606.03430](https://pdf.arxiv.org/pdf/1606.03430); [Tilburg](https://research.tilburguniversity.edu/en/publications/quantum-like-models-cannot-account-for-the-conjunction-fallacy/)
- Erweiterungen (POVM) können die QQ-Gleichung verletzen (Lebedev & Khrennikov) — [arXiv 1811.00045 (Preprint)](https://arxiv.org/abs/1811.00045v1)
- Yukalov & Sornette, Quantum Decision Theory: Wahrscheinlichkeit = Nutzenfaktor + Attraktionsfaktor (Interferenzterm). Attraktionsfaktoren summieren sich zu null. "Quarter law": nicht-informativer Prior für den mittleren Betrag des Attraktionsfaktors = 1/4 — [Favre et al., arXiv 1602.04058](https://arxiv.org/pdf/1602.04058)
- Empirie zum Quarter Law: gute Übereinstimmung auf Gruppenebene, auf Individualebene Verfeinerung nötig — [Favre et al., arXiv 1602.04058 (Preprint; Suchauszug)](https://arxiv.org/pdf/1602.04058)

### Inferences
- In Quantenkognition beschreibt der Interferenzterm, wie Menschen vom klassischen Gesetz der totalen Wahrscheinlichkeit abweichen. Pothos & Busemeyer (2013, Abschn. 5) gehen weiter: Quanten-Inferenz sei vielleicht das beste praktische rationale Schema, wenn man nicht allen Möglichkeiten zugleich Wahrscheinlichkeiten geben kann. Kritiker im selben Heft halten das für unbelegt oder sehen nur nachträgliche Beschreibungen (z. B. Navarro & Fuss; Houston & Wiesner; Kaznatcheev & Shultz). Ein Genauigkeitsvorteil ist nicht nachgewiesen. Für die Brutmaschine folgt daraus kein Genauigkeitsversprechen.
- Der Brutmaschine-Charakter |Σ w e^(iφ)|² ähnelt formal dem Born-Betragsquadrat. Anders als in Quantenkognition gibt es aber keine Projektion, keine Normierung über Alternativen und keine inkompatiblen Messungen. Die Analogie ist also nur oberflächlich.
- Ein Reihenfolgeeffekt existiert in der Brutmaschine trotzdem: Beobachtungen ziehen Phasen, also hängt das Ergebnis von der Reihenfolge der Beobachtungen ab. Das ist aber ein klassischer, dynamischer Effekt.

### Gaps
- Busemeyer & Bruza 2012 und Pothos & Busemeyer im Volltext nicht gelesen.
- Keine Metaanalyse gefunden, die die empirische Gesamtstützung quantifiziert.

## 4. Quantengrenzen (Tsirelson, CHSH) und "Schrödinger" auf klassischen Rechnern

### Takeaway
Eine klassisch simulierte Phasenmechanik ist ein klassisches Modell. Sie hat keinen Quantenvorteil. Die Tsirelson-Grenze 2√2 betrifft getrennte Messstationen ohne Kommunikation. Innerhalb eines einzigen Programms gilt sie gar nicht, man kann jede Korrelation direkt hinschreiben. Eine echte "Schrödinger-Grenze" gibt es für dieses System nicht. Die bekannte Grenze ist nur: Quantenzustände vieler verschränkter Teile brauchen klassisch exponentiell viel Speicher — das betrifft die Brutmaschine nicht, weil sie keinen Tensorproduktraum benutzt.

### Cited Findings
- Cirel'son (Tsirelson), "Quantum generalizations of Bell's inequality", Lett. Math. Phys. 4(2), 93–100 (1980): maximale Quantenverletzung der CHSH-Ungleichung = 2√2; lokal-klassisch ist die Grenze 2 — [Wikipedia: Tsirelson's bound](https://en.wikipedia.org/wiki/Tsirelson%27s_bound); [Originaltext (Information Philosopher)](https://www.informationphilosopher.com/solutions/scientist/bell/Tsirelson.pdf)
- 2√2 wird mit antikommutierenden Observablen an maximal verschränkten Qubits erreicht — [arXiv 2603.28973 (Preprint)](https://arxiv.org/pdf/2603.28973)
- Für Korrelations-Bell-Ungleichungen ist das Verhältnis Tsirelson-Grenze zu lokaler Grenze durch die Grothendieck-Konstante beschränkt (Tsirelson 1987) — [Wikipedia: Tsirelson's bound](https://en.wikipedia.org/wiki/Tsirelson%27s_bound)
- Feynman, "Simulating physics with computers", Int. J. Theor. Phys. 21, 467–488 (1982): Ausgangspunkt der Idee, dass Quantensysteme klassisch schwer zu simulieren sind — [Springer](https://link.springer.com/article/10.1007/BF02650179) (Volltext hinter Paywall; das Exponential-Argument dort nicht selbst geprüft)
- Spreeuw, "Classical wave-optics analogy of quantum-information processing", Phys. Rev. A 63, 062302 (2001): Klassische Wellen können Interferenz und eine Analogie von Verschränkung nachbilden. Quanten-Nichtlokalität ist der wesentliche Bestandteil eines Quantencomputers. Die nötigen Ressourcen (Strahlen, Bauteile) wachsen exponentiell mit der Zahl der "cebits" — [arXiv quant-ph/0009066](https://arxiv.org/abs/quant-ph/0009066)
- Bhattacharya, van Linden van den Heuvell, Spreeuw, "Implementation of quantum search algorithm using classical Fourier optics", Phys. Rev. Lett. 88, 137901 (2002): Grover-Suche mit klassischer Optik; Verschränkung ist für den Algorithmus nicht nötig, nur Welleneigenschaften — [arXiv quant-ph/0110034](https://arxiv.org/abs/quant-ph/0110034)

### Inferences
- Bell/CHSH-Grenzen sind Aussagen über räumlich getrennte Systeme ohne Signalaustausch. In einer Simulation teilen alle Worker denselben Speicher. Deshalb ist dort jede Korrelation erlaubt, auch über 2√2 hinaus. Tsirelson ist für die Brutmaschine keine Grenze und auch kein erreichbarer "Vorteil".
- Die Brutmaschine hat F reelle Phasen pro Worker, also lineare Größe. Echte Quantenzustände von n Teilen haben 2^n Amplituden. Diese exponentielle Struktur fehlt. Darum gibt es weder Quantenbeschleunigung noch eine besondere Rechenhürde. Es gelten nur gewöhnliche Rechengrenzen.
- Interferenz allein ist klassisch (Wellenoptik). Mit klassischen Wellen braucht Grovers Suche in N Einträgen etwa √N Schritte, ganz ohne Verschränkung (Bhattacharya et al. 2002). Der Aufwand an Moden, Strahlen oder Bauteilen wächst aber exponentiell mit der Zahl der Bits. Darum bleibt die Datenbank klein. Nach Spreeuw (2001) ist Quanten-Nichtlokalität der wesentliche Bestandteil eines Quantencomputers.
- "Grenze aus Schrödinger": Der nächstliegende echte Inhalt ist (a) Unitarität — die Gesamtnorm bleibt erhalten, Interferenz verteilt Gewicht nur um, erzeugt keins; bei der Brutmaschine entspricht dem die harte Schranke max = (Σw)² und Mittel = Σw²; (b) die Speckle-Statistik. Eine darüber hinausgehende Grenze ist nicht bekannt.

### Gaps
- Keine Quelle gefunden, die explizit "Schrödinger-Grenze" für rekursive Kaskaden nennt. Der Begriff hat in der Literatur keine feste Bedeutung in diesem Kontext.
- Popescu-Rohrlich-Boxen (CHSH-Wert 4, nicht-signalisierend) nicht separat recherchiert.

## 5. Bekannte Ergebnisse: interferenzbasiertes Abstimmen und Genauigkeit

### Takeaway
Ich habe kein peer-reviewtes Ergebnis gefunden, das zeigt, dass komplexe (interferierende) Stimmen-Addition die Genauigkeit gegenüber normaler Mittelung oder Mehrheitswahl verbessert. Die verwandten gesicherten Ergebnisse sprechen eher dafür, dass Genauigkeit von unabhängiger, wahrheitsbezogener Information und von Vielfalt kommt, nicht von Phasen.

### Cited Findings
- Ensemble-Vorteil hängt an Vielfalt (Ambiguity-Zerlegung) — [arXiv 1803.00314](https://arxiv.org/pdf/1803.00314.pdf)
- Sozialer Einfluss kann Vielfalt verringern ohne Genauigkeit zu steigern — [Lorenz et al. 2011, PMC3107299](https://pmc.ncbi.nlm.nih.gov/articles/PMC3107299); unter bestimmten Netzstrukturen auch umgekehrt — [Becker et al. 2017, PMC5495222](https://pmc.ncbi.nlm.nih.gov/articles/PMC5495222)
- Quantenkognition modelliert menschliche Abweichungen von klassischer Wahrscheinlichkeit. Pothos & Busemeyer (Abschn. 5) halten Quanten-Inferenz zudem für vielleicht das beste praktische rationale Schema — [Pothos & Busemeyer 2013](https://openaccess.city.ac.uk/id/eprint/2428/). Kritiker im selben Heft halten das für unbelegt — [Autoren-PDF mit den Kommentaren](https://jbusemey.pages.iu.edu/quantum/BBSQP2012.pdf). Ein Genauigkeitsvorteil ist nicht nachgewiesen.

### Inferences
- Bei zufälligen Phasen, die nichts mit der Richtigkeit einer Hypothese zu tun haben, ist |Σ|² eine Zufallsgröße mit Mittel Σw² (normiert 1). Sie trägt dann keine Information über die Wahrheit. Daraus folgt keine Trefferquote nahe 50 %: Bei einstimmigen Urteilen gewinnt weiter die richtige Antwort, und bei Gleichstand in Π entscheidet die Zahl der Urteile (`src/cascade/soul.rs:400–445`). Ein Vergleich „Seele gegen Kontrolle“ nahe 1/2 passt zu Phasen ohne Information, schließt aber keinen Fehler im Pilot aus. Prüfen lässt sich das nur im Vergleich mit der Kompetenz der Worker und mit einer Abstimmung ohne Phasen.
- Interferenz kann nur dann helfen, wenn Phasen systematisch mit Qualität verknüpft sind (z. B. richtige Hypothesen tendieren zu gleicher Phase, falsche zu zufälliger). Dann gilt: richtige Signale wachsen kohärent wie N², Rauschen nur wie N. Das ist das Prinzip der kohärenten Signalintegration. Ob die Brutmaschine eine solche Verknüpfung hat, muss sie selbst zeigen.
- Kopplung (κ) verknüpft Phasen mit Beobachtungsreihenfolge, nicht mit Wahrheit. Sie erzeugt Kohärenz, aber nicht unbedingt korrekte Kohärenz.
- Testvorschlag (abgeleitet): Kontrollbedingung mit inkohärenter Addition (Summe der Intensitäten) und mit gleichverteilten Zufallsphasen ohne Vererbung. Liegt der Endpunkt gleich, dann trägt die Phase keine Information.

### Gaps
- Keine Literatur gefunden zu "complex-valued voting", "phasor voting" oder Condorcet-Jury-Theorem mit komplexen Stimmen. Condorcet nicht separat recherchiert.
- Komplexwertige neuronale Netze (Phasen als Merkmale) nicht recherchiert; dort gibt es eventuell verwandte Ergebnisse.
