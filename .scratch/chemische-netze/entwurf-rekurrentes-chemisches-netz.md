# Entwurf: ein rekurrentes chemisches Netz nach dem Vorbild des Gehirns

Entwurfspapier vom 2026-11-04. Nichts ist gebaut, und nichts ist entschieden. Die Karte dieser
Arbeit ist `map.md`, die Forschungsnotiz ist `research.md`, die Tickets liegen in `issues/`.

**Auftrag.** Laut Auftrag des Workflows schrieb der Eigentümer (englisch, wörtlich): „ais are
linear. but not really intertwined like our brain. model the keras network of our brain
chemically“. Wir lesen das so: Heutige KI-Netze sind meist geschichtete Vorwärtsstapel. Das
Gehirn ist dicht rückgekoppelt, und seine Synapsen arbeiten chemisch. Gewünscht ist ein Netz nach
der Chemie des Gehirns, rekurrent und verflochten, mit einem Weg zu Keras. Das Zitat steht nur im
Auftrag des Workflows, nicht in der weitergeleiteten Frage der Nutzerin bzw. des Nutzers. Wir
konnten es nicht prüfen.

**Grundlage.** Dieses Papier fasst drei Literaturberichte, drei Entwürfe und zwei Gutachten vom
2026-11-04 zusammen. Die Berichte behandeln den Stand des Repositorys, Chemie, Biophysik und
Training sowie Verschaltung und Chemie des Gehirns. Die Entwürfe heißen „Biophysik zuerst“
(Entwurf 1), „Chemie zuerst“ (Entwurf 2) und „Brutmaschine zuerst“ (Entwurf 3). Keine dieser
Vorlagen liegt im Repository. Abschnitt 9 nennt deshalb jede genutzte Quelle vollständig.

**Konventionen.**

- **Quellen.** [R39] ist Quelle 39 von `research.md`, Abschnitt 10. [Q12] ist Quelle 12 dieses
  Papiers, Abschnitt 9.3. Ein Ort wie [R39, Gl. 50] bezieht sich auf die gelesene Fassung, die
  die Quellenliste nennt.
- **Vorabdrucke.** Jede Nutzung eines Vorabdrucks sagt es.
- **Repository.** Datei:Zeile verweist auf den Stand von Commit `0e58853`. Einige Zeilenangaben in
  `map.md` und `research.md` sind älter, etwa `src/config.rs:34-67` für die Limits, die heute in
  `src/config.rs:71-107` stehen.
- **Eigenes.** „Nach unserer Ableitung“, „nach unserer Rechnung“, „unsere Schätzung“ und „unser
  Vorschlag“ markieren eigene Arbeit. „Nachgerechnet“ heißt: Wir haben es am 2026-11-04 mit
  numpy 2.5.3 unter Python 3.13.18 im Speicher gerechnet, ohne eine Datei zu schreiben.
- **Weitergegeben.** „Laut Hirn-Bericht“ und „laut Chemie-Bericht“ heißt: Die Angabe stammt aus
  einem der Berichte, und wir haben sie nicht neu geprüft.
- **Zahlen.** Dezimalpunkt und Zehnerpotenzen wie 2e10, nach `docs/SCHREIBSTIL.md`.

## 1. Kurzfassung

Heutige KI-Netze sind nicht linear, aber meist geschichtet und vorwärtsgerichtet, während das
Gehirn flach, dicht rückgekoppelt und chemisch ist. Wir schlagen ein Reaktionsnetz vor, dessen
Knoten Dual-Rail-Paare sind: Synapsen sind Katalyse, Hemmung ist Auslöschung, und ein einziger
Treibstoffgradient Δμ treibt das Netz. Jede Reaktion ist reversibel, folgt der Massenwirkung und
hat Raten aus ganzzahligen Energien, sodass Positivität, Detailed Balance und die Energiebilanz für
jedes Genom gelten und das Netz ohne Antrieb nachweislich nichts rechnet. Eine zweite Stufe ersetzt
die sofortige Synapse durch elementare Synapsenchemie mit gemessenen Zeitskalen von 0.1 ms bis
etwa 1 s, und sie überträgt nur, solange ein zweiter Treibstoff das Recycling antreibt. Die
Verdrahtung folgt Konnektom-Statistik und wird gegen Zufallsgraph und Konfigurationsmodell
geprüft. Der Weg nach Keras ist eine eigene RNN-Zelle mit dem JAX-Backend auf der CPU in float64,
der Weg in die Brutmaschine eine dritte Modellart `crn` ohne neue Crate. Der kleinste Prototyp prüft
in Keras, ob das Netz ein Bit gegen Störungen hält, ob dieses Gedächtnis bei Δμ = 0 verschwindet
und ob es seinen azyklischen Zwilling schlägt. Der vorregistrierbare Endpunkt misst im Kern auf
motif-1d gegen den azyklischen Zwilling und gegen einen CNN-Lauf mit TCN-Basismodell bei gleichem
Budget.

## 2. Warum heutige Netze anders sind als das Gehirn

### 2.1 Nicht linear, aber geschichtet

- **Heutige Netze sind nichtlinear.** Ein ReLU-Netz berechnet genau die stetigen, stückweise
  affinen Funktionen [R74, Thm. 2.1]. Ohne Nichtlinearität fiele ein Stapel linearer Schichten zu
  einer einzigen linearen Abbildung zusammen (unsere Anmerkung). Das Wort „linear“ trifft also
  nicht zu.
- **Der Transformer ist ein Stapel.** Er verzichtet „auf Rekurrenz und Faltungen ganz“ [Q1,
  Abstract]. Encoder und Decoder bestehen aus je N = 6 gleichen Schichten [Q1, Abschn. 3.1]. Jede
  Schicht enthält das Netz max(0, x·W1 + b1)·W2 + b2 [Q1, Gl. 2].
- **Rückgekoppelt ist nur die Erzeugung.** Der Decoder nimmt die schon erzeugten Symbole als
  zusätzliche Eingabe [Q1, Abschn. 3]. Innerhalb eines Schritts läuft die Rechnung durch eine feste
  Folge von Schichten (unsere Lesart).
- **Rekurrente Netze gibt es.** LSTM und GRU galten 2017 als Stand der Technik für Folgen [Q1,
  Abschn. 1]. Zeitkontinuierliche Zellen wie LTC und CfC sind ebenfalls rekurrent [Q2, Q3]. Keines
  dieser Netze ist chemisch.
- **Das Repository.** MLP und CNN sind azyklisch. Der TCN-Block ist kausal, hat aber keine
  Rückkopplung (`src/cnn/genome.rs:244-247`). Kaskaden laufen Stufe für Stufe vorwärts
  (`CONTEXT.md:172`). Der Name „rnn“ erscheint nur in drei Tests als Beispiel einer unbekannten
  Modellart (`src/config.rs:636`, `src/main.rs:1615`, `src/hidden_eval.rs:1280`). Ein Verbot
  rekurrenter Modelle steht nirgends.

### 2.2 Das Gehirn ist rückgekoppelt und flach

- **Fliege.** 93.3 % der Neurone liegen in einer einzigen stark zusammenhängenden Komponente. Der
  mittlere gerichtete kürzeste Weg ist 4.42 Schritte, und die Autoren nennen das Gehirn „relativ
  flach“ (laut Hirn-Bericht nach [Q44]).
- **Gegenseitigkeit.** In der Fliege wird eine Verbindung mit der Wahrscheinlichkeit 0.138
  erwidert. Das ist Faktor 858 über dem Zufallsgraphen nach Erdős und Rényi (ER) und Faktor 43.8
  über dem Konfigurationsmodell (CFG), das die Grade erhält [Q44, Tab. 2]. Im Wurm sind es 0.372,
  Faktor 10.4 und Faktor 5.03 [Q44, Tab. 2].
- **Kortex.** Die Mehrheit der Synapsen im Kortex stammt von kortikalen Neuronen [Q30]. In
  Schicht 4 der Katze kommen nur etwa 6 % der erregenden Synapsen aus dem Thalamus [Q31].
  Funktionell liefert der Thalamus in Schicht 4 der Maus ein Drittel der Erregung [Q32]. Die
  lokalen Schaltkreise verstärken diesen Eingang etwa um den Faktor 3 [Q33]. Im Makaken liegen
  80 % der markierten Eingangsneurone im eigenen Areal [Q34]. Alle vier Angaben stammen aus dem
  Hirn-Bericht.
- **Lesart.** Ein Kortexneuron hört vor allem auf seine Nachbarn, nicht auf eine Schicht davor
  (unsere Lesart).

### 2.3 Synapsen sind Chemie mit eigenen Zeitskalen

- **Bindung.** Transmitter und Rezeptor bilden reversibel einen Komplex [Q9]. Glutamat erreicht im
  Spalt eine Spitze von 1.1 mM und fällt mit τ = 1.2 ms ab [Q12].
- **Zeitskalen.** AMPA-Ströme zerfallen mit 4 bis 8 ms, NMDA-Ströme mit 60 bis 150 ms [Q14].
  GABA_B aktiviert mit 45.2 ms und inaktiviert mit 110.2 ms (Anteil 84 %) und 516.2 ms
  (Anteil 16 %) [Q15]. Synaptische Ressourcen erholen sich in 450 bis 800 ms [Q13]. Langsame
  Übertragung über Second Messenger dauert hunderte Millisekunden bis Minuten [Q17]. Diese Zahlen
  stammen aus den Berichten.
- **Gedächtnis in Synapsen.** Arbeitsgedächtnis kann in calciumvermittelter Fazilitation liegen
  [Q18]. Trainierte RNNs mit Kurzzeitplastizität halten Information auch ohne Aktivität, wenn die
  Verzögerung kurz ist [Q19].
- **Die Lücke.** Ein Skalargewicht kennt keine dieser Zeitskalen (unsere Lesart). Die Autoren von
  CfC nennen drei Mechanismen, die tiefe Netze weglassen: kontinuierliche Dynamik, Synapsen als
  Freisetzung und Rezeptoraktivierung statt Skalargewicht, sowie Rückkopplung und Gedächtnis [Q3,
  S. 3-4, laut Chemie-Bericht].

### 2.4 Eine chemische Schicht ohne Draht

- **Peptide und Monoamine.** Im Wurm hat das Netz der Neuropeptide die Dichte 0.34 bis 0.44, das
  synaptische Netz nur 0.025 [Q42]. Monoamine senden über wenige Knoten mit vielen Verbindungen
  [Q41]. Neuromodulatoren konfigurieren Schaltkreise um [Q60].
- **Anatomie ist nicht Funktion.** Im Wurm sagt die Anatomie die gemessene Signalausbreitung
  schlecht voraus [Q40]. In der Fliege gelingen konnektomgestützte Modelle [Q54, Q55].

### 2.5 Vorzeichen, Dendriten und Energie

- **Das Vorzeichen setzt der Rezeptor.** Das glutamaterge Sinnesneuron AWC erregt das Interneuron
  AIB über AMPA-artige Rezeptoren. Dasselbe Neuron hemmt AIY über glutamatgesteuerte
  Chloridkanäle [Q20]. Das Prinzip von Dale betrifft also die Identität des Transmitters, nicht das
  Vorzeichen. Ein Vorzeichen je Neuron ist eine Modellvereinfachung, wie sie etwa Song, Yang und
  Wang für trainierte RNNs nutzen [Q64].
- **Ein Neuron ist ein kleines Netz.** Ein Modell einer CA1-Zelle verhielt sich wie ein
  zweischichtiges Netz [Q51]. Ein Modell einer L5-Pyramidenzelle brauchte ein zeitliches
  Faltungsnetz mit 5 bis 8 Schichten; ohne NMDA genügte eine verborgene Schicht [Q52].
- **Signale kosten Energie.** Im Signalbudget der grauen Substanz von Nagern entfallen 47 % auf
  Aktionspotentiale, 34 % auf die postsynaptische Glutamatwirkung, 13 % auf das Ruhepotential und
  3 % auf das Recycling [Q56]. Höchstens 15 % der Neurone sind zugleich aktiv [Q56]. Im
  menschlichen Kortex können weniger als 1 % zugleich stark aktiv sein [Q57].

### 2.6 Was ein Modell daraus übernehmen muss

1. **Zyklen.** Die Rechnung läuft in der Zeit über einen Graphen mit Schleifen, nicht durch
   Schichten.
2. **Chemie.** Zustände sind Konzentrationen, Kopplungen sind Reaktionen mit Massenwirkung.
3. **Antrieb.** Ein geschlossenes, detailliert balanciertes Netz endet im eindeutigen
   Gleichgewicht [R39, Abschn. II D]. Gedächtnis, Schalter und Takt brauchen daher Antrieb, und sie
   kosten Energie (`research.md` Abschnitt 6.6; [Q27, Q28, Q29]).
4. **Verdrahtung.** Dichte, Gegenseitigkeit, Cluster und Hemmanteil folgen gemessener Statistik.
5. **Vorzeichen je Synapse.** Der Transmitter gehört zum Neuron, das Vorzeichen zum Rezeptor.

## 3. Das vorgeschlagene Modell

### 3.1 Überblick

- **Basis.** Das Modell baut auf Entwurf 2 („Chemie zuerst“) auf. Gutachten 1 wählt ihn,
  Gutachten 2 wählt Entwurf 3. Abschnitt 7.3 gibt beide Seiten wieder und begründet unsere Wahl.
- **Stufe K, der Kern.** Jeder Knoten ist ein Dual-Rail-Paar. Synapsen sind Katalyse, Hemmung ist
  Auslöschung, und ein einziger Treibstoffgradient Δμ treibt das Netz.
- **Stufe S, die Synapsenchemie.** Sie ersetzt die sofortige Synapse durch Freisetzung, Räumung,
  Recycling mit Treibstoff, Rezeptorbindung und eine NMDA-artige Koinzidenz. Die Idee stammt aus
  Entwurf 1. Wir haben sie in elementare Reaktionen umgebaut, weil die Tickets höchstens zwei
  Spezies je Seite erlauben (`issues/01-deterministic-crn-simulator-spike.md:18`,
  `issues/02-thermodynamically-consistent-genome.md:19-20`).
- **Später.** Modulatoren, Spaltkopplung und Treibstoff-Domänen nach dem Vorbild der Glia
  (Abschnitt 3.12).
- **Weg in die Brutmaschine.** Er stammt aus Entwurf 3 (Abschnitt 4.2).

### 3.2 Einheiten und Raten

- **Konzentration.** Alle Spezies tragen eine einzige Bezugskonzentration c° = 1 mM [R42, Gl. 25;
  `research.md` Abschnitt 8.2]. Damit passen gemessene Bindungsraten in mM⁻¹·ms⁻¹ ohne Umrechnung.
- **Zeit.** Die Zeiteinheit ist t₀ = 1 ms. Ein Eingabesymbol dauert τ_in = 10 ms (unser
  Vorschlag). Eine Folge von motif-1d mit 64 Symbolen dauert dann 640 ms.
- **Energie.** Energien zählen in Schritten von RT·ln2/4. Das sind 0.173 RT oder 0.447 kJ/mol bei
  310 K (nach unserer Rechnung).
- **Raten aus Energien.** Für jede Reaktion gilt G‡ = max(G_R, G_P) + b mit einem ganzzahligen
  Barrierenüberschuss b ≥ 0 (`research.md` Abschnitt 6.3). Die Raten sind k⁺ = κ·2^(−(G‡ − G_R)/4)
  und k⁻ = κ·2^(−(G‡ − G_P)/4). Daraus folgt k⁺/k⁻ = 2^((G_R − G_P)/4) = exp(−ΔG°/RT) für jeden
  Genwert [R39, Gl. 50].
- **Größte Rate.** κ = 16 t₀⁻¹ für erste Ordnung und 16 c°⁻¹·t₀⁻¹ für zweite Ordnung (unser
  Vorschlag). Keine Quelle der Forschungsnotiz nennt diese Schranke (`research.md` Abschnitt 6.3).
  Der schnellste gemessene Schritt, der Abtransport aus dem Spalt mit 1e4 s⁻¹ = 10 t₀⁻¹ [Q10,
  laut Chemie-Bericht], liegt darunter.
- **Ohne exp.** Eine Rate ist κ mal ein Eintrag einer Tabelle aus vier Bitmustern 2^(−r/4) mit
  r = 0 bis 3, mal einer exakten Zweierpotenz (unser Vorschlag, nach Entwurf 3). Zur Laufzeit läuft
  kein exp (`research.md` Abschnitt 7.3). Das Verhältnis k⁺/k⁻ stimmt dann bis auf die Rundung der
  vier Tabelleneinträge; die Wegscheider-Bedingungen gelten exakt auf den ganzen Zahlen der Gene.
- **Genauigkeit.** Ein Schritt ändert eine Rate um den Faktor 2^(1/4) = 1.19. Ein gemessener Wert
  lässt sich mit ganzzahligen Genen also bis auf den halben Schritt treffen, den Faktor
  2^(1/8) = 1.09, das heißt bis auf 9 % (nach unserer Rechnung).

### 3.3 Stufe K: Spezies und Reaktionen

| Zeichen | Bedeutung | Einheit |
|---|---|---|
| Y⁺ᵢ, Y⁻ᵢ | die zwei Schienen von Knoten i, mit Konzentrationen y⁺ᵢ und y⁻ᵢ | c° |
| P, W | Treibstoff und Abfall, als Chemostaten festgehalten | c° |
| U⁺ₖ, U⁻ₖ | Eingang k als Dual-Rail-Paar von Katalysatoren; die Aufgabe setzt u⁺ₖ = max(vₖ, 0) und u⁻ₖ = max(−vₖ, 0) | c° |
| Sᵢⱼ und andere | feste Katalysatoren wie Rezeptoren, Transporter und Enzyme; ihre Menge steckt in den Ratenkonstanten | c° |

| Nr. | Reaktion | Zweck |
|---|---|---|
| K1 | Y⁺ⱼ + P ⇌ Y⁺ⱼ + Y^σᵢ, katalysiert durch Sᵢⱼ | Synapse j → i; das Vorzeichen σ folgt der Rezeptorklasse |
| K2 | U^±ₖ + P ⇌ U^±ₖ + Y^(±σ)ᵢ | Eingang k → i mit Vorzeichen σ |
| K3 | P ⇌ Y^±ᵢ, katalysiert | Grundbildung, also der Bias |
| K4 | Y^±ᵢ ⇌ W, katalysiert | Abbau; er setzt die Zeitkonstante |
| K5 | Y⁺ᵢ + Y⁻ᵢ ⇌ 2 W, katalysiert durch ein Enzym | Auslöschung; sie koppelt die Schienen |

- **Molekularität.** Jede Reaktion ist reversibel und hat höchstens zwei veränderliche Spezies je
  Seite. Chemostaten und feste Katalysatoren stecken in Pseudo-Ratenkonstanten. Ticket 03 fragt,
  ob sie gegen die Grenze zählen (`issues/03-compile-mlp-into-crn.md:40-43`). Unser Vorschlag: Sie
  zählen nicht.
- **Zusammensetzung.** P, W und jedes Y tragen je eine Baustein-Einheit u. Jede Reaktion K1 bis K5
  erhält u (Ableitung aus Entwurf 2).
- **Wert und Wirkung.** Der Wert des Knotens ist dᵢ = y⁺ᵢ − y⁻ᵢ. Weiter wirkt nur y⁺ᵢ, und zwar
  als Katalysator. Ein negatives Gewicht als Abfluss wäre keine Chemie [R43; `research.md`
  Abschnitt 6.1].
- **Eingänge sind Katalysatoren.** Ein Katalysator darf die Konzentration 0 haben. Ein
  Eingangsreservoir mit Pegel 0 hätte dagegen das chemische Potential −∞, und die Energiebilanz
  bräche (Korrektur aus Gutachten 1 zu Entwurf 3).

### 3.4 Stufe K: Gleichungen

Mit der Eichung μ°(Y) = 0 folgen vier abgeleitete Größen (Ableitung aus Entwurf 2):

- **A = p·exp(μ°_P/RT).** Bei diesem Pegel steht eine Schiene im Gleichgewicht mit P.
- **ε = w·exp(μ°_W/RT).** Bei diesem Pegel steht sie im Gleichgewicht mit W. Wir setzen
  ε = 2^(−10) c°, etwa 1 µM (unser Vorschlag).
- **γ = ε².** Bei diesem Produkt steht die Auslöschung im Gleichgewicht. Die Thermodynamik legt γ
  fest; es ist kein freier Parameter.
- **Δμ = RT·ln(A/ε).** Das ist der Antrieb, die einzige Batterie des Kerns.

```
dy⁺ᵢ/dt = g⁺ᵢ·(A − y⁺ᵢ) + δᵢ·(ε − y⁺ᵢ) − kᵃᵢ·(y⁺ᵢ·y⁻ᵢ − ε²)
dy⁻ᵢ/dt = g⁻ᵢ·(A − y⁻ᵢ) + δᵢ·(ε − y⁻ᵢ) − kᵃᵢ·(y⁺ᵢ·y⁻ᵢ − ε²)

g⁺ᵢ = β⁺ᵢ + Σ_{j→i, σ=+} κᵢⱼ·y⁺ⱼ + Σ_{k→i} κᵢₖ·u⁽⁺⁾ₖᵢ
g⁻ᵢ = β⁻ᵢ + Σ_{j→i, σ=−} κᵢⱼ·y⁺ⱼ + Σ_{k→i} κᵢₖ·u⁽⁻⁾ₖᵢ
u⁽⁺⁾ₖᵢ = u⁺ₖ, wenn σₖᵢ = +, sonst u⁻ₖ; für u⁽⁻⁾ₖᵢ umgekehrt
κᵢⱼ = nᵢⱼ·κ_c          (Synapsenzahl nᵢⱼ von 1 bis 36, Einheitsleitwert κ_c der Klasse c)

Einheiten: y, A, ε in c°; g, β, δ in t₀⁻¹; κ und kᵃ in c°⁻¹·t₀⁻¹
Auslese:   dₒ = y⁺ₒ − y⁻ₒ an den Motor-Knoten o, am Ende des letzten Symbols
```

- **Herkunft des Synapsenterms.** Es gilt k⁺·p·y⁺ⱼ − k⁻·y⁺ⱼ·y = κ·y⁺ⱼ·(A − y) mit κ = k⁻, weil
  k⁺·p/k⁻ = A ist (Entwurf 2).
- **Gewicht als Synapsenzahl.** Im Wurm ist die Zahl der Synapsen ein guter Stellvertreter für
  ihre Größe (laut Hirn-Bericht nach [Q39]). Ein Ganzhirnmodell der Fliege nutzte nur
  Synapsenzahlen und Vorzeichen [Q55].
- **Zeitkonstanten.** δ = κ·2^(−n/4) mit n von 0 bis 64 deckt Zeitkonstanten von 0.06 ms bis
  4.1 s ab (nach unserer Rechnung). Der Hirn-Bericht verlangt Klassen von 1 ms bis 10 s. Die
  langsamsten davon, Modulatoren und Glia mit zig Sekunden, gehören zu den späteren Lagen
  (Abschnitt 3.12).

### 3.5 Stufe K: Grenzfälle

Alle Ableitungen dieses Abschnitts stammen aus Entwurf 2; die Korrekturen sind markiert.

- **Ohne Auslöschung.** Mit kᵃ → 0 folgt jede Schiene dy/dt = −[δ + g(t)]·y + g(t)·A + δ·ε. Das
  ist die Form des LTC-Neurons dx/dt = −[1/τ + f]·x + f·A [Q2, Gl. 1]. Das LTC-Neuron stammt aus
  einem Wurmmodell, in dem eine Synapse den Strom g·(E − V) liefert [Q6, laut Chemie-Bericht]. Der
  Unterschied: g ist hier linear in der Quelle, nicht sigmoid.
- **Ein Pegel je Schiene ist gewählt.** Alle Synapsen treiben eine Schiene zum selben Pegel A. Das
  folgt aus der Wahl eines einzigen Treibstoffs P, nicht aus der Physik (Korrektur aus
  Gutachten 1). Ein zweiter Treibstoff P′ mit eigenem Potential gäbe einen zweiten Pegel. Da er
  dieselbe Einheit u trägt, bräche er keine weitere Erhaltung und erzeugte einen weiteren
  emergenten Zyklus mit eigener Affinität [R39, Abschn. II D, |S_y| = |λ_b| + |ε|, nachgelesen].
- **Hemmung ohne negatives Umkehrpotential.** Negative Umkehrpotentiale gibt es chemisch nicht
  (Lesart des Chemie-Berichts). Hier bildet eine hemmende Synapse die Minus-Schiene, und diese
  löscht die Plus-Schiene aus.
- **Schnelle Auslöschung bei festen Leitwerten.** Dann gilt
  y⁺* = [A·Δg + √(A²·Δg² + 4·(δ + g⁺)·(δ + g⁻)·γ)] / (2·(δ + g⁺)) mit Δg = g⁺ − g⁻. Gutachten 1 hat
  die Formel nachgerechnet: Simulation und Formel stimmen auf 8 Stellen.
- **Zusätzlich g ≪ δ.** Dann ist y⁺* = φ_γ(A·Δg/δ) mit φ_γ(x) = (x + √(x² + 4γ))/2. Das ist die
  geglättete ReLU von [R9, Gl. 4, Prop. 4.9]; Dack et al. nutzen sie als chemisches Perzeptron
  [R17, Gl. 3-4]. Hier legt die Thermodynamik die Glättung fest: γ = ε².
- **Starker Antrieb (A ≫ y).** Dann gilt ḋᵢ ≈ −δᵢ·dᵢ + Σⱼ Wᵢⱼ·φ_γ(dⱼ) + bᵢ + Σₖ Vᵢₖ·uₖ mit
  Wᵢⱼ = σᵢⱼ·A·κᵢⱼ. Das ist ein zeitkontinuierliches RNN mit festen Vorzeichen; solche Netze lassen
  sich mit Gradienten trainieren [Q64].
- **Bezug zu Ticket 04.** Ticket 04 sucht eine reversible, zusammensetzungserhaltende Variante, die
  den stationären Zustand von [R9, Prop. 4.9] behält
  (`issues/04-gradients-through-the-integrator.md:34-35`). Der Dual-Rail-Knoten ist ein Kandidat.
  Er braucht nur bimolekulare Schritte statt der trimolekularen von [R9, Tab. 1]. Anders als in
  [R9] ist die Glättung nicht frei wählbar.

### 3.6 Stufe S: Synapsenchemie

Stufe S ersetzt K1. Je Sender j gibt es drei Transmitter-Pools, je Synapse j → i einen Rezeptor.

| Zeichen | Bedeutung | Einheit |
|---|---|---|
| Rⱼ, Eⱼ, Iⱼ | Transmitter des Senders j: bereit, im Spalt, in der Rückgewinnung | c° |
| Qᵢⱼ, Bᵢⱼ | freier und gebundener Rezeptor der Synapse j → i; Summe ρᵢⱼ = nᵢⱼ·ρ₀ | c° |
| Oᵢⱼ | offener Koinzidenz-Komplex der langsamen erregenden Klasse | c° |
| F, F′ | Treibstoff des Recyclings und sein Abfall, als Chemostaten; Antrieb D | c° |

| Nr. | Reaktion | Bedeutung | Vorbild |
|---|---|---|---|
| S1 | Rⱼ + Y⁺ⱼ ⇌ Eⱼ + Y⁺ⱼ | Freisetzung, katalysiert durch die Aktivität des Senders | Freisetzungskette [Q10, Q11] |
| S2 | Eⱼ ⇌ Iⱼ, katalysiert | Räumung durch Transporter | Glutamat verlässt den Extrazellulärraum nur über Transporter [Q21] |
| S3 | Iⱼ + F ⇌ Rⱼ + F′ | Recycling mit Treibstoff | Zyklus aus wirksam, inaktiv und erholt [Q13] |
| S4 | Eⱼ + Qᵢⱼ ⇌ Bᵢⱼ | Rezeptorbindung; sie verbraucht Transmitter | Zweizustandsschema [Q9] |
| S5 | Bᵢⱼ + P ⇌ Bᵢⱼ + Y^σᵢ | offener Kanal der schnellen Klassen bildet die Schiene σ des Ziels | Vorzeichen nach Rezeptor [Q20] |
| S6 | Bᵢⱼ + Y⁺ᵢ ⇌ Oᵢⱼ und Oᵢⱼ + P ⇌ Oᵢⱼ + Y⁺ᵢ | NMDA-artige Koinzidenz von Transmitter und Aktivität des Ziels | Mg-Block als Koinzidenzdetektor [Q10] |
| S7 | 2 Y⁺ⱼ ⇌ Y²ⱼ und 2 Y²ⱼ ⇌ Y⁴ⱼ; Y⁴ⱼ ersetzt Y⁺ⱼ als Katalysator in S1 | Kooperativität, optional | 4 Ca²⁺ binden bei der Freisetzung [Q10, Q11] |

```
drⱼ/dt = −y⁺ⱼ·(k₁⁺·rⱼ − k₁⁻·eⱼ) + (k₃⁺·iⱼ − k₃⁻·rⱼ)
deⱼ/dt = +y⁺ⱼ·(k₁⁺·rⱼ − k₁⁻·eⱼ) − (k₂⁺·eⱼ − k₂⁻·iⱼ) − Σᵢ Jᴮᵢⱼ
diⱼ/dt = (k₂⁺·eⱼ − k₂⁻·iⱼ) − (k₃⁺·iⱼ − k₃⁻·rⱼ)
Jᴮᵢⱼ   = α_c·eⱼ·qᵢⱼ − β_c·bᵢⱼ,         qᵢⱼ = ρᵢⱼ − bᵢⱼ − oᵢⱼ
dbᵢⱼ/dt = Jᴮᵢⱼ − Jᴼᵢⱼ,   Jᴼᵢⱼ = k_o⁺·bᵢⱼ·y⁺ᵢ − k_o⁻·oᵢⱼ   (Jᴼ nur in der langsamen erregenden Klasse)
doᵢⱼ/dt = Jᴼᵢⱼ

Im Knoten wird Σ κᵢⱼ·y⁺ⱼ zu Σ κ_c·bᵢⱼ (schnelle Klassen) plus Σ κ_o·oᵢⱼ (langsame Klasse);
dy⁺ᵢ/dt erhält zusätzlich −Σⱼ Jᴼᵢⱼ.
Antrieb des Recyclings: ln(k₁⁺·k₂⁺·k₃⁺ / (k₁⁻·k₂⁻·k₃⁻)) = D/RT; F und F′ stecken in k₃±.

Einheiten: r, e, i, q, b, o, ρ in c°; k₁± und α, k_o⁺ in c°⁻¹·t₀⁻¹; k₂±, k₃±, β, k_o⁻ in t₀⁻¹
```

Startwerte aus Messungen; die Gene treffen jeden Wert bis auf 9 % (Abschnitt 3.2):

| Größe | Wert | Quelle |
|---|---|---|
| Abtransport aus dem Spalt | 1/k_c = 0.1 ms | [Q10], laut Chemie-Bericht |
| Inaktivierung wirksamer Ressourcen | etwa 3 ms | [Q13] |
| Erholung | τ_rec = 450 bis 800 ms | [Q13] |
| Freigesetzter Anteil je Spike, U_SE | 0.1 bis 0.95 in 33 Verbindungen | [Q13] |
| AMPA-artig | α = 1.1 c°⁻¹·ms⁻¹, β = 0.19 ms⁻¹ | [Q10] |
| NMDA-artig | α = 0.072 c°⁻¹·ms⁻¹, β = 0.0066 ms⁻¹ | [Q10] |
| GABA_A-artig | α = 5 c°⁻¹·ms⁻¹, β = 0.18 ms⁻¹ | [Q11], Korrektur aus Gutachten 1 |
| GABA_B-artig, ohne G-Protein-Stufe | α = 0.09 c°⁻¹·ms⁻¹, β = 1/(110 ms) = 0.0091 ms⁻¹ | [Q10], [Q15], Korrektur aus Gutachten 1 |

- **GABA_A.** Entwurf 1 hatte α = 1.1 vom AMPA-Rezeptor übernommen, weil der Scan unleserlich war.
  Der offizielle Code von Destexhe nennt α = 5 und β = 0.18 [Q11]. Nach einem Puls von 1 mM für
  1 ms ist der Rezeptor damit zu 96 % statt zu 62 % gebunden (nachgerechnet).
- **GABA_B.** Ohne G-Protein-Stufe fiele die langsame Hemmung mit 1/K₂ = 0.83 s ab. Gemessen sind
  110 ms (84 %) und 516 ms (16 %) [Q15]. Wir setzen β auf die dominante Zeit. Den Anstieg in 45 ms
  und den langsamen Rest liefert erst die G-Protein-Stufe. Sie läuft auf GTP und braucht damit
  einen eigenen Treibstoff; wir verschieben sie (Ticket 11, Abschnitt 4.4).
- **Vesikelinhalt.** 10 000 Moleküle je Vesikel sind ein Parameter in Destexhes Code, „nb of molec
  of T per vesicle“, keine Messung [Q11]. Stufe S braucht ihn nicht (Korrektur aus Gutachten 1).

### 3.7 Stufe S: was die Chemie erzwingt

- **Übertragung braucht Treibstoff.** Ohne Antrieb (D = 0) steht der Zyklus R ⇌ E ⇌ I im
  Gleichgewicht. Ein Katalysator verschiebt kein Gleichgewicht [R14, Anh. 7.1]. Der Anteil im Spalt
  hängt dann nicht von der Aktivität des Senders ab, und die Synapse überträgt nichts (nach unserer
  Ableitung).
- **Nachgerechnet.** Mit den Energien G_R = 0, G_E = −1 RT und G_I = −3 RT war der Anteil E bei
  D = 0 für jede Aktivität von 0.01 bis 10 gleich 0.1142. Bei D = 2 RT stieg er von 0.093 auf
  0.141, bei D = 6 RT von 0.023 auf 0.195. Die Affinität des Zyklus war jedes Mal genau D.
- **Gehirn.** Das passt zum Budget: Postsynaptische Glutamatwirkung und Recycling kosten 34 % und
  3 % [Q56].
- **Zwei Batterien.** Stufe S hat die Chemostaten P, W, F und F′. Gebrochen sind die Erhaltung von
  u und die des Treibstoffs. Also entstehen 4 − 2 = 2 emergente Zyklen mit den Affinitäten Δμ und D
  [R39, Abschn. II D] (unsere Zählung). Die Eingänge brechen nur ihre eigene, triviale Erhaltung.
- **Erhaltung.** Je Sender gilt rⱼ + eⱼ + iⱼ + Σᵢ (bᵢⱼ + oᵢⱼ) = Vⱼ. Je Synapse gilt
  qᵢⱼ + bᵢⱼ + oᵢⱼ = ρᵢⱼ.
- **Reversible Kanäle.** Die meisten, aber nicht alle Ionenkanäle gehorchen der mikroskopischen
  Reversibilität [Q23, laut Hirn-Bericht]. Reversible Bindung und reversibles Öffnen sind also eine
  gute, aber keine ausnahmslose Näherung.
- **Echte Last.** Bindung verbraucht Transmitter, nicht die Aktivität des Senders. Entwurf 3 machte
  den Botenstoff selbst zum Transmitter (2 Xⱼ ⇌ Dⱼ). Gutachten 1 nennt die Rückwirkung daraus ein
  Artefakt, weil Ca²⁺ die Freisetzung auslöst und nicht selbst Transmitter wird [Q10].
- **Koinzidenz.** S6 verknüpft präsynaptischen Transmitter mit der Aktivität des Ziels. Erst diese
  Kopplung ist NMDA-artig (Korrektur aus Gutachten 1 zu Entwurf 3). Sie gibt positive Rückkopplung
  im Ziel und damit eine Quelle für Bistabilität. NMDA-Spikes in Basaldendriten verstärken die
  Antwort um 226 ± 46 % [Q22, laut Hirn-Bericht].
- **Hill-Grad.** Im Gleichgewicht von S7 gilt [Y⁴ⱼ] = K₁²·K₂·(y⁺ⱼ)⁴ (nach unserer Ableitung). Ketten
  bimolekularer Schritte geben also effektive Hill-Grade bis 4, ohne die Grenze von zwei Spezies je
  Seite zu brechen (Korrektur aus Gutachten 1 zu Entwurf 3).

### 3.8 Der Rechenschritt

- **Knoten.** Jeder Knoten löst die implizite Euler-Gleichung für K1 bis K5 exakt, bei
  eingefrorenen Katalysatoren (Entwurf 2). Der Schritt nimmt die positive Wurzel einer quadratischen
  Gleichung. Er bleibt für jede Schrittweite h positiv.
- **Kopplung.** Katalysatoren anderer Knoten und Synapsen kommen aus dem vorigen Teilschritt
  (Jacobi-Reihenfolge).
- **Getriebene Zyklen geschlossen.** Der Zyklus R → E → I → R eines Senders löst einen
  geschlossenen impliziten Schritt ohne Subtraktion (Entwurf 1). Gutachten 1 hat den Löser
  nachgerechnet: Abweichung zu np.linalg.solve höchstens 3.6e-13.
- **Sackgassen einzeln.** S4, S6 und S7 relaxieren jede für sich auf ihr eigenes Gleichgewicht zu,
  nie darüber hinaus (Entwurf 3). Der Fluss beim Umsatz ξ ist J(ξ) = a·ξ² − S·ξ + J₀. Seine
  Nullstelle im zulässigen Bereich ist ξ* = 2J₀/(S + λ) mit λ = √(S² − 4·a·J₀). Der Schritt setzt
  ξ = ξ*·λh/(1 + λh). Für diese Relaxation gilt der zweite Hauptsatz je Teilschritt; Gutachten 2
  hat das numerisch und analytisch geprüft.
- **Regel.** Wer getriebene Zyklen geschlossen löst und Sackgassen einzeln relaxiert, erhält
  Fixpunkte, die für jedes h stationäre Zustände der Gleichungen sind. Das ist unsere Ableitung; sie
  verallgemeinert das Argument von Gutachten 1 für Entwurf 3. Den Gegenfall fand Entwurf 1: Ein
  paarweise gelöster Zyklus verfehlte getriebene stationäre Zustände um 28 %.
- **Rechenarten.** In der Schleife stehen nur +, −, ×, ÷ und √. Rust rundet sie exakt nach
  IEEE 754 und zieht a·b + c nicht zu einem FMA zusammen [R70, R71].
- **Schrittregel.** Die Kopplung zwischen Knoten ist explizit. Zu große Schritte erzeugen
  Scheinschwingungen mit der Periode 2h; Entwurf 2 fand sie in 5 von 50 Zufallsnetzen, bei
  h·A·K_max = 51, aber nicht mehr bei 5.1. Dabei ist K_max = maxᵢ Σⱼ κᵢⱼ. Wir verlangen
  h·A·K_max ≤ 1; die Zahl der Teilschritte L folgt daraus und ist kein Gen. Ticket 01 kalibriert die
  Schranke.
- **Halbschritt-Test.** Je Kandidat laufen 64 Validierungsfolgen zusätzlich mit h/2. Ändert sich das
  Argmax bei mehr als 3 % von ihnen, scheitert der Kandidat als numerisches Artefakt [R55, Abschn. 6;
  `research.md` Abschnitt 8.3]. Die Schwelle ist ein Platzhalter.
- **Quantisierte Auslese.** dₒ wird zur Null hin auf Vielfache von 2^(−20) c° gerundet (unser
  Vorschlag). Das verdeckt Rundungsrauschen und macht die Sperre bei Δμ = 0 exakt (Abschnitt 5.5).

Der Knotenschritt in Formeln (Entwurf 2; c = h·kᵃ):

```
a⁺ = 1 + h·(g⁺ + δ),   a⁻ = 1 + h·(g⁻ + δ)
r⁺ = y⁺ + h·(g⁺·A + δ·ε),   r⁻ = y⁻ + h·(g⁻·A + δ·ε)
b = a⁺·a⁻ − c·(r⁺ − r⁻),   q = a⁻·(r⁺ + c·ε²),   s = √(b² + 4·c·a⁺·q)
y⁺neu = 2q/(b + s), wenn b ≥ 0, sonst (s − b)/(2·c·a⁺)
y⁻neu = (r⁻ + c·ε²)/(a⁻ + c·y⁺neu)
```

- **Kosten.** Je Teilschritt 2E + 34N FLOPs, dazu 4M je Eingabeschritt (unsere Zählung; ÷ und √ als
  je ein FLOP). E ist die Zahl der Synapsen, M die der Eingangskanten, N die der Knoten. Je Knoten
  fallen drei Divisionen und eine Wurzel an; Ticket 01 misst ihre Kosten.
- **Nachgerechnet.** Wir haben den Schritt mit Zufallsnetzen geprüft:
  - 240 Läufe mit h von 1e-3 bis 1e3 und kᵃ von 1e-3 bis 1e6: kein negativer und kein nicht
    endlicher Wert.
  - Δμ = 0, Start bei ε, Zufallseingänge, 30 Netze: höchstens 6.7e-16 relative Abweichung von ε.
  - Fixpunkt bei h = 0.05, 0.5 und 5: gleich bis auf 1.3e-15.
  - Stationärer Zustand: T·σ und RT·Δμ·J_P gleich bis auf 3e-16 relativ.

### 3.9 Physik: Garantien für jedes Genom

| # | Gesetz | Wie das Modell es erzwingt | Prüfung |
|---|---|---|---|
| 1 | Massenwirkung und Positivität [R43, Def. 6, Lemma 1] | Jeder negative Term von dy/dt enthält y; der Schritt nimmt die positive Wurzel und bildet Quotienten positiver Größen | AT-1 |
| 2 | Lokales detailliertes Gleichgewicht [R39, Gl. 50] | Raten aus G‡ = max(G_R, G_P) + b; Katalysatoren skalieren k⁺ und k⁻ gleich [R14, Anh. 7.1] | Code-Review, AT-5 |
| 3 | Wegscheider [R40, Abschn. 1.1] | Innere Zyklen summieren die Energien zu 0, als Identität in ganzen Zahlen | Eigenschaftstest |
| 4 | Erhaltung der Zusammensetzung | Jede Vorlage erhält u; in Stufe S auch Transmitter und Rezeptor | AT-3 |
| 5 | Nur Chemostaten treiben [R39, Abschn. II D] | Stufe K: ein emergenter Zyklus mit Affinität Δμ; Stufe S: zwei, mit Δμ und D | AT-5 |
| 6 | Ohne Antrieb ein eindeutiges Gleichgewicht [R39, Abschn. II D] | Bei Δμ = 0 ist y⁺ = y⁻ = ε für jeden Eingang ein Fixpunkt; in Stufe S braucht es zusätzlich D = 0 | AT-4, AT-9 |
| 7 | Getriebener stationärer Zustand [R39, Gl. 55, 60, 61] | T·σ = RT·Δμ·J_P im stationären Zustand (Ableitung aus Entwurf 2) | AT-5 |
| 8 | Schranke | y ≤ max(y(0), A, ε) + √γ (Ableitung aus Entwurf 2; Gutachten 1 nachgerechnet) | AT-2 |
| 9 | Zweiter Hauptsatz je Teilschritt | Bewiesen für die Relaxation der Sackgassen (Entwurf 3); für den Knotenschritt fehlt ein Beweis | AT-17 |
| 10 | Keine irreversible Reaktion | Jede Rückrate ist endlich (`research.md` Abschnitte 6.4 und 6.5) | Validator |

- **Was mutiert.** Gene sind ganze Zahlen: Energien, Barrieren, Pegel, Synapsenzahlen und der
  Antrieb. Die Vorlagen legen Reaktionen, Stöchiometrie und Katalysatoren fest. Deshalb bleibt
  jeder Mutant physikalisch, wie Ticket 02 es verlangt
  (`issues/02-thermodynamically-consistent-genome.md:8-12`).
- **Wo das Modell nähert.**
  - Katalyse ohne Komplexbildung in K1, K2 und S5. Hjelmfelt et al. treffen dieselbe Annahme: Die
    Enzymmenge ist klein gegen die Effektormenge [R1, S. 10984, laut Chemie-Bericht].
  - Eine ideale, verdünnte, gut gemischte, isotherme Lösung [R39, Abschn. II].
  - Deterministische Konzentrationen; molekulares Rauschen fehlt [R39, Abschn. II].
  - Die Synapse von Stufe K wirkt sofort. So rechnen auch Wicks et al.; eine Synapsen-Zeitkonstante
    von 10 ms änderte dort die Ergebnisse nicht wesentlich [Q6, laut Chemie-Bericht].
  - Die Auslese liegt außerhalb der Chemie.

### 3.10 Antrieb, Gedächtnis und Energie

- **Ohne Antrieb kein Rechnen.** Bei Δμ = 0 gilt A = ε. Dann ist y⁺ = y⁻ = ε für jeden Eingang ein
  Fixpunkt, in der Gleichung wie im Schritt (Ableitung aus Entwurf 2). Der Endzustand hängt nicht
  vom Eingang ab. Das passt zu [R47]: Ohne Antrieb hängt der stationäre Zustand nicht von den
  Barrieren ab. Auch Gleichgewichtsnetze aus Dimeren haben einen eindeutigen Zustand und damit kein
  Gedächtnis [Q24, Vorabdruck].
- **Antriebe tragen das Lernen.** In einem Vorabdruck gelang das Training mit nur den Antrieben
  als freien Größen in 60 % der Fälle, mit nur den Barrieren in 0 % [Q25, Vorabdruck].
- **Kontrast.** Bei schneller Auslöschung liegt y⁺ zwischen ε²/(A + ε) und A + ε. Zwei Zustände
  unterscheiden sich daher höchstens um den Faktor (e^(Δμ/RT) + 1)² (Ableitung aus Entwurf 2).
- **Bistabilität.** Zwei sich gegenseitig erregende Knoten waren in einer Gittersuche erst ab
  Δμ = 1.5 RT bistabil (Entwurf 2, 475 Parametersätze, im Speicher gerechnet, nicht
  vorregistriert):

  | Δμ in RT | bistabile Sätze | bester Kontrast | Schranke (e^Δμ + 1)² |
  |---|---|---|---|
  | 0, 0.5 und 1 | 0 % | 1 | 4, 7.0 und 13.8 |
  | 1.5 | 1.3 % | 8.7 | 30 |
  | 2 | 3.2 % | 35 | 70 |
  | 4 | 17.7 % | 2676 | 3091 |
  | 8 | 58.7 % | 8.4e6 | 8.9e6 |

- **Gegenprobe.** Wir haben die Suche mit einem anderen Gitter wiederholt: 480 Sätze, δ = 1,
  kᵃ = 1e3, h = 0.25, je zwei Starts (nachgerechnet). Bistabil wurde das Paar erst ab 2 RT
  (2.3 %), bei 3 RT in 6.9 %, bei 4 RT in 10.6 % und bei 8 RT in 21.2 % der Sätze. Der beste
  Kontrast blieb stets unter der Schranke, etwa 7.05e6 gegen 8.89e6 bei 8 RT. Die Anteile hängen
  vom Gitter ab; die Richtung stimmt mit Entwurf 2 überein.

- **Gedächtnis kostet.** Im stationären Zustand gilt T·σ = RT·Δμ·J_P. Ein Zustand „an“ setzt mehr P
  um als „aus“ und kostet daher mehr (Ableitung aus Entwurf 2). Im Gehirn kostet schon das
  Ruhepotential 13 % des Signalbudgets [Q56]. Auch Schalter, Takt und Präzision kosten freie
  Energie [Q27, Q28, Q29].
- **Energiebilanz.** Abgerechnet wird die Dissipation W_c − ΔG [R39, Gl. 78-79; `research.md`
  Abschnitt 8.3]. In der Schleife zählen wir nur die Umsätze N_P und N_W. ln braucht man nur am
  Anfang und bei der Auslese (Fog 5).
- **Eingänge kosten nichts.** Ändert sich u(t), führt die Umgebung Katalysator-Moleküle zu oder ab.
  Sie tun das beim chemischen Potential, das im Netz herrscht. Da ∂G/∂Z_U = μ_U gilt [R39, Gl. 70],
  hebt sich dieser Beitrag in W_c − ΔG heraus (nach unserer Ableitung; Korrektur aus Gutachten 1
  zur Aussage von Entwurf 2, die Eingänge tauschten „keine Masse“).
- **Schranke ohne ln.** Für einen Austausch gilt k·(A − x)·ln(A/x) ≥ 2k·(A − x)²/(A + x), weil das
  logarithmische Mittel unter dem arithmetischen liegt (Ableitung aus Entwurf 1). So lässt sich eine
  untere Schranke der Dissipation ohne ln berichten.

### 3.11 Verschaltung

Der Generator zieht ein Genom deterministisch aus einem ChaCha8-Seed. Er verbindet die Ansätze von
Entwurf 1 und Entwurf 3; die Prüfungen gegen das Konfigurationsmodell und die relative Weglänge
sind Korrekturen aus Gutachten 1.

1. **Gruppen.** Wie bei NCP: sensorisch, Inter, Kommando, Motor [Q4]. Für N = 48 sind es 8, 16, 16
   und 8 Knoten (unser Vorschlag).
2. **Neigung je Gruppenpaar.** Innerhalb der Gruppen viele rekurrente Verbindungen, zwischen den
   Gruppen meist vorwärts, wie im Wurm (laut Hirn-Bericht nach [Q37]). Rekurrente Synapsen vor
   allem zwischen Kommandoknoten, wie bei NCP [Q4].
3. **Nähe.** Die Knoten liegen auf einem Ring; ein Gauß-Kern mit σ = 0.15 wichtet nahe Paare
   (Entwurf 1). Im Kortex fällt die Verbindungswahrscheinlichkeit mit einer Gauß-Breite von 125 bis
   130 µm [Q36].
4. **Hubs.** Jeder Knoten hat einen lognormalen Faktor (σ = 0.5) für Ein- und Ausgangsgrad
   zugleich (Entwurf 1). Das koppelt beide Grade, wie im Wurm mit der Korrelation 0.71 und in der
   Fliege mit 0.76 (laut Hirn-Bericht nach [Q37, Q44]).
5. **Gegenseitigkeit.** Mit P(i→j) = a und P(j→i) = b gilt
   P(beide) = max(min(a, b, λ·a·b), a + b − 1, 0). So bleiben die Randwahrscheinlichkeiten exakt
   (Ableitung aus Entwurf 1). Paare aus erregend und hemmend bekommen 2λ, weil gegenseitige Paare in
   der Fliege bevorzugt erregend und hemmend sind (laut Hirn-Bericht nach [Q44]).
6. **Dreiecke.** Ein Zug schließt mit Vorliebe Dreiecke (Entwurf 3).
7. **Transmitter und Vorzeichen.** 10 bis 12 % der Knoten tragen GABA und haben nur hemmende
   Synapsen. Im Wurm sind 8.6 % der Neurone GABAerg (26 von 302), in einer Säule der Ratte 11.5 %
   Interneurone (laut Hirn-Bericht nach [Q37, Q49]). Der Anteil GABAerger Neurone ist nicht der
   Anteil hemmender Synapsen (Korrektur aus Gutachten 1). Ein Transmitter je Knoten folgt der
   Fliege, in der Neuronen meist nur einen schnellen Transmitter tragen (laut Hirn-Bericht nach
   [Q43, Q46]). Glutamaterge Knoten tragen meist erregende Synapsen; ein Gen erlaubt hemmende, wie
   bei AWC → AIY [Q20].
8. **Synapsenzahlen.** Lognormal mit Mittel 3, ganzzahlig von 1 bis 36, wie im Wurm mit 3.15
   Synapsen je Verbindung und der Spanne 1 bis 36 [Q44, Tab. 2].
9. **Eingänge.** Jeder Kanal wirkt auf mindestens zwei sensorische Knoten mit beiden Vorzeichen.
   Ein Eingang, der nur eine Kante treibt, legt das Vorzeichen der Antwort fest [R47].
10. **Skala.** Am Arbeitspunkt liefert der Eingang etwa ein Drittel der Erregung [Q32, Q33]. Wenige
    Eingänge je Knoten verlangen höhere Gewichte, und den Gültigkeitsbereich dieser Skalierung muss
    man prüfen [Q53].

| Kennzahl | Gehirn | Ziel bei N = 48 | Prüfung |
|---|---|---|---|
| Eingangsgrad | Wurm 3242/302 = 10.7 (unsere Rechnung nach [Q44]); NCP 253/19 = 13.3 [Q4] | 8, bei N = 302 dann 10.7 | ±10 % |
| Dichte | 0.025 bis 0.054 bei 100 bis 400 Einheiten (laut Hirn-Bericht nach [Q37, Q44]) | folgt aus dem Grad: 8/47 = 0.17 | nur berichtet |
| Gegenseitigkeit gegen ER | Faktor 1.63 (Maus L2/3) bis 14.7 (Wurm, Männchen) [Q44]; Ratte L5 Faktor 4 [Q35]; Allen-Daten 3 bis 5 [Q36] | Faktor 1.6 bis 15 | AT-15 |
| Gegenseitigkeit gegen CFG | Faktor 1.33 bis 6.02 [Q44] | mindestens 1.3 | AT-15 |
| Clusterung gegen ER | Faktor 1.51 bis 6.39 [Q44] | mindestens 1.5 | AT-15 |
| Clusterung gegen CFG | Faktor 1.06 bis 2.40 [Q44] | mindestens 1.05 | AT-15 |
| Größte stark zusammenhängende Komponente | Wurm 85 % (237 von 279), Fliege 93.3 % (laut Hirn-Bericht nach [Q37, Q44]) | mindestens 85 % | AT-15, bei kleinem N trivial |
| Mittlerer Weg gegen ER | Wurm etwa Faktor 1.10 (unsere Rechnung aus C/C_ER = 2.6 und dem Small-World-Index 2.37, laut Hirn-Bericht nach [Q37]) | Faktor 1.0 bis 1.3 | AT-15 |
| Knoten auf Schleifen bis 5 Schritte | 41 % der Neurone der Fliegenlarve [Q45] | mindestens 40 % | AT-15, bei kleinem N trivial |
| Hemmanteil | 8.6 % und 11.5 % | 10 bis 12 % | exakt |
| Synapsen je Verbindung | Wurm 3.15, Spanne 1 bis 36 [Q44] | Mittel 3, 1 bis 36 | AT-15 |

- **Warum relativ.** Die Weglänge wächst mit der Netzgröße. Ein Zufallsgraph mit Eingangsgrad 8
  hat bei N = 48 im Mittel 2.06 Schritte, bei N = 64 2.19 und mit Grad 10.7 bei N = 302 2.66
  (nachgerechnet). Absolute Ziele von 3 bis 4.4 Schritten, wie Entwurf 2 und Entwurf 3 sie setzten,
  sind bei kleinem N unerreichbar (Korrektur aus Gutachten 1).
- **Warum auch CFG.** Hub-Faktoren mit gekoppelten Graden heben Gegenseitigkeit und Clusterung
  schon ohne echte Gegenseitigkeit über ER (Gutachten 1). Lin et al. messen deshalb gegen beide
  Modelle [Q44, Tab. 2].
- **Azyklischer Zwilling.** Er hat dieselben Knoten, Vorzeichen, Synapsenzahlen und FLOPs wie das
  rekurrente Netz (Entwurf 3). Jeder Knoten trägt einen Rang: zuerst nach Gruppe, innerhalb der
  Gruppe zufällig. Synapsen gegen den Rang fallen weg. Vorwärtssynapsen füllen auf, bis E wieder
  gleich ist. Jede stark zusammenhängende Komponente des Zwillings hat die Größe 1.
- **Zufallszwilling.** Gleiche Dichte und Vorzeichen, aber ohne Hirnstatistik (Entwurf 2). Bei NCP
  schnitten zufällig oder voll verdrahtete Netze mit 19 Neuronen schlechter ab: Fehler 5.19 und
  5.18 gegen 3.22 [Q4, Tab. 1, laut Chemie-Bericht].

### 3.12 Was das Modell nicht übernimmt

- **Eingangszahl.** Eine CA1-Zelle hat etwa 30 000 erregende und 1700 hemmende Eingänge [Q50]. Hier
  sind es 8, etwa Faktor 4000 weniger (nach unserer Rechnung). Beim Verkleinern bleiben mittlere
  Raten und Korrelationen nur in einem begrenzten Bereich zugleich erhalten [Q53].
- **Dendriten.** Nur die Koinzidenz S6 bleibt [Q51, Q52].
- **Zelltypen.** Die Fliege hat 8453 Zelltypen (laut Hirn-Bericht). Hier sind es vier Gruppen und
  zwei Transmitter.
- **Spikes.** Zeitpunkte der Freisetzung unter 0.1 ms fehlen; die Antwort beginnt 150 µs nach dem
  Start des Spikes [Q16].
- **Unterschiede zwischen Tieren.** Im Wurm sind 43 % der Verbindungen variabel [Q39].
- **Echte Konnektome.** Der Wurm mit 302 Neuronen passt in die Größe [Q38]. Seine Verdrahtung könnte
  später ein Basismodell geben (Entwurf 1). Datenzugang und Lizenz haben wir nicht geprüft, und in
  Wurmgröße sprengen 496 Kandidaten die Laufgrenze (Abschnitt 5.2).
- **Glia.** Ob Astrozyten unter physiologischen Bedingungen Transmitter freisetzen, ist umstritten
  [Q65, Q66]. Wir lassen es weg.
- **Später möglich, mit elementaren Reaktionen.**
  - **Modulator.** Eine Spezies M, die wenige Hub-Knoten bilden [Q41], bindet Rezeptoren zu einem
    stärker leitenden Komplex, B + M ⇌ BM (unser Vorschlag). Sie trifft 34 bis 44 % der Knoten, so
    dicht wie das Peptid-Netz im Wurm [Q42]. Ein Faktor in der Rate wie in Entwurf 1 wäre ein
    effektives Ratengesetz.
  - **Spaltkopplung.** Y⁺ₐ ⇌ Y⁺_b mit gleichem μ° hat die Gleichgewichtskonstante 1; die Physik
    erzwingt also die Symmetrie (Entwurf 2). Sie verbindet vor allem Hemmzellen gleichen Typs,
    bei Pvalb-Zellen 8.2 % der Paare [Q36, Q47, Q48].
  - **Treibstoff-Domänen.** Knoten einer Domäne teilen einen inneren Vorrat P_d, der über P ⇌ P_d
    nachfließt. Ein Astrozyt umhüllt im Mittel 4 Zellkörper [Q58] und integriert über zig Sekunden
    [Q59] (Entwurf 2).

### 3.13 Gradienten und Stabilität

- **Exakter Gradient.** BPTT durch den diskreten Schritt liefert den Gradienten des simulierten
  Modells [R57, Abschn. IV-V]. Der Adjoint scheitert bei steifen, dissipativen Systemen [R53,
  Abschn. III.A; R57, Abschn. III]. Die LTC-Arbeit wählt aus demselben Grund BPTT [Q2, Tab. 1].
- **Verschwinden und Explodieren.** Je Teilschritt ist der Diagonalfaktor 1/(1 + h·(g + δ))
  kleiner als 1. Die Jacobi-Matrix des Netzschritts hat aber Nebendiagonalen von etwa
  h·κᵢⱼ·(A − yᵢ)/(1 + h·(g + δ)). Ihr Spektralradius kann über 1 liegen, gerade nahe der
  Bistabilität (Korrektur aus Gutachten 1 zu Entwurf 1 und Entwurf 2). Gradienten können also
  verschwinden und explodieren [Q8].
- **Zeitkontinuierliche RNNs.** Dort verschwinden oder explodieren Gradienten bei jedem Löser [Q7,
  Status in Abschnitt 9.3].
- **Gegenmittel.** Globales Normclipping auf 1.0 [Q8]. Rechnen in float64, weil steife Probleme in
  32 Bit NaN-Gradienten riskieren [Q63]. Gleiche Startverteilung von W = A·κ in allen Armen. Ein
  Lehrplan von kurzen zu langen Verzögerungen (Entwurf 2).
- **Langsame Moden.** In Stufe S tragen langsame Zustände Geschichte. Über 128 Teilschritte zu
  2.5 ms schrumpft ein Gradient entlang der Erholung um den Faktor 0.49 bis 0.67, entlang NMDA um
  0.12 (nachgerechnet für τ = 450 bis 800 ms und 152 ms). Entwurf 1 nannte 0.49 bis 0.59;
  Gutachten 1 hat das richtiggestellt.

## 4. Zwei Wege

### 4.1 Der Keras-Prototyp

**Pakete.** PyPI am 2026-11-04 [Q62]:

| Paket | Version, Datum | Wheel für cp313, x86_64 | Rolle |
|---|---|---|---|
| keras | 3.15.1, 2026-07-29 | 2.4 MB | Schichten und Training; Python ≥ 3.11 |
| jax | 0.11.0, 2026-07-16 | 3.26 MB | Backend; verlangt numpy ≥ 2.1 und scipy ≥ 1.15 |
| jaxlib | 0.11.0, 2026-07-16 | 87.26 MB | Backend für die CPU |
| scipy | 1.18.1 | 35.31 MB | von jax verlangt; fehlt in der Umgebung |
| h5py, optree, ml-dtypes, opt-einsum | 3.16.0, 0.20.0, 0.6.0, 3.4.0 | 5.4, 0.49, 0.41 und 0.07 MB | Abhängigkeiten |
| Ersatz: jax und jaxlib | 0.12.0.1, 2026-10-21 | 3.47 und 86.85 MB | neueste Fassung, jünger als keras 3.15.1 |
| nicht nötig: tensorflow-cpu | 2.21.0 | 274.27 MB | anderes Backend |
| nicht nötig: torch | 2.15.0 | 539.17 MB | anderes Backend |
| nicht nötig: ncps | 1.0.1, 2024-08-14 | 0.06 MB | LTC und CfC für Keras |

- **Größe.** Zusammen etwa 137 MB Wheels, mit rich, absl-py, namex und packaging (nach unserer
  Summe aus [Q62]; die kleinen Pakete nach Entwurf 3). Installiert ist es größer. numpy 2.5.3 ist
  schon da.
- **Platz.** Am 2026-11-04 zeigte `df` auf `/` 30 GB frei. Die Sitzung hat aber ein festes
  Kontingent. Prüfen Sie es vor der Installation.
- **Backend JAX.** Es ist das kleinste Backend für die CPU. `KERAS_BACKEND=jax` muss vor dem ersten
  Import von keras gesetzt sein. float64 braucht `jax_enable_x64` und
  `keras.config.set_floatx("float64")`. Steife Probleme brauchen 64 Bit [Q63].
- **Versionen.** keras 3.15.1 erschien 13 Tage nach jax 0.11.0 [Q62]. Ob das Paar zusammen getestet
  ist, wissen wir nicht. Der erste Abnahmetest K0 prüft deshalb Import und einen Gradientenschritt.
- **ncps wird nicht gebraucht.** Die Keras-3-Zelle LTC übergibt bei unregelmäßigen Zeitschritten
  das ganze Tupel an `_map_inputs` (`ncps/keras/ltc_cell.py:326`) [Q5]. Sie rechnet den Zeitschritt
  in float32 (`ltc_cell.py:271-273`) [Q5]. Beides haben wir am Code geprüft.
- **Ohne Backend.** Keras 3.6.0 (1.19 MB, 2024-10-03) bringt ein NumPy-Backend mit [Q61, Q62]. Es
  hat `rnn`, aber `fit` wirft „fit not implemented for NumPy backend.“
  (`keras/src/backend/numpy/trainer.py:169`) [Q61]. Es taugt nur als Gegenprobe vorwärts.
- **Ort.** Der Prototyp liegt außerhalb des Repositorys, wie Ticket 01 es für Wegwerf-Code verlangt
  (`issues/01-deterministic-crn-simulator-spike.md:16-17`). `.scratch/` ist versioniert: 112
  Dateien in git (nachgezählt).
- **Was trainiert.** Nur Konzentrationen von Katalysatoren: Synapsen, Eingänge, Grundbildung,
  Abbau und das Enzym der Auslöschung. Energien und Δμ bleiben feste Gene
  (`issues/04-gradients-through-the-integrator.md:20-21`; `research.md` Abschnitt 8.5). Ein
  Katalysator skaliert k⁺ und k⁻ gleich. Kein Trainingsschritt kann also Detailed Balance verletzen
  (Ableitung aus Entwurf 2). Entwurf 1 trainierte Energien und Barrieren stetig; das widerspricht
  Ticket 04 (Korrektur aus Gutachten 2).

**Gerüst der Zelle für Stufe K.** Ungetestet; dieselbe Mathematik lief in numpy (Abschnitt 3.8).

```python
import os
os.environ["KERAS_BACKEND"] = "jax"            # before the first import of keras
import jax
jax.config.update("jax_enable_x64", True)      # the default is float32
import numpy as np
import keras
from keras import ops
keras.config.set_floatx("float64")


class StepRule(keras.constraints.Constraint):
    """Keeps h·A·sum_j kappa_ij <= 1 for every target node i (section 3.8)."""

    def __init__(self, mask, bound):
        self.mask, self.bound = mask, bound

    def __call__(self, w):
        into = ops.sum(ops.exp(w) * self.mask, axis=0)
        return w - ops.log(ops.maximum(into / self.bound, 1.0))[None, :]


class ChemCell(keras.layers.Layer):
    """One input step of stage K with L substeps. State: both rails of every node."""

    def __init__(self, wiring, drive_steps, eps=2.0**-10, substeps=4, **kw):
        super().__init__(**kw)
        self.n = wiring["n"]
        self.m_pos, self.m_neg = wiring["mask_pos"], wiring["mask_neg"]   # (n, n): row source j
        self.in_pos, self.in_neg = wiring["in_pos"], wiring["in_neg"]     # (m, n): input edges
        self.out = np.asarray(wiring["out"])                              # two output nodes
        self.eps = eps
        self.A = eps * 2.0 ** (drive_steps / 4.0)    # drive = drive_steps · RT·ln2/4, fixed
        self.L, self.h = substeps, 1.0 / substeps    # time unit: one input step
        self.state_size = [self.n, self.n]
        self.output_size = len(self.out)

    def build(self, input_shape):
        n, m = self.n, input_shape[-1]
        w0 = -np.log(self.A)                         # same start law of W = A·kappa in all arms
        self.log_k = self.add_weight(
            shape=(n, n), name="log_kappa",
            initializer=keras.initializers.RandomNormal(w0 - 1.0, 1.0),
            constraint=StepRule(self.m_pos + self.m_neg, 1.0 / (self.h * self.A)))
        self.log_kin = self.add_weight(
            shape=(m, n), name="log_kappa_in",
            initializer=keras.initializers.RandomNormal(w0, 1.0))
        self.th_delta = self.add_weight(shape=(n,), name="theta_delta", initializer="zeros")
        self.log_beta = self.add_weight(
            shape=(2, n), name="log_beta", initializer=keras.initializers.Constant(-3.0))
        self.log_ann = self.add_weight(
            shape=(), name="log_annihilation_enzyme",
            initializer=keras.initializers.Constant(3.0))

    def call(self, u, states):
        yp, ym = states
        A, eps, h = self.A, self.eps, self.h
        gam = eps * eps
        k = ops.exp(self.log_k)
        kp, km = k * self.m_pos, k * self.m_neg
        kin = ops.exp(self.log_kin)
        dl = 0.125 + (4.0 - 0.125) * ops.sigmoid(self.th_delta)   # delta in [1/8, 4] per step
        bp, bm = ops.exp(self.log_beta[0]), ops.exp(self.log_beta[1])
        c = h * ops.exp(self.log_ann)
        up, um = ops.relu(u), ops.relu(-u)                          # dual-rail input
        gin_p = ops.matmul(up, kin * self.in_pos) + ops.matmul(um, kin * self.in_neg)
        gin_m = ops.matmul(up, kin * self.in_neg) + ops.matmul(um, kin * self.in_pos)
        for _ in range(self.L):                                     # Jacobi: old catalysts
            gp = bp + gin_p + ops.matmul(yp, kp)
            gm = bm + gin_m + ops.matmul(yp, km)
            ap, am = 1.0 + h * (gp + dl), 1.0 + h * (gm + dl)
            rp = yp + h * (gp * A + dl * eps)
            rm = ym + h * (gm * A + dl * eps)
            b = ap * am - c * (rp - rm)
            q = am * (rp + c * gam)
            s = ops.sqrt(b * b + 4.0 * c * ap * q)
            pos = b >= 0
            den1 = ops.where(pos, b + s, 1.0)                      # safe denominators, so
            den2 = ops.where(pos, 1.0, 2.0 * c * ap)                # no branch yields inf
            yp = ops.where(pos, 2.0 * q / den1, (s - b) / den2)
            ym = (rm + c * gam) / (am + c * yp)
        d = ops.take(yp, self.out, axis=1) - ops.take(ym, self.out, axis=1)
        return d, [yp, ym]

    def get_initial_state(self, batch_size=None):
        e = ops.full((batch_size, self.n), self.eps)
        return [e, e]                                               # equilibrium: y = eps


def chem_model(wiring, drive_steps, m):
    cell = ChemCell(wiring, drive_steps)
    x = keras.Input(shape=(None, m))
    d = keras.layers.RNN(cell)(x)
    z = keras.layers.Rescaling(8.0 / cell.A)(d)                     # fixed read-out, no free gain
    model = keras.Model(x, z)
    model.compile(optimizer=keras.optimizers.Adam(1e-2, global_clipnorm=1.0),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    return model
```

- **Schnittstelle.** `keras.layers.RNN` verlangt `state_size`, `output_size`,
  `get_initial_state(batch_size)` und `call(inputs, states)` mit Rückgabe `(output, new_states)`.
  Die Entwürfe 1 und 2 haben das an der Quelle von Keras geprüft; wir nicht.
- **Feste Auslese.** Die Logits sind z = (8/A)·dₒ an zwei Ausgangsknoten. Die Auslese hat keine
  freie Verstärkung, damit die Chemie den Kontrast selbst erzeugen muss (Entwurf 2).
- **Dichte Masken.** Keras rechnet mit dichten, maskierten Matrizen. Die FLOPs zählen wir trotzdem
  dünn, nach Abschnitt 3.8.
- **Referenz.** Eine numpy-Fassung rechnet denselben Schritt in float64, in fester Reihenfolge:
  Kanten nach Ziel, dann nach Quelle. Sie dient den Physiktests, den finiten Differenzen und später
  dem Bitvergleich mit Rust.
- **Stufe S.** Sie folgt als eigene Zelle in Ticket 11. Ihre Sackgassen-Relaxation braucht
  ebenfalls sichere Nenner.

### 4.2 Die Modellart `crn` in der Brutmaschine

**Genom.** Ein Genom ist reine Daten und wird nie ausgeführt (`src/genome.rs:1-4`). Unbekannte
Felder werden abgelehnt (`src/genome.rs:75-78`). Unser Vorschlag, nur aus ganzen Zahlen, damit
`Arch` weiter `Eq` ableitet (`src/genome.rs:57`):

```rust
#![forbid(unsafe_code)]
/// A recurrent reaction network (`Arch::Crn`): dual-rail nodes, counted synapses and integer
/// energy levels. Integers only, so `Arch` keeps deriving `Eq`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrnGenome {
    pub groups: [u16; 4],              // sensory, inter, command, motor (NCP order)
    pub nodes: Vec<CrnNode>,           // as many as the groups hold, at most 64
    pub edges: Vec<CrnEdge>,           // strictly increasing by (post, pre); pre != post; <= 1024
    pub inputs: Vec<CrnInput>,         // channel to sensory node; at most 64
    pub drive: u8,                     // drive in steps of RT·ln2/4, 0..=64
    pub unit_levels: [u8; 3],          // kappa of excitatory, inhibitory and input edges
    pub settle: u8,                    // symbols without input after the last one, 0..=4
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub synapses: Option<CrnSynapses>, // stage S; absent in stage K
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Transmitter { Glutamate, Gaba }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Sign { Plus, Minus }

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrnNode {
    pub transmitter: Transmitter, // identity (Dale); a GABA node has only Minus edges
    pub decay: u8,                // delta = kappa·2^(-decay/4)
    pub basal: [u8; 2],           // beta+ and beta- as levels
    pub annihilation: u8,         // enzyme of K5 as a level
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrnEdge { pub pre: u16, pub post: u16, pub sign: Sign, pub count: u8 } // 1..=36

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrnInput { pub channel: u16, pub post: u16, pub sign: Sign, pub count: u8 }
```

- **Kein Trainingsfeld im Genom.** Entwurf 2 trug `train: Option<{optimizer, lr, epochs}>` im
  CRN-Genom. `Genome` hat diese Gene schon (`src/genome.rs:79-98`), und ein Gleitkommafeld bräche
  das `Eq` von `Arch` (Korrektur aus Gutachten 2). In Phase 1 stehen die Trainingsgene von `Genome`
  auf festen Werten, und `hidden` bleibt leer. Sonst hätten gleiche Netze verschiedene
  Content-IDs (Entwurf 1).
- **Feste Größen.** ε = 2^(−10) c°, τ_in = 10 ms und κ = 16 sind Konstanten des Laufs, keine Gene.
  Die Zahl der Teilschritte folgt aus der Schrittregel.

**Naht nach dem Muster des CNN.**

- `Arch::Crn(CrnGenome)` neben `Mlp` und `Cnn` (`src/genome.rs:58-65`), mit Zweigen in
  `param_count`, `train_flops` und `validate` (`src/genome.rs:195`, `212`, `232`).
- `ModelKinds::Crn` mit dem Namen `crn` für `--models crn` (`src/config.rs:31-69`). Den Namen „rnn“
  nutzen drei Tests als Beispiel einer unbekannten Art; ein eigener Name vermeidet den Konflikt.
- Ein eigener Zweig in `Config::for_models` (`src/config.rs:210-231`). Für Nicht-MLP-Arten setzt er
  heute die Aufgabe shapes, 8 Zufallsgenome, 8 Kinder je Generation, 5 Generationen und 3600 s
  (`src/config.rs:221-227`). Für `crn` braucht es motif-1d und eine eigene Laufgröße (Korrektur aus
  Gutachten 1 und Gutachten 2).
- `Config::validate` weist flache Aufgaben für jede Nicht-MLP-Art ab (`src/config.rs:452`). In v1
  bleibt das so; `crn` läuft nur auf Folgenaufgaben (Korrektur aus Gutachten 2).
- Ein `Model`-Fall mit `"kind":"crn"` (`src/model.rs:1-36`) und eine Prüfung in `brut-eval` gegen
  1e11 FLOPs und 60 s (`src/hidden_eval.rs:84`, `91`).
- Der Simulator liegt im Crate, weil `brut-eval` dieselben Bits braucht (`research.md`
  Abschnitt 8.1).

**Grenzen.**

- **Gemeinsame Limits.** 20 000 Parameter, 2e10 Trainings-FLOPs, 30 s je Kandidat, 300 s und 2e12
  FLOPs je Lauf (`src/config.rs:94-107`).
- **Eigene Obergrenzen** (unser Vorschlag): 64 Knoten, 1024 Synapsen, 64 Eingangskanten,
  Synapsenzahl 1 bis 36, Pegel und Antrieb 0 bis 64, höchstens 64 Teilschritte je Symbol,
  Spitzenkonzentration A + ε ≤ 64 c°.
- **Zeit.** Die geschätzte Zeit muss unter max_train_flops/1e9 s bleiben, wie beim CNN
  (`src/genome.rs:322-331`; `src/cnn/shape.rs:532`). Das Zeitmodell rechnet mit der schlechtesten
  FLOP-Rate, die Ticket 01 misst. Dann greift keine Uhrgrenze, wie ADR 0007 es für gleiche Bits
  verlangt.
- **Laufzeit des Laufs.** `max_total_seconds` steht auf 3600 s, wie bei CNN-Läufen, damit die Uhr
  keinen Lauf beendet.

**Determinismus.**

- Zustand in f64; in der Schleife nur +, −, ×, ÷ und √ [R70, R71].
- Raten aus der Tabelle der vier Bitmuster; kein exp, ln oder powf in der Schleife [R71].
- Feste Reihenfolge: Synapsen nach (post, pre).
- Ein Thread je Kandidat; Kandidaten parallel, Ergebnisse in Slot-Reihenfolge
  (`docs/STAGE1_CONTRACT.md:42-45`).
- Die Auslese löst Ridge per Cholesky; √ rundet exakt [R71].
- NaN beendet einen Kandidaten als Fehlschlag und wird nie gehasht (`research.md` Abschnitt 7.3).
- **Ganze Quanten als Variante.** Entwurf 3 hält den Zustand in ganzen Quanten von 2^(−40) c°.
  Dann gilt die Erhaltung bitgenau. Offen ist dort, wie ein Austauschschritt seine Quanten auf
  mehrere Reservoire verteilt, damit die Buchung exakt bleibt (Gutachten 2). Wir übernehmen die
  Zweierpotenz-Raten sofort und die Quanten als Variante B von Ticket 01.

**Crates.** Es kommt keine Crate hinzu. Invariante I1 und ihr Test bleiben unverändert
(`docs/STAGE1_CONTRACT.md:33-35`; `tests/learner_security.rs:95-106`). Jede neue Datei beginnt mit
`#![forbid(unsafe_code)]` (I2); Bibliothekscode panikt nicht (I5); Aufzeichnungen enthalten nur
endliche Zahlen (I8) (`docs/STAGE1_CONTRACT.md:36-53`).

**Lernen im Kandidaten.**

- **Phase 1.** Zucht setzt alle Gene. Im Leben lernt nur die Auslese: eine affine Abbildung der
  quantisierten dₒ der Motor-Knoten auf die Klassen, geschlossen per Ridge auf dem Train-Split
  gefittet. Reine Zucht ohne trainierte Auslese hätte für Klassen keinen Lernschritt (Gutachten 2
  zu Entwurf 3).
- **Belege für Zucht.** Gerichtete Evolution formte chemische Gleichgewichtsnetze meist in höchstens
  etwa 100 Runden [Q24, Vorabdruck]. Ein genetischer Algorithmus fand gute Netze aus DNA-Kügelchen
  mit 100 bis 300 Individuen in 10 bis 40 Generationen [Q26, Vorabdruck]. Beides ist nur simuliert.
- **Startzustand.** Der Ruhezustand ohne Eingang wird je Kandidat einmal berechnet, über 16 Symbole.
  Er gilt dann für jede Folge.
- **Risiko.** Entwurf 1 fand für untrainierte Zufallsnetze mit Ridge-Auslese auf motif-1d nur 49 bis
  53 % (3 Seeds, im Speicher gerechnet, nicht vorregistriert). Ein MLP mit gleichem Kopf erreichte
  0.513, ein flaches 1-D-CNN 0.803
  (`.scratch/cnn-basismodelle/issues/02-model-seam-and-grid-tasks.md:153`, `248`).
- **Phase 2.** BPTT im Kandidaten für die Konzentrationen der Katalysatoren, mit den Trainingsgenen
  von `Genome`. Die trainierten Werte sind Gewichte in der Modelldatei, keine Gene. Jedes Kind
  trainiert von vorn (ADR 0008:76-81). Phase 2 kommt nur, wenn der Brutversuch P2 die Schwelle aus Abschnitt 6.5 verfehlt.
- **Keine fremden Genome im Endpunkt.** In Keras trainierte Genome tragen Rechenzeit, die kein Lauf
  bezahlt hat. Sie dürfen nur in Erkundungsläufen als benannte `base_genomes` einziehen
  (`src/config.rs:157-162`), nie in Läufen des Endpunkts. Die Beförderung prüft die Herkunft aus
  einem Lauf (ADR 0008:112-120); für extern trainierte Genome gibt es keine Regel. Gutachten 2 nannte
  diese Brücke bei Entwurf 1 fatal; wir schließen sie für den Endpunkt aus.

**Fitnesskosten.**

- Die Formel bleibt: Hauptwert − 0.005·log10(Parameter) − 0.002·log10(FLOPs)
  (`src/score.rs:396-415`).
- **Parameter.** Die Zahl der numerischen Gene: E + M + 4N + 4, dazu die Auslese mit
  (Motor-Knoten + 1) mal Klassen. Bei N = 48, E = 384 und M = 8 sind das 606 (nach unserer
  Rechnung).
- **FLOPs.** Simulation auf dem Train-Split, Fit der Auslese und Halbschritt-Test, zusammen 7.0e8
  (Abschnitt 5.2). Die Validierung zählt nicht, wie bei MLP und CNN: Der Lauf bucht nur
  `entry.train.flops` (`src/evolve.rs:791`). Entwurf 3 hätte den Valid-Lauf mitgebucht (Korrektur
  aus Gutachten 2).
- **Abgabe.** 0.0139 für die Parameter und 0.0177 für die FLOPs, zusammen 0.0316. Das TCN-
  Basismodell zahlt 0.0385, ein MLP an den Grenzen 0.0421 (nach unserer Rechnung).
- **Dissipation.** Sie wird je Folge berichtet, aber nicht abgezogen (Entscheidung 5).

**Operatoren.** Genau einer je Kind; ein ungültiges Kind wird bis zu 16-mal neu gezogen, dann kommt
der nächste Operator (ADR 0008:41-59). Unsere Liste nach Entwurf 3:

1. Synapsenzahl ±1.
2. Synapse hinzufügen, mit Vorliebe gegenseitig oder ein Dreieck schließend.
3. Synapse entfernen.
4. Ein Ende einer Synapse in die Nähe umhängen.
5. Vorzeichen einer Synapse eines glutamatergen Senders kippen.
6. Transmitter eines nicht-sensorischen Knotens wechseln; seine Synapsen folgen; der GABA-Anteil
   bleibt zwischen 5 und 30 %.
7. Ein Pegel ±1: Abbau, Grundbildung, Auslöschung oder Einheitsleitwert.
8. Antrieb ±1.
9. Einen Knoten samt Synapsen verdoppeln oder löschen, nach dem Vorbild von BioJazz [R63].

Die Kreuzung nimmt für jeden Knoten seine eingehenden Synapsen von Elternteil A oder B. Im Lauf des
azyklischen Zwillings bleiben nur Vorwärtssynapsen. Ohne Erhaltungsregel waren gute künstliche
Stoffwechselnetze sehr schwer zu finden [R64, Abschn. VI.A]; unser Genom erhält per Bau.

**Byte-Treue.** `tests/mlp_golden.rs` und die E2E-Tests bleiben unverändert (ADR 0008:21-24). Neu
kommt ein Test „a short crn run on motif is reproducible byte for byte“ hinzu, nach dem Vorbild
des CNN (ADR 0008:76-81).

### 4.3 Nötige ADRs

Die nächste freie Nummer ist 0012; `docs/adr/` endet bei 0011. Titel englisch, wie bei den
bisherigen ADRs.

| ADR | Titel | Entscheidet |
|---|---|---|
| 0012 | Reaction networks are a third Model kind, bred in the MLP mode | `Arch::Crn`, `--models crn`, Zweig in `for_models` mit motif-1d und eigener Laufgröße, nur Folgenaufgaben, Modelldatei `"kind":"crn"`, Prüfung in `brut-eval`, feste Trainingsgene in Phase 1, Ridge-Auslese, FLOP-Buchung wie MLP und CNN, keine extern trainierten Genome in Endpunktläufen, Beförderung nach ADR 0008 |
| 0013 | Reaction networks obey reversible mass action, and only chemostats drive them | Vorlagen K1 bis K5 und S1 bis S7, Energieschritt RT·ln2/4, κ = 16, Bereiche, Vorzeichen je Synapse bei fester Transmitter-Identität, Energiebilanz nur berichtet |
| 0014 | Reaction networks step by closed implicit solves of driven cycles and relaxation of dead ends | f64, Tabelle aus vier Bitmustern, Jacobi-Kopplung, Schrittregel, Halbschritt-Test, quantisierte Auslese, Fehlerregeln, FLOP-Gewichte, Zeitmodell |
| 0015 | A reaction-network child changes by one operator, and wiring starts from brain statistics | Operatoren, Kreuzung je Knoten, 16 Neuzüge, Generator, azyklischer Zwilling als Laufschalter |
| 0016 | Keras stays outside the core | nur nötig, wenn Keras-Code ins Repository soll: eigener Ordner, feste Versionen mit Hashes, nie von `brut` gestartet, kein Rückfluss in Endpunktläufe; bisher nutzen Python-Werkzeuge nur die Standardbibliothek (`tools/rearc_to_jsonl.py:7`) |

### 4.4 Nötige Tickets

Alle lägen in `.scratch/chemische-netze/issues/` und starteten mit `Status: needs-triage`
(`docs/agents/triage-labels.md`). Dieses Papier legt keines an.

| Ticket | Typ | Blockiert durch | Frage |
|---|---|---|---|
| 01, Ergänzung | prototype | keins | Knotenschritt als viertes Schema; Sperre bei Δμ = 0; Schrittregel kalibrieren; ÷ und √ messen; Variante B mit ganzen Quanten |
| 02, Ergänzung | research | keins | Vorlagen K1 bis K5 und S1 bis S7; Energieschritt; Vorzeichen je Synapse bei fester Transmitter-Identität; zwei Batterien |
| 04, Ergänzung | prototype | 01 | Dual-Rail-Knoten als reversible Variante von [R9]; Rückwärtsmodus für Phase 2 |
| 05-keras-chem-cell-prototype | prototype | keins | Der kleinste Prototyp, P0 und P1 (Abschnitt 6) |
| 06-breeding-spike-recurrent-vs-acyclic | prototype | 01, 05 | Der Brutversuch P2 (Abschnitt 6.5) |
| 07-brain-statistics-generator | task | 02 | Generator, Zwillinge, Kennzahlen gegen ER und CFG |
| 08-reaction-network-model-kind | task | 06, 07; ADR 0012 bis 0015 angenommen | Einbau nach dem Muster des CNN |
| 09-prereg-crn-recurrence | task | 08 | `docs/PREREG_CRN_RECURRENCE.md` und Pilot |
| 10-glossary-reaction-networks | grilling | 02 | Begriffe per /domain-modeling: Reaction network, Chemostat, Dual-rail pair (`map.md:40-43`), dazu Drive, Synapse count und Acyclic twin |
| 11-synapse-chemistry-stage-s | prototype | 05 | Stufe S in numpy und Keras; Tests ohne Treibstoff, Erhaltung, Fixpunkte; G-Protein-Stufe |
| 12-energy-bill-in-fitness | research | 06 | Soll die Dissipation in die Fitness eingehen? (Fog 5) |

Ticket 03 bleibt: Es prüft den kompilierten Weg, nicht diesen.

## 5. Aufgaben, Vergleich und Endpunkt

### 5.1 Aufgaben

| Aufgabe | Form | Rolle |
|---|---|---|
| motif-1d | 64 Symbole über 4 Kanäle, 2 Klassen, gleichverteiltes Rauschen ±0.3, Splits 1000, 1000 und 2000 (`src/task.rs:40`, `152-158`, `512-514`, `929-960`) | Endpunkt im Kern; die einzige Folgenaufgabe des Repositorys |
| Bit halten mit Störung | 3 Kanäle (Setzen, Löschen, Störer), 80 Schritte, Puls bei t₀ von 4 bis 11, drei Störpulse, Rauschen ±0.3 (Entwurf 2) | nur im Prototyp P1 |
| Echo-k | Folgen wie motif-1d; Klasse 1, wenn das Symbol k Schritte vor dem Ende 0 oder 1 ist; k aus 1, 4 und 16 (Entwurf 3) | nur im Brutversuch P2; Echo-1 als Positivkontrolle |
| Parität als Folge | 4 bis 8 Bits als Pulse zu unregelmäßigen Zeiten | später; LTC blieb beim ereignisbasierten Bitstrom-XOR bei 49.11 %, CfC erreichte 99.42 % [Q3, Tab. 4-7, laut Chemie-Bericht] |
| flache Aufgaben | Eingang über die Lesezeit konstant | nur im Prototyp; `Config::validate` weist sie für Nicht-MLP-Arten ab (`src/config.rs:452`) |
| Bildaufgaben | 256 Eingänge | ausgeschlossen; 512 Eingangsspezies sind zu groß (`research.md` Abschnitt 8.6) |

- **Ausgeglichene Klassen.** In motif-1d ist die Klasse jeder Folge `i % 2` (`src/task.rs:936`). Jeder
  Split hat also genau gleich viele Folgen je Klasse. Ein Modell, dessen Ausgabe nicht vom Eingang
  abhängt, trifft genau 50 %.
- **Rauschen.** Jeder Wert erhält gleichverteiltes Rauschen der halben Breite 0.3
  (`src/task.rs:40`, `682-689`). Die Werte liegen in [−0.3, 1.3); Dual-Rail ist also nötig.

### 5.2 Kosten je Aufgabe

Nach unserer Rechnung, für Stufe K mit L = 4 Teilschritten:

| Netz | FLOPs je Teilschritt | je Folge | Train-Split | Kandidat gesamt |
|---|---|---|---|---|
| Kern auf motif-1d, N = 48, E = 384, M = 8 | 2400 | 6.2e5 | 6.2e8 | 7.0e8 |
| Prototyp auf „Bit halten“, N = 24, E = 144, M = 12 | 1104 | 4.3e5, mit 16 Schritten Vorlauf | nicht im Kern | nicht im Kern |
| Wurmgröße, N = 302, E = 3242, M = 16 | 16 752 | 4.4e6 | 4.4e9 | etwa 5e9 |
| Stufe S, N = 48, E = 384 | etwa 12 900 | 3.3e6 | 3.3e9 | etwa 3.7e9 |

- **Vergleich mit dem TCN.** Das TCN-Basismodell kostet je Folge 9.65e9/(3·1000·5) = 6.4e5 FLOPs
  (nach unserer Rechnung aus `.scratch/cnn-basismodelle/issues/04-families-b.md:165`). Der Kern
  liegt damit 4 % darunter.
- **Lauf.** 496 Kandidaten des Kerns kosten 3.5e11 FLOPs, unter der Grenze von 2e12. Bis zur Grenze
  passen etwa 2900 Kandidaten. In Wurmgröße sprengen 496 Kandidaten die Grenze.
- **Bewertung.** Der Hidden-Split kostet 1.2e9 FLOPs, unter 1e11 (`src/hidden_eval.rs:84`).
- **Stufe S** ist etwa fünfmal so teuer wie der Kern. Sie braucht kleinere Läufe oder weniger
  Teilschritte.

### 5.3 Vergleichsbasen bei gleichem Budget

- **Im Kern.** Das Glossar definiert die Vergleichsbasis als Standardmethode mit derselben
  Rechenmenge (`CONTEXT.md:133-135`). Für motif-1d ist das ein CNN-Lauf mit dem TCN-Basismodell und
  seinen 14 Zwillingen (ADR 0008:35-37). Das TCN allein erreichte auf Valid 0.951, 0.961 und 0.957
  (`.scratch/cnn-basismodelle/issues/04-families-b.md:157-170`).
- **Spezifische Kontrolle.** Der azyklische Zwilling misst, ob die Rückkopplung trägt (Entwurf 3).
  Er ist exakt rechengleich: gleiche N, E und FLOPs.
- **Untergrenze.** Ein MLP-Lauf; ein MLP mit gleichem Kopf erreichte 0.513 (Ticket 02 der
  CNN-Basismodelle, Zeile 248).
- **Im Prototyp.** Eine GRU mit gleichen FLOPs je Schritt: H = 25 gibt 4500 FLOPs gegen 4464 des
  Netzes, also 0.8 % mehr, mit 2250 Parametern (nach unserer Rechnung).
- **Gleiches Budget.** Alle Arme eines Seeds bekommen dieselben Grenzen: 2e10 Trainings-FLOPs je
  Kandidat, 2e12 FLOPs und 3600 s je Lauf. Jeder Arm hat genug Generationen, um die FLOP-Grenze zu
  erreichen. Wir berichten den tatsächlichen Verbrauch.

### 5.4 Primärer Endpunkt

Vorregistrierbar. Die Datei `docs/PREREG_CRN_RECURRENCE.md` legt ihn vor dem ersten Lauf fest, wie
`docs/PREREG_STEP1_CIRCUITS.md` es für Schritt 1 tat.

- **Aufgabe.** motif-1d, unverändert.
- **Arme je Seed.** R: `crn` mit dem Hirnstatistik-Generator. A: `crn` als azyklischer Zwilling.
  B: `cnn` mit den Basismodellen.
- **Messgröße.** Die Genauigkeit auf dem Hidden-Split, gemessen von `brut-eval`, für den Finalisten
  mit der besten Validierungs-Fitness (ADR 0008:83-96).
- **Seeds.** 5001 bis 5008. Nach unserer Suche nennt kein Dokument diese Zahlen als Seeds. Schritt 2
  reserviert 1002 und folgende (`docs/PREREG_STEP2_SOUL.md:128-131`); die Seeds 1001 bis 1008 aus
  Entwurf 1 sind also nicht frei (Korrektur aus Gutachten 2). Pilotseeds 5901 bis 5908 zählen nie.
- **H-R, Rückkopplung.** Erfolg, wenn R auf mindestens 7 von 8 Seeds über A liegt und der Median von
  R − A mindestens 0.05 beträgt. Gleichstände und ungültige Läufe zählen gegen R. Der exakte
  Vorzeichentest gibt P(mindestens 7 von 8 | p = 1/2) = 9/256 = 0.035 (nachgerechnet).
- **H-B, Vergleichsbasis.** Nicht-Unterlegenheit gegen B. Mit dᵢ = Rᵢ − Bᵢ gilt
  L = mean(d) − 1.895·sd(d)/√8. Erfolg, wenn L > −0.05. Dabei ist 1.895 das 95-%-Quantil der
  t-Verteilung mit 7 Freiheitsgraden (nachgerechnet).
- **Erfolg der Modellart.** H-R und H-B müssen beide gelten. Weil beide gelten müssen, braucht es
  keine Korrektur für mehrfaches Testen (unsere Lesart des Prinzips „intersection-union“).
- **Gültigkeitstor.** Arm B erreicht im Mittel mindestens 0.85. Sonst ist der Lauf ungültig, nicht
  negativ (Entwurf 1).

### 5.5 Sperren vor dem Lesen

1. **Ohne Antrieb.** Ein Lauf von R mit Antrieb 0 ergibt auf dem Hidden-Split genau 0.500. Die
   Auslese ist dann für jeden Eingang dieselbe: Alle quantisierten dₒ sind 0, die Ridge-Gewichte
   sind 0, und die Logits sind gleich. Bei Gleichstand nimmt der Scorer den ersten Index
   (`src/score.rs:204-209`), also immer Klasse 0. Er trifft damit genau die Hälfte (Ableitung nach
   Entwurf 2; Klassenbalance aus `src/task.rs:936`).
2. **Halbschritt.** Jeder Finalist besteht den Halbschritt-Test.
3. **Physik.** Die Abnahmetests der Tickets 01 und 02 bestehen am eingefrorenen Commit.

### 5.6 Sekundäre Endpunkte

Beschreibend, ohne Anspruch:

- Genauigkeit gegen Antrieb, für Antriebe von 0 bis 46 Schritten (0 bis 8 RT).
- Dissipation je richtig erkannter Folge, als Front gegen die Genauigkeit.
- Kennzahlen der Finalisten gegen die Ziele aus Abschnitt 3.11: Bleiben gezüchtete Netze
  hirnähnlich?
- Anteil aktiver Knoten. Im Gehirn sind höchstens 15 % zugleich aktiv [Q56].
- Hidden-Lücke je Arm.
- R gegen den MLP-Lauf und gegen den Zufallszwilling.
- Später Stufe S gegen Stufe K.

## 6. Der kleinste Prototyp

### 6.1 Frage

Die größte offene Frage lautet: Lernt ein Netz, das Detailed Balance streng einhält und nur über
Chemostaten angetrieben wird, durch seinen Integrator ein Gedächtnis, das Störungen übersteht?
Verschwindet dieses Gedächtnis ohne Antrieb, wie die Physik es verlangt? Und schlägt das rekurrente
Netz seinen azyklischen Zwilling? Fällt die Antwort negativ aus, lohnt sich der Bau in Rust nicht.

### 6.2 Aufbau

- **P0, nur numpy, ohne Installation.** Referenzschritt, Generator mit beiden Zwillingen,
  Physiktests, FLOP-Zähler und der Erzeuger der Aufgabe „Bit halten“. Teile davon liefen schon im
  Speicher (Abschnitt 3.8).
- **P1, Keras mit JAX auf der CPU in float64.** Training auf „Bit halten mit Störung“.
- **Netz.** N = 24 Knoten, davon 21 erregend und 3 hemmend; Eingangsgrad 6, also E = 144; 3 Kanäle
  mit je 4 Kanten, also M = 12; 2 Ausgangsknoten; L = 4 Teilschritte.
- **Zeit im Prototyp.** Die Zeiteinheit ist ein Eingabeschritt. δ liegt zwischen 1/8 und 4 je
  Schritt. Die Zeitkonstante eines Knotens ist also höchstens 8 Schritte. Über mindestens 66 Schritte
  trägt das Bit nur die Dynamik des Netzes (Entwurf 2).
- **Widerspruch aufgelöst.** Abschnitt 3.4 erlaubt Zeitkonstanten über fast fünf Größenordnungen,
  der Prototyp nur über den Faktor 32. Das ist Absicht: Der Prototyp prüft Gedächtnis im Netz, nicht in
  einem einzelnen langsamen Knoten (Korrektur aus Gutachten 1, das den Widerspruch in Entwurf 2
  fand).
- **Arme.**
  - C(Δμ) mit Antrieben von 0, 6, 12, 23 und 46 Schritten, also 0, 1.04, 2.08, 3.99 und 7.97 RT,
    je 5 Seeds.
  - C_DAG, der azyklische Zwilling, bei 46 Schritten, 5 Seeds.
  - GRU mit H = 25, 5 Seeds.
- **Training.** Adam mit Lernrate 1e-2 und `global_clipnorm` 1.0, Batch 64, 2000 Schritte. Die
  Verzögerung wächst im Lauf von 16 auf 66 Schritte. Vor jeder Folge laufen 16 Schritte ohne
  Eingang. Eine einzige Lernrate hält das Budget fest; Entwurf 2 sah 8 Sätze je Arm vor, die nicht
  budgetiert waren (Korrektur aus Gutachten 2).
- **Rechenmenge.** Etwa 1.65e11 FLOPs je Lauf mit BPTT, 35 Läufe, zusammen 5.8e12 (nach unserer
  Rechnung). Auf 4 Kernen mit je 1 GFLOP/s wären das 24 min. Mit dem Aufwand von JAX schätzen wir
  0.5 bis 3 Stunden; gemessen ist das nicht.
- **Aufwand.** Etwa 300 Zeilen numpy und 300 Zeilen Keras (Schätzung aus Entwurf 2).

### 6.3 Abnahmetests

Sie müssen bestehen, bevor ein Ergebnis gelesen wird. In der Spalte „Herkunft“ steht E1 bis E3 für
Entwurf 1 bis 3, mit dessen Testnummer.

| ID | Test | Schwelle | Herkunft | Schon gerechnet? |
|---|---|---|---|---|
| K0 | Import von keras 3.15.1 mit jax 0.11.0; ein Gradientenschritt in float64 | läuft, kein NaN | neu | nein |
| AT-1 | Positivität: 1e4 Zufallsnetze, h von 1e-3 bis 1e3, 150 Schritte | kein negativer und kein nicht endlicher Wert | E2 P1, E1 AT1 | ja, 240 Läufe |
| AT-2 | Schranke max y ≤ max(y(0), A, ε) + √γ | gilt | E2 P2 | in Entwurf 2, Verhältnis 0.99999 |
| AT-3 | Bilanz \|ΔΣy − (N_P − N_W)\| | ≤ 1e-8 des Bruttoflusses | E2 P3 | in Entwurf 2, 1.4e-9 |
| AT-4 | Ohne Antrieb: Start bei ε, Zufallseingänge | \|y − ε\|/ε ≤ 1e-12 | E2 P4 | ja, 6.7e-16 |
| AT-5 | Stationär: T·σ = RT·Δμ·J_P; Gl. 60 und 61 von [R39] | relativ ≤ 1e-10 | E2 P5, E1 AT4 | ja, 3e-16 |
| AT-6 | Fixpunkt unabhängig von h | relativ ≤ 1e-12 | E1 AT5; Korrektur zu E3 A5 | ja, 1.3e-15 |
| AT-7 | Halbschritt: Ausgaben bei h/2 | Änderung ≤ 1 %; kein Muster der Periode 2h | E2 P6 | nein |
| AT-8 | BPTT gegen zentrale finite Differenzen, float64, N = 8, T = 20, 20 Parameter | relativ ≤ 1e-6 | E2 P7, E1 AT8 | nein |
| AT-9 | Antrieb 0 auf „Bit halten“ | genau 50 %; Gleichstand \|Δz\| < 1e-9·A zählt als Klasse 0 | E2 P8 | nein |
| AT-10 | Antrieb 46 Schritte | mindestens 95 % auf mindestens 4 von 5 Seeds | E2 P9 | nein |
| AT-11 | Monotonie | der Median fällt mit wachsendem Antrieb nirgends um mehr als 2 Prozentpunkte | E2 P10 | nein |
| AT-12 | Kontrast des gelernten Speichers | unter (e^Δμ + 1)² | E2 P11 | nein |
| AT-13 | Lösbarkeit | GRU mindestens 95 % | E2 P12 | nein |
| AT-14 | Übertrag | numpy und Keras vorwärts ≤ 1e-12; zwei numpy-Läufe bitgleich; Runden auf ganze Gene kostet ≤ 2 Prozentpunkte | E2 P13, E1 AT12, AT13 | nein |
| AT-15 | Verdrahtung | Ziele aus Abschnitt 3.11; Zwilling azyklisch mit gleichem E | E1 AT11, korrigiert | nein |
| AT-16 | FLOPs gezählt gegen 2E + 34N je Teilschritt plus 4M je Eingabeschritt | ≤ 1 % | E1 AT9 | nein |
| AT-17 | Zweiter Hauptsatz je Schritt: W_c − ΔG ≥ 0 | kein negativer Wert über 1e-12 relativ | E3 A4 | nein |

- **Verdrahtungstest korrigiert.** Entwurf 1 verlangte in seinem Test AT11 r/p ≤ 12. Sein eigener
  Generator lieferte bei N = 302 aber 12.1 (Gutachten 1). AT-15 misst nun gegen ER und CFG mit den
  Spannen der Tabelle in Abschnitt 3.11, und λ ist ein Parameter des Generators.

### 6.4 Entscheidungen nach P1

- **Einer der Tests AT-1 bis AT-9 scheitert.** Code oder Physik haben einen Fehler. Wir halten an
  und suchen.
- **AT-10 scheitert, AT-13 besteht.** Das Netz ist schwer zu trainieren. Wir prüfen Lehrplan,
  Startwerte, einen stärkeren Antrieb und eine getrennte Speicherzelle [Q3, Q7]. Chemisch wäre das
  ein bistabiler Schalter aus Kinase und Phosphatase (Lesart aus Entwurf 2).
- **C besteht, C_DAG fällt unter 60 %.** Die Rückkopplung trägt das Gedächtnis. Wir bauen P2.
- **C und C_DAG bestehen beide.** Die Aufgabe ist für einen azyklischen Stapel aus langsamen Knoten
  zu leicht. Wir verlängern die Verzögerung.
- **Alles besteht.** Die Tickets 01, 02 und 04 werden um den Knotenschritt erweitert, und P2
  beginnt.

### 6.5 Danach: der Brutversuch P2

- **Frage.** Findet Zucht ohne Gradienten unter den Grenzen der Brutmaschine ein rekurrentes Netz,
  das auf einer Gedächtnisaufgabe seinen azyklischen Zwilling schlägt (Entwurf 3)?
- **Umfang.** Eine Wegwerf-Crate außerhalb des Repositorys mit Stufe K, Generator, Zwilling,
  Operatoren 1 bis 8, Ridge-Auslese, Echo-k und motif-1d. Sie nutzt nur Crates, die das Repository
  schon erlaubt.
- **Budget.** Je Arm und Seed 2e12 FLOPs auf motif-1d und 5e11 je k auf Echo-k, 8 Pilotseeds.
  Zusammen etwa 5.6e13 FLOPs, also 3.9 Stunden auf den 4 Kernen dieser Umgebung bei 1 GFLOP/s je
  Kern. Entwurf 3 schätzte 2 Stunden auf 8 Kernen; `nproc` zeigt hier 4 (Korrektur aus
  Gutachten 2).
- **Positivkontrolle.** Auf Echo-1 erreichen beide Arme auf mindestens 6 von 8 Pilotseeds
  mindestens 0.9. Sonst ist die Zucht defekt.
- **Regel.** Weiter zu den Tickets 08 und 09, wenn auf motif-1d der Median von R mindestens 0.60
  und der Median von R − A mindestens 0.05 beträgt. Halt, wenn der Median von R − A höchstens 0
  ist. Dazwischen entscheidet die Kurve über k. Liegt R unter 0.60, kommt zuerst Phase 2.

## 7. Was die Gutachter fanden

### 7.1 Bewertungen

| Entwurf | Gutachten | Physik | Hirnnähe | Repo-Passung | Machbarkeit | Messbarkeit | Summe |
|---|---|---|---|---|---|---|---|
| 1, Biophysik zuerst | 1 | 8 | 8 | 7 | 6 | 8 | 37 |
| 1, Biophysik zuerst | 2 | 7 | 8 | 5 | 5 | 7 | 32 |
| 2, Chemie zuerst | 1 | 9 | 5 | 8 | 8 | 8 | 38 |
| 2, Chemie zuerst | 2 | 8 | 5 | 6 | 7 | 8 | 34 |
| 3, Brutmaschine zuerst | 1 | 7 | 6 | 9 | 6 | 9 | 37 |
| 3, Brutmaschine zuerst | 2 | 9 | 6 | 9 | 6 | 7 | 37 |

Über beide Gutachten summiert: Entwurf 1 hat 69, Entwurf 2 hat 72, Entwurf 3 hat 74 Punkte.

### 7.2 Behobene Fehler

| # | Entwurf | Fehler | Gutachten | Korrektur in diesem Papier |
|---|---|---|---|---|
| 1 | 1 | GABA_A mit α = 1.1 vom AMPA-Rezeptor | 1 | α = 5, β = 0.18 [Q11]; 96 % statt 62 % gebunden (Abschnitt 3.6) |
| 2 | 1 | GABA_B ohne G-Protein fällt mit 0.83 s ab | 1 | β auf 110 ms; G-Protein-Stufe in Ticket 11 (Abschnitt 3.6) |
| 3 | 1 | Gradientenfaktor der Erholung 0.49 bis 0.59 | 1 | 0.49 bis 0.67 (Abschnitt 3.13) |
| 4 | 1, 2 | „Faktor 1/(1 + hk) < 1, also verschwinden Gradienten“ | 1 | Nebendiagonalen; Explodieren möglich [Q8] (Abschnitt 3.13) |
| 5 | 1, 2, 3 | Dale verwechselt Transmitter und Vorzeichen | 1 | Vorzeichen je Synapse, Transmitter je Knoten [Q20] (Abschnitte 2.5, 3.11) |
| 6 | 1 | 10 000 Moleküle je Vesikel als Beleg | 1 | ein Parameter im Code [Q11] (Abschnitt 3.6) |
| 7 | 1 | AT11 verlangt r/p ≤ 12, der Generator liefert 12.1 | 1 | Spannen gegen ER und CFG (Abschnitt 6.3) |
| 8 | 1 | Gegenseitigkeit und Clusterung nur gegen ER | 1 | auch gegen CFG [Q44, Tab. 2] (Abschnitt 3.11) |
| 9 | 1, 2, 3 | Lechner und Hasani als Vorabdruck | 1 | Status offen, siehe [Q7] in Abschnitt 9.3 |
| 10 | 1 | Kernpfad hängt an einer Keras-Brücke, die die Beförderung nicht kennt | 2 (fatal) | Kern lernt selbst; fremde Genome nie im Endpunkt (Abschnitt 4.2) |
| 11 | 1 | Seeds 1001 bis 1008 nicht frei | 2 | 5001 bis 5008 (Abschnitt 5.4) |
| 12 | 1 | Keras trainiert Energien und Barrieren | 2 | nur Katalysatoren (Abschnitt 4.1) |
| 13 | 1 | Prototyp im versionierten `.scratch/` | 2 | außerhalb des Repositorys (Abschnitt 4.1) |
| 14 | 1 | Keras-Code im Repository ohne Regel | 2 | ADR 0016 (Abschnitt 4.3) |
| 15 | 1, 2 | Endpunkt auf neu erzeugten Daten außerhalb des Kerns | 2 | Endpunkt im Kern mit `brut-eval` (Abschnitt 5.4) |
| 16 | 1 | TCN-Zwilling ungeprüft gegen `src/cnn` | 2 | geprüft: Stamm 16 Kanäle, Kern 3; drei Restblöcke mit Dilatation 1, 2 und 4; Batch-Norm, ReLU, globales Mittel (`src/cnn/families.rs:661-683`) |
| 17 | 1 | Hill-Grad 4 und Produkte wie r·B·φ als effektive Ratengesetze | 2 | Stufe S nur elementar (Abschnitt 3.6) |
| 18 | 1, 2 | Dezimalkomma | 2 | Dezimalpunkt nach `docs/SCHREIBSTIL.md` |
| 19 | 2 | „Die Physik erzwingt einen Pegel je Schiene“ | 1 | Folge der Wahl eines Treibstoffs (Abschnitt 3.5) |
| 20 | 2 | Mittlerer Weg 3 bis 4.4 Schritte als Ziel | 1 | relativ zu ER; nachgerechnet 1.90 bis 2.66 (Abschnitt 3.11) |
| 21 | 2 | „Eingänge tauschen keine Masse aus“ | 1 | keine Einheit u; Zufuhr hebt sich in W_c − ΔG heraus (Abschnitt 3.10) |
| 22 | 2 | Bootstrap über 10 Seeds | 1, 2 | t-Schranke und exakter Vorzeichentest (Abschnitt 5.4) |
| 23 | 2 | Zeitskalen 1e4 im Text, Faktor 32 im Prototyp | 1 | Widerspruch benannt und begründet (Abschnitt 6.2) |
| 24 | 2 | Trainingsfeld im Genom bricht `Eq` | 2 | Genom nur aus ganzen Zahlen (Abschnitt 4.2) |
| 25 | 2 | 496 Kandidaten gelten nur für MLP-Läufe | 2 | eigener Zweig in `for_models` (Abschnitt 4.2) |
| 26 | 2 | Endpunkt kostet das 14-fache von `max_train_flops` | 2 | Endpunkt unter den Grenzen des Kerns (Abschnitt 5.4) |
| 27 | 2 | 8 Hyperparameter-Sätze je Arm unbudgetiert | 2 | eine Lernrate im Prototyp (Abschnitt 6.2) |
| 28 | 2, 3 | Checkerboard und andere flache Aufgaben im Kern | 2 | v1 nur Folgenaufgaben (Abschnitt 5.1) |
| 29 | 3 | Eingangsreservoire mit Pegel 0 | 1 | Eingänge als Katalysatoren (Abschnitt 3.3) |
| 30 | 3 | „Der Aufteilungsfehler verschiebt stationäre Zustände“ | 1 | Fixpunkte exakt; AT-6 prüft Gleichheit (Abschnitt 3.8) |
| 31 | 3 | Schranke x ≤ max(x(0), a_max) ohne Bedingung an den Start | 1 | Start im Ruhezustand des Kandidaten (Abschnitt 4.2) |
| 32 | 3 | Paarbildung als Koinzidenzdetektor | 1 | Koinzidenz mit der Aktivität des Ziels, S6 (Abschnitt 3.7) |
| 33 | 3 | Grenze von zwei Teilchen je Seite kappt den Hill-Grad auf 2 | 1 | Ketten geben Grad 4, S7 (Abschnitt 3.7) |
| 34 | 3 | Botenstoff ist zugleich Transmitter | 1 | eigener Transmitter-Pool (Abschnitt 3.7) |
| 35 | 3 | „Das Repository lehnt rnn ausdrücklich ab“ | 1, 2 | nur Testbeispiele (Abschnitt 2.1) |
| 36 | 3 | Endpunkt nur gegen eine Ablation | 2 | H-B gegen den CNN-Lauf (Abschnitt 5.4) |
| 37 | 3 | Valid-Lauf als Trainings-FLOPs gebucht | 2 | Buchung wie MLP und CNN (Abschnitt 4.2) |
| 38 | 3 | Reine Zucht ohne trainierte Auslese | 2 | Ridge-Auslese; Phase 2 als Rückfall (Abschnitt 4.2) |
| 39 | 3 | Keras-Skizze mit undefinierten Hilfsfunktionen | 2 | vollständiges Gerüst für Stufe K (Abschnitt 4.1) |
| 40 | 3 | Verteilung ganzer Quanten auf mehrere Reservoire offen | 2 | Quanten als Variante B von Ticket 01 (Abschnitt 4.2) |

### 7.3 Wo die Gutachten sich widersprechen

- **Welche Basis.**
  - *Gutachten 1* wählt Entwurf 2. Er habe die wenigsten Fehler in Gleichungen und Thermodynamik.
    Er sei das einzige echte Massenwirkungsnetz ohne Hill- oder Torfunktionen. Sein Prototyp prüfe
    die physikalische Kernaussage billig, und er beantworte die offene Frage von Ticket 04.
  - *Gutachten 2* wählt Entwurf 3. Nur er messe den Endpunkt im Kern, mit gleichen Bits,
    versiegeltem Hidden-Split und `brut-eval`. Er brauche keine Crate, kein exp und keinen
    Download. Er ordne jede Regel des Repositorys einem ADR oder Ticket zu.
  - *Unsere Wahl.* Wir nehmen Entwurf 2 als Basis des Modells und Entwurf 3 als Basis des Wegs in
    den Kern. Gründe: Die Fehler von Entwurf 2 betreffen Aussagen, nicht Gleichungen. Nur sein
    Keras-Pfad trainiert, und der Auftrag nennt Keras. Seine Eingänge als Katalysatoren vermeiden
    das Potential −∞, an dem das Modell von Entwurf 3 scheitert. Der Vorsprung von Entwurf 3 in der
    Summe, 74 gegen 72, stammt aus Repo-Passung und Messbarkeit. Beides übernehmen wir ganz.
- **Physik von Entwurf 3.** Gutachten 1 gibt 7 Punkte: Pegel 0, falsche Koinzidenz, falscher Hill-
  Grad, Botenstoff als Transmitter. Gutachten 2 gibt 9: elementare, reversible Schritte, bitgenaue
  Erhaltung, zweiter Hauptsatz je Teilschritt. Aus unserer Sicht haben beide recht. Die Numerik ist
  gut, das Modell hat Fehler. Wir übernehmen die Relaxation für Sackgassen, nicht die Spezies.
- **Fataler Fehler in Entwurf 1.** Gutachten 2 nennt den Kernpfad fatal; Gutachten 1 sieht keinen
  fatalen Fehler. Aus unserer Sicht betrifft der Einwand den Weg, nicht das Modell. Wir beheben ihn
  in Abschnitt 4.2.
- **Endpunkt.** Gutachten 1 lobt den azyklischen Zwilling mit Vorzeichentest als beste Idee.
  Gutachten 2 nennt ihn eine Ablation, keine Vergleichsbasis im Sinne des Glossars. Wir machen beide
  Hypothesen zu Pflichten (Abschnitt 5.4). Sie entscheiden darüber in Entscheidung 7.
- **Ganze Quanten.** Beide Gutachten zählen sie zu den besten Ideen von Entwurf 3. Wir übernehmen
  sie nur als Variante, weil ihr Austauschschritt ungelöst ist und f64 die Erhaltung in den
  Rechnungen der Entwürfe bis 1.9e-14 hielt. Hier weichen wir von beiden Gutachten ab.
- **Repo-Passung von Entwurf 1 und Entwurf 2.** Gutachten 1 gibt 7 und 8, Gutachten 2 gibt 5 und 6.
  Gutachten 2 zählt Laufgröße, Genom und Budget mit, Gutachten 1 vor allem die Naht.

### 7.4 Was wir selbst geprüft haben

- **Quellen.** Tabelle 2 von Lin et al. auf nature.com [Q44]; das Abstract von Chalasani et al. laut
  PubMed [Q20]; Verlagsseite von Pascanu et al. [Q8]; Destexhes `gabaa.mod` und `release.mod`
  [Q11]; Abschnitte 1 bis 3 von Vaswani et al. [Q1]; Abschnitt II D von Rao und Esposito [R39]; den
  Code von ncps [Q5] und von Keras 3.6.0 [Q61]; PyPI [Q62].
- **Repository.** Alle Zeilenangaben dieses Papiers am Commit `0e58853`, dazu `nproc` (4) und die
  Seeds aller Vorregistrierungen.
- **Rechnungen.** Positivität, Sperre ohne Antrieb, Fixpunkte, Dissipation, Abhängigkeit der
  Freisetzung vom Treibstoff, Weglängen im Zufallsgraphen, GABA_A-Bindung, Gradientenfaktoren,
  t-Quantil, Vorzeichentest, FLOPs und Fitnesskosten.
- **Nicht geprüft.** Die Verlagsseite der NeurIPS 2020 zu [Q7] war nicht erreichbar. Die übrigen
  Hirn- und Chemiezahlen geben wir aus den Berichten weiter.

## 8. Entscheidungen für die Eigentümerin bzw. den Eigentümer

Diese Fragen entscheiden Sie. Jede nennt Optionen, unsere Empfehlung und den Punkt des Fog in
`map.md`; Fog 11 ist neu. Bis Sie entscheiden, bleibt jede Wahl dieses Papiers ein Vorschlag.

**Entscheidung 1. Welche Semantik? (Fog 1)**
- Optionen: (a) kompilierte, ratenunabhängige Netze; (b) Massenwirkung mit Energien und Treibstoff,
  deterministisch; (c) stochastische Netze.
- Empfehlung: (b). Nur sie kennt Gedächtnis durch Antrieb und hat eine Energiebilanz. Ob
  rekurrente ReLU-Netze und rekurrente Reaktionsnetze gleichwertig sind, ist offen (`research.md`
  Abschnitt 9.1). (a) bleibt über Ticket 03 als Vergleich, (c) später.

**Entscheidung 2. Wie streng ist die Physik? (Fog 2)**
- Optionen: (a) streng: alles reversibel, Antrieb nur über Chemostaten; (b) idealisiert
  irreversible Reaktionen erlaubt; (c) streng im Kern, irreversibel nur als Messarm.
- Empfehlung: (c). Bei 8 RT ist A/ε = e^8, etwa 3000; das Netz liegt dann nahe am irreversiblen
  Grenzfall (nach unserer Rechnung). Laufen
  Rückreaktionen dauerhaft, ist kein Zustand stabil, und Rechnen ist dann schwer zu definieren [R19,
  Fn. 2, Vorabdruck]. Wir lesen deshalb zu einer festen Zeit aus.

**Entscheidung 3. Wie lernt ein Kandidat? (Fog 3)**
- Optionen: (a) nur Zucht mit geschlossener Ridge-Auslese; (b) zusätzlich BPTT im Kandidaten
  (Phase 2); (c) in Keras trainierte Genome als Basis.
- Empfehlung: (a) zuerst. (b), wenn der Brutversuch P2 die Schwelle verfehlt. (c) nur in
  Erkundungsläufen, nie im Endpunkt. Keine Quelle vergleicht Zucht und Gradienten bei gleicher
  Rechenmenge (`map.md:79-81`).

**Entscheidung 4. Wie wird es deterministisch? (Fog 4)**
- Optionen für den Energieschritt: RT·ln2 (exakt, ohne Tabelle), RT·ln2/4 (vier Bitmuster) oder
  RT·ln2/16 (sechzehn). Ein Messwert lässt sich damit bis auf 41 %, 9 % oder 2.2 % treffen (nach
  unserer Rechnung). Für den Zustand: f64 oder ganze Quanten. Für die Schleife: √ zulassen oder
  nicht.
- Empfehlung: RT·ln2/4, f64, √ zulassen; √ rundet exakt [R71]. Ganze Quanten als Variante B in
  Ticket 01. libm ist nicht nötig.

**Entscheidung 5. Was zählt als Größe, was kostet? (Fog 5)**
- Optionen: (a) Parameter als numerische Gene, Dissipation nur berichtet; (b) zusätzlich eine
  Abgabe auf die Dissipation, mit eigenem ADR.
- Empfehlung: (a) in v1. Ticket 12 prüft (b). Ein Abzug würde Sparsamkeit belohnen wie im Gehirn
  [Q56], änderte aber die Fitness aller Läufe mit `crn`.

**Entscheidung 6. Welche Aufgaben? (Fog 6)**
- Optionen: (a) im Kern nur motif-1d; (b) zusätzlich eine Gedächtnisaufgabe wie Echo-k in
  `src/task.rs`; (c) flache Aufgaben mit konstantem Eingang.
- Empfehlung: (a) in v1. (b), falls motif-1d die Rückkopplung in P2 nicht trennt. (c) verlangt eine
  Änderung von `Config::validate` und bringt für Rückkopplung wenig.

**Entscheidung 7. Was heißt Erfolg? (Fog 7)**
- Optionen: (a) H-R und H-B beide als Pflicht; (b) H-R primär, H-B nur beschreibend; (c) nur H-B.
  Dazu: Marge 0.05 oder anders; gleiche Grenzen oder gleicher Verbrauch.
- Empfehlung: (a), Marge 0.05, 8 Seeds 5001 bis 5008, gleiche Grenzen mit berichtetem Verbrauch.
  H-B dürfte in Phase 1 schwer sein, weil das TCN schon allein 0.95 erreicht. Ein Netz, das dem CNN
  um mehr als 0.05 unterliegt, lohnt den dauerhaften Einbau aber nicht.

**Entscheidung 8. Wie viel Laborrealismus? (Fog 8)**
- Optionen: (a) formale Netze mit Motiven aus der Neurobiologie; (b) nur gebaute Motive mit ihren
  gemessenen Grenzen.
- Empfehlung: (a). Die Raten stammen aus Synapsen-Messungen [Q10, Q11, Q13], nicht aus einem
  Laborbaukasten.

**Entscheidung 9. Soll ein Sprachmodell Genome schreiben? (Fog 9)**
- Empfehlung: später, nach dem Architekten für CNNs. Die Physik-Vorgabe wäre Abschnitt 3.9 in
  Worten (`research.md` Abschnitt 8.8).

**Entscheidung 10. Jenseits des MLP-Modus? (Fog 10)**
- Empfehlung: später. Ein Netz mit Gedächtnis könnte ein Werkzeug mit Zustand in einer Kaskade
  werden.

**Entscheidung 11. Darf die Modellart rekurrent sein? (neu, Fog 11)**
- Keiner der zehn Punkte fragt nach Rückkopplung.
- Optionen: (a) ja, mit dem azyklischen Zwilling als Kontrolle; (b) nein, nur vorwärts.
- Empfehlung: (a). Das ist der Kern des Auftrags.

**Entscheidung 12. Welcher Keras-Weg? (zu Fog 3)**
- Optionen: (a) JAX auf der CPU, etwa 137 MB; (b) Keras 3.6.0 mit NumPy-Backend, nur vorwärts, ohne
  Training; (c) TensorFlow auf der CPU, 274 MB, wie in der CI von ncps.
- Empfehlung: (a), außerhalb des Repositorys. Prüfen Sie vorher den freien Platz.

**Entscheidung 13. Wann kommt die Synapsenchemie? (zu Fog 2)**
- Optionen: (a) v1 nur Stufe K, Stufe S danach in Ticket 11; (b) beides zugleich.
- Empfehlung: (a). Stufe S ist etwa fünfmal so teuer und hat mehr Fehlerquellen.

**Entscheidung 14. Wo sitzt das Vorzeichen? (zu Fog 2)**
- Optionen: (a) je Neuron, als Vereinfachung; (b) je Synapse nach Rezeptorklasse, bei fester
  Transmitter-Identität je Neuron.
- Empfehlung: (b). Das entspricht dem Befund an AWC [Q20]. GABA-Knoten bleiben rein hemmend.

## 9. Quellen

### 9.1 Repository

Primärquelle, nicht begutachtet. Stand Commit `0e58853`.

- Notizen und Tickets: `.scratch/chemische-netze/map.md`, `research.md`,
  `issues/01-deterministic-crn-simulator-spike.md` bis `issues/04-gradients-through-the-integrator.md`;
  `.scratch/cnn-basismodelle/issues/02-model-seam-and-grid-tasks.md`, `04-families-b.md`.
- Code: `src/config.rs`, `src/genome.rs`, `src/model.rs`, `src/evolve.rs`, `src/train.rs`,
  `src/score.rs`, `src/task.rs`, `src/hidden_eval.rs`, `src/main.rs`, `src/cnn/genome.rs`,
  `src/cnn/families.rs`, `src/cnn/shape.rs`, `tests/learner_security.rs`,
  `tools/rearc_to_jsonl.py`, `Cargo.toml`.
- Entscheidungen und Regeln: ADR 0007 und 0008, `CONTEXT.md`, `docs/STAGE1_CONTRACT.md`,
  `docs/PREREG_STEP2_SOUL.md`, `docs/SCHREIBSTIL.md`, `docs/agents/triage-labels.md`.

### 9.2 Aus research.md

Status und gelesene Fassung stehen in `research.md`, Abschnitt 10. Hier nicht neu geprüft, außer
[R39].

- [R1] Hjelmfelt, Weinberger, Ross 1991, PNAS; begutachtet. Seite 10984 laut Chemie-Bericht im
  Volltext gelesen.
- [R9] Anderson, Joshi, Deshpande 2021, J R Soc Interface; begutachtet.
- [R14] Poole et al. 2017, DNA 23; begutachtete Konferenz.
- [R17] Dack, Qureshi, Ouldridge, Plesa 2026, Cell Systems; begutachtet.
- [R19] Kini, Doty 2026; Vorabdruck.
- [R39] Rao, Esposito 2016, Phys Rev X 6:041064; begutachtet. Abschnitt II D, Beispiel 6 und
  Gl. 69 bis 79 haben wir am 2026-11-04 in arXiv 1602.07257v3 nachgelesen.
- [R40] Gorban, Yablonsky 2011; begutachtet.
- [R42] Ouldridge 2018, Natural Computing; begutachtet, eine Übersicht.
- [R43] Craciun et al. 2020; begutachtet.
- [R47] Floyd et al. 2025, Nat Commun; begutachtet.
- [R53] Kim et al. 2021, Chaos; begutachtet.
- [R55] Lee, Buse, Winfree 2025, DNA 31; begutachtete Konferenz.
- [R57] Gholaminejad, Keutzer, Biros 2019, IJCAI; begutachtete Konferenz.
- [R63] Feng et al. 2015, BioJazz, Nucleic Acids Research; begutachtet.
- [R64] Lones et al. 2014, IEEE Trans Evol Comput; begutachtet.
- [R70] Rust RFC 3514; Dokumentation, nicht begutachtet.
- [R71] Rust-Standardbibliothek, f64; Dokumentation, nicht begutachtet.
- [R74] Arora et al. 2018, ICLR; begutachtete Konferenz.

### 9.3 Neue Quellen dieses Papiers

„Selbst geprüft“ heißt: am 2026-11-04 an der genannten Stelle gelesen. Alle anderen Angaben stammen
aus den Berichten vom 2026-11-04 und sind dort als gelesen vermerkt.

1. **[Q1]** Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, Kaiser Ł, Polosukhin I
   (2017). Attention is all you need. Advances in Neural Information Processing Systems 30
   (NIPS 2017). https://papers.nips.cc/paper/7181-attention-is-all-you-need. Begutachtete
   Konferenz. Selbst geprüft: Abstract auf der Verlagsseite; Abschnitte 1 bis 3 in arXiv
   1706.03762v7.
2. **[Q2]** Hasani R, Lechner M, Amini A, Rus D, Grosu R (2021). Liquid time-constant networks. Proc
   AAAI 35:7657-7666. https://ojs.aaai.org/index.php/AAAI/article/view/16936. Begutachtete
   Konferenz. Laut Chemie-Bericht gelesen als arXiv 2006.04439v4.
3. **[Q3]** Hasani R, Lechner M, Amini A, Liebenwein L, Ray A, Tschaikowski M, Teschl G, Rus D
   (2022). Closed-form continuous-time neural networks. Nat Mach Intell 4(11):992-1003.
   https://doi.org/10.1038/s42256-022-00556-7. Begutachtet. Laut Chemie-Bericht gelesen als arXiv
   2106.13898v2.
4. **[Q4]** Lechner M, Hasani R, Amini A, Henzinger TA, Rus D, Grosu R (2020). Neural circuit
   policies enabling auditable autonomy. Nat Mach Intell 2(10):642-652.
   https://doi.org/10.1038/s42256-020-00237-3. Begutachtet. Laut Chemie-Bericht nur S. 642 bis 645
   gelesen.
5. **[Q5]** ncps 1.0.1, PyPI (2024-08-14, Apache 2.0), und das Repository mlech26l/ncps, Zweig
   master. Code, nicht begutachtet. Selbst geprüft: `ncps/keras/ltc_cell.py`, Zeilen 271 bis 273
   und 319 bis 326; der CI-Workflow mit einem Lauf für das Backend Torch. Den Lauf für TensorFlow
   nennt der Chemie-Bericht.
6. **[Q6]** Wicks SR, Roehrig CJ, Rankin CH (1996). A dynamic network simulation of the nematode
   tap withdrawal circuit. J Neurosci 16(12):4017-4031.
   https://doi.org/10.1523/JNEUROSCI.16-12-04017.1996. Begutachtet. Laut Chemie-Bericht im Volltext
   gelesen.
7. **[Q7]** Lechner M, Hasani R (2020). Learning long-term dependencies in irregularly-sampled time
   series. arXiv 2006.04418v4 (4 Dec 2020). Status offen: Gutachten 1 nennt NeurIPS 2020. Selbst
   geprüft haben wir nur Sekundärbelege: den Eintrag bei researchr
   (https://researchr.org/publication/LechnerH20), der die Arbeit in Advances in Neural
   Information Processing Systems 33 führt, und eine Vortragsseite der NeurIPS 2020 bei SlidesLive.
   Die Verlagsseite war nicht erreichbar. Das offizielle Code-Repository mlech26l/ode-lstms zitiert
   den arXiv-Vorabdruck. Wir behandeln die Arbeit deshalb wie einen Vorabdruck.
8. **[Q8]** Pascanu R, Mikolov T, Bengio Y (2013). On the difficulty of training recurrent neural
   networks. Proc 30th ICML, PMLR 28(3):1310-1318. https://proceedings.mlr.press/v28/pascanu13.html.
   Begutachtete Konferenz. Selbst geprüft: Verlagsseite mit Abstract und Bibliografie.
9. **[Q9]** Destexhe A, Mainen ZF, Sejnowski TJ (1994). An efficient method for computing synaptic
   conductances based on a kinetic model of receptor binding. Neural Comput 6(1):14-18.
   https://doi.org/10.1162/neco.1994.6.1.14. Begutachtet. Laut Chemie-Bericht als Scan gelesen.
10. **[Q10]** Destexhe A, Mainen ZF, Sejnowski TJ (1998). Kinetic models of synaptic transmission.
    In: Koch C, Segev I (Hrsg.), Methods in Neuronal Modeling, 2. Aufl., MIT Press, S. 1-25.
    Buchkapitel in einem Herausgeberband; die Begutachtung ist nicht angegeben. Laut
    Chemie-Bericht als Autoren-PDF gelesen.
11. **[Q11]** ModelDB, Eintrag 18198 (Destexhe), Dateien `gabaa.mod` und `release.mod`, Spiegel
    https://github.com/ModelDBRepository/18198. Code, nicht begutachtet. Selbst geprüft:
    Alpha = 5 /(ms·mM), Beta = 0.18 /ms, Cmax = 1 mM, Cdur = 1 ms; `nt = 10000` als Parameter
    „nb of molec of T per vesicle“.
12. **[Q12]** Clements JD et al. (1992). The time course of glutamate in the synaptic cleft.
    Science 258:1498-1501. https://doi.org/10.1126/science.1359647. Begutachtet. Laut
    Chemie-Bericht nur Abstract.
13. **[Q13]** Tsodyks MV, Markram H (1997). PNAS 94(2):719-723.
    https://doi.org/10.1073/pnas.94.2.719. Begutachtet. Laut Chemie-Bericht Modellteil und Abb. 1
    gelesen.
14. **[Q14]** Hestrin S et al. (1990). J Physiol 422:203.
    https://doi.org/10.1113/jphysiol.1990.sp017980. Begutachtet. Laut Hirn-Bericht.
15. **[Q15]** Otis TS, De Koninck Y, Mody I (1993). J Physiol 463:391.
    https://doi.org/10.1113/jphysiol.1993.sp019600. Begutachtet. Laut Hirn-Bericht.
16. **[Q16]** Sabatini BL, Regehr WG (1996). Nature 384:170. https://doi.org/10.1038/384170a0.
    Begutachtet. Laut Hirn-Bericht.
17. **[Q17]** Greengard P (2001). Science 294:1024. https://doi.org/10.1126/science.294.5544.1024.
    Begutachtet, eine Übersicht. Laut Hirn-Bericht.
18. **[Q18]** Mongillo G, Barak O, Tsodyks M (2008). Synaptic theory of working memory. Science
    319:1543-1546. https://doi.org/10.1126/science.1150769. Begutachtet. Laut Chemie-Bericht nur
    Abstract.
19. **[Q19]** Masse NY et al. (2019). Nat Neurosci 22(7):1159-1167.
    https://doi.org/10.1038/s41593-019-0414-3. Begutachtet. Laut Chemie-Bericht nur Abstract.
20. **[Q20]** Chalasani SH, Chronis N, Tsunozaki M, Gray JM, Ramot D, Goodman MB, Bargmann CI (2007).
    Dissecting a circuit for olfactory behaviour in Caenorhabditis elegans. Nature
    450(7166):63-70. https://doi.org/10.1038/nature06292. Begutachtet. Selbst geprüft: Abstract
    laut PubMed (PMID 17972877).
21. **[Q21]** Danbolt NC (2001). Prog Neurobiol 65:1.
    https://doi.org/10.1016/s0301-0082(00)00067-8. Begutachtet, eine Übersicht. Laut Hirn-Bericht.
22. **[Q22]** Schiller J et al. (2000). Nature 404:285. https://doi.org/10.1038/35005094.
    Begutachtet. Laut Hirn-Bericht.
23. **[Q23]** Colquhoun D et al. (2004). Biophys J 86:3510.
    https://doi.org/10.1529/biophysj.103.038679. Begutachtet. Laut Hirn-Bericht: Die meisten, aber
    nicht alle Ionenkanäle gehorchen der mikroskopischen Reversibilität.
24. **[Q24]** Tkachenko AV, Mognetti BM, Maslov S. arXiv 2506.14006v2. Vorabdruck. Laut
    Chemie-Bericht.
25. **[Q25]** Tuccio M, Rocks JW, Goldford JE. arXiv 2609.17339v1 (15.09.2026). Vorabdruck. Laut
    Chemie-Bericht.
26. **[Q26]** Pandey T, Feketa P, Steinkühler J. arXiv 2509.04000v2. Vorabdruck. Laut
    Chemie-Bericht.
27. **[Q27]** Qian H, Reluga TC (2005). PRL 94:028101.
    https://doi.org/10.1103/PhysRevLett.94.028101. Begutachtet. Laut Chemie-Bericht nur Abstract.
28. **[Q28]** Cao Y et al. (2015). Nat Phys 11(9):772-778. https://doi.org/10.1038/nphys3412.
    Begutachtet. Laut Chemie-Bericht nur Abstract.
29. **[Q29]** Barato AC, Seifert U (2015). PRL 114:158101.
    https://doi.org/10.1103/PhysRevLett.114.158101. Begutachtet. Laut Chemie-Bericht nur Abstract.
30. **[Q30]** Douglas RJ, Koch C, Mahowald M, Martin KAC, Suarez HH (1995). Science 269:981.
    https://doi.org/10.1126/science.7638624. Begutachtet. Laut Hirn-Bericht.
31. **[Q31]** Ahmed B et al. (1994). J Comp Neurol 341:39. https://doi.org/10.1002/cne.903410105.
    Begutachtet. Laut Hirn-Bericht.
32. **[Q32]** Lien AD, Scanziani M (2013). Nat Neurosci 16:1315. https://doi.org/10.1038/nn.3488.
    Begutachtet. Laut Hirn-Bericht.
33. **[Q33]** Li YT et al. (2013). Nat Neurosci 16:1324. https://doi.org/10.1038/nn.3494.
    Begutachtet. Laut Hirn-Bericht.
34. **[Q34]** Markov NT et al. (2011). Cereb Cortex 21:1254. https://doi.org/10.1093/cercor/bhq201.
    Begutachtet. Laut Hirn-Bericht.
35. **[Q35]** Song S et al. (2005). PLoS Biol 3:e68. https://doi.org/10.1371/journal.pbio.0030068.
    Begutachtet. Laut Hirn-Bericht im Volltext gelesen.
36. **[Q36]** Campagnola L et al. (2022). Science 375:eabj5861.
    https://doi.org/10.1126/science.abj5861. Begutachtet. Laut Hirn-Bericht im Volltext gelesen.
37. **[Q37]** Varshney LR et al. (2011). PLoS Comput Biol 7:e1001066.
    https://doi.org/10.1371/journal.pcbi.1001066. Begutachtet. Laut Hirn-Bericht als PDF gelesen.
38. **[Q38]** Cook SJ et al. (2019). Nature 571:63. https://doi.org/10.1038/s41586-019-1352-7.
    Begutachtet. Laut Hirn-Bericht im Volltext gelesen.
39. **[Q39]** Witvliet D et al. (2021). Nature 596:257. https://doi.org/10.1038/s41586-021-03778-8.
    Begutachtet. Laut Hirn-Bericht im Volltext gelesen.
40. **[Q40]** Randi F et al. (2023). Nature 623:406. https://doi.org/10.1038/s41586-023-06683-4.
    Begutachtet. Laut Hirn-Bericht.
41. **[Q41]** Bentley B et al. (2016). PLoS Comput Biol 12:e1005283.
    https://doi.org/10.1371/journal.pcbi.1005283. Begutachtet. Laut Berichten nur Abstract.
42. **[Q42]** Ripoll-Sánchez L et al. (2023). Neuron 111:3570.
    https://doi.org/10.1016/j.neuron.2023.09.043. Begutachtet. Laut Hirn-Bericht im Volltext
    gelesen.
43. **[Q43]** Dorkenwald S et al. (2024). Nature 634:124. https://doi.org/10.1038/s41586-024-07558-y.
    Begutachtet. Laut Hirn-Bericht im Volltext gelesen.
44. **[Q44]** Lin A et al. (2024). Network statistics of the whole-brain connectome of Drosophila.
    Nature 634:153. https://doi.org/10.1038/s41586-024-07968-y. Begutachtet. Selbst geprüft:
    Tabelle 2 auf nature.com; Text in PMC11446825.
45. **[Q45]** Winding M et al. (2023). Science 379:eadd9330. https://doi.org/10.1126/science.add9330.
    Begutachtet. Laut Hirn-Bericht im Volltext gelesen.
46. **[Q46]** Eckstein N et al. (2024). Cell 187:2574. https://doi.org/10.1016/j.cell.2024.03.016.
    Begutachtet. Laut Hirn-Bericht.
47. **[Q47]** Galarreta M, Hestrin S (1999). Nature 402:72. https://doi.org/10.1038/47029.
    Begutachtet. Laut Hirn-Bericht.
48. **[Q48]** Gibson JR, Beierlein M, Connors BW (1999). Nature 402:75.
    https://doi.org/10.1038/47035. Begutachtet. Laut Hirn-Bericht.
49. **[Q49]** Meyer HS et al. (2011). PNAS 108:16807. https://doi.org/10.1073/pnas.1113648108.
    Begutachtet. Laut Hirn-Bericht.
50. **[Q50]** Megías M et al. (2001). Neuroscience 102:527.
    https://doi.org/10.1016/s0306-4522(00)00496-6. Begutachtet. Laut Hirn-Bericht.
51. **[Q51]** Poirazi P, Brannon T, Mel BW (2003). Neuron 37:989.
    https://doi.org/10.1016/s0896-6273(03)00149-1. Begutachtet. Laut Hirn-Bericht.
52. **[Q52]** Beniaguev D, Segev I, London M (2021). Neuron 109:2727.
    https://doi.org/10.1016/j.neuron.2021.07.002. Begutachtet. Laut Hirn-Bericht.
53. **[Q53]** van Albada SJ, Helias M, Diesmann M (2015). PLoS Comput Biol 11:e1004490.
    https://doi.org/10.1371/journal.pcbi.1004490. Begutachtet. Laut Hirn-Bericht.
54. **[Q54]** Lappalainen JK et al. (2024). Nature 634:1132.
    https://doi.org/10.1038/s41586-024-07939-3. Begutachtet. Laut Hirn-Bericht.
55. **[Q55]** Shiu PK et al. (2024). Nature 634:210. https://doi.org/10.1038/s41586-024-07763-9.
    Begutachtet. Laut Hirn-Bericht.
56. **[Q56]** Attwell D, Laughlin SB (2001). J Cereb Blood Flow Metab 21(10):1133-1145.
    https://doi.org/10.1097/00004647-200110000-00001. Begutachtet. Laut Berichten nur Abstract.
57. **[Q57]** Lennie P (2003). Curr Biol 13:493. https://doi.org/10.1016/s0960-9822(03)00135-0.
    Begutachtet. Laut Hirn-Bericht.
58. **[Q58]** Halassa MM et al. (2007). J Neurosci 27:6473.
    https://doi.org/10.1523/JNEUROSCI.1419-07.2007. Begutachtet. Laut Hirn-Bericht.
59. **[Q59]** Mu Y et al. (2019). Cell 178:27. https://doi.org/10.1016/j.cell.2019.05.050.
    Begutachtet. Laut Hirn-Bericht.
60. **[Q60]** Bargmann CI (2012). BioEssays 34(6):458-465. https://doi.org/10.1002/bies.201100185.
    Begutachtet. Laut Berichten nur Abstract.
61. **[Q61]** Keras, offizielles Repository keras-team/keras, Tag v3.6.0:
    `keras/src/backend/numpy/trainer.py` (Zeile 169) und `keras/src/backend/numpy/rnn.py`
    (Zeile 6). Code, nicht begutachtet. Selbst geprüft.
62. **[Q62]** PyPI, JSON-Metadaten vom 2026-11-04 zu keras 3.6.0 und 3.15.1, jax und jaxlib 0.11.0,
    0.12.0.1 und 0.4.34, scipy 1.18.1, h5py 3.16.0, optree 0.20.0, ml-dtypes 0.6.0, opt-einsum
    3.4.0, tensorflow-cpu 2.21.0, torch 2.15.0 und ncps 1.0.1. Nicht begutachtet. Selbst geprüft.
63. **[Q63]** Diffrax, Dokumentation: „Stiff ODE“ und „How to choose a solver“. Dokumentation,
    nicht begutachtet. Laut Chemie-Bericht: 64 Bit empfohlen, sonst drohen NaN-Gradienten.
64. **[Q64]** Song HF, Yang GR, Wang X-J (2016). PLoS Comput Biol 12(2):e1004792.
    https://doi.org/10.1371/journal.pcbi.1004792. Begutachtet. Laut Chemie-Bericht nur Abstract.
65. **[Q65]** Fiacco TA, McCarthy KD (2018). J Neurosci 38:3.
    https://doi.org/10.1523/JNEUROSCI.0016-17.2017. Begutachtet, ein Standpunkt. Laut Hirn-Bericht.
66. **[Q66]** Savtchouk I, Volterra A (2018). J Neurosci 38:14.
    https://doi.org/10.1523/JNEUROSCI.0017-17.2017. Begutachtet, ein Standpunkt. Laut Hirn-Bericht.

**Nicht erreichbar oder nur teilweise gelesen.** Die Verlagsseite der NeurIPS 2020 zu [Q7]
(Zeitüberschreitung); dblp.org (vom Proxy abgelehnt). Die Tabellenwerte von [Q44] fehlen im Text
bei PMC; wir haben sie auf nature.com gelesen.
