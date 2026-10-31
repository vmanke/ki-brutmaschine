# Erste Läufe der Kaskaden-Zucht mit echten Modellen (Stufe 2c)

Stand: 31. Oktober 2026. Wir messen auf einer gemieteten VM mit einer NVIDIA H200
(141 GB HBM3e, 700 W, Compute Capability 9.0), 16 virtuellen Kernen und 196 GB
Arbeitsspeicher. Die Modelle liefert Ollama 0.40.13 auf 127.0.0.1:11434; die Läufe
startet `brut-breed-ollama` aus dem Paket `backend-ollama` (PR #33, Commit `f8d3478`).
Den Vertrag des Adapters beschreibt [STAGE2_BACKEND_CONTRACT.md](STAGE2_BACKEND_CONTRACT.md),
die Zucht selbst [STUFE2C_KASKADEN_ZUCHT.md](STUFE2C_KASKADEN_ZUCHT.md).

Diese Läufe sind ein Funktionsnachweis mit echten Modellen und kein Versuch nach
Plan. Wir haben sie weder vorab registriert noch mit Kontrollarm gefahren, weshalb
sie über den Nutzen der Zucht nichts entscheiden.

## 1. Kurzfassung

- **Die Maschine läuft mit einem echten Modell.** In drei Läufen schickten
  Kaskaden 2704 Worker-Aufrufe an `qwen2.5:7b` und `qwen2.5:32b`. Jeder Lauf
  besteht `brut-breed --verify`: Die Hash-Kette ist intakt, und der Bericht
  rechnet sich aus dem Checkpoint gleich nach.
- **Die Kaskaden sprechen miteinander.** Wir zählen 661 Übergaben von Stufe zu
  Stufe (`PREVIOUS ANSWER k`; 30, 463 und 168 je Lauf) und 406 blinde Urteile
  einer Kaskade über die Antworten zweier anderer, nicht verwandter Kaskaden.
- **`qwen2.5:7b` liegt am Boden.** Im großen Lauf lösten 40 von 52 Kaskaden kein
  einziges der 12 Such-Rätsel und die übrigen 12 genau eines. Die Selektion hat
  damit kein Signal.
- **`qwen2.5:32b` hebt die Seeds vom Boden.** Die vier handgeschriebenen Seeds
  lösen 1 bis 6 von 24 ungesehenen Rätseln (im Mittel 15.6 %).
- **Ein Zuchtgewinn ist nicht belegt.** Die beste gezüchtete Kaskade löste im
  Such-Panel 5 von 16 Rätseln gegen 2 von 16 beim besten Seed. Auf den 24
  ungesehenen Rätseln erreichen beide 6 von 24. Den Vorsprung im Such-Panel
  halten wir damit für Zufall der kleinen Stichprobe.
- **Die Richter raten.** Wo eine der beiden Antworten richtig war, wählten die
  Richter sie im 32B-Lauf in 6 von 18 Fällen (33 %), im großen 7B-Lauf in 7 von 11
  Fällen. Beides ist von einem Münzwurf nicht zu unterscheiden.

## 2. Die Läufe

Alle Läufe nutzen die Grid-Domäne, die Darwin-Gödel-Auswahl der Eltern und die
übrigen Werte von `BreedConfig::default()`; jeweils ein Modell steht auf der
Erlaubnisliste, mit 0.002 Fitness-Einheiten je Aufruf.

| Lauf | Modell | Seed | Generationen | Kinder je Gen. | Such- / Auswahl-Rätsel | Kaskaden | Aufrufe | Urteile | Laufzeit |
|---|---|---|---|---|---|---|---|---|---|
| q1 | qwen2.5:7b | 1 | 3 | 4 | 10 / 10 | 12 | 292 | 68 | 1.6 min |
| q2 | qwen2.5:7b | 2 | 9 | 6 | 12 / 12 | 52 | 1814 | 272 | 10.5 min |
| q3 | qwen2.5:32b | 3 | 3 | 4 | 16 / 24 | 12 | 598 | 66 | 16.6 min |

Lauf q3 hielt an seiner Zeitgrenze von 900 s während des Auswahl-Panels an. Wir
setzten ihn mit `--resume` fort, wobei der Adapter die 526 schon aufgezeichneten
Aufrufe abspielte, statt das Modell erneut zu fragen (Vertrag I7).

## 3. Lösungsquoten

| Lauf | Seeds im Such-Panel | Bestes Ergebnis im Such-Panel | Kaskaden ohne Lösung | Bestes Seed, ungesehen | Beste Zucht, ungesehen |
|---|---|---|---|---|---|
| q1 | 1, 0, 1, 1 von 10 | 1 von 10 | 5 von 12 | 1 von 10 | 1 von 10 |
| q2 | 0, 0, 1, 0 von 12 | 1 von 12 | 40 von 52 | 0 von 12 | 1 von 12 |
| q3 | 1, 1, 2, 2 von 16 | 5 von 16 | 0 von 12 | 6 von 24 | 6 von 24 |

Das 95-%-Wilson-Intervall für 6 von 24 reicht von 12 % bis 45 %. Bei Panels dieser
Größe trennen wir damit nur Unterschiede von mehr als etwa 30 Prozentpunkten.

## 4. Beobachtungen

- **Ohne Strategie-Hinweis besser, vorläufig.** Die beste Kaskade in q3 entstand
  durch `section_drop`: Sie strich den Hinweis „Look for mirror images…“ und
  behielt nur Rolle und Format. Die zwei Kaskaden ohne Hinweis lösten 6 von 32
  Such-Rätseln (18.8 %), die zehn mit Hinweis 19 von 160 (11.9 %). Wir nehmen an,
  dass ein festgelegter Hinweis das Modell auf eine einzige Transformation lenkt;
  bei zwei Kaskaden ist das eine Vermutung, kein Befund.
- **Mehr Stufen lösen nicht mehr.** In q2 lösten 4 der 28 mehrstufigen Kaskaden
  ein Rätsel, aber 8 der 24 einstufigen. Liegt schon die erste Stufe falsch, gibt
  die zweite den Fehler eher weiter, als ihn zu beheben.
- **Format.** Von 2704 Antworten endeten 2546 sauber (94.2 %); die übrigen
  schnitt die Längengrenze ab, und der Orchestrator wertet sie nicht.

## 5. Rechenzeit und GPU

Wir messen die GPU einmal je Sekunde mit `nvidia-smi`.

| Modell | Zeit je Aufruf | GPU-Last unter Last | Leistung (Mittel / Spitze) | Speicher |
|---|---|---|---|---|
| qwen2.5:7b | 0.35 s | 84 % | 431 W / 465 W | 6.8 GB |
| qwen2.5:32b | 1.7 s | 92 % | 509 W / 566 W | 27.1 GB |

Ollama arbeitet die Anfragen nacheinander ab, obwohl die Zucht die Rätsel eines
Panels parallel schickt. Acht gleichzeitige Anfragen an `qwen2.5:32b` dauern
12.0 s gegen 1.5 s für eine, also Faktor 8; die Dekodierrate bleibt bei 72 bis
75 Token/s. Mit `OLLAMA_NUM_PARALLEL=8` meldete ein zweiter Server acht Plätze,
lieferte in unserer Messung aber keinen höheren Durchsatz. Von 141 GB belegen die
Modelle höchstens 27 GB.

## 6. Was als Nächstes zählt

| Hebel | Zweck | Stand |
|---|---|---|
| vLLM statt Ollama | Durchsatz: parallele Anfragen gebündelt dekodieren | Installation auf der VM steht aus |
| Schlichte Seeds (nur Rolle und Format) | Startpunkt ohne lenkenden Hinweis | berührt `src/cascade/`, nach Absprache |
| Stärkeres oder denkendes Modell | Seeds auf 30 bis 50 % heben, damit die Selektion ein Signal hat | `qwen2.5:72b` passt; Denkmodelle brauchen I5 im Adapter |
| Mutator-Runden (`--llm-every`) | Neue Prompts schreiben statt nur Phrasen zu mischen | ungenutzt |
| Panels mit 100 Rätseln, Kontrollarm `--selection uniform`, drei Seeds | Unterschiede von etwa 10 Prozentpunkten messbar machen | nicht begonnen |
| Kleineres `judge_weight` (heute 0.2) | Rauschen der Urteile aus der Fitness nehmen | Konfigurationswert |

Den Bericht mit den Zwiegesprächen der Kaskaden, der Urteilsmatrix und den
GPU-Kurven halten wir als Artefakt außerhalb des Repositorys; die Rohdaten der Läufe
(`events.jsonl`, `llm/worker-calls.jsonl`) liegen auf der VM und nicht in Git.
