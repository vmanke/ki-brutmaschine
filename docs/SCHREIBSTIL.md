# Schreibstil der deutschen Berichte

Diese Vorgabe gilt für die deutschen Berichte des Repositorys: README.md,
docs/ARCHITEKTUR.md, docs/SICHERHEIT_UND_RECHT.md, docs/PLAN_GENERALISIERUNG.md,
docs/ERGEBNISSE_STUFE1.md, docs/ENTSCHEIDUNGEN_STUFE2A.md,
docs/POSITIVKONTROLLE.md, docs/STUFE2A_PILOT.md,
docs/STUFE2B_PR_MUTATION.md, docs/STUFE2C_KASKADEN_ZUCHT.md,
docs/ERGEBNISSE_STUFE2C_ERSTE_LAEUFE.md und docs/SCHRITT1_SCHALTUNGEN.md. Der Code, seine
Kommentare, das Begriffswörterbuch CONTEXT.md, der Bauvertrag und die ADRs
bleiben englisch, damit sie zu den Code-Begriffen passen. Die Vorgabe folgt dem
Schreibstil des Projektplans Self-Mixing-Diagnose; wir übernehmen ihn, damit das
Repository in einer Stimme spricht.

## Haltung

Wir schreiben nüchtern und belegt. Jede Aussage trägt, wo sie quantitativ ist,
eine Zahl und eine Einheit, und wo sie von außen kommt, eine Quelle. Wir trennen
dabei stets, was wir gemessen, und was wir angenommen haben. Wir werben nicht,
füllen nicht und schreiben keine Bilder, wo ein Fachwort genügt.

## Register

- **Wir-Form, aktiv.** Das Team beschreibt seine eigene Arbeit: „wir messen“,
  „wir setzen … an“, „wir nehmen an“. Die Sie-Form nutzen wir nur dort, wo ein
  Text die Leserin oder den Leser unmittelbar anweist, etwa bei den Startbefehlen
  der README.
- **Verkettete, aber kontrollierte Sätze.** Ein Satz darf eine Ursache und ihre
  Folge tragen, verbunden mit „da“, „sodass“, „womit“, „weshalb“ oder „wobei“.
  Wir reihen jedoch nicht endlos, sondern setzen einen Punkt, sobald der Gedanke
  steht.
- **Relativsätze mit „welche/welcher/welches“** sind erlaubt und erwünscht, wo
  sie den Satz ordnen; „die/der/das“ bleibt daneben zulässig.

## Zahlen und Einheiten

- **Punkt als Dezimaltrennzeichen:** 0.9 mW, 33.3 %, 1.6 Stunden.
- **Leerzeichen vor der Einheit und vor dem Prozentzeichen:** 1 mW, 655 nm,
  70 %, 25 °C.
- **Leerzeichen als Tausendertrennzeichen:** 50 000 Werte, 2000 Aufgaben.
- **Bereiche mit „bis“:** 3 bis 5 mW. **Vielfache mit „Faktor“:** Faktor 2.
- **Zehnerpotenzen** schreiben wir als 1e25 oder als 10^25, einheitlich je
  Dokument.

## Belege und Ehrlichkeit

- Gemessenes kennzeichnen wir mit „wir messen“, Angenommenes mit „wir nehmen an“,
  „wir setzen … an“, „nach unserer Rechnung“ oder „aus unserer Sicht“.
- Quellen stehen im Text, als benannter Verweis („laut Datenblatt“, „nach
  Zhang et al.“) oder als Markdown-Link. Vorabdrucke kennzeichnen wir als solche.
- Grenzen, offene Fragen und negative Ergebnisse nennen wir klar, nicht am Rand.

## Form

- **Listen mit fettem Einstieg:** „**Laser.** Die Laserdiode …“.
- **Kurze, nummerierte Abschnitte.** Eine Frage je Abschnitt; keine Überschrift
  erscheint zweimal.
- **Tabellen** für Kriterien, Risiken und Vergleiche, mit knappen Spalten.
- Keine Werbeworte („leistungsstark“, „modern“), keine leeren Verstärker
  („wirklich“, „deutlich“, wo keine Zahl folgt), kein Gedankenstrich als
  Mode.

## Beispiel

Statt: „Das System ist wirklich sehr sicher und läuft super schnell.“

So: „Der Not-Aus greift nach 0.17 s, da wir die STOP-Datei während des Trainings
alle 64 Schritte prüfen. Ein voller Lauf dauert 17 bis 26 s auf vier Kernen,
gemessen über drei abwechselnde Läufe.“
