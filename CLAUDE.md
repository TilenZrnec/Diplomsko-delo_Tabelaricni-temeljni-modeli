# Diplomsko delo — besedilo (FRI)

Repozitorij vsebuje besedilo diplomske naloge v LaTeXu (uradna predloga FRI):
`diploma-predloga.tex`, viri v `literatura.bib`, slike v `slike/`, predloga in
naslovne strani v `podporno/`. Koda in eksperimenti so v ločenem repozitoriju
`../Diplomsko-delo_Koda` (glej njegov `CLAUDE.md`).

Spodnje smernice so **uradne smernice FRI za pisanje diplome**. Upoštevaj jih pri
vsakem pisanju ali popravljanju besedila naloge — tudi kadar te uporabnik na njih
ne opozori posebej.

## Slog pisanja

- **Prva oseba množine.** Pišemo »razvili smo«, »opazimo«, »v nadaljevanju
  predstavimo«. Nikoli v prvi osebi ednine in ne v poljudnem/blogovskem tonu.
- **Akademski slog**, ne poljuden. Brez marketinškega besedišča, vzklikov,
  retoričnih vprašanj in pretiravanja.
- **Dosledna terminologija.** Ko je izraz enkrat izbran, velja do konca naloge.
  Ne mešamo slovenskega in angleškega izraza za isti pojem (npr. *strojno
  učenje* proti *machine learning*). Angleški izvirnik navedemo ob **prvi**
  pojavitvi v obliki `strojno učenje (angl. machine learning)`, nato dosledno le
  slovenski izraz.
- **Ista ugotovitev na več mestih, na različnih ravneh podrobnosti:**
  visokonivojsko v povzetku, motivacijsko v uvodu, tehnično v jedru, sintetično
  v zaključku. To ni podvajanje, ampak zahtevana struktura.
- **Vsaka trditev mora biti preverljiva** — bodisi iz lastne izpeljave oz.
  meritve bodisi z navedbo vira. Trditev, ki ni ne eno ne drugo, iz besedila
  odstranimo.

## Citiranje

- **Kdaj citiramo:** vsakič, ko uporabimo tujo idejo, ugotovitev, sliko ali
  besedilo. Splošno znanih dejstev ne citiramo. **Ob dvomu vir navedemo.**
- **Povzemanje je boljše od dobesednega navajanja.** Dobesedne prepise damo v
  narekovaje in dodamo citat; praviloma pa prebrano preoblikujemo s svojimi
  besedami in citat dodamo na koncu povedi.
- **Tilda pred `\cite`, vedno.** Pišemo `beseda~\cite{oznaka}`, nikoli
  `beseda \cite{oznaka}`. Tilda je nedeljivi presledek in prepreči, da bi oznaka
  citata (npr. [1]) sama padla v novo vrstico.
- **BibTeX je edini način vodenja referenc.** Vsak nov vir gre takoj v
  `literatura.bib` skupaj z zapisom BibTeX, že v fazi branja — ne šele ob
  pisanju.
- **Nobenega izmišljenega vira.** Vnos v `literatura.bib` sme nastati le iz
  zapisa, ki ga je uporabnik dejansko videl oz. ga je mogoče preveriti (DOI,
  arXiv ID, povezava). Če vira nimam, to povem in ga ne rekonstruiram po
  spominu — LLM-ji si vire in podatke izmišljujejo, zato je kritično preverjanje
  obvezno.

## Delo z literaturo

- **Tabela virov.** Vsak prebran članek se zabeleži z: naslovom, avtorji, letom;
  glavnimi doprinosi (1–2 povedi); lastnimi opažanji (zakaj je pomemben za to
  diplomo, kakšne so pomanjkljivosti); zapisom BibTeX.
- **Kje iščemo:** Google Scholar (tudi za hitro preverjanje števila citatov),
  IEEE Xplore, ACM Digital Library, arXiv (prednatisi), raziskovalne funkcije
  LLM-jev (s kritičnim preverjanjem) in iskanje po referencah (angl.
  *snowballing*) — koga članek citira in kdo citira njega.
- **Vrstni red branja članka:** povzetek → diagrami in slike → zaključek in
  rezultati → metodologija. Izjema: če je članek zunaj primarnega področja,
  začnemo z uvodom in sorodnimi deli.

## Metodološke zahteve, ki se odražajo v besedilu

- **Osnovni primerjalni model (angl. baseline) — namenoma ga ne uporabljamo.
  Odločeno 2026-08-16, ne odpiramo znova.** Smernice sicer navajajo primerjavo z
  naivno metodo kot spodnjo mejo, a je ta naloga ne potrebuje: raziskovalno
  vprašanje je, kako se tabelarični temeljni modeli odrežejo *v primerjavi z
  uveljavljenimi drevesnimi ansambli*, in štirje ansambli (RandomForest,
  XGBoost, LightGBM, CatBoost) že sami služijo kot referenčna točka. Naivni
  klasifikator (`DummyClassifier`) je po konstrukciji pri ROC-AUC 0,5 in ne bi
  prispeval nobene informacije. V besedilu torej ne trdimo, da primerjamo z
  naivno spodnjo mejo; namesto tega v metodologiji **eksplicitno utemeljimo**,
  zakaj vlogo referenčne točke prevzamejo drevesni ansambli. Usklajeno z
  `../Diplomsko-delo_Koda/CLAUDE.md`, kjer v `REGISTRY` ni in ne bo vnosa
  tipa `DummyClassifier`.
- **Ponovljivost.** V besedilu navedemo fiksirana naključna semena, vse vhodne
  parametre, verzije knjižnic in okolje, tako da lahko nekdo drug (ali avtor čez
  tri mesece) dobi identične rezultate.
- **Vsaka številka v besedilu ima izvor v generiranih rezultatih.** Številk,
  tabel in grafov ne prepisujemo na pamet — vzamemo jih iz datotek z rezultati v
  repozitoriju s kodo, skupaj z navedbo, katera datoteka jih je proizvedla.
