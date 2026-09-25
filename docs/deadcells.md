# Dead Cells: lettura dello schermo e albero delle build

Cartella [`deadcells/`](../deadcells/): un programma che descrive in pochi millisecondi cosa c'è
sullo schermo di Dead Cells (vita, fiaschette, cellule, oro, minimappa con percorsi, nemici e
oggetti con posizione e velocità) in un JSON compatto che Rizzo Flow può leggere, più la tabella
delle build migliori con le domande al modello per scegliere armi, pergamene, mutazioni e bioma.

> **Stato al 25 settembre 2026.** Codice e test verificati **solo su frame sintetici** in un
> container Linux senza gioco. **Mai provato sul gioco vero.** Le posizioni dell'HUD e i colori
> della minimappa nel layout predefinito sono **ipotesi** da calibrare su uno screenshot (nessuna
> fonte li documenta). Il rilevatore di nemici e oggetti **non è ancora addestrato**: senza un
> modello, `threats`, `loot` e `places` restano vuoti. Nessun numero di qualità esiste ancora.

## Indice

1. [Come funziona](#1-come-funziona)
2. [Latenza](#2-latenza)
3. [Installazione (Windows, dove gira il gioco)](#3-installazione)
4. [Calibrazione](#4-calibrazione)
5. [Rilevatore di nemici e oggetti](#5-rilevatore-di-nemici-e-oggetti)
6. [Cosa riceve il modello](#6-cosa-riceve-il-modello)
7. [Tabella e albero delle build](#7-tabella-e-albero-delle-build)
8. [Grafo dei biomi](#8-grafo-dei-biomi)
9. [Limiti e prossimi passi](#9-limiti-e-prossimi-passi)

## 1. Come funziona

```
schermo ──► cattura (dxcam, ~1 frame di ritardo) ──► ultimo frame solo (i vecchi si scartano)
            │
            ├─► HUD: barra vita (maschera di colore), fiaschette (blob), cellule/oro (template cifre)
            ├─► minimappa: griglia di celle ► giocatore, zone, obiettivi, percorso BFS ("up 2, right 16")
            ├─► rilevatore YOLO (ONNX) ► tracker (ID stabili, velocità in px/s)
            └─► descrizione: posizioni relative al giocatore in "altezze del giocatore"
                              │
                              └─► (ogni 0.25 s, su un altro thread) POST /v1/decisions
```

| Modulo | Cosa fa |
| --- | --- |
| `capture.py` | `DxcamSource` (Windows, DXGI Desktop Duplication: la via a latenza minima), `MssSource` (portabile), `FileSource` (immagini), `find_window("Dead Cells")`, `LatestFrame` (thread di cattura che tiene solo l'ultimo frame). |
| `layout.py` | Regioni dello schermo in frazioni del frame (valgono a ogni risoluzione 16:9) e classi di colore; `calibrated: false` finché non si verifica su uno screenshot. |
| `hud.py` | `bar_fraction` (vita), `count_blobs` (fiaschette), `DigitReader` (cifre per confronto con template imparati da uno screenshot: niente OCR, < 0.1 ms). |
| `minimap.py` | Classificazione per celle con `cv2.inRange`, zone connesse, BFS vettoriale (una dilatazione per anello), percorsi compressi. |
| `detect.py` | `OnnxDetector` (YOLO v8/11 esportato in ONNX; CUDA, DirectML, CoreML o CPU), `ColorDetector` (blob di colori calibrati, senza training), NMS. |
| `track.py` | Associazione frame-a-frame per classe, velocità con media esponenziale. |
| `scene.py` | Il JSON per il modello: minacce ordinate per tempo d'impatto, bottino, luoghi, mappa. |
| `pipeline.py` | Tutto insieme, con i tempi di ogni fase. |
| `decide.py` | Richieste per `/v1/decisions`: azione in combattimento + pericolo, prossimo obiettivo sulla mappa. |
| `builds.py` | Tabella/albero delle build, copertura dell'inventario, domande per oggetti, pergamene, mutazioni, bioma. |

Il modello non guarda lo schermo: legge solo testo. Tutta la "vista" è in questo programma.

## 2. Latenza

Misurata con `python -m deadcells bench` su un frame sintetico 1920×1080, CPU del container di
sviluppo (Intel Xeon 2.8 GHz, 4 core), caso peggiore per la mappa (tutta esplorata, 98×73
celle, obiettivi negli angoli opposti). Millisecondi, p50 / p95 su 300 frame:

| Fase | ms |
| --- | ---: |
| HUD (vita, fiaschette) | 0.33 / 0.50 |
| Minimappa, invariata dal frame prima (cache) | 1.7 / 2.8 |
| Minimappa, ricalcolo completo (media di frame con e senza cache) | 5.7 / 7.8 |
| `ColorDetector` (4 colori, frame ridotto 1/4) | 2.5 / 3.4 |
| Tracker + descrizione | < 0.1 |
| **Totale senza rilevatore** | **2.1 / 3.3** |

Il ricalcolo completo della mappa costa ~10 ms nel caso peggiore e avviene solo quando il
marcatore del giocatore cambia cella. Il rilevatore YOLO non è misurato (nessun modello): per un
YOLO11n a 640 px su una GPU recente ci si può aspettare qualche millisecondo, ma va misurato.
A questo va sommato il ritardo della cattura (con dxcam il frame arriva quando il gioco lo
presenta). Una decisione del modello costa ~45–50 ms (4B Q8_0, llama.cpp/CUDA, RTX 5060 Ti):
per questo gira su un thread separato e legge l'ultima descrizione, mentre la percezione
continua a frame rate pieno.

## 3. Installazione

Il gioco gira su Windows: l'ambiente va creato lì, separato dal `.venv` del progetto (come
`training/`).

```bash
cd deadcells
uv venv .venv-dc && uv pip install --python .venv-dc -r requirements.txt
uv pip install --python .venv-dc onnxruntime-directml   # oppure onnxruntime-gpu (NVIDIA) / onnxruntime
.venv-dc/Scripts/python -m pytest -q tests              # 28 test, frame sintetici
.venv-dc/Scripts/python -m deadcells bench              # latenza su questa macchina
```

Poi, con il gioco aperto in finestra (o a schermo intero senza bordi):

```bash
.venv-dc/Scripts/python -m deadcells live --layout calibration/layout.json --digits calibration/digits
# con il modello: in un altro terminale `rizzo serve`, poi
.venv-dc/Scripts/python -m deadcells live --layout calibration/layout.json --decide http://127.0.0.1:8017 --route
```

Ogni riga stampata è un JSON (`frame`, `state`, e quando arriva `decision` con `decision_ms`);
all'uscita (Ctrl+C) i tempi di ogni fase e i frame scartati vanno su stderr.

## 4. Calibrazione

1. Fare uno screenshot del gioco alla risoluzione in uso (PNG, senza compressione).
2. `python -m deadcells calibrate shot.png --out calibration` scrive `overlay.png` (le regioni
   disegnate sullo screenshot), un ritaglio per regione e `colours.json` con i colori più
   frequenti di ogni regione.
3. Partire dal layout predefinito (`python -m deadcells layout > calibration/layout.json`),
   correggere le regioni finché `overlay.png` combacia, copiare i colori da `colours.json`
   (vita: la wiki dice verde per la vita attuale, arancione per quella recuperabile, giallo
   Malaise, blu vita bonus), poi mettere `"calibrated": true`.
4. Minimappa: una classe per colore (giocatore `kind: "player"`, pavimento esplorato
   `kind: "walkable"`, ogni icona `kind: "poi"`); l'ordine conta, le prime vincono.
   `minimap_cell` è il lato di una cella in frazione dell'altezza del frame. L'opzione di HUD
   size (50–150 %) cambia le dimensioni: calibrare con quella che si usa.
5. Cifre: `python -m deadcells learn-digits shot.png --roi cells --value 1234` (il valore
   letto a schermo), ripetere con altri screenshot finché tutte e 10 le cifre sono note.

## 5. Rilevatore di nemici e oggetti

"Descrivere esattamente" nemici, proiettili e oggetti richiede un rilevatore addestrato sul
gioco: i colori da soli non bastano. Classi previste (`detect.CLASSES`, l'ordine è l'indice
del modello): `player, enemy, elite, boss, projectile, telegraph, trap, weapon_drop, scroll,
food, gold, cell, chest, door, exit, teleporter, npc, breakable_wall`.

1. Raccolta: `python -m deadcells collect --every 0.5 --count 2000` salva frame in
   `dataset/raw/` mentre si gioca (variare biomi, boss, effetti).
2. Etichettatura in formato YOLO (CVAT, Label Studio o simili) con le classi sopra.
3. Training (ambiente a parte con `ultralytics`), per esempio:
   `yolo detect train model=yolo11n.pt data=dataset.yaml imgsz=640 epochs=100`
4. Export: `yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=640`
5. Uso: `python -m deadcells live --model best.onnx`. `OnnxDetector` controlla che il numero di
   classi coincida.

Niente di questo è stato fatto: nessun dataset, nessun modello, nessuna misura di precisione.

## 6. Cosa riceve il modello

Esempio reale di `python -m deadcells describe` su un frame sintetico:

```json
{"units": "positions relative to the player in player heights; dx>0 right, dy>0 up",
 "player": {"health_percent": 60, "flask_charges": 2, "cells": 234567, "gold": 9876,
            "position_estimated": true},
 "threats": {"total": 0, "listed": []}, "loot": {"total": 0, "listed": []},
 "places": {"total": 0, "listed": []},
 "map": {"player_found": true, "explored_cells": 158, "separate_areas": 1,
         "targets": {"total": 2, "listed": [
           {"type": "door", "path_cells": 18, "route": "up 2, right 16", "dx": 16, "dy": 2},
           {"type": "exit", "path_cells": 41, "route": "up 2, right 26, down 5, right 8", "dx": 34, "dy": -3}]}},
 "warning": "screen layout not calibrated: HUD and map values may be wrong"}
```

Ogni minaccia ha `dx`, `dy`, `distance`, `side`, `level` (sopra/sotto/stesso piano) e, se
tracciata per almeno due frame, `speed`, `approaching` e `seconds_to_reach`. Le liste sono
ordinate per urgenza e limitate (`--limit`, default 6) dichiarando sempre il totale: nessun
troncamento silenzioso. Il testo verso il modello è in inglese come il resto dei prompt.

Domande pronte (tutte validate contro `rizzo_flow.schema` nei test):

| Funzione | Domanda |
| --- | --- |
| `decide.combat_request` | Azione ora (attacco, schivata, salto, abilità, movimento, cura, attesa) + pericolo 0–2 |
| `decide.route_request` | Quale obiettivo raggiungibile sulla mappa |
| `builds.pickup_request` | Quale oggetto prendere (o lasciare), date statistiche e build più vicine |
| `builds.scroll_request` | Quale statistica alzare con una pergamena (anche doppie) |
| `builds.mutation_request` | Quale mutazione |
| `builds.biome_request` | Quale uscita, con rune e Boss Stem Cell controllate |

Manca lo strato che esegue le azioni (tastiera/pad) e la lettura delle schermate di scelta
(nomi degli oggetti offerti): oggi quelle liste vanno passate a mano.

## 7. Tabella e albero delle build

Dati in `deadcells/deadcells/data/builds.json`, rigenerabili con
`python -m deadcells builds` (tabella), `--format tree` (albero), `--format json`.
Versione del gioco: 3.5 "The End is Near". **Scaling di ogni oggetto e colore di ogni
mutazione presi dalle tabelle Cargo della wiki ufficiale** (deadcells.wiki.gg, controllate il
25 settembre 2026); **le build sono scelte della community** (guide citate per ogni build in
`sources`), non ufficiali, e le guide non sempre concordano. Nomi in inglese come nel gioco.

¹ build composta da noi a partire dalle note di sinergia della wiki, non presa da una guida.
² oggetto che non scala con il colore della build (usa un'altra statistica).

| Build | Colore | Armi | Abilità | Mutazioni | Affissi | Stile di gioco |
| --- | --- | --- | --- | --- | --- | --- |
| Backstab assassin | Brutalità | Assassin's Dagger, Meat Skewer | Phaser, Smoke Bomb | Scheme (Brutalità), Predator (Brutalità), Initiative (Brutalità), Instinct of the Master of Arms (incolore) | — | Phaser behind the target, then a burst of critical hits from behind. |
| Bleed | Brutalità | Blood Sword, Sadist's Stiletto, Hemorrhage, Throwing Knife | Knife Dance, Cleaver, Sinew Slicer, Leghugger, Corrosive Cloud | Open Wounds (Brutalità), Combo (Brutalità), Killer Instinct (Brutalità) | Bleed Damage (+60%), Bleed on Hit | Stack bleeding, then finish with a weapon that crits on bleeding targets (Sadist's Stiletto, Hemorrhage). |
| Curse ¹ | Brutalità | Spite Sword, Misericorde, Anathema ² | Indulgence | Demonic Strength (incolore), Cursed Flask (incolore), Damned Vigor (incolore) | — | Stay cursed on purpose for the damage bonuses; Indulgence purges curse on kills. |
| Fire / Oil | Brutalità | Oiled Sword, Torch, Vampire Killer, Fire Blast, Pyrotechnics, Firebrands | Oil Grenade, Fire Grenade, Flamethrower Turret, Holy Water | Killer Instinct (Brutalità), Combo (Brutalità), Vengeance (Brutalità) | Fire Damage (+40% on burning), Blue Fire Damage (+100% on burning oil), Oil on Use / Oil on Deploy | Oil the enemies, set them on fire, then crit with weapons that crit on burning or oiled targets. |
| Speed | Brutalità | Swift Sword | Vampirism, Lightspeed | Frenzy (Brutalità), Velocity (incolore), Get Rich Quick (incolore) | — | Chain kills to keep a speed buff up (Swift Sword crits while it lasts) and run between fights. |
| Spite / Vengeance | Brutalità | Spite Sword, Front Line Shield, Bloodthirsty Shield | Lacerating Aura | Vengeance (Brutalità), Instinct of the Master of Arms (incolore) | — | Take small hits on purpose to keep Spite Sword crits and Vengeance active. |
| Bow / arrows | Tattica | Quick Bow, Multiple-nocks Bow, Marksman's Bow, Bow and Endless Quiver, Infantry Bow | Wolf Trap, Grappling Hook ² | Barbed Tips (Tattica), Ripper (Tattica), Point Blank (Tattica), Tranquility (Tattica) | — | Stick arrows into enemies and rip them out; keep distance (Infantry Bow and Point Blank want close range instead). |
| Electric ¹ | Tattica | Lightning Bolt, Electric Whip, Thunder Shield | Tesla Coil, Lightning Rods, Wings of the Crow, Magnetic Grenade | Support (Tattica), Hunter's Instinct (Tattica), Tranquility (Tattica) | Shock Damage (+40%) | Shock zones and chained damage. |
| Poison | Tattica | Alchemic Carbine, Hokuto's Bow, Blowgun, Snake Fangs | Corrosive Cloud, Barnacle | Networking (Tattica), Tranquility (Tattica), Hunter's Instinct (Tattica) | Poison Damage (+80%), Poison on Hit | Keep poison and Hokuto's marks up from range. |
| Turrets | Tattica | Alchemic Carbine, Parry Shield | Heavy Turret, Double Crossb-o-matic, Barnacle, Tesla Coil, Flamethrower Turret, Sinew Slicer, Crusher, Scavenged Bombard | Support (Tattica), Hunter's Instinct (Tattica), Tranquility (Tattica) | — | Deploy two turrets and fight next to them. |
| Freeze / root control | Sopravvivenza | Nutcracker, Ice Bow, Frost Blast, Ice Crossbow, Repeater Crossbow, Baseball Bat | Ice Grenade, Root Grenade, Stun Grenade, Ice Armor, Wolf Trap | Heart of Ice (Sopravvivenza), Frostbite (incolore), Soldier's Resistance (Sopravvivenza) | Ice Damage (+175% on frozen), Root Damage (+75%), Stun Damage (+70%) | Lock enemies down (freeze, root, stun), then crit with weapons that crit on immobilised targets. |
| Heavy melee / Giantkiller | Sopravvivenza | Broadsword, Symmetrical Lance, Giantkiller, War Spear, Flawless, Shovel | Powerful Grenade, Telluric Shock, Tonic, Ice Armor | Necromancy (Sopravvivenza), Berserker (Sopravvivenza), Soldier's Resistance (Sopravvivenza), Kill Rhythm (Sopravvivenza) | — | Tank hits and land slow, heavy critical hits (Giantkiller crits on elites and bosses). |
| Parry / shield | Sopravvivenza | Punishment, Spiked Shield, Rampart, Ice Shield, Iron Staff, Alucard's Shield | Cocoon, Tonic | Counterattack (Sopravvivenza), Spite (Sopravvivenza), Blind Faith (Sopravvivenza), What Doesn't Kill Me (Sopravvivenza) | — | Parry everything: damage and healing come from parries. |

Regole di scaling (wiki): ogni punto in una statistica dà +15 % di danno agli oggetti di quel
colore (composto, 1.15^(n−1)); gli oggetti a doppio colore usano la statistica più alta; gli
oggetti incolori (forzieri maledetti, progetti appena comprati) e leggendari scalano con la
statistica più alta (a pari merito Brutalità > Tattica > Sopravvivenza); la vita cresce con
tutte e tre, di più con Sopravvivenza.

```
Brutalità (red)
├── Bleed
│   ├── armi: Blood Sword [Brutalità], Sadist's Stiletto [Brutalità/Tattica], Hemorrhage [Brutalità/Tattica], Throwing Knife [Brutalità/Tattica]
│   ├── abilità: Knife Dance [Brutalità/Tattica], Cleaver [Brutalità], Sinew Slicer [Brutalità/Tattica], Leghugger [Brutalità/Tattica], Corrosive Cloud [Brutalità/Tattica]
│   └── mutazioni: Open Wounds, Combo, Killer Instinct
├── Fire / Oil
│   ├── armi: Oiled Sword [Brutalità], Torch [Brutalità], Vampire Killer [Brutalità/Tattica], Fire Blast [Brutalità/Tattica], Pyrotechnics [Brutalità/Tattica], Firebrands [Brutalità/Tattica]
│   ├── abilità: Oil Grenade [Brutalità/Tattica], Fire Grenade [Brutalità], Flamethrower Turret [Brutalità/Tattica], Holy Water [Brutalità/Sopravvivenza]
│   └── mutazioni: Killer Instinct, Combo, Vengeance
├── Speed
│   ├── armi: Swift Sword [Brutalità]
│   ├── abilità: Vampirism [Brutalità/Sopravvivenza], Lightspeed [Brutalità/Tattica]
│   └── mutazioni: Frenzy, Velocity, Get Rich Quick
├── Backstab assassin
│   ├── armi: Assassin's Dagger [Brutalità], Meat Skewer [Brutalità]
│   ├── abilità: Phaser [Brutalità/Tattica], Smoke Bomb [Brutalità/Tattica]
│   └── mutazioni: Scheme, Predator, Initiative, Instinct of the Master of Arms
├── Spite / Vengeance
│   ├── armi: Spite Sword [Brutalità], Front Line Shield [Brutalità/Sopravvivenza], Bloodthirsty Shield [Brutalità/Sopravvivenza]
│   ├── abilità: Lacerating Aura [Brutalità/Tattica]
│   └── mutazioni: Vengeance, Instinct of the Master of Arms
└── Curse
    ├── armi: Spite Sword [Brutalità], Misericorde [Brutalità/Tattica], Anathema [Tattica/Sopravvivenza]
    ├── abilità: Indulgence [Brutalità/Sopravvivenza]
    └── mutazioni: Demonic Strength, Cursed Flask, Damned Vigor
Tattica (purple)
├── Turrets
│   ├── armi: Alchemic Carbine [Tattica], Parry Shield [Tattica/Sopravvivenza]
│   ├── abilità: Heavy Turret [Tattica], Double Crossb-o-matic [Tattica], Barnacle [Tattica], Tesla Coil [Tattica], Flamethrower Turret [Brutalità/Tattica], Sinew Slicer [Brutalità/Tattica], Crusher [Tattica/Sopravvivenza], Scavenged Bombard [Tattica/Sopravvivenza]
│   └── mutazioni: Support, Hunter's Instinct, Tranquility
├── Bow / arrows
│   ├── armi: Quick Bow [Tattica], Multiple-nocks Bow [Tattica], Marksman's Bow [Tattica], Bow and Endless Quiver [Tattica], Infantry Bow [Brutalità/Tattica]
│   ├── abilità: Wolf Trap [Tattica/Sopravvivenza], Grappling Hook [Brutalità]
│   └── mutazioni: Barbed Tips, Ripper, Point Blank, Tranquility
├── Poison
│   ├── armi: Alchemic Carbine [Tattica], Hokuto's Bow [Tattica], Blowgun [Tattica], Snake Fangs [Brutalità/Tattica]
│   ├── abilità: Corrosive Cloud [Brutalità/Tattica], Barnacle [Tattica]
│   └── mutazioni: Networking, Tranquility, Hunter's Instinct
└── Electric
    ├── armi: Lightning Bolt [Tattica], Electric Whip [Tattica], Thunder Shield [Tattica/Sopravvivenza]
    ├── abilità: Tesla Coil [Tattica], Lightning Rods [Tattica], Wings of the Crow [Tattica], Magnetic Grenade [Tattica]
    └── mutazioni: Support, Hunter's Instinct, Tranquility
Sopravvivenza (green)
├── Parry / shield
│   ├── armi: Punishment [Sopravvivenza], Spiked Shield [Sopravvivenza], Rampart [Sopravvivenza], Ice Shield [Sopravvivenza], Iron Staff [Brutalità/Sopravvivenza], Alucard's Shield [Brutalità/Sopravvivenza]
│   ├── abilità: Cocoon [Tattica/Sopravvivenza], Tonic [Sopravvivenza]
│   └── mutazioni: Counterattack, Spite, Blind Faith, What Doesn't Kill Me
├── Freeze / root control
│   ├── armi: Nutcracker [Sopravvivenza], Ice Bow [Tattica/Sopravvivenza], Frost Blast [Tattica/Sopravvivenza], Ice Crossbow [Tattica/Sopravvivenza], Repeater Crossbow [Tattica/Sopravvivenza], Baseball Bat [Brutalità/Sopravvivenza]
│   ├── abilità: Ice Grenade [Sopravvivenza], Root Grenade [Sopravvivenza], Stun Grenade [Sopravvivenza], Ice Armor [Sopravvivenza], Wolf Trap [Tattica/Sopravvivenza]
│   └── mutazioni: Heart of Ice, Frostbite, Soldier's Resistance
└── Heavy melee / Giantkiller
    ├── armi: Broadsword [Sopravvivenza], Symmetrical Lance [Sopravvivenza], Giantkiller [Brutalità/Sopravvivenza], War Spear [Brutalità/Sopravvivenza], Flawless [Brutalità/Sopravvivenza], Shovel [Brutalità/Sopravvivenza]
    ├── abilità: Powerful Grenade [Brutalità/Sopravvivenza], Telluric Shock [Brutalità/Sopravvivenza], Tonic [Sopravvivenza], Ice Armor [Sopravvivenza]
    └── mutazioni: Necromancy, Berserker, Soldier's Resistance, Kill Rhythm
```

`builds.fits(data, inventario)` ordina le build per copertura (fino a 2 armi, 2 abilità, 3
mutazioni per build): è il contesto che accompagna ogni domanda al modello.

## 8. Grafo dei biomi

`deadcells/deadcells/data/biomes.json`: 80 uscite da Prisoners' Quarters ai boss finali (fonte:
wiki, pagine Biomes e Runes), con runa richiesta (Vine, Teleportation, Ram, Spider), condizioni
(Boss Stem Cell, "after beating Dracula", chiavi, Cultist Outfit) e boss di ogni bioma.
`builds.exits` controlla rune e Boss Stem Cell; le altre condizioni si controllano passando
`unlocked`, altrimenti restano nel testo dell'opzione. The Bank (bioma bonus che sostituisce un
bioma normale) non è nel grafo.

## 9. Limiti e prossimi passi

- **Mai provato sul gioco.** Prima cosa da fare su Windows: screenshot, `calibrate`,
  `learn-digits`, `bench`, poi `live` e controllo a occhio del JSON.
- Rilevatore da addestrare (sezione 5): senza, il modello non sa dove sono i nemici.
- La minimappa descrive la mappa disegnata, non le collisioni reali (salti, piattaforme,
  passaggi a senso unico); le icone non documentate (uscite, forzieri, pergamene) vanno
  riconosciute per colore dopo la calibrazione, e la mappa nasconde di proposito molti segreti.
- La posizione del giocatore senza rilevatore è stimata al centro dello schermo
  (`position_estimated: true`).
- Nessuno strato di input: le decisioni vengono stampate, non eseguite. Anche con quello, ~50 ms
  a decisione non bastano per schivate al frame: servirebbe uno strato di riflessi a regole
  (schiva se `seconds_to_reach` < soglia) con il modello per le scelte di medio periodo.
- Le build non sono misurate con il modello: nessun dato dice che le sue scelte siano buone.
