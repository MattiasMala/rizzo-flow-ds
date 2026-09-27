Stand-in for Dead Cells' object graph, with the class and field names the game uses (from the
probe's types.txt): `pr.Game.ME` -> `hero: en.Hero`, `curLevel: pr.Level` -> `map:
level.LevelMap` (`collisions`, `wid`, `hei`, `id`) and `entities: Array<Entity>`. Compiled to
`dcmock.hl` with `haxe -hl dcmock.hl -main Main` (Haxe 4.3.3) to test `game_state.py` on a live
HashLink process. Not game code: only the names match.
