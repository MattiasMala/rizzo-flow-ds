Small Haxe program (`Main.hx`, `en/*.hx`) compiled to HashLink bytecode (`test.hl`, Haxe 4.3.3)
to test `deadcells/hashlink.py` on a live HashLink process: `en.Hero extends en.Entity` with
fields of every scalar kind and a String, five `en.Mob` pointing at the hero, and a `Game`
singleton holding an `Array<en.Mob>`. Rebuild with `haxe -hl test.hl -main Main`.
The live test runs only when a HashLink VM is available (`HASHLINK_BIN` or `hl` on the PATH).
