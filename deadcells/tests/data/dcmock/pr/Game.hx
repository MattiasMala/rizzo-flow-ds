package pr;

class Game {
	public static var ME : Game;
	public var hero : en.Hero;
	public var curLevel : pr.Level;
	public function new() {
		ME = this;
		curLevel = new pr.Level(this);
		hero = new en.Hero();
		curLevel.entities.push(hero);
		for (i in 0...3) {
			var m = new en.Mob(10 + i * 2, i == 1);
			if (i == 2) m.destroyed = true;
			curLevel.entities.push(m);
		}
	}
}
