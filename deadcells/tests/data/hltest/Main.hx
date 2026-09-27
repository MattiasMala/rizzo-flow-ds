class Game {
	public static var ME : Game;
	public var hero : en.Hero;
	public var mobs : Array<en.Mob>;
	public var frame : Int;
	public var wid : Int;
	public var hei : Int;
	public var collisions : Array<Int>;
	public function new() {
		wid = 6;
		hei = 4;
		collisions = [for (i in 0...wid * hei) i % 3 == 0 ? 1 : 0];
		ME = this;
		hero = new en.Hero();
		mobs = [for (i in 0...5) new en.Mob(i, hero)];
		frame = 0;
	}
}

class Main {
	static function main() {
		var g = new Game();
		Sys.println("ready");
		while (true) {
			g.frame++;
			g.hero.xr = (g.frame % 100) / 100;
			Sys.sleep(0.01);
		}
	}
}
