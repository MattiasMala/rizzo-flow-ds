package en;

class Mob extends Entity {
	public var target : Hero;
	public function new(i:Int, hero:Hero) {
		super("mob" + i, 100 + i, 7);
		target = hero;
	}
}
