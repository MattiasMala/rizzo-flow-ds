package en;

class Mob extends Entity {
	public var elite : Bool;
	public function new(cx:Int, elite:Bool) {
		super(cx, 8, 60);
		this.elite = elite;
	}
}
