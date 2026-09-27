package en;

class Hero extends Entity {
	public var flasks : Int;
	public var dx : Float;
	public var onGround : Bool;
	public var speed : hl.F32;
	public function new() {
		super("hero", 12, 34);
		flasks = 3;
		dx = -1.5;
		onGround = true;
		speed = 2.5;
	}
}
