package en;

class Hero extends Entity {
	public var cells : Int;
	public function new() {
		super(3, 8, 250);
		cells = 42;
	}
}
