package level;

class LevelMap {
	public var collisions : Array<Int>;
	public var wid : Int;
	public var hei : Int;
	public var id : String;
	public function new() {
		wid = 20;
		hei = 10;
		id = "PrisonStart";
		collisions = [for (i in 0...wid * hei) (Std.int(i / wid) == hei - 1 || i % wid == 0) ? 1 : 0];
	}
}
