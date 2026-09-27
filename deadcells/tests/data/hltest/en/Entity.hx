package en;

class Entity {
	public var cx : Int;
	public var cy : Int;
	public var xr : Float;
	public var yr : Float;
	public var life : Int;
	public var name : String;
	public function new(name:String, cx:Int, cy:Int) {
		this.name = name;
		this.cx = cx;
		this.cy = cy;
		xr = 0.25;
		yr = 0.75;
		life = 100;
	}
}
