class Entity {
	public var cx : Int;
	public var cy : Int;
	public var xr : Float;
	public var yr : Float;
	public var dx : Float;
	public var dy : Float;
	public var life : Int;
	public var maxLife : Int;
	public var destroyed : Bool;
	public function new(cx:Int, cy:Int, life:Int) {
		this.cx = cx;
		this.cy = cy;
		xr = 0.5;
		yr = 1.0;
		dx = 0;
		dy = 0;
		this.life = life;
		maxLife = life;
		destroyed = false;
	}
}
