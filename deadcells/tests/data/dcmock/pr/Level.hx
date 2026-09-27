package pr;

class Level {
	public var map : level.LevelMap;
	public var game : pr.Game;
	public var nbTotalMobs : Int;
	public var nbMobsLeft : Int;
	public var entities : Array<Entity>;
	public function new(g:pr.Game) {
		game = g;
		map = new level.LevelMap();
		entities = [];
		nbTotalMobs = 3;
		nbMobsLeft = 2;
	}
}
