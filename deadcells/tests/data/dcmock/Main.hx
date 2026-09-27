class Main {
	static function main() {
		var g = new pr.Game();
		Sys.println("ready");
		var t = 0;
		while (true) {
			t++;
			g.hero.xr = (t % 100) / 100;
			g.hero.dx = 0.3;
			Sys.sleep(0.01);
		}
	}
}
