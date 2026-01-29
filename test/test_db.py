from utils.db import db
import os

if __name__ == "__main__":
  #os.makedirs("data", exist_ok=True); open("data/cc_cache.db", "a").close()  
  cache_db = db()
  x = cache_db.run_get("links", None)
  assert x == {}
  cache_db.run_put("links", "cam1", "https://link1")
  cache_db.run_put("links", "cam2", "https://link2")
  x = cache_db.run_get("links", "cam1")
  assert x == "https://link1"
  x = cache_db.run_get("links", None)
  assert x == {'cam1': 'https://link1', 'cam2': 'https://link2'}
  cache_db.run_delete("links", "cam2")
  x = cache_db.run_get("links", None)
  assert x == {'cam1': 'https://link1'}
  cache_db.run_put("alerts", "cam1", "a", replace=False)
  cache_db.run_put("alerts", "cam1", "b", replace=False)
  x = cache_db.run_get("alerts", "cam1")
  assert x == {'1': 'a', '2': 'b'}
  cache_db.run_put("alerts", "cam1", "b", replace=True)
  x = cache_db.run_get("alerts", "cam1")
  assert x == 'b'
  cache_db.run_delete("alerts", "cam1")
  x = cache_db.run_get("alerts", "cam1")
  assert x == {}
  cache_db.run_put("settings", "cam1", "x", id="zone")
  cache_db.run_put("settings", "cam1", "y", id="det")
  x = cache_db.run_get("settings", "cam1")
  assert x == {"zone": "x", "det": "y"}
  x = cache_db.run_get("settings", "cam1", "zone")
  assert x == "x"
  cache_db.run_delete("settings", "cam1", "zone")
  x = cache_db.run_get("settings", None)  
  assert x == {'cam1': {'det': 'y'}}
  x = cache_db.run_get("settings", "cam1", None)
  assert x == {'det': 'y'}