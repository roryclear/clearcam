from utils.db import db
import os
import threading

if __name__ == "__main__": # todo, use different file
  os.remove("data/cc_cache.db") if os.path.exists("data/cc_cache.db") else None
  os.makedirs("data", exist_ok=True); open("data/cc_cache.db", "a").close()  
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
  
  # test with threading
  cache_db.run_put("max_storage", "all", 256)
  for _ in range(100):
    responses = []
    def get_cross_thread(id):
        try:
            if id == 0:
              resp = cache_db.run_get("links", None, timeout=2)
              responses.append(("links", resp))
            elif id == 1:
              resp = cache_db.run_get("max_storage", "all", timeout=2)
              responses.append(("max_storage", resp))
            else:
              resp = cache_db.run_get("settings", None, timeout=2)
              responses.append(("settings", resp))
        except Exception as e:
            responses.append((f"thread{id}", f"ERROR: {e}"))
    threads = []
    responses.clear()
    for i in range(3):
        t = threading.Thread(target=get_cross_thread, args=(i,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    wrong_type_count = 0
    for table, resp in responses:
        if table == "max_storage" and isinstance(resp, dict):
            wrong_type_count += 1
        elif table != "max_storage" and not isinstance(resp, dict):
            wrong_type_count += 1
    assert wrong_type_count == 0