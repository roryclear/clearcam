import os, sqlite3, contextlib, string, pickle, functools
from typing import Any, overload, TypeVar

_db_tables = set()
cache_dir: str = "data/"
CACHEDB = os.path.abspath(os.path.join(cache_dir, "cache.db"))
_db_connection = None
VERSION = 1
def db_connection():
  global _db_connection
  if _db_connection is None:
    _db_connection = sqlite3.connect(CACHEDB, timeout=60, isolation_level="IMMEDIATE")
    with contextlib.suppress(sqlite3.OperationalError): _db_connection.execute("PRAGMA journal_mode=WAL").fetchone()
  return _db_connection

def diskcache_put(table: str, key: dict|str|int, val: Any,  id: int|str|None = None, prepickled=False, replace=True):
  if isinstance(key, (str, int)): key = {"key": key}
  conn = db_connection()
  cur = conn.cursor()
  table_name = f"{table}_{VERSION}"
  if table not in _db_tables:
    TYPES = {str: "text", bool: "integer", int: "integer", float: "numeric", bytes: "blob"}
    ltypes = ', '.join(f"{k} {TYPES[type(key[k])]}" for k in key.keys())
    cur.execute(f"""CREATE TABLE IF NOT EXISTS '{table_name}' (id TEXT, {ltypes}, val BLOB, PRIMARY KEY (id, {', '.join(key.keys())}))""")
    _db_tables.add(table)
  if replace:
    if id is None:
      cur.execute(f"""DELETE FROM '{table_name}' WHERE {' AND '.join([f'{x}=?' for x in key.keys()])}""", tuple(key.values()))
      id = "1"
    else:
      cur.execute(f"""DELETE FROM '{table_name}' WHERE id=? AND {' AND '.join([f'{x}=?' for x in key.keys()])} """,
      tuple([str(id)] + list(key.values())))
  else:
    if id is None:
      cur.execute(f""" SELECT COALESCE(MAX(CAST(id AS INTEGER)), 0) + 1 FROM '{table_name}' WHERE {' AND '.join([f'{x}=?' for x in key.keys()])}""", 
        tuple(key.values()))
      result = cur.fetchone()
      id = str(result[0]) if result and result[0] else "1"
    else:
      id = str(id)
  columns = ["id"] + list(key.keys()) + ["val"]
  placeholders = ["?"] * len(columns)
  values = [id] + list(key.values()) + [val if prepickled else pickle.dumps(val)]
  cur.execute(f"""INSERT INTO '{table_name}' ({', '.join(columns)}) VALUES ({', '.join(placeholders)})""", tuple(values))
  conn.commit()
  cur.close()
  return val, id


def diskcache_get(table: str, key: str|None, id: str|None = None) -> Any:
  cur = db_connection().cursor()
  if key is None:
    try: res = cur.execute(f"SELECT * FROM '{table}_{VERSION}'")
    except sqlite3.OperationalError: return None
    out = {}
    for row in res.fetchall():
        row_id, user_key, pickled_val = row[0], row[1], row[-1]
        val = pickle.loads(pickled_val)
        if user_key not in out: out[user_key] = {row_id: val} if row_id != "1" else val
        elif isinstance(out[user_key], dict): out[user_key][row_id] = val
        else: out[user_key] = {"1": out[user_key], row_id: val}
    return {k: (v["1"] if isinstance(v, dict) and len(v) == 1 and "1" in v else v) 
            for k, v in out.items()}
  if id is not None:
      try:
        res = cur.execute(f"SELECT val FROM '{table}_{VERSION}' WHERE key=? AND id=?", (key, str(id)))
        row = res.fetchone()
        return pickle.loads(row[0]) if row else None
      except sqlite3.OperationalError:
          return None
  else:
      try:
          res = cur.execute(f"SELECT id, val FROM '{table}_{VERSION}' WHERE key=?", (key,))
          rows = res.fetchall()
      except sqlite3.OperationalError:
          return None
      if not rows: return None
      if len(rows) == 1 and rows[0][0] == "1":
          return pickle.loads(rows[0][1])
      return {row_id: pickle.loads(val) for row_id, val in rows}

diskcache_put("links", "cam1", "https://link1")
diskcache_put("links", "cam2", "https://link2")
x = diskcache_get("links","cam1")
print(x)
assert x == "https://link1"
x = diskcache_get("links", None)
print(x)
assert x == {'cam1': 'https://link1', 'cam2': 'https://link2'}
diskcache_put("alerts","cam1","a", replace=False)
diskcache_put("alerts","cam1","b", replace=False)
x = diskcache_get("alerts","cam1")
print(x)
assert x == {'1': 'a', '2': 'b'}
diskcache_put("alerts","cam1","b", replace=True)
x = diskcache_get("alerts","cam1")
print(x)
assert x == 'b'
diskcache_put("alerts","cam1", None, replace=True)
x = diskcache_get("alerts","cam1")
print(x)
assert x == None

diskcache_put("settings","cam1","x", id="zone")
diskcache_put("settings","cam1","y", id="det")
x = diskcache_get("settings","cam1")
assert x == {"zone": "x", "det": "y"}
x = diskcache_get("settings", "cam1","zone")
assert x == "x"

