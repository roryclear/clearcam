import subprocess
for i in range(10):
  subprocess.run( ["python", "test/test_yolo.py"],
      env={
          **__import__("os").environ,
          "PYTHONPATH": ".",
          "BEAM": "2",
          "IGNORE_BEAM_CACHE": "1",
      },
      check=True,
  )
  print("passed",(i+1))
