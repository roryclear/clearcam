import subprocess
for x in ["y", "r"][:1]: subprocess.run(["python", "test/run_mot.py", x]) # one for now