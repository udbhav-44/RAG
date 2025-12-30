import subprocess
import os
import time
import traceback

VENV_PYTHON = os.path.join("./test", "bin", "python3")  

while True:
    try:
        print(f"Starting pw_new.py in {VENV_PYTHON}...")
        process = subprocess.Popen([VENV_PYTHON, "pw_new.py"])
        process.wait()

        print("pw_new.py has exited.")
        print("Restarting pw_new.py...")
        time.sleep(5)  

    except KeyboardInterrupt:
        print("Received KeyboardInterrupt. Exiting...")
        break
    except subprocess.CalledProcessError as e:
        print("An error occurred while running pw_new.py:")
        print(e)
        traceback.print_exc()
        time.sleep(5)  # Wait before retrying
    else:
        break  # Exit loop if the script runs successfully