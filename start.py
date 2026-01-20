import os
import subprocess
import time
import logging
from logging.handlers import RotatingFileHandler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(BASE_DIR, "pw_new.py")
VENV_PYTHON = os.getenv("VENV_PYTHON", os.path.join(BASE_DIR, "test", "bin", "python3"))
PATHWAY_BIN = os.getenv("PATHWAY_BIN", os.path.join(BASE_DIR, "test", "bin", "pathway"))
THREADS = os.getenv("PW_NEW_THREADS", "8")
PROCESSES = os.getenv("PW_NEW_PROCESSES", "1")
FIRST_PORT = os.getenv("PW_NEW_FIRST_PORT", "")

def _pick_executable(candidate: str, fallback: str) -> str:
    return candidate if os.path.exists(candidate) else fallback

python_bin = _pick_executable(VENV_PYTHON, "python3")
pathway_bin = _pick_executable(PATHWAY_BIN, "pathway")

RUN_COMMAND = [pathway_bin, "spawn", "-t", str(THREADS), "-n", str(PROCESSES)]
if FIRST_PORT:
    RUN_COMMAND.extend(["--first-port", str(FIRST_PORT)])
RUN_COMMAND.extend([python_bin, SCRIPT_PATH])
INITIAL_RESTART_DELAY = float(os.getenv("PW_NEW_RESTART_DELAY", "5"))
MAX_RESTART_DELAY = float(os.getenv("PW_NEW_RESTART_MAX_DELAY", "60"))

LOG_DIR = os.getenv("PW_NEW_LOG_DIR", "./logs")
LOG_FILE = os.path.join(LOG_DIR, "pw_new_runner.log")


def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger("pw_new_runner")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


logger = setup_logging()
restart_delay = INITIAL_RESTART_DELAY

while True:
    process = None
    try:
        logger.info("Starting %s", " ".join(RUN_COMMAND))
        process = subprocess.Popen(RUN_COMMAND, cwd=BASE_DIR)
        return_code = process.wait()

        if return_code == 0:
            logger.info("pw_new.py exited cleanly with code 0; stopping.")
            break

        logger.error("pw_new.py exited with code %s; restarting in %.1fs", return_code, restart_delay)
        time.sleep(restart_delay)
        restart_delay = min(restart_delay * 2, MAX_RESTART_DELAY)

    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt. Exiting...")
        if process and process.poll() is None:
            process.terminate()
        break
    except subprocess.CalledProcessError as e:
        logger.error("An error occurred while running pw_new.py: %s", e)
        logger.exception("Runner crash")
        time.sleep(restart_delay)
        restart_delay = min(restart_delay * 2, MAX_RESTART_DELAY)
    except Exception:
        logger.exception("Runner crash")
        time.sleep(restart_delay)
        restart_delay = min(restart_delay * 2, MAX_RESTART_DELAY)
    else:
        break  # Exit loop if the script runs successfully
