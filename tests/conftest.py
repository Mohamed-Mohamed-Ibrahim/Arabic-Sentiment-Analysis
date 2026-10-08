import subprocess
import time

import pytest
import requests


@pytest.fixture(scope="session")
def base_url():
    return "http://0.0.0.0:8000"


@pytest.fixture(scope="session", autouse=True)
def start_server(base_url):
    process = subprocess.Popen(
        ["python", "main.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    server_ready = False
    for _ in range(30): 
        try:
            response = requests.get(base_url, timeout=0.5)
            if response.status_code:
                server_ready = True
                break
        except requests.exceptions.ConnectionError:
            time.sleep(0.1)

    if not server_ready:
        process.terminate()
        raise RuntimeError("Server failed to start on port 8081")

    yield

    process.terminate()
    process.wait()
