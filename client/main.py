# client/main.py
import subprocess
import os

def launch_ui():
    ui_path = os.path.join("client", "ui", "interface.py")
    subprocess.run(["streamlit", "run", ui_path])

if __name__ == "__main__":
    launch_ui()
