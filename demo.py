# demo.py
import subprocess
import sys
import os

# Path to your main Streamlit app
app_file = os.path.join(os.path.dirname(__file__), "app.py")  # replace app.py with your actual file name

# Function to run Streamlit
def run_streamlit():
    try:
        # Call Streamlit via subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")

if __name__ == "__main__":
    run_streamlit()
