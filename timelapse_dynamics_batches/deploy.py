import os
import subprocess
from pyngrok import ngrok

# Start the Streamlit app
streamlit_process = subprocess.Popen(["streamlit", "run", "st_app.py", "--server.port","8506"])

# Start ngrok
public_url = ngrok.connect(8506)
print(f"Public URL: {public_url}")

try:
    # Keep the app running
    streamlit_process.wait()
except KeyboardInterrupt:
    print("Stopping...")
    streamlit_process.terminate()
    ngrok.disconnect(public_url)
    ngrok.kill()
