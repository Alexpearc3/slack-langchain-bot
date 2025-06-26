import os
import requests
import zipfile
import subprocess

CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
INSTALL_PATH = os.path.join(os.path.expanduser("~"), "cloudflared")
BINARY_PATH = os.path.join(INSTALL_PATH, "cloudflared.exe")

# Create install directory if it doesn't exist
os.makedirs(INSTALL_PATH, exist_ok=True)

# Download cloudflared
print("Downloading cloudflared...")
r = requests.get(CLOUDFLARED_URL)
with open(BINARY_PATH, "wb") as f:
    f.write(r.content)

print(f"cloudflared downloaded to: {BINARY_PATH}")

# Run the tunnel (change port if needed)
print("Starting tunnel...")
subprocess.Popen([BINARY_PATH, "tunnel", "--url", "http://localhost:3000"])