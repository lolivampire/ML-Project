# generate_traffic.py — simpan dan run: python generate_traffic.py
import requests
import time

BASE = "http://localhost:8000"
print("Generating traffic...")

for i in range(200):
    requests.get(f"{BASE}/")
    requests.get(f"{BASE}/error")   # ini yang trigger HighErrorRate
    requests.get(f"{BASE}/slow")
    if i % 20 == 0:
        print(f"  {i}/200 requests sent...")
    time.sleep(0.05)

print("Done. Cek http://localhost:9090/alerts dalam 1-2 menit.")