# stress_500.py
import requests, time

BASE = "http://localhost:8000"
print("Hammering error endpoint...")

for i in range(500):
    try:
        requests.post(f"{BASE}/predictions/", json={"input": "test"}, timeout=2)
    except:
        pass
    if i % 50 == 0:
        print(f"  {i}/500")
    time.sleep(0.01)  # lebih cepat dari sebelumnya

print("Done. Pantau /alerts sekarang.")