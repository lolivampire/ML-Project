# load_test.py
# Jalankan: python load_test.py

import time
import statistics
import concurrent.futures
import urllib.request
import urllib.error
import json

# ── CONFIG ──────────────────────────────────────────
BASE_URL = "https://ml-project-production-f8a9.up.railway.app"  # ganti dengan URL kamu
CONCURRENT_USERS = 5   # simulasi x user bersamaan
TOTAL_REQUESTS   = 15   # total request yang dikirim

SAMPLE_PAYLOAD = {
  "feature_1": 1.5,
  "feature_2": -0.5,
  "feature_3": 2.1,
  "feature_4": 0.8
}

# ── FUNGSI SATU REQUEST ──────────────────────────────
def send_request(request_id: int) -> dict:
    """
    Kirim satu POST request ke /predict.
    Return dict berisi: request_id, status, duration, error.
    """
    url     = f"{BASE_URL}/api/v1/predict"
    payload = json.dumps(SAMPLE_PAYLOAD).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    req   = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    start = time.time()

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            duration = time.time() - start
            return {
                "request_id": request_id,
                "status":     resp.status,        # 200, 422, dll
                "duration":   round(duration, 3), # dalam detik
                "error":      None
            }
    except urllib.error.HTTPError as e:
        duration = time.time() - start
        return {
            "request_id": request_id,
            "status":     e.code,
            "duration":   round(duration, 3),
            "error":      str(e)
        }
    except Exception as e:
        duration = time.time() - start
        return {
            "request_id": request_id,
            "status":     None,
            "duration":   round(duration, 3),
            "error":      str(e)
        }

# ── JALANKAN LOAD TEST ──────────────────────────────
def run_load_test(n_concurrent: int, total: int) -> None:
    print(f"\n{'='*50}")
    print(f"Load Test: {total} requests, {n_concurrent} concurrent")
    print(f"Target: {BASE_URL}/predict")
    print(f"{'='*50}\n")

    results   = []
    wall_start = time.time()

    # ThreadPoolExecutor: jalankan send_request() secara paralel
    # max_workers = jumlah thread = simulasi concurrent users
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_concurrent) as executor:
        futures = [executor.submit(send_request, i) for i in range(total)]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)

            # print progress real-time
            status_icon = "✓" if result["status"] == 200 else "✗"
            print(f"  [{status_icon}] req#{result['request_id']:02d} "
                  f"status={result['status']} "
                  f"time={result['duration']:.3f}s")

    wall_time = time.time() - wall_start

    # ── ANALISIS HASIL ──────────────────────────────
    successful = [r for r in results if r["status"] == 200]
    failed     = [r for r in results if r["status"] != 200]
    durations  = [r["duration"] for r in successful]

    print(f"\n{'='*50}")
    print("HASIL LOAD TEST")
    print(f"{'='*50}")
    print(f"Total requests  : {total}")
    print(f"Berhasil (200)  : {len(successful)} ({len(successful)/total*100:.1f}%)")
    print(f"Gagal           : {len(failed)} ({len(failed)/total*100:.1f}%)")
    print(f"Wall time       : {wall_time:.2f}s")
    print(f"Throughput      : {total/wall_time:.1f} req/s")

    if durations:
        print(f"\nResponse time (request yang berhasil):")
        print(f"  Min    : {min(durations)*1000:.0f}ms")
        print(f"  Max    : {max(durations)*1000:.0f}ms")
        print(f"  Avg    : {statistics.mean(durations)*1000:.0f}ms")
        print(f"  Median : {statistics.median(durations)*1000:.0f}ms")
        # p95: 95% request selesai lebih cepat dari angka ini
        sorted_d = sorted(durations)
        p95_idx  = int(len(sorted_d) * 0.95)
        print(f"  p95    : {sorted_d[p95_idx]*1000:.0f}ms")

    if failed:
        print(f"\nError detail:")
        for r in failed[:5]:  # tampilkan max 5 error
            print(f"  req#{r['request_id']:02d}: status={r['status']} error={r['error']}")

    print(f"{'='*50}\n")

# ── MAIN ──────────────────────────────────────────
if __name__ == "__main__":
    run_load_test(
        n_concurrent=CONCURRENT_USERS,
        total=TOTAL_REQUESTS
    )