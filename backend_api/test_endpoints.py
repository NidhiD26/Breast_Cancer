
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_endpoints():
    print(f"Testing connectivity to {BASE_URL}...")
    
    # Test Root
    try:
        resp = requests.get(f"{BASE_URL}/")
        print(f"Root: {resp.status_code} {resp.json()}")
    except Exception as e:
        print(f"Root failed: {e}")
        return

    # Test Analysis
    try:
        resp = requests.get(f"{BASE_URL}/analysis")
        if resp.status_code == 200:
            data = resp.json()
            cors_header = resp.headers.get('access-control-allow-origin')
            print(f"Analysis: OK. CORS: {cors_header}")
        else:
            print(f"Analysis: Failed {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"Analysis failed: {e}")

    # Test Samples
    try:
        resp = requests.get(f"{BASE_URL}/samples")
        if resp.status_code == 200:
            data = resp.json()
            print(f"Samples: OK. Count: {len(data)}")
        else:
            print(f"Samples: Failed {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"Samples failed: {e}")

if __name__ == "__main__":
    test_endpoints()
