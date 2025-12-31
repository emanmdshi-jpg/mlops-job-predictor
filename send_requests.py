"""
Traffic Generator for MLOps Dashboard Demo.
Sends random prediction requests to the inference service to populate the dashboard.
"""
import requests
import time
import random
import sys

URL = "http://localhost:8000/predict"

SAMPLE_DATA = [
    {"skills": "Python, Machine Learning, Docker", "qualification": "M.Sc", "experience_level": "Senior"},
    {"skills": "Java, Spring Boot, SQL", "qualification": "B.Tech", "experience_level": "Mid"},
    {"skills": "React, CSS, JavaScript", "qualification": "B.Sc", "experience_level": "Junior"},
    {"skills": "Excel, Word", "qualification": "High School", "experience_level": "Entry"},
    {"skills": "Kubernetes, Go, Terraform, AWS", "qualification": "PhD", "experience_level": "Executive"},
    {"skills": "Python, Pandas, Scikit-learn", "qualification": "M.Sc", "experience_level": "Mid"},
]

def send_traffic(count=20, delay=1.0):
    print(f"🚀 Sending {count} requests to {URL}...")
    
    success_count = 0
    
    try:
        for i in range(count):
            payload = random.choice(SAMPLE_DATA)
            try:
                response = requests.post(URL, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status", "Unknown")
                    print(f"[{i+1}/{count}] Role: {data['predicted_role']} | Conf: {data['confidence']:.2f} | Status: {status}")
                    success_count += 1
                else:
                    print(f"[{i+1}/{count}] Failed: {response.text}")
            except requests.exceptions.ConnectionError:
                print(f"❌ Could not connect to {URL}. Is the server running?")
                print("Run: 'uvicorn inference_service:app' in a separate terminal.")
                return

            time.sleep(delay)
            
    except KeyboardInterrupt:
        print("\nStopping traffic generator...")

    print(f"\n✅ Completed. Sent {success_count} successful requests.")

if __name__ == "__main__":
    # Check if user passed arguments
    requests_count = 100
    if len(sys.argv) > 1:
        try:
            requests_count = int(sys.argv[1])
        except ValueError:
            pass
            
    print("Press Ctrl+C to stop early.")
    send_traffic(requests_count, delay=0.8)
