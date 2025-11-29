"""
Signal test - check if external interrupt is being sent
"""
import signal
import time
import sys

def handler(sig, frame):
    print(f"\n[CAUGHT] Signal {sig}")
    sys.exit(0)

signal.signal(signal.SIGINT, handler)

print("Waiting for 2 minutes...")
print("If this gets interrupted, an external signal is being sent.")

for i in range(120):
    print(f"\r{i+1}/120 seconds", end="", flush=True)
    time.sleep(1)

print("\nCompleted without interrupt!")
