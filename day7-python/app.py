import socket 
import time

hostname = socket.gethostname()

print(f"Container hostname: {hostname}")
print("AI infrastructure Day 7 - version 2")

while True:
	print("heartbeat")
	time.sleep(10)

echo '# tiny source change' >> app.py
