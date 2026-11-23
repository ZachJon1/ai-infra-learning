
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class AlertReceiver(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))

        print("\n=== ALERTMANAGER NOTIFICATION ===", flush=True)
        print("Group status:", payload.get("status"), flush=True)

        for alert in payload.get("alerts", []):
            print({
                "alertname": alert.get("labels", {}).get("alertname"),
                "severity": alert.get("labels", {}).get("severity"),
                "status": alert.get("status"),
            }, flush=True)

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")


server = HTTPServer(("127.0.0.1", 19094), AlertReceiver)
print("Listening for alerts on port 19094", flush=True)
server.serve_forever()
