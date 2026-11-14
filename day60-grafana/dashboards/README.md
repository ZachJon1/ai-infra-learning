# Export the your Grafana dashboard if you cannot share 

This folder intentionally has no fabricated dashboard JSON.

1. Save all panels and return to the dashboard in Grafana.
2. Click **Export** in the top toolbar (or **Share → Export**, depending on UI).
3. Select **Export as code**, choose **Classic** JSON for portability.
4. Enable the option to share with another instance where available, to remove instance-specific identifiers.
5. Download and save as `ai-infra-monitoring.json` in this directory.
6. Validate locally: `python3 -m json.tool dashboards/ai-infra-monitoring.json >/dev/null`.

Do not commit service-account tokens or confidential links.
