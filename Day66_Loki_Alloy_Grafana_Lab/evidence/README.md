# Evidence captures — Day 66

These cropped screenshots were taken as part of the November 21, 2026 learning session. They show:

- `loki-vllm-volume.png`: a LogQL label selector for actual Docker vLLM logs and nonempty Loki log volume.
- `recovered-outage-log.png`: entries for both `day66-alloy-outage-test` and `day66-alloy-recovered` visible after Alloy restarted.

**Interpretation:** These images show post-restart retrieval; they are not proof of zero log loss in all circumstances, nor do they contain a separate during-outage Loki query result. Inspect images before posting publicly.
