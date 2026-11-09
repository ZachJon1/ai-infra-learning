# Interview / portfolio talk track

## 60-second narrative

I built a GPU-backed model-serving lab on Ubuntu using vLLM 0.33.0 and Qwen2.5-0.5B-Instruct on an RTX 3070. I verified end-to-end inference, wrote Python benchmark scripts to measure non-streaming latency and throughput, and ran a concurrency sweep from 1 to 16. My highest tested throughput was 19.149 requests per second on a repeated short-prompt workload; I also saw a p95-latency tradeoff as concurrency increased. Then I compared prefix caching ON and OFF. With a long shared prefix, throughput improved 21.6% in our experimental run and p95 latency decreased 16.0%. Finally, I reproduced and recovered four classes of failure: invalid API schema, inference server shutdown, context-length violation, and an untolerated Kubernetes node taint.

## Interview questions and defensible answers

**Q: Why did throughput increase with concurrency?** A: vLLM can batch and overlap execution for requests, making better use of available compute. The sweep is consistent with this, but does not isolate continuous batching as the only cause; thermal conditions, clock speeds, and caching matter.

**Q: Where was the knee of the latency/throughput tradeoff?** A: Lowest measured p95 was 684.592 ms at concurrency 4. At 16, p95 rose to 860.079 ms while RPS rose from 6.069 to 19.149. Choosing a level requires a concrete SLO and realistic traffic mix.

**Q: How did you prove prefix caching helped?** A: Compared the same prompt generator, request count, concurrency and model with caching ON/OFF. For shared-prefix prompts: 5.454 vs 4.486 RPS and 761.477 vs 906.434 ms p95. This is preliminary, not a production causal estimate: only one run per setting and clock/power order might matter.

**Q: Why isn't context overflow a GPU OOM?** A: The server rejected a >2048-token input with HTTP 400 and remained healthy, so the failure was a request-level context limit before successful generation. CUDA OOM requires allocation/runtime evidence.

**Q: Why did Kubernetes preemption not fix the Pending Pod?** A: The Pod lacked a matching toleration for the node's `dedicated=gpu:NoSchedule` taint. Evicting other workloads would not change that policy. After adding the toleration, the Pod scheduled and ran.

**Q: Is this fully production-ready or Kubernetes-deployed?** A: No. Inference was host-local vLLM; the separate Kubernetes lab tested GPU resource registration and Pod scheduling. Productionization would require containerizing vLLM, Deployments/Services, monitoring, auth/security, resource policies, deployment automation, realistic load testing, and resilience validation.

**Q: What surprised you?** A: The GPU was configured with a 55 W power limit vs a driver-reported 125 W default, and GPU utilization did reach 100% in sustained cache tests. I treated this as a possible performance constraint but did not modify hardware limits without understanding cooling/firmware considerations.

## CV bullet draft (edit to fit evidence)

- Built a local GPU model-serving and benchmarking capstone with **vLLM / Qwen2.5 / RTX 3070**, measuring **19.15 requests/s at 16 concurrent clients** on a controlled short-prompt workload and quantifying **21.6% shared-prefix throughput improvement** with KV-prefix caching in an exploratory A/B test.
- Diagnosed and recovered four API/inference/Kubernetes fault scenarios, including `FailedScheduling` from an untolerated GPU-node taint; documented root causes, remediation, and end-to-end recovery verification.
