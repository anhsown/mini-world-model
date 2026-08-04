# Qwen2-VL-2B × MMAD: deep-analysis summary

- Coverage: 39,669/39,670 parse-valid answers (99.997%).
- Overall micro accuracy: 64.73% (Wilson 95% CI 64.26%–65.20%).
- Macro task accuracy: 65.03%.
- Strongest task: Object Classification (93.82%, n=3,155).
- Weakest task: Defect Classification (41.42%, n=4,688).
- Strongest source: MVTec-AD (75.25%).
- Weakest source: MVTec-LOCO (55.22%).
- Anomaly detection: precision 74.28%, recall 48.07%, F1 58.37%; miss rate 51.93%, overkill 25.85%.
- Largest answer-position shift: D (+3.10% prediction share versus truth share).
- Median latency: 0.687s; p95: 0.773s per record.

## Decision
Qwen2-VL-2B is strong at object recognition and descriptive/analytic questions, but is not yet reliable as an industrial anomaly gate because the false-negative rate is high. The next comparison must reuse the exact manifest and evaluator for Cosmos 3. Probability outputs must be collected in a future run before making calibration or abstention claims.
