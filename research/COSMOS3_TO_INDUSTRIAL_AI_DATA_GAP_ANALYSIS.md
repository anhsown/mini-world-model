# Cosmos 3 to Industrial AI Data Gap Analysis

Date: 2026-07-23  
Owner: Mai Anh Son — Dataset track  
Task: Analyze Cosmos-to-Industrial AI data gaps  
Status: Complete for research review  

## 1. Decision summary

Cosmos 3 provides a strong foundation for Industrial AI when the task depends on visual understanding, spatial grounding, physical dynamics, human/robot activity, video generation, or action-conditioned prediction. Its public ecosystem is particularly relevant to warehouses, safety events, rigid-body interactions, robot manipulation, and multi-view video.

It is not, by itself, a complete factory dataset foundation. The largest gaps are:

1. synchronized PLC/SCADA, vibration, electrical, thermal, acoustic and process time series;
2. explicit machine commands and interventions aligned with observations;
3. long-horizon degradation, maintenance and remaining-useful-life histories;
4. product quality, scrap, rework, downtime and production outcomes;
5. real factory coverage across machines, products, shifts and sites;
6. subtle factory faults that are not visible in RGB video;
7. leakage-safe splits and deployment-realistic validation at machine/site level.

The correct adaptation is therefore not to replace Cosmos data. It is to retain the Cosmos multimodal and action-oriented data contract, then add factory-native state, control and outcome streams.

## 2. Scope assumption

This analysis assumes that “Industrial AI” includes:

- fixed-camera factory and warehouse monitoring;
- product and surface inspection;
- machine condition monitoring;
- manual and robotic assembly;
- intralogistics and worker safety;
- process monitoring and control;
- predictive maintenance.

Autonomous driving and healthcare are excluded except where their data patterns transfer to mobile industrial robots or multi-sensor control.

The final company scope must still be frozen in the next task, **Define Industrial AI data requirements and scope**.

## 3. What Cosmos 3 already covers well

| Capability | Cosmos 3 evidence | Transfer value |
|---|---|---|
| General visual/language foundation | Reasoner pre-training includes OCR, grounding, VQA, captioning and reasoning | High for signs, dashboards, documents and inspection prompts |
| Spatial grounding | 2D/3D grounding, RGB-D warehouse QA and camera-relative boxes | High for object localization, distances, zones and free space |
| Temporal event understanding | Video QA, dense temporal captions, event localization and anomaly reasoning | High for process steps, incidents and operator activities |
| Physical dynamics | SDG-PhyxSim provides RGB, depth, masks, velocity, rotation and camera state | High for collision, motion and object-interaction priors |
| Industrial safety scenes | SDG-Warehouse includes near-miss, fire, shelf collision and box pickup | High for rare warehouse safety events |
| Robot embodiment and contact | RobotSim, MimicGen, DROID and action modes | High for manipulation and robot-policy adaptation |
| Multi-view observations | Warehouse, driving, robotics and surgery streams | High for synchronized CCTV and robot cameras |
| Video/audio/action architecture | Cosmos 3 treats vision, audio and action as core modalities | Architecturally suitable, although public industrial audio/action coverage is limited |
| Synthetic controllability | Isaac Sim/Omniverse pipelines provide reproducible seeds and dense labels | High for targeted rare-event expansion after real-data admission |

## 4. Gap scoring

### 4.1 Severity scale

| Score | Meaning |
|---:|---|
| 5 | Blocking: a factory world model cannot be trusted without this data |
| 4 | Major: substantial loss of capability or transferability |
| 3 | Moderate: useful Cosmos capability exists but factory adaptation is incomplete |
| 2 | Minor: mostly covered, requiring domain-specific refinement |
| 1 | Low: directly reusable with limited changes |

### 4.2 Priority scale

- **P0:** must be addressed before model training or any claim of factory readiness.
- **P1:** required for a useful pilot and robust evaluation.
- **P2:** expansion after the first validated pilot.

## 5. Cosmos-to-Industrial AI gap matrix

| ID | Gap | Cosmos 3 coverage | Factory requirement | Severity | Priority | Evidence needed to close |
|---|---|---|---|---:|---|---|
| G01 | PLC/SCADA and machine telemetry | Weak. Cosmos is dominated by image/video/text; action data is mainly robotics | Timestamped setpoints, states, currents, pressures, temperatures, flow, vibration, alarms and counters | 5 | P0 | Real synchronized historian export from representative machines |
| G02 | Explicit control actions and interventions | Partial. Robot actions are represented, but factory commands are not broadly covered | Start/stop, speed, valve, heater, recipe, reset, reject, re-route and maintenance interventions | 5 | P0 | Command log aligned to pre-state and post-state |
| G03 | Outcome and business labels | Partial success/failure exists in robotics; factory outcomes are missing | Quality pass, defect, scrap, rework, downtime, throughput, energy, safety consequence and recovery time | 5 | P0 | Traceability from operation/asset to verified outcome |
| G04 | Long-horizon degradation | Weak. Most public visual clips span seconds or minutes | Weeks/months of machine health, wear, maintenance, failure and remaining useful life | 5 | P0 | Leakage-safe run-to-failure or maintenance-cycle histories |
| G05 | Real factory domain anchor | Partial. Public Cosmos industrial releases are heavily synthetic or warehouse-focused | Real cameras, machines, products, operators, lighting, noise, dust and occlusion | 5 | P0 | Held-out real episodes from target factory/site |
| G06 | Factory fault taxonomy | Partial. Cosmos covers visible incidents and robot failures more than subtle machine faults | Bearing wear, leakage, cavitation, sensor drift, overheating, tool wear, misalignment, contamination and control faults | 5 | P0 | SME-approved hierarchical taxonomy with confirmed cases |
| G07 | Sensor/video/audio synchronization | Architecture supports modalities; public industrial examples rarely provide all streams together | Common physical clock, timestamp uncertainty, sampling rates and missing-data indicators | 5 | P0 | Synchronization audit and measured clock error |
| G08 | Product and process diversity | General visual diversity is high, target-factory diversity is unknown | Product variants, recipes, tooling, materials, line speeds, shifts, sites and seasonal conditions | 4 | P1 | Coverage matrix and minimum sample counts by condition |
| G09 | Counterfactual and causal supervision | Forward/inverse/policy modes help, but observational factory data does not identify causality | What happened after an intervention and what likely happens without it | 4 | P1 | Controlled trials, simulator pairs or matched historical interventions |
| G10 | Fine-grained inspection labels | Grounding and segmentation are strong foundations | Microscopic defects, tolerance measurements, severity and disposition | 4 | P1 | Calibrated inspection images, masks/measurements and QA disposition |
| G11 | Industrial audio and vibration | Cosmos supports audio, but public factory audio/action mixtures are not documented at sufficient depth | Raw waveform, microphone geometry, vibration axes, RPM/load and machine identity | 4 | P1 | Real normal/fault recordings under varied load and noise |
| G12 | Manual work instructions and procedure state | Temporal reasoning exists; factory SOP semantics are missing | Step ID, expected/observed action, omission, correction, tool and assembly state | 4 | P1 | SOP-linked multi-view episodes with step boundaries |
| G13 | OT data quality and sensor failure | Generic filtering exists, but factory telemetry failure modes differ | Dropout, stale values, drift, saturation, recalibration, replacement and unit changes | 4 | P1 | Data-quality flags and injected/real sensor-fault examples |
| G14 | Normal-operation coverage | Web/video scale is broad, but target normal process distributions are absent | Stable cycles across machines, shifts, recipes and environmental conditions | 4 | P1 | Representative normal baseline before anomaly oversampling |
| G15 | Site/machine leakage control | Cosmos describes semantic deduplication, not target-factory split policy | No frames or cycles from the same asset/episode leaking across splits | 5 | P0 | Split manifest grouped by site, machine, batch and operation |
| G16 | Calibration and metric geometry | Strong in synthetic datasets and some RGB-D sources | Real camera intrinsics/extrinsics, depth accuracy and calibration revisions | 3 | P1 | Calibration files and reprojection/depth-error audit |
| G17 | Human privacy and labor sensitivity | Some Cosmos streams blur PII, but factory governance is company-specific | Face/badge redaction, consent, retention, access and worker-impact review | 5 | P0 | Approved governance and redaction protocol |
| G18 | OT security and proprietary-process protection | Outside Cosmos training design | Network isolation, export approval, secrets removal and restricted recipes | 5 | P0 | Security classification and access-control plan |
| G19 | Sim-to-real admission | Cosmos shows synthetic value but also persistent domain penalties | Synthetic data must improve held-out real metrics without damaging worst-case subgroups | 5 | P0 | Real-only versus real+synthetic A/B test |
| G20 | Deployment-rate data | Cosmos data may be high resolution but does not define target edge constraints | Target FPS, sensor latency, packet loss, compute limits and action deadline | 4 | P1 | Replay benchmark under production timing constraints |
| G21 | Uncertainty and abstention labels | HUE uses conservative judgments; factory confidence calibration is not directly supplied | Unknown/unclear state, OOD condition and safe abstention/escalation | 4 | P1 | Human adjudication and calibration set with ambiguous cases |
| G22 | Rare catastrophic events | Synthetic datasets cover selected events, real tail coverage remains sparse | Fire, collision, entanglement, spill, unsafe restart and compound failures | 4 | P2 | Scenario hazard analysis followed by controlled simulation |

## 6. Modality-level comparison

| Modality | Cosmos 3 | Industrial AI gap | Recommended role |
|---|---|---|---|
| Language | Strong prompts, QA, captions and reasoning | Factory terminology, SOPs, alarm manuals and maintenance language | Add controlled vocabulary and document grounding |
| RGB image/video | Very strong | Target machine/product/site appearance | Keep Cosmos foundation; adapt on real factory video |
| Depth/segmentation/boxes | Strong in synthetic industrial datasets | Real calibration and label cost | Use simulation for dense labels, validate on real RGB-D |
| Audio | Architecture supports synchronous audio | Limited public factory-aligned audio/action data | Add raw machine audio with RPM/load and identity |
| Robot action | Strongest action coverage | Factory PLC/process actions and non-robot actuators | Generalize action schema to commands and interventions |
| Machine telemetry | Not a core released Cosmos stream | Major missing physical-state signal | Introduce dedicated time-series tokens/encoder |
| Maintenance history | Largely absent | Required for degradation and RUL | Add work orders, inspections and component replacements |
| Quality outcome | Limited task success/failure | Missing product/process economics | Add pass, defect, rework, scrap and downtime |
| Geometry/calibration | Strong in simulation | Real measurement noise and calibration drift | Store revisions and uncertainty |

## 7. Temporal-scale gap

Cosmos-style samples are strongest at short and medium horizons:

- frame-level perception;
- seconds-long physical interaction;
- video event understanding;
- short robot action chunks;
- generated visual futures.

Industrial AI requires a nested temporal hierarchy:

1. **milliseconds:** vibration, current and control-loop response;
2. **seconds:** contact, tool motion, collision and anomaly onset;
3. **minutes:** production cycle, recipe and recovery;
4. **hours:** shift conditions, thermal behavior and throughput;
5. **weeks/months:** degradation, maintenance and failure.

A factory sample cannot be represented only as an isolated clip. The data contract must link:

`frame/window -> operation -> batch -> shift -> machine life -> maintenance event`.

## 8. Action and causality gap

Cosmos 3 defines action as a causal variable connecting adjacent world states. This principle transfers directly, but the factory action vocabulary differs.

### Cosmos-style robotics actions

- ego pose delta;
- end-effector pose delta;
- grasp state;
- joint/robot policy action.

### Required factory actions

- PLC command and setpoint change;
- machine start, stop, pause and reset;
- speed/feed/temperature/pressure adjustment;
- valve, heater, motor and conveyor actuation;
- reject/divert/re-route;
- operator correction;
- maintenance, calibration and component replacement.

For every intervention sample, the dataset should preserve:

- state before action;
- action value and timestamp;
- authorization source;
- state trajectory after action;
- verified outcome;
- action latency;
- whether the action was automatic, operator-driven or simulated.

Without these fields, the model learns correlation rather than controllable world dynamics.

## 9. Real-versus-synthetic gap

### Reusable synthetic strengths

- exact geometry and camera state;
- dense masks, depth and object identity;
- rare-event controllability;
- deterministic replay;
- counterfactual action variants;
- no risk to workers or equipment.

### Synthetic limitations

- simplified material response, wear and deformation;
- unrealistic human and controller behavior;
- insufficient appearance variation;
- missing dirt, glare, vibration blur, sensor failure and process noise;
- inaccurate long-term degradation;
- mismatch between simulated labels and operational outcomes.

### Admission rule

Synthetic data should be admitted only if a real-only versus real+synthetic experiment satisfies all of the following:

1. no regression on the primary held-out real metric;
2. no regression on the worst machine/site/product subgroup beyond tolerance;
3. improvement on the intended capability;
4. no increase in overconfidence or unsafe false negatives;
5. no train/test scene or asset leakage.

## 10. Public datasets that demonstrate the missing factory signals

These sources are evidence of the types of data Industrial AI needs. Selection and licensing are handled in the later task **Identify suitable public Industrial AI datasets**.

| Source | Missing signal it demonstrates | Important limitation |
|---|---|---|
| [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) | Real industrial images and pixel-level anomaly localization | Non-commercial terms; mostly static inspection |
| [MVTec 3D-AD](https://www.mvtec.com/research-teaching/datasets/mvtec-3d-ad) | RGB/3D defect geometry | Non-commercial; no action dynamics |
| [MIMII](https://zenodo.org/records/3384388) | Real factory machine sounds, noise and anomalies | Audio only; limited machine types |
| [CWRU Bearing Data](https://engineering.case.edu/bearingdatacenter/download-data-file) | Vibration, RPM and bearing faults | Laboratory rig rather than complete factory process |
| [Assembly101](https://assembly-101.github.io/) | Multi-view procedure steps, mistakes and corrections | Toy assembly; non-commercial |
| [NVIDIA IndustReal](https://developer.nvidia.com/blog/transferring-industrial-robot-assembly-tasks-from-simulation-to-reality/) | Contact-rich assembly and sim-to-real control | Narrow robot/task scope |
| [Tennessee Eastman Process](https://github.com/mv-per/tennessee-eastman-dataset) | Process variables, control modes and faults | Simulated chemical process |
| [NASA PCoE / C-MAPSS](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) | Long-horizon degradation and RUL | Turbofan simulation; not direct factory footage |

## 11. Minimum factory-native data unit

The smallest useful Industrial-AI world-model episode should contain:

```text
identity:
  site, line, machine, product/batch, operation

time:
  absolute timestamps, sampling rates, synchronization uncertainty

observation:
  RGB/depth/video/audio + PLC/SCADA + vibration/electrical/process sensors

state:
  machine mode, process phase, asset/part pose, health and environment

action:
  command/intervention, actor, parameters and timestamp

future:
  next sensor state, next visual state, event and action consequence

outcome:
  quality, safety, downtime, energy, throughput, maintenance

annotation:
  events, boxes/masks/tracks, fault cause, confidence and provenance
```

This preserves Cosmos 3’s unified observation/action/world-transition concept while making it meaningful for factory operations.

## 12. Highest-risk assumptions

| Assumption | Risk |
|---|---|
| Warehouse data represents all Industrial AI | Factories include process, machine and product-quality signals absent from warehouse video |
| More synthetic video will solve data scarcity | It may increase visual volume without adding real faults, controls or outcomes |
| RGB contains enough machine state | Many faults are detectable first in vibration, current, audio or process variables |
| Fault labels alone teach causality | A label does not show which intervention changes the future |
| Random frame splits are valid | They cause severe leakage across the same machine, batch and operation |
| Robot action schema directly covers factory control | PLC and process actions have different semantics, rates and safety constraints |
| One factory/site is sufficient | The model can memorize layout, camera and machine signatures |

## 13. Gap closure order

### P0 — before training

1. freeze the target factory use cases;
2. define the factory episode and synchronization contract;
3. identify real held-out machines/sites/products;
4. build the fault, action and outcome taxonomy;
5. establish privacy, security and licensing rules;
6. enforce leakage-safe splits;
7. define synthetic-data admission against real data.

### P1 — before pilot

1. collect representative normal operation;
2. add audio/vibration/process telemetry;
3. link SOPs and procedure steps;
4. capture calibration and sensor-quality metadata;
5. build uncertainty/OOD cases;
6. test production latency and missing-sensor behavior.

### P2 — expansion

1. generate rare hazards and counterfactual interventions;
2. broaden sites, machines and product variants;
3. add long-tail compound failures;
4. expand long-horizon maintenance histories.

## 14. Definition of done for this task

The task **Analyze Cosmos-to-Industrial AI data gaps** is complete when the team has reviewed and accepted:

- the Industrial AI boundary assumption;
- the 22-gap matrix and P0/P1/P2 priorities;
- the modality and temporal-scale gaps;
- the distinction between reusable Cosmos data and required factory-native data;
- the minimum factory episode contract;
- the list of unresolved scope decisions passed to the next task.

## 15. Inputs required for the next task

For **Define Industrial AI data requirements and scope**, the team must decide:

1. first deployment use case;
2. target factory/site and machine classes;
3. required prediction horizon;
4. allowed sensors and camera placement;
5. available intervention/control logs;
6. measurable operational outcome;
7. safety-critical failure tolerance;
8. data governance and retention boundary.

## 16. Primary references

- NVIDIA Cosmos 3 technical report: https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf
- NVIDIA Cosmos 3 project: https://research.nvidia.com/labs/cosmos-lab/cosmos3/
- NVIDIA Cosmos code: https://github.com/nvidia/cosmos
- NVIDIA SDG-Warehouse: https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes
- NVIDIA SDG-PhyxSim: https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes
- NVIDIA SDG-RobotSim: https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Embodied-Robot-Scenes
- NVIDIA Cosmos3-DROID: https://huggingface.co/datasets/nvidia/Cosmos3-DROID
- MVTec AD: https://www.mvtec.com/company/research/datasets/mvtec-ad
- MVTec 3D-AD: https://www.mvtec.com/research-teaching/datasets/mvtec-3d-ad
- MIMII: https://zenodo.org/records/3384388
- CWRU Bearing Data Center: https://engineering.case.edu/bearingdatacenter/download-data-file
- Assembly101: https://assembly-101.github.io/
- NVIDIA IndustReal: https://developer.nvidia.com/blog/transferring-industrial-robot-assembly-tasks-from-simulation-to-reality/
- Tennessee Eastman Process dataset: https://github.com/mv-per/tennessee-eastman-dataset
- NASA Prognostics Center of Excellence: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
