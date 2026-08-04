# Industrial Dataset Landscape Compared with Cosmos 3

Date: 2026-07-23  
Owner: Mai Anh Son — Dataset track  
Status: Research shortlist complete; legal and pilot admission still required

## 1. Executive conclusion

No public Industrial AI dataset is a drop-in replacement for the Cosmos 3 data curriculum.

Cosmos 3 is broad across language, image, video, audio and robot action, and it supplies strong priors for spatial reasoning, physical dynamics, generation and embodied control. Public industrial datasets are narrower but expose signals that Cosmos 3 public data does not cover deeply: microscopic defects, vibration, motor current, force/torque, PLC/SCADA state, setpoint changes, process faults, degradation histories and product outcomes.

The recommended strategy is therefore **compositional specialization**:

1. keep the Cosmos 3 representation and multimodal episode contract;
2. specialize each missing capability with the best real industrial source;
3. align those specialists using a small company-owned dataset in which video, telemetry, controls and outcomes share a clock;
4. admit synthetic data only when a real-only versus real-plus-synthetic A/B test improves held-out real metrics and the worst subgroup.

## 2. How the datasets differ from Cosmos 3

| Dataset family | What it contributes beyond Cosmos 3 | What it still cannot provide |
|---|---|---|
| Visual inspection | Real defects, pixel masks, logical errors, small-defect and lighting-shift tests | Machine state, causal control, degradation and future video |
| 3D inspection | Surface geometry and depth-only defects | Process history, action and long-term health |
| Audio/vibration | Hidden machine condition not visible in RGB | Spatial scene understanding and explicit interventions |
| Predictive maintenance | Degradation and remaining-useful-life trajectories | Rich visual context and detailed operator actions |
| PLC/process/ICS | State-action time series, setpoints, actuators and fault intervals | Product appearance, human activity and multimodal grounding |
| Procedural assembly | Steps, mistakes, corrections and action anticipation | Factory PLC state and asset health |
| Contact-rich robotics | Synchronized RGB/audio/force/action with success and failure | Broad factory process diversity and long-horizon maintenance |

This complementarity is the main result: a factory world model cannot be trained by simply increasing the number of inspection images. It needs aligned observations, latent physical state, interventions and outcomes.

## 3. Shortlist by Industrial AI capability

### 3.1 Visual quality and defect inspection

| Priority | Dataset | Why it matters | Cosmos role retained |
|---|---|---|---|
| P0 | VisA | 10,821 real images, pixel labels and CC BY 4.0 make it a practical first anchor | VLM reasoning, grounding and anomaly explanation |
| P0 | MVTec AD 2 | Hard OOD inspection: tiny defects, transparent/overlapping objects and unseen lighting | Robust visual representation and abstention |
| P0 | Real-IAD | 151,050 multi-view production-line images across 30 objects | Multi-view reasoning and real-domain adaptation |
| P1 | MVTec LOCO AD | Logical anomalies such as missing or misplaced components | Semantic reasoning beyond texture |
| P1 | MVTec 3D-AD / Real-IAD D3 | Adds 3D shape and surface geometry | Cosmos depth and spatial token pathway |

Use this family for image-level anomaly detection, segmentation, defect description and multi-view consistency. Do not use it alone to claim physical-world prediction.

### 3.2 Acoustic and vibration condition monitoring

| Priority | Dataset | Why it matters | Cosmos gap closed |
|---|---|---|---|
| P0 | MIMII | Real machine sound under different noise conditions | Industrial audio and machine identity |
| P0 | DCASE 2025 Task 2 | First-shot and unseen-machine generalization | OOD detection and calibrated abstention |
| P1 | Paderborn | Vibration plus motor current and varied loads | Hidden mechanical/electrical state |
| P1 | CWRU | Simple and reproducible bearing-fault baseline | Initial vibration encoder validation |

These datasets require a dedicated high-rate time-series encoder or spectrogram tokenizer. Treating vibration as ordinary text tokens would destroy local frequency structure.

### 3.3 Degradation and predictive maintenance

| Priority | Dataset | Why it matters | Cosmos gap closed |
|---|---|---|---|
| P0 | XJTU-SY | True run-to-failure vibration sequences | Long-horizon degradation |
| P1 | NASA C-MAPSS | Standard RUL benchmark across fleets and fault modes | Asset-held-out lifetime prediction |
| P2 | AI4I 2020 | Lightweight schema and pipeline smoke test | Failure/outcome fields |

Use asset-level, forward-in-time splits. Random row splits leak machine identity and future degradation into training.

### 3.4 Process state, control and intervention

| Priority | Dataset | Why it matters | Cosmos gap closed |
|---|---|---|---|
| P0 | Tennessee Eastman | 28 faults, six modes, setpoint changes and mode transitions under CC0 | Forward/inverse process dynamics |
| P0 | SWaT | Real testbed with 51 sensors/actuators and attack intervals | PLC state-action trajectories |
| P0 | HAI | Hundreds of hours of HIL-based SCADA and target-aware anomalies | Process graph and time-aware anomaly reasoning |
| P1 | Bosch Production Line | Rare product failure labels at production scale | Outcome and route supervision |

Tennessee Eastman is the cleanest source for algorithm development because it contains explicit modes and interventions. SWaT and HAI are stronger domain anchors, but cyberattacks are only proxies for equipment and process faults.

### 3.5 Assembly and multimodal action

| Priority | Dataset | Why it matters | Cosmos gap closed |
|---|---|---|---|
| P0 | REASSEMBLE | RGB, event camera, audio, force/torque, proprioception and robot actions share timestamps | Contact-rich world-action learning |
| P0 | RH20T | Over 110,000 real contact-rich sequences across tasks, robots and views | Omnimodal action generalization |
| P1 | Assembly101 | 513 hours with mistakes, corrections, anticipation and 12 views | Human procedural understanding |

REASSEMBLE and RH20T are structurally closest to the Cosmos 3 goal. They should influence the company data schema even when their task domain is not identical to the target factory.

## 4. Dataset-to-Cosmos operational-mode mapping

| Cosmos 3 mode | Industrial data needed | Suitable public sources |
|---|---|---|
| VLM / language answer | image/video plus defect, procedure, alarm or fault explanation | VisA, Real-IAD, MVTec LOCO AD, Assembly101 |
| Text-to-image/video | real factory visual distribution plus structured captions | Real-IAD, VisA; company camera data is still required |
| Video-to-video / forward dynamics | state and action before future observation | REASSEMBLE, RH20T; Tennessee Eastman for nonvisual dynamics |
| Inverse dynamics | observed transition plus causal command/action | REASSEMBLE, RH20T, SWaT, HAI, Tennessee Eastman |
| Joint video-action policy | synchronized observations, actions and success/failure | Cosmos3-DROID, REASSEMBLE, RH20T |
| Synchronous audio-video | aligned machine audio, camera and state | Public coverage remains weak; collect company data |
| Long-horizon world state | degradation, maintenance and failure outcome | XJTU-SY, C-MAPSS; company CMMS/historian data |

## 5. Recommended curriculum

### Stage A — Cosmos foundation

Retain general language, visual grounding, temporal reasoning, video generation and physical-dynamics priors from Cosmos-style data.

### Stage B — Industrial perception specialists

- Visual: VisA + Real-IAD + hard OOD evaluation on MVTec AD 2.
- Acoustic: MIMII + DCASE unseen-machine split.
- Mechanical: Paderborn/CWRU for vibration and electrical signals.

### Stage C — Industrial state and dynamics

- Tennessee Eastman for process forward/inverse dynamics.
- SWaT and HAI for realistic, coupled state-action anomaly sequences.
- XJTU-SY/C-MAPSS for long-horizon degradation and RUL.

### Stage D — Procedures and action

- Assembly101 for human procedure, mistakes and anticipation.
- REASSEMBLE/RH20T for synchronized contact-rich robot action.

### Stage E — Company alignment

Collect the minimum factory-native record:

`episode_id + timestamps + camera/audio + sensor state + command/action + asset/product identity + outcome + uncertainty`

The company set is not optional. It is the bridge that teaches the model which public capabilities correspond to the actual machines, products, shifts and business outcomes.

## 6. Validation and admission criteria

| Check | Required decision rule |
|---|---|
| Provenance and rights | License/access documented per source; incompatible data is not redistributed or merged blindly |
| Realism | Public/synthetic samples are compared with target-factory feature and condition distributions |
| Synchronization | Clock error and missing-sample rate measured for each modality |
| Leakage | Splits grouped by site, machine, batch, episode and time |
| Label validity | SME audit of fault taxonomy, defect severity and outcome |
| OOD | Held-out machine/product/site/lighting/load conditions |
| Synthetic admission | Real+synthetic must beat real-only without degrading the worst source/subgroup |
| Calibration | Confidence/ECE and abstention measured alongside task score |
| Temporal quality | Event and anomaly metrics tolerate delay and duration, not only point accuracy |

## 7. Metrics by family

| Family | Primary metrics | Required secondary checks |
|---|---|---|
| Visual anomaly | AUROC, AUPR, AU-PRO, pixel F1 | per-defect and per-lighting worst group |
| Audio/vibration | AUC, pAUC, macro F1 | unseen machine/load, calibration, false alarms/hour |
| Process anomaly | eTaPR, event F1, detection delay | per-fault recall and false alarm duration |
| RUL | MAE/RMSE, NASA score | asset-held-out error and interval calibration |
| Procedure | segmental F1, edit score, anticipation recall | mistake/correction recall and cross-view transfer |
| World/action | action error, success rate, video-state consistency | counterfactual consistency and closed-loop safety |

## 8. Legal and operational caution

- CC BY and CC0 sources are generally easier to reuse, but company legal review is still required.
- CC BY-SA introduces share-alike obligations.
- CC BY-NC/CC BY-NC-SA datasets are research-only for this project unless separate permission is obtained.
- SWaT and competition datasets have request or competition-specific terms.
- “Publicly downloadable” does not automatically mean “commercially trainable” or “redistributable.”

The machine-readable inventory is in `industrial_dataset_landscape.csv`; rows marked `needs_legal_check` must not enter a commercial training mixture until reviewed.

## 9. Final recommendation

For the first Industrial AI world-model pilot, use:

1. **VisA + Real-IAD** for visual inspection;
2. **MIMII + DCASE Task 2** for machine sound;
3. **Tennessee Eastman + HAI/SWaT** for process state and intervention;
4. **XJTU-SY or C-MAPSS** for degradation;
5. **REASSEMBLE or RH20T** for synchronized multimodal action;
6. a small **company factory alignment set** joining observations, controls and outcomes.

This mixture fills the largest Cosmos 3 industrial gaps while preserving the architecture’s strongest advantage: one shared representation for understanding, simulation and action.

## 10. Primary sources

- Cosmos 3: [project page](https://research.nvidia.com/labs/cosmos-lab/cosmos3/) and [technical report](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf)
- Cosmos physical/industrial data: [SDG-Warehouse](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes), [SDG-PhyxSim](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes), [SDG-RobotSim](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Embodied-Robot-Scenes), [Cosmos3-DROID](https://huggingface.co/datasets/nvidia/Cosmos3-DROID)
- Visual inspection: [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad), [MVTec AD 2](https://www.mvtec.com/company/research/datasets/mvtec-ad-2), [MVTec LOCO AD](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad), [MVTec 3D-AD](https://www.mvtec.com/research-teaching/datasets/mvtec-3d-ad), [VisA](https://registry.opendata.aws/visa/), [Real-IAD](https://huggingface.co/datasets/Real-IAD/Real-IAD)
- Acoustic/vibration: [MIMII](https://zenodo.org/records/3384388), [DCASE 2025 Task 2](https://dcase.community/challenge2025/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring), [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter/download-data-file), [Paderborn Bearing Data Center](https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter)
- Degradation and maintenance: [XJTU-SY](https://biaowang.tech/xjtu-sy-bearing-datasets/), [NASA PCoE repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/), [AI4I 2020](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- Process/control: [Tennessee Eastman reference data](https://data.dtu.dk/articles/dataset/Tennessee_Eastman_Reference_Data_for_Fault-Detection_and_Decision_Support_Systems/13385936), [SWaT](https://www.sutd.edu.sg/itrust/itrust-labs/datasets/dataset-characteristics/swat/), [HAI](https://github.com/icsdataset/hai)
- Procedure/action: [Assembly101](https://assembly-101.github.io/), [REASSEMBLE](https://tuwien-asl.github.io/REASSEMBLE_page/), [RH20T](https://rh20t.github.io/)
