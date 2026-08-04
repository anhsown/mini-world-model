# Cosmos 3 Dataset Review for Industrial AI

Date: 2026-07-22  
Owner: Mai Anh Son (Dataset track)  
Scope: dataset identification and curation only. New synthetic-data generation is paused until this map is reviewed.

## 1. Executive conclusion

Cosmos 3 is not trained from one downloadable dataset. It uses two coordinated data curricula:

1. **Reasoner:** about **24.17M** vision-language/text samples: 22.00M pre-training samples and 2.17M supervised fine-tuning (SFT) samples.
2. **Generator:** about **767M retained images**, **347.7M retained video clips**, 139M video-audio samples in pre-training, followed by smaller high-quality image/video/audio/action/control mixtures in mid- and post-training.

The exact Cosmos 3 corpus cannot be reproduced from public data alone. NVIDIA names and releases several important specialist datasets, but the web-scale image/video corpus and multiple AV, surveillance, healthcare, and human-annotation streams are internal or are described only as mixtures. The practical company task is therefore to reproduce the **data contract and curriculum**, not claim a byte-identical replica.

The official reference is the [Cosmos 3 technical report](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf), supported by the [Cosmos 3 project page](https://research.nvidia.com/labs/cosmos-lab/cosmos3/), [official code](https://github.com/nvidia/cosmos), and [official Hugging Face collection](https://huggingface.co/collections/nvidia/cosmos3).

## 2. Evidence labels

| Label | Meaning |
|---|---|
| **Used** | The Cosmos 3 technical report explicitly states that this source or mixture is used in training. |
| **Released** | NVIDIA or the original owner provides public files or a gated public dataset card. |
| **Internal** | The report explicitly identifies the source as internal or does not provide a public release. |
| **Supporting release** | Appears in the official Cosmos 3 collection but is not enough evidence by itself that the entire release was consumed in the reported run. |

## 3. Curriculum at a glance

### 3.1 Reasoner branch

| Stage | Image-text | Video-text | Text-only | Total |
|---|---:|---:|---:|---:|
| Pre-training | 18,814,952 | 1,016,299 | 2,170,762 | 22,002,013 |
| SFT | 1,051,513 | 1,079,200 | 40,960 | 2,171,673 |
| **Total** | **19,866,465** | **2,095,499** | **2,211,722** | **24,173,686** |

Pre-training is dominated by OCR (42.9%), 2D grounding (16.5%), visual QA (11.3%), image reasoning (7.5%), text QA (6.1%), image captioning (5.9%), and video QA (4.5%). SFT shifts heavily toward video, temporal reasoning, spatial grounding, robotics, autonomous driving, and smart infrastructure.

### 3.2 Generator branch

| Training mode | Pre-training | Mid-training | Specialist post-training |
|---|---:|---:|---:|
| Image / text-to-image | 767M | 16M | 8M |
| Video: T2V, I2V, V2V | 347.7M | 75M | 20K I2V |
| Video + audio | 139M | 19M | — |
| Action/video-action | — | 8M | 58K policy samples |
| Video transfer/control | — | 4M | — |

These counts are training samples/modes rather than a count of unique source assets; one asset may be transformed into multiple tasks.

## 4. Dataset map — Reasoner

| Stage/domain | Source used by Cosmos 3 | Scale reported for Cosmos use | Modality and supervision | Access status | Industrial-AI value |
|---|---|---:|---|---|---|
| Pre-train, general | Nemotron Nano 2 collection subset | 19.7M samples | image-text, video-text, text-only; OCR, QA, grounding, reasoning | Mixture not released as an exact Cosmos subset | Medium: broad visual/language foundation |
| Pre-train, added capabilities | Additional math, video, spatial grounding, instruction data | 2.3M | multimodal instructions and QA | Exact composition not disclosed | Medium |
| SFT, 2D grounding | LocateAnything-derived data, majority of the 2D grounding stream | Exact Cosmos subset not stated | boxes, points, referring expressions, OCR/layout, counting | The paper reports a 138M-sample LocateAnything data engine; exact Cosmos subset/release requires confirmation ([paper](https://research.nvidia.com/labs/lpr/locate-anything/LocateAnything.pdf)) | **High**: precise localization and inspection |
| SFT, 3D grounding | 3D scanned scenes converted to camera-relative boxes | Not stated | 3D center, dimensions, orientation, category | Source mixture not named | **High** |
| SFT, temporal | Dense human temporal captions | 55K videos, 2.6K hours, 743K event triplets | start/end timestamps and atomic action captions | Annotation stream not identified as public | **High**: activity/event understanding |
| SFT, temporal | FoundationMotion pipeline | Not stated | four-way motion questions | Pipeline cited; exact Cosmos data not released | Medium |
| SFT, physics | Cosmos human-evaluation annotations | 13.5K tuples from 1K generated videos | Yes/No/Unclear physical and visual judgments | Closely related evaluation release: [Cosmos-HumanEval-v1](https://huggingface.co/datasets/nvidia/Cosmos-HumanEval-v1) under OpenMDW-1.1; training tuple release not confirmed | **High** |
| SFT, physics | VideoPhy-2 | 3.4K videos, 200 actions | 1–5 physical-adherence score converted to QA | Public official [repository/data links](https://github.com/Hritikbansal/videophy) | **High**: physical plausibility |
| SFT, AV | Human-labeled internal driving logs | >10K videos | decision, critical objects, weather, rules, causal CoT | **Internal** | Medium unless AV is in company scope |
| SFT, AV | Auto-labeled internal driving logs | ~1.1M videos | structured decisions and concise reasoning | **Internal** | Medium |
| SFT, AV | Nexar dashcam footage | >24K videos reported in Cosmos stream | temporal events, dense captions, ego behavior | A smaller official public collision set exists ([HF card](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction)); it is not identical to the reported >24K stream | Medium |
| SFT, AV | MADS | sampled at ~1 FPS for grounding; underlying set 1.1M seven-camera samples | cameras, intrinsics/extrinsics, ego pose, world-scenario map, 3D boxes | Public availability not established from the report | Medium/High for mobile industrial systems |
| SFT, robotics | Robot Action-CoT | Not stated | objects, affordance points, collision-free regions, 2D waypoints | Generated using Qwen3-VL-72B, Molmo-7B and MolmoAct/tracked DROID; exact data not released | **High** |
| SFT, robotics | MimicGen rerenders | 3.6K videos across 6 tasks | temporal subtask boundaries from trajectories/kinematics | MimicGen code and datasets are public ([official project](https://mimicgen.github.io/)); Cosmos rerendered subset not separately identified | **High** |
| SFT, robotics | BEHAVIOR-1K | 83K samples | frame + candidate actions -> correct action | Public simulator/benchmark ([official project](https://behavior.stanford.edu/)) | **High**: long-horizon planning |
| SFT, robotics | ERQA from EO-Data-1.5M | Cosmos subset not stated | planning, affordance, failure detection, commonsense, grounding, trajectory | Public Apache-2.0 dataset; 1,422,808 rows / 201GB ([official card](https://huggingface.co/datasets/IPEC-COMMUNITY/EO-Data1.5M)) | **High** |
| SFT, healthcare | Robotic-surgery VQA | 398K conversations over 2.2M images | multi-view VQA, tools, steps, tracking metadata | **Internal**; ORQA-inspired | Low unless medical robotics is in scope |
| SFT, smart infrastructure | PhysicalAI Spatial Intelligence Warehouse | Cosmos samples 80K; source described as 93K RGB-D / 873K QA in report | RGB-D, masks, spatial relations, metric distance, counting, grounding | Public but gated; current card lists ~95K RGB-D and 499K train QA, CC-BY-4.0 ([official card](https://huggingface.co/datasets/nvidia/PhysicalAI-Spatial-Intelligence-Warehouse)) | **Very high** |
| SFT, smart infrastructure | Dense pedestrian localization | 208K images, 5.6M human boxes | pedestrian boxes; PII blurred | **Internal** | **Very high**: worker safety/tracking |
| SFT, smart infrastructure | CARLA collision data + Cosmos Transfer augmentation | 3.4K pair queries | binary vehicle-pair collision prediction | Derived Cosmos set not released | **High** |
| SFT, smart infrastructure | Traffic Anomaly Reasoning (TAR) | 3.6K videos, 26 hours, 44K annotations | QA, temporal/causal reasoning, description, summary | Public official dataset card exists ([TAR](https://huggingface.co/datasets/nvidia/PhysicalAI-Traffic-Anomaly-Reasoning)); source-video licenses must also be followed | **Very high** |
| SFT, smart infrastructure | Tailgating surveillance clips | 1K clips | binary anomaly verification | **Internal** | **Very high** |

## 5. Dataset map — Generator

| Stage | Source used by Cosmos 3 | Reported scale | Modalities/labels | Access status | Industrial-AI value |
|---|---|---:|---|---|---|
| Pre-train | Raw web/licensed image pool | 7.8B raw -> 767M retained | images + captions/metadata | Exact corpus and licenses not disclosed as one release | Medium foundation; not directly reproducible |
| Pre-train | Raw web/licensed video pool | 3B source videos -> 347.7M retained clips | video + captions; scene-split and quality filtered | Exact corpus not released | **High**, but not directly reproducible |
| Pre-train | Video-audio pool | 139M samples | synchronous video and stereo audio | Exact sources not disclosed | Medium |
| Mid-train | High-quality image/video pools | 16M image; 75M video; 19M audio-video | T2I/T2V/I2V/V2V/audio | Exact mixture not released | **High** |
| Mid-train, transfer | Pre-training video subset with generated controls | 3M of 4M transfer samples described | Canny/blur, Video Depth Anything depth, SAM2 segmentation -> RGB | Derived controls not released as one corpus | **Very high** |
| Mid-train, transfer | MADS world-scenario-map | 1.1M seven-view samples | map, lanes, boundaries, lights, dynamic 3D boxes -> RGB | Availability unclear | Medium/High |
| Mid-train, SDG | SDG-PhyxSim | 76,489 simulation runs | RGB, metric depth, instance masks, physics state, cameras, captions | Public OpenMDW-1.1; 16.4TB ([official card](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes)) | **Very high** |
| Mid-train, SDG | SDG-RobotSim | 208,022 clips in training table; public release currently ~373.7K clips | robot collision, manipulation, humanoid motion; partial simulator state/captions | Public OpenMDW-1.1; ~2.08TB ([official card](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Embodied-Robot-Scenes)) | **Very high** |
| Mid-train, SDG | SDG-DriveSim | 264K clips, ~1,467 hours | 4K/24FPS multiview RGB + captions | Public OpenMDW-1.1; ~8.24TB ([official card](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Autonomous-Driving-Scenarios)) | Medium |
| Mid-train, SDG | SDG-SynHuman | 236,937 clips, 5,841 hours | 1080p/30FPS RGB, metric depth, cameras | Public OpenMDW-1.1 ([official card](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Digital-Human-Scenes)) | **High**: human/worker motion, with sim-to-real caveat |
| Mid-train, SDG | SDG-Warehouse | ~122,952 clips, ~412 hours | 1080p/30FPS RGB, depth, segmentation, edges, 2D/3D boxes, cameras | Public OpenMDW-1.1 ([official card](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes)) | **Very high** |
| Mid-train, action | Mixed video-action data | 8M samples | forward dynamics, inverse dynamics, joint video-action prediction | Exact mixture not disclosed | **Very high** |
| Policy post-train | DROID | 58K policy samples in curriculum | 3 RGB views, robot states/actions, language instructions | Public source DROID is 76K demonstrations / 350h ([official project](https://droid-dataset.github.io/)); NVIDIA conversion is [Cosmos3-DROID](https://huggingface.co/datasets/nvidia/Cosmos3-DROID), OpenMDW-1.1 | **Very high** |
| I2V post-train | High-quality I2V set | 20K | conditioning image + target video/text | Exact set not disclosed | High |
| T2I post-train | High-quality image set | 8M | prompt + target image | Exact set not disclosed | Medium |

## 6. Curation protocol that must be copied before data volume

The most reusable part of Cosmos 3 is not a particular dataset name; it is the admission process:

1. **Semantic deduplication:** conversation-level image/text embeddings; video embeddings; K-means partitioning; remove near duplicates over cosine similarity 0.95.
2. **Three-axis AI judging:** Faithfulness, Completeness, Correctness, each scored independently from 1–5.
3. **All-axis threshold:** retain only when every score meets the threshold; do not average away one severe failure.
4. **Stage-dependent quality:** broad pre-training uses threshold 2 (reported 78% retention); precision-sensitive SFT uses threshold 5 (46% retention).
5. **Distribution audit:** inspect retention by capability/domain so strict filters do not eliminate rare industrial skills.
6. **Leakage-safe splits:** split by source scene/episode/site before deriving clips, captions, questions, or control maps.
7. **Real-world validation:** simulation can supplement rare physical events, but it cannot replace held-out real industrial video. Cosmos reports a persistent synthetic-human sim-to-real penalty.

## 7. Recommended Industrial-AI priority order

This is a review order, not permission to start generating synthetic data.

| Priority | Dataset family | Why inspect first |
|---:|---|---|
| 1 | PhysicalAI Spatial Intelligence Warehouse | Public, directly aligned with warehouse geometry, RGB-D, masks, QA, distance and counting |
| 2 | TAR + public real anomaly/event sources | Real temporal and causal supervision for safety events |
| 3 | DROID + MimicGen + EO-Data-1.5M | Connect perception, reasoning and action; includes failures/affordances/trajectories |
| 4 | SDG-Warehouse | Dense labels and controllable industrial hazards; must be anchored to real data |
| 5 | SDG-PhyxSim | Clean physical dynamics and collision supervision |
| 6 | SDG-RobotSim | Embodiment/contact diversity; large storage footprint |
| 7 | VideoPhy-2 / Cosmos-HUE | Physical-plausibility quality gate rather than primary volume source |
| 8 | General OCR/grounding/video data | Necessary foundation, but should be sampled around industrial documents, signage, tools and PPE |

## 8. Open questions for the team/NVIDIA

- Which exact component datasets form the 19.7M Nemotron subset and the 2.3M additional Reasoner data?
- Which licenses and provenance classes make up the 767M-image and 347.7M-video Generator corpora?
- Are the 55K temporal-caption videos and 208K pedestrian images releasable, or must equivalents be sourced?
- Is MADS publicly obtainable, and under what terms?
- Which portion of LocateAnything-Data was used, and are the transformed Cosmos samples distributable?
- Are counts in the paper counts of unique assets, clips, conversations, or repeated task formulations?
- What is the company’s exact Industrial-AI boundary: fixed CCTV, mobile robot, manipulation, warehouse operations, or all four?

## 9. Immediate next actions

1. Review this map with the research lead and freeze the target Industrial-AI boundary.
2. Create a license/provenance record for each candidate public source before downloading it.
3. Download only small metadata/sample subsets first and verify schema, labels, quality, and domain match.
4. Define the real held-out validation set before any synthetic expansion.
5. After approval, design synthetic data only for measured coverage gaps and require an A/B admission test against real validation data.

## 10. Important caveats

- Public dataset cards can change after the technical report. For example, the current RobotSim release count differs from the paper’s training-table count; both are recorded rather than silently merged.
- A dataset appearing in the official Cosmos 3 Hugging Face collection is evidence of ecosystem relevance, but not automatically proof that every sample was used in the reported training run.
- No third-party dataset should be copied into the project repository. Store manifests, checksums, licenses and deterministic download/build scripts instead.
