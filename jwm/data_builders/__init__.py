"""Data builders — mirror of Cosmos 3 §3 data organization (2 branches):

Reasoner branch (2 types)              Generator branch (3 types)
  reasoner_pretrain  broad QA            generator_image   T2I caption->scene
  reasoner_sft       Physical-AI hard    generator_video   FD frame pairs
                                         generator_action  grounding bbox

(Cosmos' generator branch is Image&Video / Audio / Action. JWM deliberately has
no audio modality — JARVIS handles speech at the system layer via speech.py and
voice.py — so the three generator types here are Image / Video / Action.)

Every builder shares one validated camera model (autotuned against real JARVIS
frames until 5 Wasserstein statistics pass) and the 3-axis programmatic judge.
Outputs land in data/jwm_v3/ with per-sample quality scores; post-training
tiers are strict-threshold subsets (the micro analog of AI-judge >=2 vs >=5).
"""
