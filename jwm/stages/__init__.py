"""Staged training pipeline — mirror of Cosmos 3 §4 (two branches, 7 stages):

Reasoner training                      Generator training
  r1_reasoner_pretrain                   g1_generator_pretrain   (tower-copy init)
  r2_reasoner_sft                        g2_generator_midtrain   (action enters)
                                         g3_post_text2image
                                         g4_post_image2video
                                         g5_post_policy          (-> deployable brain)

Each stage is its own file, reads the previous stage's checkpoint, trains with
periodic shutdown-safe partials, evaluates, and writes stage_<name>.pt + a JSON
report. jwm/stages/run_pipeline.py chains them into one unified training run.
"""

STAGE_ORDER = [
    "r1_reasoner_pretrain",
    "r2_reasoner_sft",
    "g1_generator_pretrain",
    "g2_generator_midtrain",
    "g3_post_text2image",
    "g4_post_image2video",
    "g5_post_policy",
]
