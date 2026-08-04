import torch

from jwm import JWM, JWMConfig
from jwm.checkpoint_utils import warmstart_eye_v322, warmstart_reader_v31
from jwm.configs import eye_physical_v32_smoke_scale


def _cfg(decoder):
    return JWMConfig(d_model=32, n_layers=1, n_heads=4, head_dim=8,
                     ffn_hidden=64, rope_sections=(2, 1, 1), image_size=16,
                     patch=8, vision_stem="local", vision_local_layers=1,
                     vision_local_heads=4, vision_window=2,
                     reader_ctc_weight=1.0, reader_box_weight=0.2,
                     reader_decoder=decoder, reader_roi_width=16)


def test_reader_corrective_warmstart_preserves_backbone_resets_heads(tmp_path):
    old = JWM(_cfg("full_page_ctc"))
    with torch.no_grad():
        old.tok_emb.weight.fill_(0.123)
        old.ocr_head.weight.fill_(9.0)
    path = tmp_path / "old.pt"
    torch.save({"model": old.state_dict()}, path)
    new = JWM(_cfg("line_roi_ctc"))
    report = warmstart_reader_v31(new, path)
    assert torch.allclose(new.tok_emb.weight,
                          torch.full_like(new.tok_emb.weight, 0.123))
    assert not torch.allclose(new.ocr_head.weight,
                              torch.full_like(new.ocr_head.weight, 9.0))
    assert report["loaded_tensors"] > 0
    assert report["reinitialized_tensors"] > 0


def test_eye_v322_reuses_tracker_but_resets_collapsed_heads(tmp_path):
    old = JWM(eye_physical_v32_smoke_scale())
    with torch.no_grad():
        old.geometry.tracker.context.weight.fill_(0.123)
        old.geometry.depth[-1].weight.fill_(9.0)
        old.geometry.pose_head[0].weight.fill_(8.0)
    path = tmp_path / "v321.pt"
    torch.save({"version": "jwm-eye-v3.2.1-robust-causal-geometry",
                "model": old.state_dict()}, path)
    new = JWM(eye_physical_v32_smoke_scale())
    report = warmstart_eye_v322(new, path)
    assert torch.allclose(new.geometry.tracker.context.weight,
                          torch.full_like(new.geometry.tracker.context.weight, 0.123))
    assert not torch.allclose(new.geometry.depth[-1].weight,
                              torch.full_like(new.geometry.depth[-1].weight, 9.0))
    assert not torch.allclose(new.geometry.pose_head[0].weight,
                              torch.full_like(new.geometry.pose_head[0].weight, 8.0))
    assert report["retained_tracker_tensors"] > 0
    assert report["corrective_reset_tensors"] > 0
