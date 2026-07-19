import torch

from jwm import JWM, JWMConfig
from jwm.checkpoint_utils import warmstart_reader_v31


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
