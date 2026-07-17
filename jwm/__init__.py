"""JWM — Jarvis World Model: micro Cosmos-3-style omnimodal world model."""

from .config import JWMConfig
from .model import JWM, ConvAE, merge_latent, unmerge_latent

__all__ = ["JWM", "JWMConfig", "ConvAE", "merge_latent", "unmerge_latent"]
