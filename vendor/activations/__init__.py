"""Minimal residual-stream activation extraction and projection helpers.

Adapted from the `temporal-manifolds-last-position` project (which in turn
copies its hook helpers from https://github.com/SD-interp/mech-interp-toolkit).
Only the code needed to load a Hugging Face model, read `layer_out/21` at prompt
token position -1, and place that vector in the saved PLS/residual basis is kept.

Import the submodules directly: `extraction` needs torch and transformers,
`residual_pca` does not.
"""
