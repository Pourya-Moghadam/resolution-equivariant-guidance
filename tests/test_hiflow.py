"""HiFlow helpers (reg.models.lumina): the Butterworth mask against the official double loop, and the unsharp mask."""
import torch

from reg.models.lumina import _butterworth, _unsharp


def test_butterworth_matches_official_loop():
    H, W, ratio, n = 12, 20, 0.2, 4
    ref = torch.zeros(H, W)
    for h in range(H):
        for w in range(W):
            d_square = (2 * h / H - 1) ** 2 + (2 * w / W - 1) ** 2
            ref[h, w] = 1 / (1 + (d_square / ratio ** 2) ** n)
    m = _butterworth(H, W, ratio, "cpu")
    assert torch.allclose(m, ref, atol=1e-6)
    assert m[H // 2, W // 2] == 1.0 and m[0, 0] < 1e-5      # DC (centre after fftshift) passes, the corner does not
    x = torch.randn(2, 3, H, W)
    low = torch.fft.ifft2(torch.fft.ifftshift(torch.fft.fftshift(torch.fft.fft2(x)) * m)).real
    assert torch.allclose(low.mean((-2, -1)), x.mean((-2, -1)), atol=1e-5)   # the mean survives the low-pass


def test_unsharp_keeps_flat_and_sharpens_edges():
    flat = torch.full((1, 3, 8, 8), 0.3)
    assert torch.allclose(_unsharp(flat, 1.05, 1.0), flat, atol=1e-6)
    step = torch.zeros(1, 1, 8, 8); step[..., 4:] = 1.0
    out = _unsharp(step, 1.05, 1.0)
    assert out[0, 0, 0, 4] > 1.0 and out[0, 0, 0, 3] < 0.0 and torch.allclose(out[..., 0], step[..., 0])
