import numpy as np


def datasets(n, seed=0):
    """Known-answer point clouds in R^1024 at matched N: (X, true_ID-or-None)."""
    rng = np.random.default_rng(seed)
    out = {}

    v = rng.standard_normal((n, 6))
    v /= np.linalg.norm(v, axis=1, keepdims=True)          # uniform on S^5 -> ID 5
    Z = np.zeros((n, 1024), np.float32); Z[:, :6] = v
    out["sphere_id5"] = (Z, 5)

    t = 1.5 * np.pi * (1 + 2 * rng.random(n)); h = 21 * rng.random(n)
    sr = np.stack([t * np.cos(t), h, t * np.sin(t)], 1)     # swiss roll -> ID 2
    Zr = np.zeros((n, 1024), np.float32); Zr[:, :3] = (sr - sr.mean(0)) / sr.std(0)
    out["swissroll_id2"] = (Zr, 2)

    B = rng.standard_normal((5, 1024)); c = rng.standard_normal((n, 5))
    lin = (c @ B + 0.01 * rng.standard_normal((n, 1024))).astype(np.float32)   # linear ID 5
    out["linear_id5"] = (lin, 5)

    a = rng.standard_normal((n // 2, 1024)).astype(np.float32) * 0.3
    b = rng.standard_normal((n - n // 2, 1024)).astype(np.float32) * 0.3 + 6.0  # 2 blobs -> beta0=2
    out["twoblobs"] = (np.vstack([a, b]), None)
    return out
