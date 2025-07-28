import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Tuple

# =========================
# 1) 归一化函数
# =========================
def normalize(df: pd.DataFrame,
              cols,
              method: str = 'minmax',
              q_low: float = 0.05,
              q_high: float = 0.95,
              eps: float = 1e-12) -> pd.DataFrame:
    """
    对指定列做归一化，返回一个新的 DataFrame（不修改原 df）
    method:
        - 'minmax': (x - min) / (max - min)
        - 'robust': (x - q_low) / (q_high - q_low)（更稳健，推荐小样本下使用）
        - 'zscore': (x - mean) / std
    """
    df_norm = df.copy()
    X = df[cols].astype(float).values
    if method == 'minmax':
        min_v = X.min(axis=0)
        max_v = X.max(axis=0)
        Xn = (X - min_v) / (max_v - min_v + eps)
    elif method == 'robust':
        ql = np.quantile(X, q_low, axis=0)
        qh = np.quantile(X, q_high, axis=0)
        Xn = (X - ql) / (qh - ql + eps)
        Xn = np.clip(Xn, 0.0, 1.0)
    elif method == 'zscore':
        mu = X.mean(axis=0)
        std = X.std(axis=0, ddof=1)
        Xn = (X - mu) / (std + eps)
        # 为了后续与 [0,1] 逻辑一致，也可以再做一次 minmax，但不是必须
    else:
        raise ValueError("Unknown method: {}".format(method))

    df_norm[cols] = Xn
    return df_norm

# =========================
# 2) CRITIC 权重
# =========================
def critic_weights(df: pd.DataFrame,
                   cols,
                   corr: str = 'spearman',
                   eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 CRITIC 权重
    返回:
        w: 权重 (len(cols),)
        Xn: (n_samples, n_metrics) 归一化后的矩阵（这里假定 df[cols] 已经是 0-1 归一化好的，
            如果你传的是未归一化的，函数会先用 minmax 做一次）
    """
    X = df[cols].astype(float).values
    # 如果不是 0-1，可以在这里强制再做一次 minmax，避免 CRITIC 用到的 std/corr 不可比
    Xn = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + eps)

    sigma = Xn.std(axis=0, ddof=1)

    if corr == 'spearman':
        R, _ = spearmanr(Xn, axis=0)
        R = R[:len(cols), :len(cols)]
    elif corr == 'pearson':
        R = np.corrcoef(Xn, rowvar=False)
    else:
        raise ValueError("corr must be 'spearman' or 'pearson'.")

    conflict = np.sum(1 - np.abs(R), axis=1)
    c = sigma * conflict
    w = c / (c.sum() + eps)
    return w, Xn

# =========================
# 3) 几何平均聚合 + EER_AV
# =========================
def compute_EER_AV(df: pd.DataFrame,
                   metric_cols = ['DS', 'DE', 'DC'],
                   energy_col: str = None,
                   norm_method: str = 'robust',
                   corr: str = 'spearman',
                   *,
                   # —— 偏好相关可选参数 ——
                   ds_weight_boost: float = 1.0,   # 方案A：放大 DS 权重的倍数 (>1 生效)
                   ds_eta: float = 1.0,            # 方案B：对 DS 做凸增益映射的指数 (>1 生效)
                   prior_w: np.ndarray = None,     # 主观先验权重，如 np.array([0.6, 0.2, 0.2])
                   mix_lambda: float = None        # 与 CRITIC 权重做凸组合的 λ ∈ [0,1]
                   ):
    """
    计算：
      1) 归一化后的指标
      2) CRITIC 权重（可与主观权重混合）
      3) （可选）放大 DS 权重 or 对 DS 做凸增益映射
      4) 几何平均质量分数 Q
      5) 若提供 energy_col，则返回 EER_AV = Q / E

    偏好控制：
      - ds_weight_boost > 1：放大 DS 在几何平均中的指数（方案 A）
      - ds_eta > 1：对 DS 的值做 x^eta 的凸映射（方案 B）
      - prior_w 与 mix_lambda：主客观混合权重
    """
    df_clean = df.copy()

    # 清洗可能带逗号的小数
    for col in metric_cols + ([energy_col] if energy_col else []):
        if col is not None:
            df_clean[col] = (df_clean[col].astype(str)
                                           .str.replace(",", "")
                                           .astype(float))

    # 先做一次你选择的归一化
    df_norm = normalize(df_clean, metric_cols, method=norm_method)

    # 用归一化后的数据计算 CRITIC 权重
    w, Xn = critic_weights(df_norm, metric_cols, corr=corr)

    # ========= 主客观混合（可选） =========
    if prior_w is not None and mix_lambda is not None:
        prior_w = np.asarray(prior_w, dtype=float)
        if prior_w.shape[0] != len(metric_cols):
            raise ValueError("prior_w length must match metric_cols.")
        if not (0 <= mix_lambda <= 1):
            raise ValueError("mix_lambda must be in [0, 1].")
        # 归一化一下主观权重
        prior_w = prior_w / (prior_w.sum() + 1e-12)
        w = mix_lambda * w + (1 - mix_lambda) * prior_w
        w = w / (w.sum() + 1e-12)

    # ========= 偏好方案 A：放大 DS 的指数（再归一化）=========
    if ds_weight_boost > 1.0:
        ds_idx = metric_cols.index('DS')
        w_prime = w.copy()
        w_prime[ds_idx] *= ds_weight_boost
        w = w_prime / (w_prime.sum() + 1e-12)

    # ========= 偏好方案 B：对 DS 值做凸增益映射 =========
    if ds_eta > 1.0:
        ds_idx = metric_cols.index('DS')
        Xn[:, ds_idx] = np.power(Xn[:, ds_idx], ds_eta)

    # ========= 几何平均质量分数 =========
    Q = np.prod(Xn ** w, axis=1)

    df_out = df_clean.copy()
    # 保存归一化后的列
    for i, c in enumerate(metric_cols):
        df_out[f"n_{c}"] = Xn[:, i]
    # 保存权重
    for wi, c in zip(w, metric_cols):
        df_out[f"critic_w_{c}"] = wi
    df_out["Q"] = Q

    # EER_AV
    if energy_col is not None:
        E = df_out[energy_col].values
        df_out["EER_AV"] = df_out["Q"] / (E + 1e-12)
    else:
        df_out["EER_AV"] = df_out["Q"]

    return df_out, w

# =========================
# 4) Bootstrap (可选)
# =========================
def bootstrap_critic(df, metric_cols, n_boot=1000, norm_method='robust', corr='spearman', random_state=42):
    rng = np.random.default_rng(random_state)
    W = []
    for _ in range(n_boot):
        # 自助采样
        idx = rng.integers(0, len(df), size=len(df))
        df_b = df.iloc[idx].reset_index(drop=True)
        _, w = compute_EER_AV(df_b, metric_cols=metric_cols, energy_col=None,
                              norm_method=norm_method, corr=corr, ds_weight_boost=2.0)
        W.append(w)
    W = np.array(W)
    w_mean = W.mean(axis=0)
    w_ci_l = np.percentile(W, 2.5, axis=0)
    w_ci_u = np.percentile(W, 97.5, axis=0)
    return w_mean, w_ci_l, w_ci_u, W


import pandas as pd
from io import StringIO

raw = """DS\tDE\tDC
33.016\t154\t0.388
38.921\t161.64\t0.293
32.136\t141.82\t0.322
27.95\t140.3597453\t0.385
29.571\t149.745\t0.359
29.537\t143.916\t0.391
35.598\t152.355\t0.324
34.104\t142.92\t0.437
39.886\t157.36\t0.406
38.151\t158.878912\t0.418
18.67\t136.9213231\t0.646
29.858\t146.757\t0.548
19.161\t134.103\t0.645
35.831\t150.438\t0.427
32.01\t147.09\t0.619
35.737\t159.273\t0.436
33.737\t160.354\t0.421
34.049\t161.555\t0.4
35.335\t156.475\t0.418
37.125\t149.319\t0.363
40.242\t151.795\t0.423
39.371\t150.103\t0.421
43.223\t155.313\t0.38
45.643\t153.227\t0.375
48,095\t157.632\t0.283
44.454\t154.8\t0.33
34.104\t142.92\t0.437
39.922\t166.262\t0.544
42.631\t60.34\t0.471
"""

df = pd.read_csv(StringIO(raw), sep=r"\s+", engine="python")
num_cols = ["DS", "DE", "DC"]
df[num_cols] = df[num_cols].replace({",": ""}, regex=True).astype(float)



print(df.head())
print(df.shape)

metric_cols = ['DS', 'DE', 'DC']

# 如果你暂时没有能耗列，就先不传 energy_col
df_out, w = compute_EER_AV(df, metric_cols=metric_cols, energy_col=None,
                           norm_method='robust', corr='spearman',ds_weight_boost=5.0)

print("CRITIC 权重：")
for c, wi in zip(metric_cols, w):
    print(f"  {c}: {wi:.4f}")

print("\n前 5 行结果：")
print(df_out[['DS', 'DE', 'DC', 'n_DS', 'n_DE', 'n_DC', 'Q', 'EER_AV']].head())

w_mean, w_ci_l, w_ci_u, W_all = bootstrap_critic(df, metric_cols, n_boot=1000)
print("\nCRITIC 权重（bootstrap 95% CI）：")
for c, m, l, u in zip(metric_cols, w_mean, w_ci_l, w_ci_u):
    print(f"  {c}: {m:.4f}  (95% CI: {l:.4f} ~ {u:.4f})")