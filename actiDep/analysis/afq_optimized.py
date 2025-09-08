import numpy as np
from scipy.stats import t as student_t

def _welch_t(group0, group1):
    """
    group0, group1: arrays shape (n0, p), (n1, p)
    Retourne t (p,), df (p,)
    """
    n0 = group0.shape[0]
    n1 = group1.shape[0]
    mean0 = group0.mean(axis=0)
    mean1 = group1.mean(axis=0)
    var0 = group0.var(axis=0, ddof=1)
    var1 = group1.var(axis=0, ddof=1)
    denom = np.sqrt(var0 / n0 + var1 / n1)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (mean1 - mean0) / denom
    # Welch df
    a = var0 / n0
    b = var1 / n1
    with np.errstate(divide='ignore', invalid='ignore'):
        df = (a + b) ** 2 / ((a**2) / (n0 - 1) + (b**2) / (n1 - 1))
    return t, df

def _p_from_t(t, df):
    # Bilatéral
    with np.errstate(over='ignore', invalid='ignore'):
        p = 2 * student_t.sf(np.abs(t), df)
    return p

def _perm_min_p_group(X, y, nperm, rng):
    """
    X: (n, p), y: binaire (0/1)
    Retourne distribution des min p (nperm,)
    """
    n, p = X.shape
    idx = np.arange(n)
    n1 = int(y.sum())
    n0 = n - n1
    # Préallocation
    min_p = np.empty(nperm, dtype=float)
    for k in range(nperm):
        rng.shuffle(idx)  # in-place
        mask1 = y[idx].astype(bool)  # permuter étiquettes via réindexation
        # plus rapide: sélectionner lignes
        g1 = X[mask1]
        g0 = X[~mask1]
        t, df = _welch_t(g0, g1)
        pvals = _p_from_t(t, df)
        min_p[k] = np.nanmin(pvals)
    return min_p

def _perm_min_p_corr(X, y, nperm, rng):
    """
    X: (n, p), y: continu
    Distribution des min p sous permutation de y.
    """
    n, p = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    sx = Xc.std(axis=0, ddof=1)
    yc = y - y.mean()
    sy = yc.std(ddof=1)
    denom = (n - 1) * sx * sy
    min_p = np.empty(nperm, dtype=float)
    idx = np.arange(n)
    for k in range(nperm):
        rng.shuffle(idx)
        yp = y[idx]
        ypc = yp - yp.mean()
        r = (ypc @ Xc) / denom
        # Convertir en t
        with np.errstate(divide='ignore', invalid='ignore'):
            t = r * np.sqrt((n - 2) / np.clip(1 - r**2, 1e-15, None))
        pvals = 2 * student_t.sf(np.abs(t), n - 2)
        min_p[k] = np.nanmin(pvals)
    return min_p

def AFQ_MultiCompCorrection(X, y, alpha, nperm=5000, seed=None):
    """
    Implémentation optimisée (vectorisation + permutations en place) de la correction multi-comparaisons.
    Paramètres:
      X: ndarray (subjects, points)
      y: ndarray (subjects,) (binaire pour test de groupe ou continu pour corrélation)
      alpha: niveau alpha global
      nperm: nb permutations (inchangé)
      seed: optionnel pour reproductibilité
    Retour:
      alphaFWE: seuil p (pointwise) contrôlant FWE
      statFWE: None (réservé compat)
      clusterFWE: np.nan (non utilisé ici)
      stats: dictionnaire {'t': ..., 'p': ...} ou {'r': ..., 't': ..., 'p': ...}
    Remarque:
      clusterFWE non estimé (get_significant_areas est appelé avec cluster_size=1 dans ce pipeline).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n, p = X.shape
    rng = np.random.default_rng(seed)

    unique_y = np.unique(y[~np.isnan(y)])
    is_group = (unique_y.size == 2) and set(unique_y) <= {0,1}

    if is_group:
        # Stat réel
        g1 = X[y == 1]
        g0 = X[y == 0]
        t_real, df_real = _welch_t(g0, g1)
        p_real = _p_from_t(t_real, df_real)
        # Distribution min p
        min_p_dist = _perm_min_p_group(X, y, nperm, rng)
        alphaFWE = np.quantile(min_p_dist, alpha)
        stats = {'t': t_real, 'p': p_real, 'df': df_real}
    else:
        # Corrélation
        n_eff = n - np.isnan(y).sum()
        X_valid = X
        y_valid = y
        # Centrage
        Xc = X_valid - X_valid.mean(axis=0, keepdims=True)
        sx = Xc.std(axis=0, ddof=1)
        yc = y_valid - y_valid.mean()
        sy = yc.std(ddof=1)
        denom = (n - 1) * sx * sy
        r_real = (yc @ Xc) / denom
        with np.errstate(divide='ignore', invalid='ignore'):
            t_real = r_real * np.sqrt((n - 2) / np.clip(1 - r_real**2, 1e-15, None))
        p_real = 2 * student_t.sf(np.abs(t_real), n - 2)
        min_p_dist = _perm_min_p_corr(X_valid, y_valid, nperm, rng)
        alphaFWE = np.quantile(min_p_dist, alpha)
        stats = {'r': r_real, 't': t_real, 'p': p_real, 'df': n - 2}

    clusterFWE = np.nan
    statFWE = None
    return alphaFWE, statFWE, clusterFWE, stats
