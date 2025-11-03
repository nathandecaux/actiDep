# Rapport d'analyse - group_analysis

Date de génération : 2025-10-01 10:24:56

## Résumé des analyses

### Features significatives (p < 0.05)

**Features 12h significatives :** 6/60

| Feature | t-statistic | p-value |
|---------|-------------|---------|
| acti_freq_min_12h_3 | 3.0042 | 0.003956 |
| acti_oadl_mean_12h_3 | -2.5280 | 0.017640 |
| acti_oadl_max_12h_3 | -2.4681 | 0.019371 |
| acti_activity_mean_12h_3 | -2.2563 | 0.030702 |
| acti_oadl_fft_mean_12h_1 | 2.2906 | 0.031345 |
| acti_walk_fft_mean_12h_3 | -2.1456 | 0.036247 |

**Features 3d significatives :** 1/10

| Feature | t-statistic | p-value |
|---------|-------------|---------|
| acti_freq_min_3d | 2.1120 | 0.039315 |

### Top 10 groupes/features par performance F1

| Rang | Feature/Groupe | Type | F1 Score | Détails |
|------|----------------|------|----------|---------|
| 1 | acti_walk_fft_mean_12h | 12h_group | 0.6265 | 6 feature(s) |
| 2 | acti_freq_min_12h | 12h_group | 0.5581 | 6 feature(s) |
| 3 | acti_activity_min_3d | individual | 0.5543 | feature individuelle |
| 4 | acti_oadl_fft_mean_12h | 12h_group | 0.5464 | 6 feature(s) |
| 5 | acti_freq_std_12h | 12h_group | 0.5437 | 6 feature(s) |
| 6 | acti_freq_mean_12h | 12h_group | 0.5401 | 6 feature(s) |
| 7 | acti_oadl_fft_max_12h | 12h_group | 0.5377 | 6 feature(s) |
| 8 | acti_activity_mean_12h | 12h_group | 0.5356 | 6 feature(s) |
| 9 | acti_oadl_max_12h | 12h_group | 0.5352 | 6 feature(s) |
| 10 | acti_activity_rate_3d | individual | 0.5322 | feature individuelle |

### Performance des modèles avec les meilleures features

#### Features 12h sélectionnées
**Nombre de features :** 60

| Modèle | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| MLP | 0.4925 | 0.4641 | 0.4620 | 0.4542 |
| DT | 0.6943 | 0.6836 | 0.6703 | 0.6734 |
| RF | 0.7126 | 0.7294 | 0.6658 | 0.6646 |
| GBM | 0.7293 | 0.7547 | 0.7214 | 0.7129 |
| Ada | 0.6454 | 0.6254 | 0.6025 | 0.5970 |

#### Features 3d sélectionnées
**Nombre de features :** 10

| Modèle | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| MLP | 0.5925 | 0.6327 | 0.5123 | 0.4377 |
| DT | 0.4908 | 0.4796 | 0.4788 | 0.4781 |
| RF | 0.5759 | 0.5505 | 0.5437 | 0.5408 |
| GBM | 0.5592 | 0.5550 | 0.5568 | 0.5527 |
| Ada | 0.4902 | 0.4577 | 0.4665 | 0.4599 |

#### Features combinées (12h + 3d)
**Nombre total de features :** 70

| Modèle | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| MLP | 0.5259 | 0.5284 | 0.5290 | 0.5212 |
| DT | 0.6764 | 0.6660 | 0.6618 | 0.6634 |
| RF | 0.7287 | 0.7354 | 0.6928 | 0.6974 |
| GBM | 0.6598 | 0.6622 | 0.6679 | 0.6569 |
| Ada | 0.6264 | 0.6048 | 0.5940 | 0.5916 |

### Meilleurs résultats globaux

**Meilleur F1-Score :** 0.7129 (GBM)
**Meilleure Accuracy :** 0.7293 (GBM)
