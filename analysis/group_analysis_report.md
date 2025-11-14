# Rapport d'analyse - group_analysis

Date de génération : 2025-11-14 11:06:23

## Résumé des analyses

### Features significatives (p < 0.05)

**Features 12h significatives :** 8/60

| Feature | t-statistic | p-value |
|---------|-------------|---------|
| acti_freq_min_12h_3 | 3.0042 | 0.003956 |
| acti_oadl_min_12h_3 | -2.3928 | 0.020559 |
| acti_activity_mean_12h_3 | -2.2563 | 0.030702 |
| acti_oadl_fft_mean_12h_1 | 2.2906 | 0.031345 |
| acti_walk_fft_max_12h_1 | -2.1971 | 0.032626 |
| acti_walk_fft_mean_12h_3 | -2.1456 | 0.036247 |
| acti_oadl_min_12h_5 | -2.0396 | 0.047990 |
| acti_activity_max_12h_3 | -2.0480 | 0.048185 |

**Features 3d significatives :** 1/10

| Feature | t-statistic | p-value |
|---------|-------------|---------|
| acti_freq_min_3d | 2.1120 | 0.039315 |

### Top 10 groupes/features par performance F1

| Rang | Feature/Groupe | Type | F1 Score | Détails |
|------|----------------|------|----------|---------|
| 1 | acti_oadl_fft_mean_12h | 12h_group | 0.6224 | 6 feature(s) |
| 2 | acti_walk_fft_mean_12h | 12h_group | 0.6209 | 6 feature(s) |
| 3 | acti_activity_max_12h | 12h_group | 0.6123 | 6 feature(s) |
| 4 | acti_walk_fft_max_12h | 12h_group | 0.6014 | 6 feature(s) |
| 5 | acti_freq_min_12h | 12h_group | 0.5745 | 6 feature(s) |
| 6 | acti_activity_rate_3d | individual | 0.5682 | feature individuelle |
| 7 | acti_activity_mean_12h | 12h_group | 0.5642 | 6 feature(s) |
| 8 | acti_oadl_min_12h | 12h_group | 0.5487 | 6 feature(s) |
| 9 | acti_walk_fft_min_12h | 12h_group | 0.5417 | 6 feature(s) |
| 10 | acti_walk_max_12h | 12h_group | 0.5411 | 6 feature(s) |

### Performance des modèles avec les meilleures features

#### Features 12h sélectionnées
**Nombre de features :** 60

| Modèle | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| MLP | 0.4937 | 0.4147 | 0.5490 | 0.4445 |
| DT | 0.6448 | 0.6542 | 0.6033 | 0.5975 |
| RF | 0.6454 | 0.6416 | 0.5964 | 0.5886 |
| GBM | 0.6960 | 0.7035 | 0.6658 | 0.6691 |
| Ada | 0.6948 | 0.6849 | 0.6703 | 0.6731 |

#### Features 3d sélectionnées
**Nombre de features :** 10

| Modèle | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| MLP | 0.5931 | 0.2966 | 0.5000 | 0.3723 |
| DT | 0.5431 | 0.5436 | 0.5437 | 0.5365 |
| RF | 0.5420 | 0.4959 | 0.4959 | 0.4826 |
| GBM | 0.5431 | 0.5261 | 0.5237 | 0.5232 |
| Ada | 0.5609 | 0.5595 | 0.5462 | 0.5414 |

#### Features combinées (12h + 3d)
**Nombre total de features :** 70

| Modèle | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| MLP | 0.6425 | 0.6152 | 0.5948 | 0.5796 |
| DT | 0.6621 | 0.6663 | 0.6172 | 0.6139 |
| RF | 0.5753 | 0.5630 | 0.5490 | 0.5442 |
| GBM | 0.6621 | 0.6697 | 0.6242 | 0.6236 |
| Ada | 0.6448 | 0.6542 | 0.6033 | 0.5975 |

### Meilleurs résultats globaux

**Meilleur F1-Score :** 0.6731 (Ada)
**Meilleure Accuracy :** 0.6960 (GBM)
