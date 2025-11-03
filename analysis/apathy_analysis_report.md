# Rapport d'analyse - apathy_analysis

Date de génération : 2025-10-01 10:25:32

## Résumé des analyses

### Features significatives (p < 0.05)

**Features 12h significatives :** 4/60

| Feature | t-statistic | p-value |
|---------|-------------|---------|
| acti_freq_mean_12h_6 | 2.9042 | 0.007856 |
| acti_oadl_fft_mean_12h_3 | 2.8429 | 0.009364 |
| acti_freq_mean_12h_3 | 2.1484 | 0.042457 |
| acti_inactivity_min_12h_6 | -2.1215 | 0.042715 |

**Features 3d significatives :** 1/10

| Feature | t-statistic | p-value |
|---------|-------------|---------|
| acti_activity_rate_3d | 2.1001 | 0.049414 |

### Top 10 groupes/features par performance F1

| Rang | Feature/Groupe | Type | F1 Score | Détails |
|------|----------------|------|----------|---------|
| 1 | acti_freq_mean_12h | 12h_group | 0.6499 | 6 feature(s) |
| 2 | acti_inactivity_mean_12h | 12h_group | 0.6395 | 6 feature(s) |
| 3 | acti_inactivity_min_12h | 12h_group | 0.5988 | 6 feature(s) |
| 4 | acti_walk_max_12h | 12h_group | 0.5857 | 6 feature(s) |
| 5 | acti_oadl_fft_mean_12h | 12h_group | 0.5615 | 6 feature(s) |
| 6 | acti_activity_min_12h | 12h_group | 0.5577 | 6 feature(s) |
| 7 | acti_activity_max_3d | individual | 0.5528 | feature individuelle |
| 8 | acti_walk_mean_12h | 12h_group | 0.5407 | 6 feature(s) |
| 9 | acti_inactivity_max_12h | 12h_group | 0.5366 | 6 feature(s) |
| 10 | acti_inactivity_min_3d | individual | 0.5338 | feature individuelle |

### Performance des modèles avec les meilleures features

#### Features 12h sélectionnées
**Nombre de features :** 60

| Modèle | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| MLP | 0.5458 | 0.5193 | 0.5152 | 0.5146 |
| DT | 0.5980 | 0.5854 | 0.5947 | 0.5818 |
| RF | 0.6291 | 0.5568 | 0.5568 | 0.5476 |
| GBM | 0.5997 | 0.5711 | 0.5758 | 0.5713 |
| Ada | 0.6291 | 0.5958 | 0.5985 | 0.5961 |

#### Features 3d sélectionnées
**Nombre de features :** 10

| Modèle | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| MLP | 0.6863 | 0.6344 | 0.6004 | 0.5909 |
| DT | 0.6307 | 0.6076 | 0.6174 | 0.6009 |
| RF | 0.7435 | 0.7212 | 0.6837 | 0.6860 |
| GBM | 0.7157 | 0.6820 | 0.6818 | 0.6709 |
| Ada | 0.6552 | 0.6321 | 0.5966 | 0.5969 |

#### Features combinées (12h + 3d)
**Nombre total de features :** 70

| Modèle | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| MLP | 0.6291 | 0.5870 | 0.5795 | 0.5768 |
| DT | 0.6552 | 0.6321 | 0.5966 | 0.5969 |
| RF | 0.6569 | 0.6037 | 0.5777 | 0.5657 |
| GBM | 0.6275 | 0.5857 | 0.5758 | 0.5744 |
| Ada | 0.6013 | 0.5756 | 0.5777 | 0.5731 |

### Meilleurs résultats globaux

**Meilleur F1-Score :** 0.6860 (RF)
**Meilleure Accuracy :** 0.7435 (RF)
