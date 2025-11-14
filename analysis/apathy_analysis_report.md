# Rapport d'analyse - apathy_analysis

Date de génération : 2025-11-14 11:06:56

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
| 1 | acti_freq_mean_12h | 12h_group | 0.6762 | 6 feature(s) |
| 2 | acti_inactivity_max_12h | 12h_group | 0.6335 | 6 feature(s) |
| 3 | acti_inactivity_mean_12h | 12h_group | 0.6200 | 6 feature(s) |
| 4 | acti_activity_rate_3d | individual | 0.6032 | feature individuelle |
| 5 | acti_freq_max_12h | 12h_group | 0.6011 | 6 feature(s) |
| 6 | acti_oadl_fft_min_12h | 12h_group | 0.5961 | 6 feature(s) |
| 7 | acti_inactivity_min_12h | 12h_group | 0.5938 | 6 feature(s) |
| 8 | acti_oadl_fft_mean_12h | 12h_group | 0.5783 | 6 feature(s) |
| 9 | acti_walk_mean_12h | 12h_group | 0.5745 | 6 feature(s) |
| 10 | acti_oadl_min_12h | 12h_group | 0.5710 | 6 feature(s) |

### Performance des modèles avec les meilleures features

#### Features 12h sélectionnées
**Nombre de features :** 60

| Modèle | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| MLP | 0.5997 | 0.5330 | 0.5379 | 0.5249 |
| DT | 0.6242 | 0.6235 | 0.5928 | 0.5931 |
| RF | 0.6552 | 0.6090 | 0.5985 | 0.6012 |
| GBM | 0.5980 | 0.5729 | 0.5739 | 0.5696 |
| Ada | 0.5703 | 0.5494 | 0.5530 | 0.5464 |

#### Features 3d sélectionnées
**Nombre de features :** 10

| Modèle | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| MLP | 0.6569 | 0.3284 | 0.5000 | 0.3964 |
| DT | 0.6275 | 0.5857 | 0.5758 | 0.5744 |
| RF | 0.6569 | 0.4653 | 0.5417 | 0.4887 |
| GBM | 0.7157 | 0.6820 | 0.6818 | 0.6709 |
| Ada | 0.6863 | 0.6464 | 0.6402 | 0.6366 |

#### Features combinées (12h + 3d)
**Nombre total de features :** 70

| Modèle | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| MLP | 0.4592 | 0.4325 | 0.4280 | 0.4260 |
| DT | 0.5964 | 0.5870 | 0.5928 | 0.5811 |
| RF | 0.7467 | 0.7346 | 0.6875 | 0.7001 |
| GBM | 0.5980 | 0.5573 | 0.5322 | 0.5161 |
| Ada | 0.5131 | 0.5083 | 0.5095 | 0.4978 |

### Meilleurs résultats globaux

**Meilleur F1-Score :** 0.7001 (RF)
**Meilleure Accuracy :** 0.7467 (RF)
