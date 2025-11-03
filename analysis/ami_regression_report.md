# Rapport d'analyse de régression - AMI

Date de génération : 2025-10-01 10:25:33

## Résumé de l'analyse

**Target :** ami
**Nombre total de features analysées :** 181

### Corrélations significatives (p < 0.05)

**Nombre de corrélations significatives :** 6/181

| Rang | Feature | Corrélation (Pearson) | p-value | Corrélation (Spearman) | p-value Spearman |
|------|---------|----------------------|---------|------------------------|------------------|
| 1 | acti_inactivity_min_12h_6 | 0.3919 | 0.026536 | 0.4070 | 0.020795 |
| 2 | acti_walk_fft_mean_12h_1 | -0.3798 | 0.032015 | -0.3164 | 0.077646 |
| 3 | acti_inactivity_max_12h_1 | 0.3699 | 0.037205 | 0.4393 | 0.011888 |
| 4 | acti_walk_max_12h_6 | -0.3666 | 0.039031 | -0.4013 | 0.022827 |
| 5 | acti_freq_mean_12h_6 | -0.3661 | 0.039303 | -0.4472 | 0.010290 |
| 6 | acti_walk_fft_min_12h_3 | 0.3527 | 0.047705 | 0.3089 | 0.085337 |

### Top 20 corrélations (toutes)

| Rang | Feature | Corrélation (Pearson) | p-value | Significative |
|------|---------|----------------------|---------|---------------|
| 1 | acti_inactivity_min_12h_6 | 0.3919 | 0.026536 | ✓ |
| 2 | acti_walk_fft_mean_12h_1 | -0.3798 | 0.032015 | ✓ |
| 3 | acti_inactivity_max_12h_1 | 0.3699 | 0.037205 | ✓ |
| 4 | acti_walk_max_12h_6 | -0.3666 | 0.039031 | ✓ |
| 5 | acti_freq_mean_12h_6 | -0.3661 | 0.039303 | ✓ |
| 6 | acti_walk_fft_min_12h_3 | 0.3527 | 0.047705 | ✓ |
| 7 | acti_inactivity_min_12h_1 | 0.3384 | 0.058196 | ✗ |
| 8 | acti_inactivity_mean_12h_1 | 0.3315 | 0.063821 | ✗ |
| 9 | acti_walk_min_12h_6 | -0.3304 | 0.064779 | ✗ |
| 10 | acti_activity_rate_3d | -0.3292 | 0.065788 | ✗ |
| 11 | acti_walk_mean_12h_6 | -0.3122 | 0.081912 | ✗ |
| 12 | acti_oadl_fft_min_12h_4 | 0.3060 | 0.088468 | ✗ |
| 13 | acti_activity_max_12h_6 | -0.2985 | 0.097029 | ✗ |
| 14 | acti_oadl_fft_std_12h_3 | 0.2794 | 0.121520 | ✗ |
| 15 | acti_walk_fft_max_12h_5 | -0.2788 | 0.122287 | ✗ |
| 16 | acti_freq_max_12h_3 | -0.2783 | 0.123034 | ✗ |
| 17 | acti_inactivity_max_12h_2 | -0.2746 | 0.128312 | ✗ |
| 18 | acti_oadl_fft_mean_12h_5 | 0.2682 | 0.137737 | ✗ |
| 19 | acti_walk_fft_std_12h_5 | -0.2498 | 0.167875 | ✗ |
| 20 | acti_activity_mean_12h_5 | 0.2440 | 0.178359 | ✗ |

### Features sélectionnées pour la modélisation

**Features 12h sélectionnées :** 6
**Features 3d sélectionnées :** 3

#### Features 12h sélectionnées

1. **acti_inactivity_min_12h_6** - r=0.3919, p=0.026536
2. **acti_walk_fft_mean_12h_1** - r=-0.3798, p=0.032015
3. **acti_inactivity_max_12h_1** - r=0.3699, p=0.037205
4. **acti_walk_max_12h_6** - r=-0.3666, p=0.039031
5. **acti_freq_mean_12h_6** - r=-0.3661, p=0.039303
6. **acti_walk_fft_min_12h_3** - r=0.3527, p=0.047705

#### Features 3d sélectionnées

1. **acti_activity_rate_3d** - r=-0.3292, p=0.065788
2. **acti_freq_max_3d** - r=-0.1686, p=0.356403
3. **acti_activity_min_3d** - r=-0.1661, p=0.363723

### Performance des modèles de régression

| Type Features | Modèle | R² | RMSE | MAE |
|---------------|--------|----|----- |----|
| 12h | RandomForest | -0.7425 | 7.3411 | 6.3817 |
| 12h | Ada | -1.1663 | 8.1913 | 6.9241 |
| 12h | GradientBoosting | -1.6820 | 8.5552 | 7.2407 |
| 12h | DT | -3.5601 | 11.1635 | 9.6381 |
| 12h | MLP | -3.9545 | 12.8606 | 11.5365 |
| 3d | RandomForest | -0.9380 | 8.4277 | 7.0747 |
| 3d | Ada | -1.1749 | 8.9818 | 7.4782 |
| 3d | GradientBoosting | -1.6632 | 9.6852 | 7.8229 |
| 3d | DT | -1.9960 | 9.8206 | 8.2524 |
| 3d | MLP | -7.1780 | 16.3785 | 14.7594 |
| combined | RandomForest | -0.4973 | 7.1525 | 6.1739 |
| combined | Ada | -0.7160 | 7.5682 | 6.3367 |
| combined | GradientBoosting | -0.9209 | 7.7691 | 6.7034 |
| combined | DT | -1.7199 | 9.5135 | 7.3667 |
| combined | MLP | -2.2070 | 10.6235 | 9.0808 |

### Meilleur modèle global

**Modèle :** RandomForest
**Type de features :** combined
**R² :** -0.4973
**RMSE :** 7.1525
**MAE :** 6.1739

### Statistiques de sélection des features

**Critère de sélection :** Corrélations significatives (p < 0.05) ou top 3 par type
**Features 12h significatives :** 6
**Features 3d significatives :** 0

### Distribution des p-values

| Plage p-value | Nombre de features |
|---------------|-------------------|
| < 0.001 | 0 |
| 0.001-0.01 | 0 |
| 0.01-0.05 | 6 |
| 0.05-0.1 | 7 |
| >= 0.1 | 168 |
