# Rapport d'analyse de régression - AES

Date de génération : 2025-10-01 10:25:35

## Résumé de l'analyse

**Target :** aes
**Nombre total de features analysées :** 181

### Corrélations significatives (p < 0.05)

**Nombre de corrélations significatives :** 15/181

| Rang | Feature | Corrélation (Pearson) | p-value | Corrélation (Spearman) | p-value Spearman |
|------|---------|----------------------|---------|------------------------|------------------|
| 1 | acti_activity_rate_3d | -0.6264 | 0.000057 | -0.6407 | 0.000034 |
| 2 | acti_walk_min_12h_6 | -0.4961 | 0.002436 | -0.6164 | 0.000081 |
| 3 | acti_inactivity_min_12h_1 | 0.4692 | 0.004462 | 0.4814 | 0.003415 |
| 4 | acti_walk_max_12h_6 | -0.4662 | 0.004761 | -0.6069 | 0.000111 |
| 5 | acti_inactivity_max_12h_1 | 0.4598 | 0.005453 | 0.5006 | 0.002194 |
| 6 | acti_inactivity_mean_12h_1 | 0.4535 | 0.006218 | 0.5219 | 0.001302 |
| 7 | acti_activity_max_12h_6 | -0.4410 | 0.008009 | -0.4480 | 0.006956 |
| 8 | acti_walk_mean_12h_6 | -0.4171 | 0.012670 | -0.4569 | 0.005797 |
| 9 | acti_freq_max_12h_3 | -0.3946 | 0.019003 | -0.2995 | 0.080465 |
| 10 | acti_activity_mean_12h_6 | -0.3894 | 0.020779 | -0.4304 | 0.009861 |
| 11 | acti_freq_std_12h_3 | 0.3889 | 0.020938 | 0.4266 | 0.010597 |
| 12 | acti_inactivity_mean_12h_2 | -0.3853 | 0.022277 | -0.3456 | 0.041972 |
| 13 | acti_inactivity_min_3d | 0.3651 | 0.031061 | 0.4773 | 0.003738 |
| 14 | acti_activity_min_12h_3 | -0.3602 | 0.033562 | -0.3493 | 0.039715 |
| 15 | acti_activity_mean_12h_1 | -0.3352 | 0.048994 | -0.3496 | 0.039546 |

### Top 20 corrélations (toutes)

| Rang | Feature | Corrélation (Pearson) | p-value | Significative |
|------|---------|----------------------|---------|---------------|
| 1 | acti_activity_rate_3d | -0.6264 | 0.000057 | ✓ |
| 2 | acti_walk_min_12h_6 | -0.4961 | 0.002436 | ✓ |
| 3 | acti_inactivity_min_12h_1 | 0.4692 | 0.004462 | ✓ |
| 4 | acti_walk_max_12h_6 | -0.4662 | 0.004761 | ✓ |
| 5 | acti_inactivity_max_12h_1 | 0.4598 | 0.005453 | ✓ |
| 6 | acti_inactivity_mean_12h_1 | 0.4535 | 0.006218 | ✓ |
| 7 | acti_activity_max_12h_6 | -0.4410 | 0.008009 | ✓ |
| 8 | acti_walk_mean_12h_6 | -0.4171 | 0.012670 | ✓ |
| 9 | acti_freq_max_12h_3 | -0.3946 | 0.019003 | ✓ |
| 10 | acti_activity_mean_12h_6 | -0.3894 | 0.020779 | ✓ |
| 11 | acti_freq_std_12h_3 | 0.3889 | 0.020938 | ✓ |
| 12 | acti_inactivity_mean_12h_2 | -0.3853 | 0.022277 | ✓ |
| 13 | acti_inactivity_min_3d | 0.3651 | 0.031061 | ✓ |
| 14 | acti_activity_min_12h_3 | -0.3602 | 0.033562 | ✓ |
| 15 | acti_activity_mean_12h_1 | -0.3352 | 0.048994 | ✓ |
| 16 | acti_activity_min_12h_6 | -0.3323 | 0.051165 | ✗ |
| 17 | acti_activity_min_3d | -0.3225 | 0.058834 | ✗ |
| 18 | acti_inactivity_max_12h_2 | -0.3182 | 0.062441 | ✗ |
| 19 | acti_walk_min_12h_3 | -0.3162 | 0.064277 | ✗ |
| 20 | acti_inactivity_min_12h_6 | 0.3154 | 0.064973 | ✗ |

### Features sélectionnées pour la modélisation

**Features 12h sélectionnées :** 13
**Features 3d sélectionnées :** 3

#### Features 12h sélectionnées

1. **acti_walk_min_12h_6** - r=-0.4961, p=0.002436
2. **acti_inactivity_min_12h_1** - r=0.4692, p=0.004462
3. **acti_walk_max_12h_6** - r=-0.4662, p=0.004761
4. **acti_inactivity_max_12h_1** - r=0.4598, p=0.005453
5. **acti_inactivity_mean_12h_1** - r=0.4535, p=0.006218
6. **acti_activity_max_12h_6** - r=-0.4410, p=0.008009
7. **acti_walk_mean_12h_6** - r=-0.4171, p=0.012670
8. **acti_freq_max_12h_3** - r=-0.3946, p=0.019003
9. **acti_activity_mean_12h_6** - r=-0.3894, p=0.020779
10. **acti_freq_std_12h_3** - r=0.3889, p=0.020938
11. **acti_inactivity_mean_12h_2** - r=-0.3853, p=0.022277
12. **acti_activity_min_12h_3** - r=-0.3602, p=0.033562
13. **acti_activity_mean_12h_1** - r=-0.3352, p=0.048994

#### Features 3d sélectionnées

1. **acti_activity_rate_3d** - r=-0.6264, p=0.000057
2. **acti_inactivity_min_3d** - r=0.3651, p=0.031061
3. **acti_activity_min_3d** - r=-0.3225, p=0.058834

### Performance des modèles de régression

| Type Features | Modèle | R² | RMSE | MAE |
|---------------|--------|----|----- |----|
| 12h | RandomForest | 0.2802 | 7.9424 | 6.5683 |
| 12h | Ada | 0.2662 | 8.0453 | 6.6736 |
| 12h | GradientBoosting | 0.1221 | 8.7184 | 6.9309 |
| 12h | DT | -0.2515 | 10.0247 | 8.2857 |
| 12h | MLP | -4.2838 | 21.7136 | 18.7055 |
| 3d | RandomForest | -0.0371 | 9.6077 | 7.6911 |
| 3d | GradientBoosting | -0.3429 | 10.9660 | 8.8980 |
| 3d | Ada | -0.3474 | 10.9145 | 8.8900 |
| 3d | DT | -0.6492 | 12.0867 | 10.0857 |
| 3d | MLP | -9.8914 | 31.3076 | 29.4813 |
| combined | RandomForest | 0.3259 | 7.6964 | 6.2917 |
| combined | Ada | 0.1943 | 8.4106 | 7.1933 |
| combined | GradientBoosting | 0.1886 | 8.4314 | 6.5761 |
| combined | DT | 0.0658 | 8.8222 | 7.1143 |
| combined | MLP | -4.1545 | 21.4287 | 18.3260 |

### Meilleur modèle global

**Modèle :** RandomForest
**Type de features :** combined
**R² :** 0.3259
**RMSE :** 7.6964
**MAE :** 6.2917

### Statistiques de sélection des features

**Critère de sélection :** Corrélations significatives (p < 0.05) ou top 3 par type
**Features 12h significatives :** 13
**Features 3d significatives :** 2

### Distribution des p-values

| Plage p-value | Nombre de features |
|---------------|-------------------|
| < 0.001 | 1 |
| 0.001-0.01 | 6 |
| 0.01-0.05 | 8 |
| 0.05-0.1 | 12 |
| >= 0.1 | 154 |
