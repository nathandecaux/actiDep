import os
from tractviewer import TractViewer
from actiDep.data.loader import Actidep
from actiDep.set_config import get_HCP_bundle_names
NO_ACTI_CSV = "/home/ndecaux/report_optimized_no_actimetry_clusterFWE/summary_results.csv"
import pandas as pd

# Tracts fronto-limbiques MANQUE SLFII
fronto_limbiques = [
    "CG_left", "CG_right",
    "UF_left", "UF_right",
    "FX_left", "FX_right",
    "FPT_left", "FPT_right",
    "T_PREF_left", "T_PREF_right",
    "ST_FO_left", "ST_FO_right",
    "ST_PREF_left", "ST_PREF_right"
]

renaud_dep_controls = ['ATR', 'CC1','SLFI','SLFIII','STPREM', 'UF']
renaud_apathy = ['CC1','CC2','CST','SLFIII','STPREM']

to_remove = []
for b in renaud_dep_controls:
    if b+'right' in list(get_HCP_bundle_names().keys()):
        renaud_dep_controls.append(b+'left')
        renaud_dep_controls.append(b+'right')
        to_remove.append(b)

renaud_dep_controls= [b for b in renaud_dep_controls if b not in to_remove]

for b in renaud_apathy:
    if b+'right' in list(get_HCP_bundle_names().keys()):
        renaud_apathy.append(b+'left')
        renaud_apathy.append(b+'right')
        to_remove.append(b)
renaud_apathy= [b for b in renaud_apathy if b not in to_remove]

# ST_PREM CC2 CC_3
# Tracts associés aux émotions
emotions = [
    "CG_left", "CG_right",
    "UF_left", "UF_right",
    "FX_left", "FX_right",
    "ATR_left", "ATR_right",
    "CA",
    "T_PREF_left", "T_PREF_right",
    "ST_FO_left", "ST_FO_right",
    "ST_PREF_left", "ST_PREF_right"
]

# Tracts moteurs
moteurs = [
    "CST_left", "CST_right",
    "CC_3", "CC_4",
    "T_PREM_left", "T_PREM_right",
    "T_PREC_left", "T_PREC_right",
    "T_POSTC_left", "T_POSTC_right",
    "ST_PREM_left", "ST_PREM_right",
    "ST_PREC_left", "ST_PREC_right",
    "ST_POSTC_left", "ST_POSTC_right",
    "ICP_left", "ICP_right",
    "MCP",
    "SCP_left", "SCP_right"
]

MIN_SIG = 1
MAX_REMOVED_SUBJ = 1
MAX_REMOVED_POINTS = 3
MIN_CORR = 0.5

fronto_limbiques=[b.replace('_','') for b in fronto_limbiques]
emotions=[b.replace('_','') for b in emotions]
moteurs=[b.replace('_','') for b in moteurs]

df=pd.read_csv(NO_ACTI_CSV)
filters={
    "fronto_limbiques": df['bundle'].isin(fronto_limbiques),
    "emotions": df['bundle'].isin(emotions),
    "moteurs": df['bundle'].isin(moteurs),
    "significant": df['n_sig_corrected'] > MIN_SIG-1,
    "all_subs": df['removed_subjects']<MAX_REMOVED_SUBJ+1,
    "all_points": df['removed_points']<MAX_REMOVED_POINTS+1,
    "metric": df['metric'].isin(['FA','IFW','IRF'])
}

#Application de tous les filtres
final_filter = filters["significant"] & filters["all_subs"] & filters["all_points"]# & (filters["emotions"] | filters["moteurs"] | filters["fronto_limbiques"]) & filters["metric"]
df_filtered = df[final_filter]
df_filtered['val']=df_filtered['type'].apply(lambda x: x.split('_')[1])
df_filtered['type']=df_filtered['type'].apply(lambda x: x.split('_')[0])
df_filtered['csv']=df_filtered.apply(lambda row: os.path.join(os.path.dirname(NO_ACTI_CSV), 'figures', f"{row['bundle']}_{row['metric']}_{row['val']}_{row['type']}_{'corrected' if 'group' in row['type'] else "partial"}.csv"), axis=1)
df_filtered['png']=df_filtered.apply(lambda row: os.path.join(os.path.dirname(NO_ACTI_CSV), 'figures', f"{row['bundle']}_{row['metric']}_{row['val']}_{row['type']}_{'corrected' if 'group' in row['type'] else "partial"}.png"), axis=1)

ACTI_CSV = "/data/ndecaux/report_actimetry_calcarine/summary_results.csv"
df_acti=pd.read_csv(ACTI_CSV)
filters_acti={
    "fronto_limbiques": df_acti['bundle'].isin(fronto_limbiques),
    "emotions": df_acti['bundle'].isin(emotions),
    "moteurs": df_acti['bundle'].isin(moteurs),
    "renaud_dep_controls": df_acti['bundle'].isin(renaud_dep_controls),
    "renaud_apathy": df_acti['bundle'].isin(renaud_apathy),
    "significant": df_acti['n_sig_corrected'] > MIN_SIG-1,
    "all_subs": df_acti['removed_subjects']<MAX_REMOVED_SUBJ+1,
    "all_points": df_acti['removed_points']<MAX_REMOVED_POINTS+1,
    "metric": df_acti['metric'].isin(['FA','IFW','IRF']),
    "min_r": df_acti['max_abs_r_partial'] > MIN_CORR
}
final_filter_acti = filters_acti["significant"] & filters_acti["all_subs"] & filters_acti["all_points"]# & filters_acti["metric"] & (filters_acti["emotions"] | filters_acti["moteurs"] | filters_acti["fronto_limbiques"])
df_acti_filtered = df_acti[final_filter_acti]
df_acti_filtered['val']=df_acti_filtered['type'].apply(lambda x: '_'.join(x.split('_')[1:]))
df_acti_filtered['type']=df_acti_filtered['type'].apply(lambda x: x.split('_')[0])
df_acti_filtered['csv']=df_acti_filtered.apply(lambda row: os.path.join(os.path.dirname(ACTI_CSV), 'figures', f"{row['bundle']}_{row['metric']}_{row['val']}_{row['type']}_{'corrected' if 'group' in row['type'] else "partial"}.csv"), axis=1)
df_acti_filtered['png']=df_acti_filtered.apply(lambda row: os.path.join(os.path.dirname(ACTI_CSV), 'figures', f"{row['bundle']}_{row['metric']}_{row['val']}_{row['type']}_{'corrected' if 'group' in row['type'] else "partial"}.png"), axis=1)

df_group = df_filtered[df_filtered['val'] == 'group']
df_psycho = df_filtered[df_filtered['type'].str.contains('corr', na=False)]
df_apathy = df_filtered[df_filtered['val'] == 'apathy']

#Get the bundles that exists in df_group, df_psycho, df_apathy and df_acti_filtered
common_bundles = set(df_acti_filtered['bundle']).intersection(set(df_psycho['bundle'])).intersection(set(df_apathy['bundle']))
print('set of group bundles:', set(df_group['bundle']))
print('set of psycho bundles:', set(df_psycho['bundle']))
print('set of apathy bundles:', set(df_apathy['bundle']))
print('set of acti bundles:', set(df_acti_filtered['bundle']))
print('common bundles:', common_bundles)

print(df_group)

# chosen_one = list(common_bundles)
# #All bundles 
# chosen_one = moteurs + emotions + fronto_limbiques
# df_group = df_group[df_group['bundle'].isin(chosen_one)]
# df_psycho = df_psycho[df_psycho['bundle'].isin(chosen_one)]
# df_apathy = df_apathy[df_apathy['bundle'].isin(chosen_one)]
# df_acti_filtered = df_acti_filtered[df_acti_filtered['bundle'].isin(chosen_one)]
# #Concatenate all dataframes
# df_final = pd.concat([df_group, df_psycho, df_apathy, df_acti_filtered], ignore_index=True)
# #Copy all csv and png to a new folder
# output_folder = '/home/ndecaux/NAS_EMPENN/share/projects/actidep/prez_seminaire'
# os.makedirs(output_folder, exist_ok=True)
# for _, row in df_final.iterrows():
#     os.system(f'cp {row["csv"]} {output_folder}/')
#     os.system(f'cp {row["png"]} {output_folder}/')

#     import numpy as np
# unique_group = df_group[['bundle','metric']].drop_duplicates()
# unique_psycho = df_psycho[['bundle','metric']].drop_duplicates()
# unique_apathy = df_apathy[['bundle','metric']].drop_duplicates()
# unique_acti = df_acti_filtered[['bundle','metric']].drop_duplicates()
# common_bundles_metrics = pd.merge(pd.merge(pd.merge(unique_group, unique_psycho, on=['bundle','metric']), unique_apathy, on=['bundle','metric']), unique_acti, on=['bundle','metric'])