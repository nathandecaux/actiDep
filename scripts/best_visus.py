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


model = "/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids/derivatives/hcp_association_50pts/sub-01002/tracto/sub-01002_bundle-<BUNDLE>_desc-associations_model-MCM_space-HCP_tracto.vtk"

import pyvista
import vtk
import numpy as np

# chosen_csv = "/home/ndecaux/NAS_EMPENN/share/projects/actidep/prez_seminaire/STPREMleft_FA_ami_corr_partial.csv"
# profile = pd.read_csv(chosen_csv)
# profile=profile[profile['sig_afq']]
# #Load the tract model
# mesh = pyvista.read(model)
# # Build mapping from AFQ point index to r value
# p2r = dict(zip(profile['point'].astype(int), profile['r'].astype(float)))

# # Fetch mesh point indices and convert to int
# if 'point_index' not in mesh.point_data:
#     raise KeyError("Mesh is missing 'point_index' point-data array.")
# pi = np.asarray(mesh.point_data['point_index']).astype(int)
# print(p2r)
# # Create r array for all mesh points (default 0 when missing)
# r_values = np.array([p2r.get(i, 0.0) for i in pi], dtype=float)
# # Attach as point data
# mesh.point_data['r'] = r_values

# #Save as vtk
# output_vtk = chosen_csv.replace('.csv', '.vtk')
# mesh.save(output_vtk)
import json 

with open('/home/ndecaux/Code/actiDep/bundle_desc_fr.json', encoding='utf-8') as f:
    bundle_desc = json.load(f)

#Remove _ from bundle_desc keys
bundle_desc = {k.replace('_',''): v for k, v in bundle_desc.items()}


for type,df_cur in [('Apathy', df_apathy), ('Psychometry', df_psycho)]:

    metric_res={}

    for metric in ['FA','IFW','IRF']:
        metric_res[metric]={}
        df_met=df_cur[df_cur['metric']==metric]
        cnt=0
        max_met=-1000
        min_met=1000
        for row in df_met.iterrows():
            cnt+=1
            # if cnt>1:
            #     break
            
            bundle = row[1]['bundle']
            print(f'Processing {metric} {bundle} ({cnt}/{len(df_met)})')
            metric_res[metric][bundle]={}
            model_b = model.replace('<BUNDLE>', bundle)
            mesh = pyvista.read(model_b)
            csv=pd.read_csv(row[1]['csv'])

            #Get clusters as a list of list of continuous significant points
            clusters = []
            current_cluster = []
            for i, val in enumerate(csv['sig_afq']):
                if val:
                    current_cluster.append(i)
                else:
                    if current_cluster:
                        clusters.append(current_cluster)
                        current_cluster = []
            if current_cluster:
                clusters.append(current_cluster)
            cluster_info = []
            for c_idx, c in enumerate(clusters):
                mean_p_raw=csv.loc[c, 'p_raw'].mean()
                n_points=len(c)
                if 'mean_dep' in csv.columns:
                    mean_dep=csv.loc[c, 'mean_dep'].mean()
                    mean_hc=csv.loc[c, 'mean_hc'].mean()
                    std_dep=csv.loc[c, 'std_dep'].mean()
                    std_hc=csv.loc[c, 'std_hc'].mean()
                    diff_mean=mean_dep-mean_hc
                    diff_std=std_dep-std_hc

                    text=f'Cluster {c_idx+1} : '
                    if diff_mean>0:
                        text+=f'HC > DEP ; '
                    else:
                        text+=f'DEP > HC ; '
                    
                    if diff_std>0:
                        text+=f'STD HC > STD DEP ; '
                    else:
                        text+=f'STD DEP > STD HC ; '

                    text+=f'n={n_points} pts ; p={mean_p_raw:.3f}'
                    # cluster_info.append((c[0], c[-1], n_points, mean_p_raw, mean_dep, mean_hc, std_dep, std_hc, diff_mean, diff_std,text))
                    cluster_info.append({'start': c[0], 'end': c[-1], 'n_points': n_points, 'mean_p_raw': mean_p_raw, 'mean_dep': mean_dep, 'mean_hc': mean_hc, 'std_dep': std_dep, 'std_hc': std_hc, 'diff_mean': diff_mean, 'diff_std': diff_std, 'text': text})
                else:
                    r=csv.loc[c, 'r'].mean()
                    text =f'Cluster {c_idx+1} : n={n_points} pts ; p={mean_p_raw:.3f} ; r={r:.3f}'
                    # cluster_info.append((c[0], c[-1], n_points, mean_p_raw, r, text))
                    cluster_info.append({'start': c[0], 'end': c[-1], 'n_points': n_points, 'mean_p_raw': mean_p_raw, 'r': r, 'text': text})        
            
            csv=csv[csv['sig_afq']]
            if 'r' in csv.columns:
                p2r = dict(zip(csv['point'].astype(int), csv['r'].astype(float)))
                if csv['r'].max()>max_met:
                    max_met=csv['r'].max()
                if csv['r'].min()<min_met:
                    min_met=csv['r'].min()
            else:
                p2r = dict(zip(csv['point'].astype(int), csv['diff'].astype(float)))
                if csv['diff'].max()>max_met:
                    max_met=csv['diff'].max()
                if csv['diff'].min()<min_met:
                    min_met=csv['diff'].min()
            pi = np.asarray(mesh.point_data['point_index']).astype(int)
            values=np.array([p2r.get(i, 0.0) for i in pi], dtype=float)
            mesh.point_data['stats'] = values

            output_vtk = row[1]['csv'].replace('.csv', '.vtk')
            mesh.save(output_vtk)
            metric_res[metric][bundle]['vtk']=output_vtk
            metric_res[metric][bundle]['png']=row[1]['png']
            metric_res[metric][bundle]['clusters']=cluster_info
            #Sort cluster_info by number of points
        
        metric_res[metric]['max']=max_met
        metric_res[metric]['min']=min_met
        print(f'Metric {metric} : min={min_met} ; max={max_met}')
    metric_res

    from tractviewer import TractViewer
    out_dir=f'/home/ndecaux/res_actidep/{type}/'

    for metric in metric_res.keys():
        out_met=os.path.join(out_dir, metric)
        print(f'Metric {metric}: min={metric_res[metric]["min"]} ; max={metric_res[metric]["max"]}')
        min=metric_res[metric]["min"]
        max=metric_res[metric]["max"]
        final_text='Métrique '+metric+f' : min={min:.3f} ; max={max:.3f}. Nombre de bundles : {len(metric_res[metric])-2}\n\n'
        viewer = TractViewer()
        
        viewer.add_dataset(
        "/home/ndecaux/Code/Data/Atlas/atlas_anat.nii.gz",
            {
            "display_array": "intensity",
            "cmap": "gray",
            "clim": (200, 800),
            "opacity": 0.1,
            "ambient": 0.6,
            "diffuse": 0.8,
            "specular": 0.1,
            "scalar_bar": False,
            "name": "anatomy",
            "style": "surface",
        })

        for b in [x for x in metric_res[metric].keys() if x not in ['min','max']]:
            
            final_text+=f'  Bundle {bundle_desc[b]}: clusters:\n'
            for c in metric_res[metric][b]['clusters']:
                # print(f"    - from {c['start']} to {c['end']}: {c['n_points']} pts, p={c['mean_p_raw']:.3f} ; {c['text']}")
                final_text+=f"    - from {c['start']} to {c['end']}: {c['n_points']} pts, p={c['mean_p_raw']:.3f} ; {c['text']}\n"
            
            #Copy png and vtk to out_met
            os.makedirs(out_met, exist_ok=True)
            out_png=os.path.join(out_met, os.path.basename(metric_res[metric][b]['png']))
            out_vtk=os.path.join(out_met, os.path.basename(metric_res[metric][b]['vtk']))
            os.system(f'cp {metric_res[metric][b]["png"]} {out_png}')
            os.system(f'cp {metric_res[metric][b]["vtk"]} {out_vtk}')
            metric_res[metric][b]['png']=out_png
            metric_res[metric][b]['vtk']=out_vtk

            viewer.add_dataset(
                metric_res[metric][b]['vtk'],
                {
                    "display_array": "stats",
                    "cmap": "coolwarm",
                    "opacity": 0.8,
                    "ambient": 0.3,
                    "diffuse": 0.6,
                    "scalar_bar": True,
                    "name": f"{b}_{metric}"
                }
            )
        out_gif=os.path.join(out_met, f'{metric}_all_bundles.gif')
        viewer.record_rotation(
            out_gif,
            n_frames=360,      # nombre d'images
            step=1.5,          # incrément azimut
            elevation=0.0,     # mettre p.ex 10 pour basculer en 1ère moitié puis -10
            fps=10,
            quality=9,         # (imageio) 0-10
            crf=18,            # (si pas de bitrate)
            supersample=2,     # rendu interne 2x puis compressé (plus net)
            window_size=(600, 400),
        )
        final_text+=f'\nAnimation de tous les bundles : {out_gif}\n'

        with open(os.path.join(out_met, 'summary.txt'), 'w', encoding='utf-8') as f:
            f.write(final_text)

