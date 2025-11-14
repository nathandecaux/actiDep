import pandas as pd
import panel as pn
import hvplot.pandas  # active l'accessor hvplot
import os, time
import sys

pn.extension('tabulator')

# CSV_PATH = '/home/ndecaux/NAS_EMPENN/share/projects/actidep/report_no_actimetry_tractseg/summary_results.csv'
# CSV_PATH = '/home/ndecaux/report_no_actimetry_tractseg/summary_results.csv'
# CSV_PATH = "/home/ndecaux/report_optimized_no_actimetry_clusterFWE/summary_results.csv"
# CSV_PATH = "/home/ndecaux/report_optimized_test_multi_no_actimetry_clusterFWE/summary_results.csv"
# CSV_PATH = "/data/ndecaux/report_hcp_association_multiclusters_frechetlong_50pts_no_actimetry_calcarine/summary_results.csv"
# CSV_PATH = "/home/ndecaux/report_amynet_hcp_association_multiclusters_umapendpoints_50pts_no_actimetry_clusterFWE/summary_results.csv"
CSV_PATH = '/data/ndecaux/report_hcp_association_50pts_actimetry_calcarine/summary_results.csv'
# CSV_PATH = '/data/ndecaux/report_hcp_association_multiclusters_umapendpoints_50pts_group_only_calcarine/summary_results.csv'

def load_data(path=CSV_PATH):
    df = pd.read_csv(path)
    df['var']  = df['type'].apply(lambda x: '_'.join(x.split('_')[1:]))
    df['type'] = df['type'].apply(lambda x: x.split('_')[0])
    df['var']  = df['var'].astype('category')
    df['type'] = df['type'].astype('category')
    cols = ['type', 'var'] + [c for c in df.columns if c not in ['type', 'var']]
    df = df[cols]
    return df

df = load_data()
if 'open' not in df.columns:
    df.insert(0, 'open', "▶")
base_dir = os.path.dirname(CSV_PATH)

table = pn.widgets.Tabulator(
    df,
    pagination='local',
    page_size=30,
    sizing_mode='stretch_both',
    show_index=False,
    selectable='row',
    header_filters=True,  # garde la ligne de filtres
    layout='fit_data',
    buttons={'print': '<i class="fa fa-print"></i>'},
    configuration={
        # Ajout headerFilter pour toutes les colonnes (persist/recréation)
        "columnDefaults": {
            "resizable": True,
            "maxWidth": 260,
            "headerFilter": "input",
            "headerFilterLiveFilter": True,  # filtrage live
        },
        "columns": [
            {"title": "", "field": "open", "hozAlign": "center", "formatter": "html",
             "headerSort": False, "width": 55, "headerFilter": False}
        ]
    },
)

detail_panel = pn.Column(pn.pane.Markdown("Sélectionnez une ligne dans l'onglet Table."))

tabs = pn.Tabs(
    ("Table", table),
    ("Détail", detail_panel),
    dynamic=True  # on conserve dynamic=True mais on gère la restauration des filtres
)

# --- Gestion persistance des filtres Tabulator quand le widget est reconstruit ---
_saved_filters = {}

def _snapshot_filters():
    global _saved_filters
    _saved_filters = {f["field"]: f for f in table.filters if f.get("value") not in (None, "", [])}

def _capture_filters(event):
    # event.new: liste de dicts [{'field':..., 'type':..., 'value':...}, ...]
    _snapshot_filters()

table.param.watch(_capture_filters, 'filters')

def _restore_filters(event):
    if event.new == 0 and _saved_filters:
        # Ne réapplique que si le widget a perdu ses filtres visibles (ex: recréation)
        if not table.filters:
            filters_list = list(_saved_filters.values())
            def _apply():
                if table.filters != filters_list:
                    table.filters = filters_list
            try:
                pn.state.curdoc.add_next_tick_callback(_apply)
            except Exception:
                _apply()

tabs.param.watch(_restore_filters, 'active')
# -------------------------------------------------------------------------------

def _build_resource(path, kind):
    if not os.path.exists(path):
        return pn.pane.Markdown(f"**Manquant:** {os.path.basename(path)}", sizing_mode='stretch_width')
    if kind == 'img':
        return pn.pane.image.Image(path, sizing_mode='stretch_width')
    if kind == 'csv':
        try:
            d = pd.read_csv(path)
            return pn.widgets.Tabulator(d, height=300, pagination='local', page_size=50,
                                        show_index=False, layout='fit_data')
        except Exception as e:
            return pn.pane.Markdown(f"Erreur lecture {os.path.basename(path)}: {e}")
    return pn.pane.Markdown("Type inconnu")

def _make_detail_panel(row):
    bundle = row.get('bundle')
    metric = row.get('metric')
    var    = row.get('var')
    type_  = row.get('type')
    centroid_id= row.get('centroid_id', -1)
    if not all([bundle, metric, var, type_]):
        return pn.pane.Markdown("Données insuffisantes pour afficher le détail.")
    def build_name(kind, ext): 
        if kind=='partial' and type_=='group':
            kind='corrected'
        if centroid_id!=-1:
            return f"{bundle}_cent{centroid_id}_{metric}_{var}_{type_}_{kind}.{ext}"
        else:
            return f"{bundle}_{metric}_{var}_{type_}_{kind}.{ext}"
    
    #If folder figures in base_dir exists, use it
    figures_dir = os.path.join(base_dir, 'figures')
    if os.path.exists(figures_dir) and os.path.isdir(figures_dir):
        base_dir_figures = figures_dir
    else:
        base_dir_figures = base_dir
   

    raw_img_path     = os.path.join(base_dir_figures, build_name('raw', 'png'))
    partial_img_path = os.path.join(base_dir_figures, build_name('partial', 'png'))
    raw_csv_path     = os.path.join(base_dir_figures, build_name('raw', 'csv'))
    partial_csv_path = os.path.join(base_dir_figures, build_name('partial', 'csv'))
    header = pn.pane.Markdown(f"### Détails: {bundle} / {metric} / {var} / {type_}", sizing_mode='stretch_width')
    raw_section = pn.Column(
        pn.pane.Markdown("#### Raw"),
        _build_resource(raw_img_path, 'img'),
        _build_resource(raw_csv_path, 'csv'),
    )
    partial_section = pn.Column(
        pn.pane.Markdown("#### Partial"),
        _build_resource(partial_img_path, 'img'),
        _build_resource(partial_csv_path, 'csv'),
    )
    back_btn = pn.widgets.Button(name="Retour au tableau", button_type='primary')
    def _back(e):
        tabs.active = 0
        # Efface sélection (évite réouverture)
        try:
            table.selection = []
        except Exception:
            pass
    back_btn.on_click(_back)
    return pn.Column(
        back_btn,
        header,
        pn.Row(raw_section, partial_section, sizing_mode='stretch_width'),
        sizing_mode='stretch_width'
    )

def _get_row_dict(idx):
    try:
        if idx in table.value.index:
            return table.value.loc[idx].to_dict()
    except Exception:
        pass
    try:
        return table.value.iloc[int(idx)].to_dict()
    except Exception:
        return None

def _cell_click(event):
    # Capture filtres avant de quitter l'onglet
    _snapshot_filters()
    # Ouvre le détail
    row_dict = _get_row_dict(event.row)
    if not row_dict:
        return
    if not all(k in row_dict for k in ('bundle','metric','var','type')):
        return
    panel = _make_detail_panel(row_dict)
    detail_panel.objects = panel.objects
    tabs.active = 1

# Reset anciens callbacks et attache
try:
    table._click_callbacks = []
except Exception:
    pass
table.on_click(_cell_click)

# Optionnel: désactiver surbrillance de sélection
table.selectable = False

# --- Filtres sidebar ---
metric_filter = pn.widgets.MultiChoice(name="Metric contient", placeholder="ex: FA", options=sorted(df['metric'].unique()), value=[])
bundle_filter = pn.widgets.MultiChoice(name="Bundle contient", options=sorted(df['bundle'].unique()), value=[])
type_filter   = pn.widgets.MultiChoice(name="Type", options=sorted(df['type'].unique()), value=[])
var_filter    = pn.widgets.MultiChoice(name="Var",  options=sorted(df['var'].unique()), value=[])
# Listes prédéfinies de bundles (identiques à report_analysis_actimetry.py)
fronto_limbiques = [
    "CG_left", "CG_right",
    "UF_left", "UF_right",
    "FX_left", "FX_right",
    "FPT_left", "FPT_right",
    "T_PREF_left", "T_PREF_right",
    "ST_FO_left", "ST_FO_right",
    "ST_PREF_left", "ST_PREF_right"
]
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

fronto_limbiques_btn = pn.widgets.Button(name="Fronto-Limbiques", button_type="primary")
emotions_btn = pn.widgets.Button(name="Émotions", button_type="success")
moteurs_btn = pn.widgets.Button(name="Moteurs", button_type="warning")

def _apply_list_to_bundle(event, bundle_list):
    # Même logique que dans report_analysis_actimetry.py
    bundle_filter.value = bundle_filter.value + [x.replace('_','') for x in bundle_list]

fronto_limbiques_btn.on_click(lambda event: _apply_list_to_bundle(event, fronto_limbiques))
emotions_btn.on_click(lambda event: _apply_list_to_bundle(event, emotions))
moteurs_btn.on_click(lambda event: _apply_list_to_bundle(event, moteurs))

# Filtres numériques (mêmes colonnes et parsing d'opérateurs)
n_sig_corrected_filter = pn.widgets.TextInput(name="n_sig_corrected (>, <)", placeholder="ex: >5")
min_p_corrected_filter = pn.widgets.TextInput(name="min_p_corrected (>, <)", placeholder="ex: <0.05")
max_abs_r_partial_filter = pn.widgets.TextInput(name="max_abs_r_partial (>, <)", placeholder="ex: >0.6")
removed_subjects_filter = pn.widgets.TextInput(name="removed_subjects (>, <)", placeholder="ex: >1")
removed_points_filter = pn.widgets.TextInput(name="removed_points (>, <)", placeholder="ex: <3")

reset_filters_btn = pn.widgets.Button(name="Réinitialiser filtres", button_type='warning')

def _apply_sidebar_filters(event=None):
    sub = df
    if metric_filter.value:
        sub = sub[sub['metric'].isin(metric_filter.value)]
    if bundle_filter.value:
        sub = sub[sub['bundle'].isin(bundle_filter.value)]
    if type_filter.value:
        sub = sub[sub['type'].isin(type_filter.value)]
    if var_filter.value:
        sub = sub[sub['var'].isin(var_filter.value)]
    # Application des filtres numériques avec seuils (comme dans report_analysis_actimetry.py)
    for col, widget in [
        ('n_sig_corrected', n_sig_corrected_filter),
        ('min_p_corrected', min_p_corrected_filter),
        ('max_abs_r_partial', max_abs_r_partial_filter),
        ('removed_subjects', removed_subjects_filter),
        ('removed_points', removed_points_filter),
    ]:
        if widget.value:
            try:
                if widget.value[0] in ('>', '<'):
                    op, val = widget.value[0], float(widget.value[1:])
                else:
                    op, val = '==', float(widget.value)
                if op == '>':
                    sub = sub[sub[col] > val]
                elif op == '<':
                    sub = sub[sub[col] < val]
                elif op == '==':
                    sub = sub[sub[col] == val]
            except Exception:
                pass
    # Met à jour le dataframe affiché
    table.value = sub

for w in (metric_filter, bundle_filter, type_filter, var_filter,
          n_sig_corrected_filter, min_p_corrected_filter, max_abs_r_partial_filter,
          removed_subjects_filter, removed_points_filter):
    w.param.watch(_apply_sidebar_filters, 'value')

def _reset_filters(event):
    metric_filter.value = ""
    bundle_filter.value = ""
    type_filter.value = []
    var_filter.value = []
    # Reset numériques
    n_sig_corrected_filter.value = ""
    min_p_corrected_filter.value = ""
    max_abs_r_partial_filter.value = ""
    removed_subjects_filter.value = ""
    removed_points_filter.value = ""
    _apply_sidebar_filters()

reset_filters_btn.on_click(_reset_filters)
# Initial
_apply_sidebar_filters()
# -----------------------------------------

template = pn.template.FastListTemplate(
    title="ActiDep - Résultats hors actimétrie",
    sidebar=[
        "Table interactive (filtrez dans les en-têtes).",
        "Cliquez une ligne pour voir les détails (onglet Détail).",
        "Dans l'onglet Détail, utilisez le bouton pour revenir.",
        ("Autoreload actif." if '--autoreload' in sys.argv else ""),
        "#### Filtres",
        metric_filter,
        bundle_filter,
        type_filter,
        var_filter,
        "#### Listes prédéfinies",
        fronto_limbiques_btn,
        emotions_btn,
        moteurs_btn,
        "#### Filtres numériques",
        n_sig_corrected_filter,
        min_p_corrected_filter,
        max_abs_r_partial_filter,
        removed_subjects_filter,
        removed_points_filter,
        reset_filters_btn,
    ],
    main=[tabs],
    accent_base_color="#1f77b4",
)

pn.config.sizing_mode = 'stretch_width'
template.servable()


if __name__ == '__main__':
    if '--autoreload' in sys.argv:
        template.show(autoreload=True)
    else:
        template.show()







