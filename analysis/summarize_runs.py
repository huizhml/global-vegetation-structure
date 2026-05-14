import wandb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
import pandas as pd
from pathlib import Path
import numpy as np
from const import BIOMES_BY_VALUE
# Choose a colormap (e.g., 'viridis')
api = wandb.Api()

def boxplot(run_ids, box_width = 0.5,box_gap = 0.1):
    run = wandb.init(project="global-vegetation-structure-v1", job_type="summary")
    run_names = {'allse0dy':'QR+LCM', 'ggi9w9b8': 'QR+LCM+zero-out', '05lp279m':'QR', '4fd8j93r':'QR+zero-out'}
    run_offset_a = box_width + box_gap
    group_offset_a = len(run_ids) * run_offset_a + 0.2
    colormap = cm.get_cmap('Pastel1')
    run_colors = {run_ids[i]: colormap(i) for i in range(len(run_ids))}
    paths = ['Table RH98_intervals', 'Table BIOME']
    metrics = [f'Residuals RH{i}' for i in [0,25,50, 75,98,100]] + ['Residuals RH_all', 'rh98']
    for path in paths:
        for name in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            run_offset = 0
            for i, run_id in enumerate(run_ids):
                group_offset = 0
                path_name = path.replace(" ", "")
                table = run.use_artifact(f'run-{run_id}-{path_name}{name.replace(" ", "")}:latest', type='run_table')
                df = table.get(f'{path}/{name}').get_dataframe()
                df = df.set_index('index')
                
                for col in df.columns:
                    box_data = [{
                        'med': df[col]['50%'],
                        'q1': df[col]['25%'],
                        'q3': df[col]['75%'],
                        'whislo': df[col]['10%'],
                        'whishi': df[col]['90%'],
                    }]
                    ax.bxp(box_data, positions=[group_offset+run_offset], widths=0.5, 
                        patch_artist=True, # Allows for box fill color
                        boxprops=dict(color=run_colors[run_id]),
                        medianprops=dict(color="black"),  # Optional: set median color
                        showfliers=False
                    )
                    group_offset += group_offset_a
                run_offset += run_offset_a
            xticks = [(i* group_offset_a) + (len(run_ids)-1)/2  for i in range(len(df.columns))]
            ax.set_xticks(xticks)
            ax.set_xticklabels(df.columns, rotation=45, ha='right')
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
            if 'RH98' in path:
                ax.axhline(y=-25, color='gray', linestyle='--', linewidth=1)
            plt.xlabel('RH')
            plt.ylabel('Residuals (m)')
            plt.title(f'{name}')
            handles = [Patch(color=run_colors[run_id], label=run_names[run_id]) for run_id in run_colors]
            plt.legend(handles=handles, title="Runs")
            plt.savefig(f'output/Boxplot_{path.split(" ")[-1]}_{name}.png')
            print()

# analysis of the prediction
def analyse_prediction(run_ids, corrected=False, ref_path:str=None, box_width = 0.5,box_gap = 0.1):
    intervals = [float('-inf')]+ np.arange(0, 55, 5).tolist() + [float('inf')]
    labels = [f"{intervals[i]}-{intervals[i+1]}" for i in range(len(intervals)-1)]
    ref_path = Path(ref_path).expanduser()
    suffix = '_corrected' if corrected else ''
    df = pd.read_parquet(f'output/canopy_height_predictions_{run_ids[0]}{suffix}.parquet')
    pred_cols = [c for c in df.columns if 'RH' in c and '_' not in c]
    ref_cols = [c for c in df.columns if '_GEDI' in c]
    ref_cols_left = [c.lower() for c in pred_cols if f'{c}_GEDI' not in ref_cols]+['wc', 'BIOME', 'rh30', 'rh98']
    df_ref = pd.read_parquet(ref_path, columns=ref_cols_left)
    df_ref['interval'] = pd.cut(df_ref['rh98'], bins=intervals, include_lowest=True, labels=labels)
    df_ref = df_ref.drop(columns=['rh98'])
    ref_cols_left.remove('rh98')
    run_config = {
        'izraz2av': 'QR',
        'uew6hxbn': 'QR_veg',
        'g7h7446j': 'QR_veg_geo',
        "ogzlapu6": "QR_veg_lc",
        'cg11fpjr': 'QR_veg_geo_lc',
    }
    colormap = cm.get_cmap('Pastel1')
    run_colors = {run_ids[i]: colormap(i) for i in range(len(run_ids))}
    run_offset_a = box_width + box_gap
    group_offset_a = len(run_ids) * run_offset_a + 0.2
    box_data = {c:None for c in run_ids}
    box_data_biome = {c:None for c in run_ids}
    for run_id in run_ids:
        if run_id != run_ids[0]:
            df = pd.read_parquet(f'output/canopy_height_predictions_{run_id}{suffix}.parquet')
        df[ref_cols_left+['interval']] = df_ref[ref_cols_left+['interval']]
        df = df.rename(columns=lambda x: f'{x.upper()}_GEDI' if x.startswith('rh') else x)
        df = df[(df['slope_mask']==1) & (df['veg_mask']==1) & (df['BIOME']<15)]
        # plot the histogram of the residuals
        
        for col in pred_cols:            
            df[f'residuals_{col}'] = df[col] - df[f'{col}_GEDI']
        residuals = df[[f'residuals_{col}' for col in pred_cols]+['interval', 'BIOME']]
        stats = residuals.groupby('interval').describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        stats_biome = residuals.groupby('BIOME').describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        box_data[run_id] = stats
        box_data_biome[run_id] = stats_biome
            
    names = ['RH98_interval', 'BIOME']
    for m, data in enumerate([box_data, box_data_biome]):
        for col in pred_cols:
            fig, ax = plt.subplots(figsize=(10, 6))
            run_offset = 0
            for i, run_id in enumerate(run_ids):
                group_offset = 0
                table = data[run_id][f'residuals_{col}'].T
                for intvl in table.columns:
                    if type(intvl) == str and 'inf' in intvl: 
                        continue
                    bx_data = [{
                        'med': table[intvl]['50%'],
                        'q1': table[intvl]['25%'],
                        'q3': table[intvl]['75%'],
                        'whislo': table[intvl]['10%'],
                        'whishi': table[intvl]['90%'],
                    }]
                    ax.bxp(bx_data, positions=[group_offset+run_offset], widths=0.5, 
                       patch_artist=True, boxprops=dict(color=run_colors[run_id]), 
                       medianprops=dict(color="black"), showfliers=False)
                    group_offset += group_offset_a
                run_offset += run_offset_a
            
            
            if m == 0:
                xticks = [(i* group_offset_a) + (len(run_ids)-1)/2  for i in range(len(table.columns)-2)]
                ax.set_xticks(xticks)
                ax.axhline(y=-25, color='gray', linestyle='--', linewidth=1)
                ax.set_xticklabels(table.columns[1:-1], ha='center')
                
            else:
                xticks = [(i* group_offset_a) + (len(run_ids)-1)/2  for i in range(len(table.columns))]
                labels = [BIOMES_BY_VALUE[col]['abbr'] for col in table.columns]
                ax.set_xticks(xticks)
                ax.set_xticklabels(labels, rotation=45, ha='right')
                plt.tight_layout()
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
  
            plt.xlabel('GEDI reference height (m)')
            plt.ylabel('Residuals (m)')
            plt.title(f'{col}')
            handles = [Patch(color=run_colors[run_id], label=run_config[run_id]) for run_id in run_colors]
            plt.legend(handles=handles, title="Runs")
            plt.savefig(f'output/Boxplot_{names[m]}_{col}.png')
            print()
    
        

def boxplot_of_sota_maps(run_id, corrected=True, sota_chm_path:str=None):
    suffix = '_corrected' if corrected else ''
    df = pd.read_parquet(f'output/canopy_height_predictions_{run_id}{suffix}.parquet')
    sota_chm_path = Path(sota_chm_path).expanduser()
    df_sota = pd.read_parquet(sota_chm_path)
    df_sota[['RH95_ours', 'RH98_ours', 'RH100_ours', 'slope_mask', 'veg_mask']] = df[['RH95', 'RH98', 'RH100', 'slope_mask', 'veg_mask']]
    df_sota = df_sota[(df_sota['slope_mask']==1) & (df_sota['veg_mask']==1) & (df_sota['BIOME']<15)]
    intervals = [float('-inf')]+ np.arange(0, 55, 5).tolist() + [float('inf')]
    labels = [f"{intervals[i]}-{intervals[i+1]}" for i in range(len(intervals)-1)]
    df_sota['interval'] = pd.cut(df_sota['RH98_GEDI'], bins=intervals, include_lowest=True, labels=labels)
    df_sota = df_sota.drop(columns=['slope_mask', 'veg_mask'])
    
    for product, name in [('RH95_UMD', 'RH95_GEDI'), ('RH98_ETH', 'RH98_GEDI'), ('RH100_UM', 'RH100_GEDI'), ('RH95_META', 'RH95_GEDI'), ('RH95_ours', 'RH95_GEDI'), ('RH98_ours', 'RH98_GEDI'), ('RH100_ours', 'RH100_GEDI')]:
        df_sota[f'residuals_{product}'] = (df_sota[product] - df_sota[name])
        # avg.groupby('interval').mean()
        # me.append((df_sota[product] - df_sota[name]).mean())
    stats = df_sota.groupby('interval').describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    fig, ax = plt.subplots(figsize=(10, 4))
    group_offset = 0
    run_offset = 0
    run_offset_a = 0.7
    group_offset_a = 7 * run_offset_a + 1
    
    colormap = cm.get_cmap('Pastel1')
    run_colors = {}
    run_colors['RH95_UMD'] = colormap(1)
    run_colors['RH95_META'] = colormap(2)
    run_colors['RH98_ETH'] = colormap(0)
    run_colors['RH100_UM'] = colormap(3)
        
    colormap = cm.get_cmap('tab20c')
    run_colors['RH95_ours'] = colormap(10)
    run_colors['RH98_ours'] = colormap(5)
    run_colors['RH100_ours'] = colormap(14)
    
    for product in ['RH95_UMD', 'RH95_META',  'RH98_ETH','RH100_UM', 'RH95_ours','RH98_ours',  'RH100_ours']:
        group_offset = 0
        table = stats[f'residuals_{product}'].T
        for intvl in table.columns:
            if type(intvl) == str and 'inf' in intvl: 
                continue
            box_data = [{
                'med': table[intvl]['mean'],
                'q1': table[intvl]['25%'],
                'q3': table[intvl]['75%'],
                'whislo': table[intvl]['10%'],
                'whishi': table[intvl]['90%'],
            }]
            
            ax.bxp(box_data, positions=[group_offset+run_offset], widths=0.5, patch_artist=True, boxprops=dict(color=run_colors[product]), medianprops=dict(color="black"), showfliers=False)
            group_offset += group_offset_a
        run_offset += run_offset_a
    xticks = [(i* group_offset_a) + (len(run_ids)-1)/2  for i in range(len(table.columns)-2)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(table.columns[1:-1], ha='center')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('GEDI reference height (m)')
    plt.ylabel('Residuals (m)')
    # plt.title(f'{col}')
    handles = [Patch(color=run_colors[run_id], label=run_id) for run_id in run_colors]
    plt.legend(handles=handles, title="Canopy height maps")
    plt.savefig(f'output/Boxplot_comparison_sota.png')
    print()
        

def export_summary_csv(ids):
    import pandas as pd
    from wandb.old.summary import SummarySubDict
    dfs = []
    for run_id in ids:
        run = api.run(f"global-vegetation-structure/{run_id}")
        summary = {}
        for key, value in run.summary.items():
            if isinstance(value, SummarySubDict) or key in ['_runtime', '_step', '_timestamp']:
                continue
            summary.update({key:[value]})
        summary['run_id'] = [run_id]
        for key, value in run.config['validate'].items():
            summary.update({key: [value]})
        dfs.append(pd.DataFrame(summary))
    df = pd.concat(dfs)
    df = df[df.columns[::-1]]
    df.to_csv('output/run_summary.csv')


def regression_results(ids):
    import pandas as pd
    from wandb.old.summary import SummarySubDict
    dfs = []
    for run_id in ids:
        run = api.run(f"global-vegetation-structure/{run_id}")
        summary = {}
        for artifact in run.logged_artifacts():
            if 'predictionvale' in artifact.name:
                print(artifact.name)


if __name__ == '__main__':
    # boxplot(run_ids)
    run_ids = ['izraz2av', 'uew6hxbn','g7h7446j','cg11fpjr' ]# 'uew6hxbn', 'g7h7446j','cg11fpjr'
    analyse_prediction(run_ids, corrected=True, ref_path='~/data/gvs/train_subsets/val_filtered_v1.parquet')
    boxplot_of_sota_maps(run_id='cg11fpjr', corrected=True, sota_chm_path='~/data/gvs/evaluation/sota_chm_val_with_gedi_biome.parquet')
    print('done')