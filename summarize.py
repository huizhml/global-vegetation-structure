import wandb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch

# Choose a colormap (e.g., 'viridis')


api = wandb.Api()

def boxplot(run_ids, box_width = 0.5,box_gap = 0.1):
    run = wandb.init(project="global-vegetation-structure", job_type="summary")
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
    run_ids = ['d30x6wk8','016hicb2','xxx1aume','bvzo4vlq', '51kmjhma','ic6pfw20', 'ggi9w9b8', '4fd8j93r', '05lp279m', 'allse0dy']
    regression_results(run_ids)
    print('done')