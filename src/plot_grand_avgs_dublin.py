import argparse
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import mne
from tqdm import tqdm
matplotlib.use('Agg')

def parse_args():
    parser = argparse.ArgumentParser(description="MNE Grand Average Processing Script")
    parser.add_argument(
        "subjects_dir", 
        type=str, 
        help="Path to the directory containing subject folders"
    )
    parser.add_argument(
        "site", 
        type=str, 
        help="Site of ecording"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="ERROR", 
        help="MNE logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    return parser.parse_args()

def get_grand_averages(subjects_dir):
    
    ## config
    config = {
        "omi": {"file": "epochs-omi.fif", "ids": ["Stimulus 4"]},
        "xx":  {"file": "epochs-xxxxx.fif", "ids": ['Stimulus 1', 'Stimulus 2', 'Stimulus 3']},
        "xy":  {"file": "epochs-xxxxy.fif", "ids": ['Stimulus/S  11', 'Stimulus/S  12', 'Stimulus/S  13']}
    }

    subject_evokeds = {key: [] for key in config}
    subjects_dir = Path(subjects_dir)
    sub_dirs = [d for d in subjects_dir.iterdir() if d.is_dir()]
    for sub_dir in tqdm(sub_dirs, desc="Subjects"):
        for key, info in config.items():
            fname = sub_dir / "epochs" / info["file"]
            
            if fname.exists():
                try:
                    epochs = mne.read_epochs(fname)
                    evoked_list = epochs.average(by_event_type=True)
                    subject_evokeds[key].append(evoked_list)
                except Exception as e:
                    print(f"Could not process {fname}: {e}")


    grand_averages = {}
    for key, info in config.items():
        for idx, event_id in enumerate(info["ids"]):
            pool = [evs[idx] for evs in subject_evokeds[key] if len(evs) > idx]
            if pool:
                grand_averages[event_id] = mne.grand_average(pool)
            else:
                print(f"Warning: No data found for {event_id}")
    return grand_averages


def plot_grand_averages(ev_grands, subjects_dir, site):

    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10})
    qc_picks = ["FC1", "FCz", "FC2", "C1", "Cz", "C2", "CP1", "CPz", "CP2"]
    ev_ids = list(ev_grands.keys())

    fig, axs = plt.subplots(len(ev_ids), 2, figsize=(10, 12), 
                            sharex=True, sharey='col', 
                            constrained_layout=True)

    for col_idx, (picks, label) in enumerate(zip([None, qc_picks], ["All Channels", "Centro-Frontal ROI"])):
        for row_idx, ev_id in enumerate(ev_ids):
            ax = axs[row_idx, col_idx]
            ev_grands[ev_id].plot(picks=picks, axes=ax, time_unit='ms', show=False)
            
            if ax.get_legend():
                ax.get_legend().remove()
                
            if col_idx == 0:
                ax.set_ylabel(f"{ev_id}\nμV")
            else:
                ax.set_ylabel("")

            if row_idx == 0:
                ax.set_title(label, fontsize=14, pad=15, fontstyle='italic')
            else:
                ax.set_title("")

            if not row_idx == len(ev_ids) - 1:
                ax.set_xlabel("")

            for text in ax.texts:
                if "N" in text.get_text():
                    text.set_visible(False)

            ax.spines[["right", "top"]].set_visible(False)
            ax.axvline(0, color="black", linestyle=":", alpha=0.5)
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.3)
            ax.grid(visible=True, axis='both', linestyle='--', linewidth=0.5, alpha=0.3, color='grey')




    handles, labels = axs[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles[:5], labels[:5], loc='lower center', 
                    bbox_to_anchor=(0.5, -0.05), ncol=5, frameon=False)
        
    output_filename = Path(subjects_dir) / "QC" / f"{site}_dublin_grand_avgs.png"
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    plt.savefig(output_filename, format='png',
                dpi=600, bbox_inches='tight',
                facecolor='white', transparent=False)


if __name__ == "__main__":
    args = parse_args()
    mne.set_log_level(args.log_level)
    ev_grands = get_grand_averages(args.subjects_dir)
    plot_grand_averages(ev_grands, args.subjects_dir, args.site)