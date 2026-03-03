import os
import numpy as np
import mne


def preprocess_dataset(
    data_path: str,
    save_path: str,
    sfreq_new: int = 125,
    epoch_duration: float = 2.0,
    epoch_overlap: float = 0.0
):
    """
    Preprocess raw EEGLAB files into normalized numpy arrays.
    """

    os.makedirs(save_path, exist_ok=True)
    subjects = sorted([s for s in os.listdir(data_path) if s.startswith("sub-")])

    for subj in subjects:
        print(f"Processing {subj}")
        subj_folder = os.path.join(data_path, subj)

        for ses_num in [1, 2]:
            eeg_folder = os.path.join(subj_folder, f"ses-{ses_num}", "eeg")
            if not os.path.exists(eeg_folder):
                continue

            set_files = [f for f in os.listdir(eeg_folder) if f.endswith(".set")]

            for f in set_files:
                try:
                    raw = mne.io.read_raw_eeglab(
                        os.path.join(eeg_folder, f),
                        preload=True
                    )
                except Exception as e:
                    print("ERROR loading:", f, e)
                    continue

                raw.filter(1., 45.)
                raw.set_eeg_reference('average')
                raw.resample(sfreq_new)

                epochs = mne.make_fixed_length_epochs(
                    raw,
                    duration=epoch_duration,
                    overlap=epoch_overlap
                )

                data = epochs.get_data().astype(np.float32)
                labels = np.full(data.shape[0], ses_num - 1, dtype=np.float32)

                # Normalize per epoch per channel
                data = (data - data.mean(axis=2, keepdims=True)) / \
                       (data.std(axis=2, keepdims=True) + 1e-6)

                np.save(os.path.join(save_path, f"{subj}_ses{ses_num}_X.npy"), data)
                np.save(os.path.join(save_path, f"{subj}_ses{ses_num}_y.npy"), labels)

                del raw, epochs, data

    print("Preprocessing complete.")