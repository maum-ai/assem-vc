# meta-metadata

This README.md describes about each metadata present in this folder.

- Metadata for LibriTTS train-clean-100 split 
  - `libritts_clean_100_train_10s_g2p.txt`
  - `libritts_clean_100_val_g2p.txt`
  - These metadata are from [NVIDIA/mellotron](https://github.com/NVIDIA/mellotron).
- Metadata for VCTK corpus
  - `vctk_22k_train_10s_g2p.txt`: List of training files that are shorter than 10 seconds
  - `vctk_22k_val_g2p.txt`: val.
  - `vctk_22k_test_g2p.txt`: test.
  - For details of making train/val/test split, see section 3.1 of our [paper](https://arxiv.org/abs/2104.00931).
- `libritts_vctk_speaker_list.txt`
  - This is a list of all speakers that are present in LibriTTS train-clean-100 and VCTK.
  - You may want to copy-paste the line below in "speakers" of global configuration yaml file.
