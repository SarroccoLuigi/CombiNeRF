# CombiNeRF #

### Requirements ###

* See requirements.txt or environment.yml if using conda

### Dataset ###

* Download [LLFF](https://drive.google.com/drive/folders/14boI-o5hGO9srnWaaogTU5_ji7wkX2S7) dataset folder and put it under `./data`.
Use `scripts/llff2nerf.py` with default parameters to convert LLFF in NeRF format.
* Download [NeRF-Synthetic](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) dataset folder and put it under `./data`.
### Run CombiNeRF ###

* Use `./run_CombiNeRF_LLFF.sh` script to run CombiNeRF on LLFF dataset
* Use `./run_CombiNeRF_NS.sh` script to run CombiNeRF on NeRF-Synthetic dataset

### Citation ###
If you use this code in your research, please cite the following paper:

M.  Bonotto, L. Sarrocco, D. Evangelista, M. Imperoli, and A. Pretto, "CombiNeRF: A Combination of Regularization Techniques for Few-Shot Neural Radiance Field View Synthesis," in International Conference on 3D Vision (3DV), 2024.

BibTeX entry:
``` 
@inproceedings{bseip3DV2024,
  author      = {Bonotto, Matteo and Sarrocco, Luigi and Evangelista, Daniele and Imperoli, Marco and Pretto, Alberto},
  title       = {{CombiNeRF: A Combination of Regularization Techniques for Few-Shot Neural Radiance Field View Synthesis}},
  booktitle   = {International Conference on 3D Vision (3DV)},
  year        = {2024}
}
```
