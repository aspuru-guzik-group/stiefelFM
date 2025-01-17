# Stiefel Flow Matching for Moment-Constrained Structure Elucidation

https://arxiv.org/abs/2412.12540

Colab: https://colab.research.google.com/drive/1kEd16obGXJtUVaeGCmCzoD-ZWfWOO9RE?usp=sharing

Code for training and sampling from models that predict all-atom 3D structure from just molecular formula and moments of inertia.

This repository contains a C++ implementation of the Stiefel exponential and logarithm, following Zimmermann & Hüper[^1].

Model checkpoints, data splits, and generated samples are located [here](https://borealisdata.ca/dataset.xhtml?persistentId=doi%3A10.5683%2FSP3%2FZ2LFNF), and can be accessed using `wget`:

```
wget --content-disposition https://borealisdata.ca/api/access/datafile/<fileID>
```

where `fileID` can be found in the URL after clicking on each file in the dataverse.

## Environment setup
```
git clone https://github.com/aspuru-guzik-group/stiefelFM
cd stiefelFM
mamba env create -f env.yml
mamba activate moment
```

For reference, this environment was prepared by
```
mamba create -n moment python=3.9
mamba install "pydantic<2" pydantic-cli wandb rdkit py3dmol einops numpy scipy=1.11.2 matplotlib lightning pytorch pytorch==2.0.1 pytorch-cuda=11.7 pyg pybind11 eigen xtb mkl=2024.0.0 -c pyg -c pytorch -c nvidia
```

Compile `stiefel_log`: (you may need to `module load gcc`, alternatively you can try `mamba install c-compiler cxx-compiler`)
```
c++ -O3 -march=native -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) src/stiefel_log.cpp -o src/stiefel_log$(python3-config --extension-suffix) -I"${CONDA_PREFIX}"/include/eigen3 -finline-functions
```

Download preprocessed data splits for QM9 and GEOM, using the same splits as KREED[^2]:
```
cd data
wget --content-disposition https://borealisdata.ca/api/access/datafile/868784
tar -xf processed.tar.gz
```
You can process the data again by uncommenting the `download()` and `process()` methods in `src/datamodule.py`. You will need to `pip install gdown`.

## Training and sampling

Assuming you are on a SLURM cluster, `cd` to the appropriate directory in `scripts/` and then `sbatch`

e.g. to train `stiefelFM` on QM9:
```
cd scripts/train_qm9
sbatch run_stiefelFM.sh
```
Modify `run_stiefelFM.sh` as needed, for changing SLURM parameters or hyperparameters.

Similarly, for large-scale generation:
```
cd scripts/gen_qm9
python make_jobs.py # you can open the file and comment out any checkpoints that you don't want to generate samples for
sbatch run.sh
```
This will divide the test set into several partitions (12 for QM9, 32 for GEOM) and create an array of SLURM jobs to handle each partition.


Download pretrained model checkpoints:

| Checkpoint Name            | Checkpoint Tag                  | Dataverse ID |
|----------------------------|---------------------------------|--------------|
| (QM9) KREED-XL             | `qm9_kreedXL`                   | 868748       |
| (QM9) Stiefel FM           | `qm9_stiefelFM`                 | 868749       |
| (QM9) Stiefel FM-OT        | `qm9_stiefelFM_OT`              | 868759       |
| (QM9) Stiefel FM-ln        | `qm9_stiefelFM_logitnormal`     | 868758       |
| (QM9) Stiefel FM-ln-OT     | `qm9_stiefelFM_logitnormal_OT`  | 868753       |
| (QM9) Stiefel FM-stoch     | `qm9_stiefelFM_stoch10`         | 868747       |
| (GEOM) KREED-XL            | `geom_kreedXL`                  | 868756       |
| (GEOM) Stiefel FM          | `geom_stiefelFM`                | 868751       |
| (GEOM) Stiefel FM-OT       | `geom_stiefelFM_OT`             | 868755       |

Place checkpoints where they are expected to be located after training:
```
wget --content-disposition https://borealisdata.ca/api/access/datafile/868755
mkdir -p scripts/train_geom/ckpt/stiefelFM_OT
mv geom_stiefelFM_OT.ckpt scripts/train_geom/ckpt/stiefelFM_OT/last.ckpt
```
Now you can continue training, or perform large-scale generation.

Continue training by modifying the appropriate sbatch script to increase `max_epochs`. QM9 models were trained for 1000 epochs and GEOM models were trained for 60 epochs. Therefore, to continue training `stiefelFM_OT` on GEOM until 80 epochs, append `--max_epochs=80` to the command in `scripts/train_geom/run_stiefelFM_OT.sh`. Appending it makes it take priority over the previous specification of `--max_epochs=60`. Continuing training will create `last-v1.ckpt`, instead of overwriting `last.ckpt`.

## Reproducing figures
First download generated samples from dataverse:
```
# in the root directory of the repo
wget --content-disposition https://borealisdata.ca/api/access/datafile/868750
tar -xf samples.tar.gz
```

Inside `samples/`, a `.pt` file stores all the samples for a given checkpoint. It is a dict that maps test set index to a length-10 list of samples. Each element of a length-10 list is a dict that stores the following keys:
['moments_rmse', 'validity', 'correctness', 'heavy_correctness', 'coord_rmse', 'heavy_coord_rmse', 'grad_norm', 'energy', 'coords', 'diversity']

Within a group of 10 generated samples, the value of diversity is copied 10 times. Note that this means that geom/stiefelFM_filter.pt has different values of diversity as compared to performing the aggregation on geom/stiefelFM_filter, geom/stiefelFM_more1.pt, geom/stiefelFM_more2.pt, because diversity has to be recomputed for the new group.

Samples for QM9: random, kreed, kreedXL, kreedXL_dps, kreedXL_proj, stiefelFM, stiefelFM_OT, stiefelFM_logitnormal, stiefelFM_logitnormal_OT, stiefelFM_stoch10

Samples for GEOM: random, kreed, kreedXL, kreedXL_proj, stiefelFM, stiefelFM_more1, stiefelFM_more2, stiefelFM_filter, stiefelFM_OT, stiefelFM_OT_more1, stiefelFM_OT_more2, stiefelFM_OT_filter

Then run any notebook in `figures`, which should all be reproducible, except for slight differences for `01_draw_fig1_fig9` and `06_log_error_fig10_fig11`.

(Optional) If you want to rerank samples, then you can delete `stiefelFM_filter` and `stiefelFM_OT_filter` and then run `cd figures/04_geom_results_table2_fig4_fig7right_fig8; python recalc_divs.py`, which will rerank the independent 30 samples of stiefelFM and stiefelFM_OT by validity and take the top-10, and then recompute diversity.

Summary metrics are also available in csv format in the dataverse:
```
wget --content-disposition https://borealisdata.ca/api/access/datafile/868757
wget --content-disposition https://borealisdata.ca/api/access/datafile/868754
```


[^1]: Zimmermann, R., & Hüper, K. (2022). Computing the Riemannian logarithm on the Stiefel manifold: Metrics, methods, and performance. SIAM Journal on Matrix Analysis and Applications, 43(2), 953-980.

[^2]: https://github.com/aspuru-guzik-group/kreed

## Citation
```
@article{cheng2024stiefel,
  title={Stiefel Flow Matching for Moment-Constrained Structure Elucidation},
  author={Cheng, Austin and Lo, Alston and Lee, Kin Long Kelvin and Miret, Santiago and Aspuru-Guzik, Al{\'a}n},
  journal={arXiv preprint arXiv:2412.12540},
  year={2024}
}
```
