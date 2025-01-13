# Stiefel Flow Matching for Moment-Constrained Structure Elucidation

https://arxiv.org/abs/2412.12540

Code for training and sampling from models that predict all-atom 3D structure from just molecular formula and moments of inertia.

This repository contains a C++ implementation of the Stiefel exponential and logarithm, following Zimmermann & Hüper[^1].

Model checkpoints, data splits, and generated samples are located at (dataverse link), and can be accessed using `wget`:

```
wget https://borealisdata.ca/api/access/datafile/<fileID>
```


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
mamba install "pydantic<2" pydantic-cli wandb rdkit py3dmol einops numpy scipy=1.11.2 matplotlib lightning pytorch pytorch==2.0.1 pytorch-cuda=11.7 pyg pybind11 eigen xtb -c pyg -c pytorch -c nvidia
```

Compile `stiefel_log`: (you may need to `module load gcc`, alternatively you can try `mamba install c-compiler cxx-compiler`)
```
c++ -O3 -march=native -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) src/stiefel_log.cpp -o src/stiefel_log$(python3-config --extension-suffix) -I"${CONDA_PREFIX}"/include/eigen3 -finline-functions
```

Download preprocessed data splits for QM9 and GEOM, using the same splits as KREED[^2]:
```
cd data
wget ...
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
| (QM9) KREED-XL             | `qm9_kreedXL`                   | XXXXX        |
| (QM9) Stiefel FM           | `qm9_stiefelFM`                 | XXXXX        |
| (QM9) Stiefel FM-OT        | `qm9_stiefelFM_OT`              | XXXXX        |
| (QM9) Stiefel FM-ln        | `qm9_stiefelFM_logitnormal`     | XXXXX        |
| (QM9) Stiefel FM-ln-OT     | `qm9_stiefelFM_logitnormal_OT`  | XXXXX        |
| (QM9) Stiefel FM-stoch     | `qm9_stiefelFM_stoch10`         | XXXXX        |
| (GEOM) KREED-XL            | `geom_kreedXL`                  | XXXXX        |
| (GEOM) Stiefel FM          | `geom_stiefelFM`                | XXXXX        |
| (GEOM) Stiefel FM-OT       | `geom_stiefelFM_OT`             | XXXXX        |

Place checkpoints where they are expected to be located after training:
```
wget https://borealisdata.ca/api/access/datafile/<fileID>
mkdir -p scripts/train_geom/ckpt/stiefelFM_OT
mv geom_stiefelFM_OT.ckpt scripts/train_geom/ckpt/stiefelFM_OT/last.ckpt
```
Now you can continue training, or perform large-scale generation.

Continue training by modifying the appropriate sbatch script to increase `max_epochs`. QM9 models were trained for 1000 epochs and GEOM models were trained for 60 epochs. Therefore, to continue training `stiefelFM_OT` on GEOM until 80 epochs, append `--max_epochs=80` to the command in `scripts/train_geom/run_stiefelFM_OT.sh`. Appending it makes it take priority over the previous specification of `--max_epochs=60`. Continuing training will create `last-v1.ckpt`, instead of overwriting `last.ckpt`.


Download generated samples from dataverse:
```
wget https://borealisdata.ca/api/access/datafile/<fileID>
tar -xf samples.tar.gz
```

Samples for QM9: random, kreed, kreedXL, kreedXL_dps, kreedXL_proj, stiefelFM, stiefelFM_OT, stiefelFM_logitnormal, stiefelFM_logitnormal_OT, stiefelFM_stoch10

Samples for GEOM: random, kreed, kreedXL, kreedXL_proj, stiefelFM, stiefelFM_more1, stiefelFM_more2, stiefelFM_filter, stiefelFM_OT, stiefelFM_OT_more1, stiefelFM_OT_more2, stiefelFM_OT_filter


[^1]: Zimmermann, R., & Hüper, K. (2022). Computing the Riemannian logarithm on the Stiefel manifold: Metrics, methods, and performance. SIAM Journal on Matrix Analysis and Applications, 43(2), 953-980.

[^2]: https://github.com/aspuru-guzik-group/kreed

Citation:
```
@article{cheng2024stiefel,
  title={Stiefel Flow Matching for Moment-Constrained Structure Elucidation},
  author={Cheng, Austin and Lo, Alston and Lee, Kin Long Kelvin and Miret, Santiago and Aspuru-Guzik, Al{\'a}n},
  journal={arXiv preprint arXiv:2412.12540},
  year={2024}
}
```
