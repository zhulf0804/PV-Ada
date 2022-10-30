# PV-Ada: Point-Voxel Adaptive Feature Abstraction for Robust Point Cloud Classification

This repository contains PyTorch implementation for **[PV-Ada: Point-Voxel Adaptive Feature Abstraction for Robust Point Cloud Classification](https://arxiv.org/pdf/2210.15514)**.

## News

- **2022-10** PV-Ada (technical report) is released on [arXiv](https://arxiv.org/abs/2210.15514).
- **2022-10** A 5-minute [presentation](https://github.com/zhulf0804/PV-Ada/tree/main/logs/PV-Ada-Report.pdf) is given in ECCV'22 Workshop on [Sensing, Understanding and Synthesizing Humans](https://sense-human.github.io). 
- **2022-09** Our solution based on PV-Ada won **the second place** in **ModelNet-C classification track** of [PointCloud-C Challenge 2022 (ECCV Workshop 2022)](https://pointcloud-c.github.io/competition.html
) .

## Dataset

- ModelNet40 [[download](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip)]
    ```
    |- modelnet40_ply_hdf5_2048
        | - ply_data_train*.h5 (#5)
        | - ply_data_test*.h5 (#2)
        | - shape_names.txt
        | - train_files.txt
        | - test_files.txt
        | - ply_data_train*.json (#5)
        | - ply_data_test*.json (#2)
    ```
- ModelNetC [[download](https://drive.google.com/file/d/1KE6MmXMtfu_mgxg4qLPdEwVD5As8B0rm/view?usp=sharing)]
  ```
    |- modelnet_c
        | - clean.h5
        | - add_global*.h5 (#5)
        | - add_local*.h5 (#5)
        | - dropout_global*.h5 (#5)
        | - dropout_local*.h5 (#5)
        | - jitter*.h5 (#5)
        | - rotate*.h5 (#5)
        | - scale*.h5 (#5)
    ```
- ExtraModelNetC [[download](https://codalab.lisn.upsaclay.fr/my/datasets/download/4828e96c-9d9f-49f9-ae31-d497a21a63b3)]
  ```
  |- cls_extra_test_data.h5
  ```


## Train

```
python train.py --pw --beta 1.0 --modelnet_root your_path_to_modelnet40 --modelnetc_root your_path_to_modelnetc --[tapering]

e.g. 
python train.py --pw --beta 1.0 --modelnet_root /mnt/ssd1/lifa_rdata/cls/modelnet40_ply_hdf5_2048 --modelnetc_root /mnt/ssd1/lifa_rdata/PointCloud-C/modelnet_c

python train.py --pw --beta 1.0 --modelnet_root /mnt/ssd1/lifa_rdata/cls/modelnet40_ply_hdf5_2048 --modelnetc_root /mnt/ssd1/lifa_rdata/PointCloud-C/modelnet_c --tapering
```

## Evaluate

```
# For ModelNet40 test set
python evaluate.py --eval --ckpt your_path/model.t7 --modelnet_root your_path_to_modelnet40 

e.g.
python evaluate.py --eval --ckpt pretrained/modelnetc.t7 --modelnet_root /mnt/ssd1/lifa_rdata/cls/modelnet40_ply_hdf5_2048


# For ModelNet40-C Public test set
python evaluate.py --eval_corrupt --ckpt your_path/model.t7 --modelnetc_root your_path_to_modelnetc

e.g.
python evaluate.py --eval_corrupt --ckpt pretrained/modelnetc.t7 --modelnetc_root /mnt/ssd1/lifa_rdata/PointCloud-C/modelnet_c
```

## Infer

```
python test.py --ckpt your_path/model.t7 --h5_path your_path_to_extra_modelnetc.h5 --saved_path your_path

e.g.
python test.py --ckpt pretrained/modelnetc.t7 --h5_path /mnt/ssd1/lifa_rdata/PointCloud-C/cls_extra_test_data.h5 --saved_path results/PCC 
```

## Citation
If you find our work useful in your research, please consider citing: 
```

```

## Acknowledgements

[ModelNet40-C](https://github.com/jiawei-ren/ModelNet-C), [PointCloud-C](https://github.com/ldkong1205/PointCloud-C), [PCT](https://github.com/MenghaoGuo/PCT) and [3DeformRS](https://github.com/gaperezsa/3DeformRS).