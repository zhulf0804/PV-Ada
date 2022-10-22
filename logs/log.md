## 提交结果记录

1. 第1/2次 (7月22号): OA=0.75

- 基于Point-MAE在shapenet数据集上进行预训练, 基于常规的PointTransformer进行finetune
- 因为有噪声, 没有进行pc_norm
- 在36个.h5数据集上进行训练, 训练和测试全部采样到1024个点
- 因为没有训练完, 此次提交是训练epoch21的结果

2. 第3次 (7月26号): OA=0.864

- 训练完300 epoch的结果

3. 第4次 (7月26号): OA=0.830

- 不采用预训练: 训练300epoch的结果

4. 第5次 (7月26号): 失败提交

- 测试在线压缩(zip)

5. 第6次提交 (7月28号): OA=47.8

- 基于ModelNet40 fixed 采样 1024个点进行训练
- 只进行了要求的两种数据增强
- 测试时对上采样的1024个点进行测试

6. 第7次提交 (7月28号): OA=0.513

- 基于ModelNet40 fixed 采样 724个点进行训练
- 只进行了要求的两种数据增强 
- 测试时对原始的724个点进行测试

7. 第8次提交 (7月28号): OA=0.739

- python PCT/main.py --exp_name=RPC_WOLFMix --num_points=1024 --use_sgd=True --pred=True --model_path /mnt/ssd1/lifa_rdata/PointCloud-C/pretrained_models/RPC_WOLFMix_final.t7 --test_batch_size 8 --model RPC
- 数据增强的RPC

8. 第9次提交 (7月28号): OA=0.634

- python PCT/main.py --exp_name=RPC --num_points=1024 --use_sgd=True --pred=True --model_path /mnt/ssd1/lifa_rdata/PointCloud-C/pretrained_models/RPC.t7 --test_batch_size 8 --model RPC
- 没有数据增强的RPC

9. 第10次提交 (7月29号): OA=0.621

- 没有数据增强的RPC
- 自己重新训练的RPC

10. 第11次提交 (7月29号): OA=0.682

- 数据增强的RPC
- 自己重新训练的RPC

11. 第12次提交 (8月1号): OA=0.712

- 基于Mink训练的网络, 没有数据增强

12. 第13次提交 (8月10号): OA=0.797

- python evaluate_cls_extra_test.py --model_path /mnt/ssd1/lifa_rdata/PointCloud-C/pretrained_models/GDANet_WOLFMix.t7 --h5_path /mnt/ssd1/lifa_rdata/PointCloud-C/cls_extra_test_data.h5
- GDANet + 数据增强 (作者提供的权重)

13. 第14次提交 (8月11号): OA=0.787

- GDANet + 数据增强 (重训)

14. 第15次提交 (8月11号): OA=0.754

- GDANet + 数据增强 (w.o. scaling and tranlation) (重训)

15. 第16次提交 (8月11号): 无效提交 (手抖了吧)

16. 第17次提交 (8月11号): OA = 0.822

- voxel + attention模型初版 (没有训练完)

17. 第18, 19次提交 (8月12号): 无效提交 (文件名错误 ?)

18. 第20次提交 (8月12号): OA=0.843

- voxel + attention模型初版 (训练完的结果)
- model: v1
- weight: /mnt/ssd1/lifa_rdata/checkpoints/modelnetc_weights/PCC/models/model.t7
- result: /home/lifa/code/PCC/results/PCC/results.h5

19. 第21次提交 (8月18号): OA=0.819

- voxel + attention + f2

20. 第22次提交 (8月25号): OA=0.848

- voxel + attention模型初版 + 选ModelNetC最好的结果
- model: v2
- weight: /mnt/ssd1/lifa_rdata/checkpoints/modelnetc_weights/PCC_repro2/models/modelnetc.t7
- result: /home/lifa/code/PCC/results/PCC_val_best/results.h5

21. 第23次提交 (8月30号): OA=0.855

- voxel + attention模型初版 + 选ModelNetC最好的结果 + mix300
- model: tag v3, branch fuse_training
- weight: /mnt/ssd1/lifa_rdata/checkpoints/modelnetc_weights/PCC_mix_300/models/modelnetc.t7
- result: /home/lifa/code/PCC/results/PCC_mix_300/result.h5

22. 第24次提交 (9月1号): OA=0.860

- voxel + attention模型初版 + 选ModelNetC最好的结果 + mix300 + point_weight
- model: tag v4
- weight: /mnt/ssd1/lifa_rdata/checkpoints/modelnetc_weights/PCC_point_weight/models/modelnetc.t7
- result: /home/lifa/code/PCC/results/PCC_weight_point/result.h5

23. 第25次提交 (9月1号): OA=0.857

- 同24次提交, 测试时但保留点由0.76 -> 0.60

24. 第26次提交 (9月1号): OA=0.855

- voxel + attention模型初版 + 选ModelNetC最好的结果 + mix300 + point_weight_shared

25. 第27次提交 (9月2号): OA=0.860

- voxel + attention模型初版 + 选ModelNetC最好的结果 + mix300 + (point weight ->) dp
- model: branch dp
- weight: /mnt/ssd1/lifa_rdata/checkpoints/modelnetc_weights/PCC_dp/models/modelnetc.t7
- result: /home/lifa/code/PCC/results/PCC_dp/result.h5

26. 第28次提交 (9月4号): OA=0.860

- voxel + attention模型初版 + 选ModelNetC最好的结果 + mix300 + point_weight_mul
- model: tag v5
- weight: /mnt/ssd1/lifa_rdata/checkpoints/modelnetc_weights/PCC_point_weight_mul/models/modelnetc.t7
- result: /home/lifa/code/PCC/results/PCC_weight_point_mul/result.h5

27. 第29次提交 (9月5号): OA=0.859

- voxel + attention模型初版 + 选ModelNetC最好的结果 + mix300 + point_weight + deeper local operator

28. 第30次提交 (9月5号): 0.858, mOA = 0.883

- voxel + attention模型初版 + 选ModelNetC最好的结果 + mix300 + point_weight_mul_repro (中间结果)
- weight: /mnt/ssd1/lifa_rdata/checkpoints/modelnetc_weights/PCC_point_weight_mul2/models/modelnetc_bak.t7

29. 第31次提交 (9月6号): 0.860, mOA = 0.884

- voxel + attention模型初版 + 选ModelNetC最好的结果 + mix300 + point_weight_mul_repro (最好结果)
- weight: /mnt/ssd1/lifa_rdata/checkpoints/modelnetc_weights/PCC_point_weight_mul2/models/modelnetc.t7

30. 第32次提交 (9月15号): 0.861, mOA = 0.883

- \+ twisting
- model: tag v6
- weight: /mnt/ssd1/lifa_rdata/checkpoints/modelnetc_weights/PCC_aug_twisting/models/modelnetc.t7
- result: /home/lifa/code/PCC/results/PCC_aug_twisting/result.h5

31. 第33次提交 (9月18号): 0.863, mOA = 0.882

- \+ tapering
- model: tag v6.2
- weight: /mnt/ssd1/lifa_rdata/checkpoints/modelnetc_weights/PCC_aug_tapering_2/models/modelnetc.t7
- result: results/PCC_aug_tapering_2/result.h5

32. 第34次提交 (9月19号): 0.865, mOA=0.88

- \+ tapering
- model: tag v6.3
- weight: /mnt/ssd1/lifa_rdata/checkpoints/modelnetc_weights/PCC_aug_tapering_3/models/modelnetc.t7
- result: /home/lifa/code/PCC/results/PCC_aug_tapering_3/result.h5


## Voxel + Attention

| Model | Extra | Clean | mOA | Scale | Jitter | Drop-G | Drop-L | Add-G | Add-L | Rotate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| PCT+WOLFMix | - | 0.934 | 0.873 | 0.906 | 0.730 | 0.906 | 0.898 | 0.912 | 0.861 | 0.895 | 
| GDANet+WOLFMix | 0.797 | 0.934 | 0.871 | 0.915 | 0.721 | 0.868 | 0.886 | 0.910 | 0.886 | 0.912 |
| RPC+WOLFMix | 0.739 | 0.933 | 0.865 | 0.905 | 0.694 | 0.895 | 0.894 | 0.902 | 0.868 | 0.897 |
| Attention + WOLFMix | - | 0.928 | 0.862 | 0.917 | 0.706 | 0.892 | 0.874 | 0.897 | 0.85 | 0.898 | 
| VoxelAttention + WOLFMix | 0.843 | 0.927 | 0.879 | 0.915 | 0.782 | 0.905 | 0.878 | 0.9 | 0.875 | 0.897 | 
| VoxelAttention_noshare + WOLFMix | - | 0.929 | 0.855 | 0.919 | 0.679 | 0.892 | 0.877 | 0.884 | 0.837 | 0.899 |
| VoxelAttention + WOLFMix + best_val | 0.848 | 0.923 | 0.881 | 0.914 | 0.778 | 0.898 | 0.88 | 0.91 | 0.886 | 0.898 | 
| VoxelAttention + WOLFMix + best_val + epoch500 | - | 0.93 | 0.877 | 0.917 | 0.767 | 0.901 | 0.877 | 0.904 | 0.868 | 0.904 |
| VoxelAttention + WOLFMix + best_val + mix300 | 0.855 | 0.921 | 0.882 | 0.903 | 0.803 | 0.894 | 0.881 | 0.911 | 0.894 | 0.889 | 
| VoxelAttention + WOLFMix + best_val + wo_mix300 | - | 0.925 | 0.871 | 0.919 | 0.769 | 0.901 | 0.873 | 0.883 | 0.842 | 0.907 |
| VoxelAttention + WOLFMix + best_val + wo_mix300 + weight_point | 0.860 | 0.92 | 0.882 | 0.903 | 0.815 | 0.892 | 0.875 | 0.912 | 0.886 | 0.888 |
| VoxelAttention + WOLFMix + best_val + wo_mix300 + weight_point + random_test | - | 0.918 | 0.875 | 0.903 | 0.793 | 0.89 | 0.873 | 0.905 | 0.878 | 0.885 | 
| VoxelAttention + WOLFMix + best_val + wo_mix300 + dp | 0.860 | 0.92 | 0.883 | 0.903 | 0.814 | 0.89 | 0.873 | 0.915 | 0.9 | 0.887 |
| VoxelAttention + WOLFMix + best_val + wo_mix300 + weight_point_0.6 | 0.857 | 0.911 | 0.877 | 0.896 | 0.82 | 0.877 | 0.865 | 0.912 | 0.89 | 0.881 |
| VoxelAttention + WOLFMix + best_val + wo_mix300 + weight_point + local operator | 0.859 | 0.923 | 0.881 | 0.898 | 0.796 | 0.886 | 0.882 | 0.919 | 0.904 | 0.884 |
| VoxelAttention + WOLFMix + best_val + wo_mix300 + weight_point_shared | 0.855 | 0.921 | 0.881 | 0.909 | 0.8 | 0.895 | 0.872 | 0.914 | 0.885 | 0.889 | 
| VoxelAttention + WOLFMix + best_val + wo_mix300 + weight_point_mul | 0.860 | 0.919 | 0.881 | 0.904 | 0.812 | 0.888 | 0.875 | 0.912 | 0.884 | 0.891 | 
| VoxelAttention + WOLFMix + best_val + wo_mix300 + weight_point_mul_repro | 0.860 | 0.923 | 0.884 | 0.911 | 0.796 | 0.9 | 0.88 | 0.915 | 0.89 | 0.896 |
| VoxelAttention + WOLFMix + best_val + wo_mix300 + dp | 0.860 | 0.92 | 0.883 | 0.903 |  0.814 | 0.89 | 0.873 | 0.915 | 0.9 | 0.887 |
| * + twisting | 0.861 | 0.915 | 0.883 | 0.896 | 0.817 | 0.894 | 0.87 | 0.91 | 0.892 | 0.9 |
| * + tapering | - | 0.915 | 0.881 | 0.904 | 0.804 | 0.892 | 0.874 | 0.913 | 0.891 | 0.889 |
| * + tapering_2 | 0.863 | 0.919 | 0.882 | 0.91 | 0.796 | 0.898 | 0.879 | 0.911 | 0.886 | 0.895 | 
| * + tapering_3 | 0.865 | 0.911 | 0.88 | 0.907 | 0.792 | 0.897 | 0.874 | 0.914 | 0.884 | 0.892 |
| * + twisting + tapering | - | 0.912 | 0.874 | 0.897 | 0.793 | 0.888 | 0.866 | 0.902 | 0.879 | 0.891 | 

