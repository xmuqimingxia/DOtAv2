# DOtA

* train detector only with communication info
```shell script
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2  --use_env opencood/tools/train.py --hypes_yaml hypes_yaml/point_pillar_intermediate_fusion_label_free.yaml
```

* initial pseudo-label generation
```shell script
python opencood/tools/inference.py --model_dir ${INITIAL_DETECTOR_CHECKPOINT_FOLDER} --fusion_method intermediate --pseudo_lable_save 0
```

* filter pseudo-label
```shell script
python opencood/tools/mvsta_multiprocess.py
```
```shell script
python opencood/tools/box_score_for_mvsta.py
```

* use pseudo-label for training
```shell script
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2  --use_env opencood/tools/train.py --hypes_yaml /mnt/32THHD/lwk/codes/OpenCOOD/opencood/hypes_yaml/point_pillar_intermediate_fusion_moma.yaml
```






