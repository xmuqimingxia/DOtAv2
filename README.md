# Learning to Detect Objects from Multi-Agent LiDAR Scans without Manual Labels

This is a official code release of [DoTA](https://arxiv.org/abs/2503.08421) (Learning to Detect Objects from Multi-Agent LiDAR Scans without Manual Labels). 

## Detection Framework
![1744802587673](https://github.com/user-attachments/assets/56aa59e3-4403-4722-ba1f-bc1bf222b021)

## Getting Started
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

### Test the model
Before you run the following command, first make sure the `validation_dir` in config.yaml under your checkpoint folder
refers to the testing dataset path, e.g. `opv2v_data_dumping/test`.

```python
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} --fusion_method ${FUSION_STRATEGY} [--show_vis] [--show_sequence]
```
Arguments Explanation:
- `model_dir`: the path to your saved model.
- `fusion_method`: indicate the fusion strategy, currently support 'early', 'late', and 'intermediate'.
- `show_vis`: whether to visualize the detection overlay with point cloud.
- `show_sequence` : the detection results will visualized in a video stream. It can NOT be set with `show_vis` at the same time.
- `global_sort_detections`: whether to globally sort detections by confidence score. If set to True, it is the mainstream AP computing method, but would increase the tolerance for FP (False Positives). **OPV2V paper does not perform the global sort.** Please choose the consistent AP calculation method in your paper for fair comparison.

The evaluation results  will be dumped in the model directory. 





