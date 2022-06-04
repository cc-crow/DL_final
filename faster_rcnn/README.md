# 1. 基于 mmdetection 实现，运行需要安装 mmdetection 包；

# 2.  训练+测试方式

## 随机初始化

修改dist_train.sh中--config 参数为 “./config/random_init_config.py”

## imagenet pretrain 初始化

修改改dist_train.sh中--config 参数为 “./config/imagenet_pre_config.py”

## coco pretrain 初始化

修改改dist_train.sh中--config 参数为 “./config/coco_pre_config.py”

```
bash dist_train.sh
```
