# Improved Lightweight YOLOv5 for Face Mask Detection
This is the code for the paper "[An Improved Lightweight YOLOv5 Model Based on Attention Mechanism for Face Mask Detection](https://link.springer.com/chapter/10.1007/978-3-031-15934-3_44)" published in ICANN 2022. Note that:

1. The implementation of the baselines are based on the code from [ultralytics/yolov5](https://github.com/ultralytics/yolov5).
2. The dataset is from [AIZOOTech](https://github.com/AIZOOTech/FaceMaskDetection).

## How to use the code
1. There are some configs in my repo, you can choose some of them to run (If you do not know how to run yolov5, please see the detailed tutorial of yolov5 by ultralytics).
2. Based on the code, you can write your own structures and configs to train your own model.
3. There are lots of comments in Chinese written by me among the code, maybe it could help you understand the code (Since it is my first work, there might be some mistakes, please be careful).
4. The yolov5 is updated frequently by ultralytics. The baseline in my repo might be not the newest, you can combine it with the newest version.

If you find this benchmark helpful, please use the citation:
```
@inproceedings{
title={An Improved Lightweight YOLOv5 Model Based on Attention Mechanism for Face Mask Detection},
author={Sheng Xu and Zhanyu Guo and Yuchi Liu and Jingwei Fan and Xuxu Liu},
booktitle={International Conference on Artificial Neural Networks},
year={2022},
url={https://link.springer.com/chapter/10.1007/978-3-031-15934-3_44}
}
```