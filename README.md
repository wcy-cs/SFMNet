# Spatial-Frequency Mutual Learning for Face Super-Resolution
![image](https://github.com/wcy-cs/SFMNet/blob/main/network.png)

## Pretrained Model
 [BaiDu]( https://pan.baidu.com/s/123BQyzubi4C5eDVA87ucDw) passward:en47, [Google Drive](https://drive.google.com/drive/folders/1yWmKxA8d2xSAopJzhtZvOEwCyJ2MxakU?usp=sharing)
## Requirement
Pytorch 1.11.0 Cuda 11.4 
## Train 
Train PSNR-oriented model:
```Python
python main_two.py --data_path data_path  --writer_name mynet --scale scale --model MYNET
```
Train GAN-based model:
```Python
python main_GAN.py --load pretrained_PSNR_model --model MYNET --scale 8 --data_path data_path  --writer_name mynetgan
```
## Test
```Python
python test.py --model MYNET --load pretrained_model_path  --data_path data_path --save_name name
```

## Test dataset
[BaiDu](https://pan.baidu.com/s/1HAj03oLERgzYeEDT3y5B4w) passward:v2uf
## Citation 
```
@InProceedings{Wang_2023_CVPR,
    author    = {Wang, Chenyang and Jiang, Junjun and Zhong, Zhiwei and Liu, Xianming},
    title     = {Spatial-Frequency Mutual Learning for Face Super-Resolution},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {22356-22366}
}
```
