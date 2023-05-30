Spatial-Frequency Mutual Learning for Face Super-Resolution


## Pretrained Model
 [BaiDu]( https://pan.baidu.com/s/123BQyzubi4C5eDVA87ucDw) passward:en47
## Train 
Train PSNR-oriented model:
```Python
python main_two.py --data_path data_path  --writer_name mynet --scale scale --model MYNET
```
Train GAN-based model:
```Python
python main_GAN.py --load pretrained_PSNR_model --model MYNMET --scale 8 --data_path data_path  --writer_name mynetgan
```
## Test
```Python
python test.py --model MYNET --load pretrained_model_path  --data_path data_path --save_name name
```
