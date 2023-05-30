Spatial-Frequency Mutual Learning for Face Super-Resolution


## Pretrained Model
 [BaiDu]( https://pan.baidu.com/s/123BQyzubi4C5eDVA87ucDw) passward:en47
## Train 
The training phase of our PSNR-oriented model.
```Python
python main_two.py --data_path data_path  --writer_name mynet --scale scale --
```
2) Train the Student Network.
```Python
python train_student.py --dir_data data_path  --writer_name Student --teacher_load pretrained_teacher_path
```
## Test
```Python
python test.py --dir_data data_path --load pretrained_model_path 
```
