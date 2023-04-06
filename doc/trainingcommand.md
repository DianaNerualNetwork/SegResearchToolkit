## ⚡  Enviroment install
You Need to confirm the version of pytorch must >=1.6.0

## ⚡  run training , validate , predict args

In train.py
```bash

        "--config" , [Required] must give yml

        '--iters', [Not Required] 

        '--batch_size', [Not Required]

        '--learning_rate', [Not Required]

        '--resume_model', [Not Required]

        '--save_dir', [Not Required] 

        '--do_eval', [Not Required]  if you want to evaluate  --do_eval

        '--use_vdl', [Not Required]  if you want to viz --use_dl


        '--seed', [Not Required] 


        '--log_iters', [Not Required] 
        
        '--num_workers', [Not Required]
       

        '--opts', [Not Required]

        '--keep_checkpoint_max', [Not Required] default:5  how many iter result to save

        '--save_interval', [Not Required] default:1000
      

```
When you want to run train.py,example:
```bash
python train.py --config xxxx.yml --do_eval --use_vdl
```
It means that run the yml to train and do evaluate and logger in tensorboard

In val.py
```bash
python val.py --config xxx.yaml --model_path xxx/output/best_model/ckpt_iters_best_model.pth 
```

In predict.py
```bash
python predict.py  --config xxx.yaml --model_path /xxx/ckpt_iters_best_model.pth --image_path  /xxx/JPEGImages/P0193.jpg
```