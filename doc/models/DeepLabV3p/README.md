<div align="center">

English| [简体中文](README_ch.md)

</div>

# Read Parper Fastly


## Archit


# How to train Deeplabv3p on your own dataset

## Data Prepare



## Training
```bash
python train.py  --config configs/RGB/deeplabv3p.yaml --do_eval --use_vdl
```

## Evaluate
```bash
python val.py --config configs/RGB/deeplabv3p.yaml  --model_path ckpt_epoch_best_model.pth --image_path Image/test.png 
```

## Predict

```bash
python predict.py --config configs/RGB/deeplabv3p.yaml \
                 --model_path ckpt_epoch_best_model.pth \
                 --image_path Image/test.png 
```