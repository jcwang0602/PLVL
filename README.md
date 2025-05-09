# Progressive Language-guided Visual Learning for Multi-Task Visual Grounding

Paper Link [ArXiv](https://arxiv.org/abs/2504.16145).

### Install
```
conda create -n plvl Python=3.8
conda activate plvl
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Data Preparation
1.You can download the images from the original source and place them in `./image_data` folder:
- RefCOCO/RefCOCO+/RefCOCOg
- Flickr30K Entities
- Visual Genome

Finally, the `./image_data` folder will have the following structure:

```angular2html
|-- ln_data
   |-- flickr30k
   |-- mscoco/images/train2014/
   |-- visual-genome
```

2.Download data labels [here](https://hkustconnect-my.sharepoint.com/:u:/g/personal/wchendb_connect_ust_hk/EbiPljj4dx5Ns-tAf3zR8_UBxiuM7kRh2VHKPoI6q58TcQ?e=WleUng) and place them in `./mask_data` folder

### Pretrained Checkpoints
Download the following checkpoints and place them in the `./checkpoints` folder.
- [ViTDet](https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_b/f325358525/model_final_435fa9.pkl)
- [bert-base-uncased](https://huggingface.co/bert-base-uncased)

### Training

1.  Training and Evaluation on RefCOCOg. 
    ```
    NGPU=4
    PORT=25449
    MODEL_NAME=PLVL
    BACKBONE=ViTDet_Dec
    
    DATASET=unc # unc/unc+/gref_umd
    OUTPUT=outputs/${DATASET}/PLVL
    
    # Train
    python -m torch.distributed.launch --nproc_per_node=${NGPU} --master_port ${PORT} --use_env train.py \
        --batch_size 20 \
        --device cuda \
        --aug_scale --aug_translate --aug_crop \
        --is_res --is_rec \
        --ca_block_indexes 2 5 8 11 \
        --model_name ${MODEL_NAME} \
        --backbone ${BACKBONE} \
        --dataset ${DATASET} \
        --output_dir ${OUTPUT}
    
    # Evaluation
    declare -a items_model=("best_checkpoint.pth" "best_mask_checkpoint.pth")
    # declare -a items_dataset=("val" "test")
    
    for item_model in "${items_model[@]}"; do
        for item_datatset in "${items_dataset[@]}"; do
            python -m torch.distributed.launch --nproc_per_node=${NGPU} --master_port ${PORT} --use_env eval.py \
                --batch_size 80 \
                --is_res --is_rec \
                --ca_block_indexes 2 5 8 11 \
                --model_name ${MODEL_NAME} \
                --backbone ${}BACKBONE} \
                --dataset ${DATASET} \
                --eval_set ${item_datatset} \
                --device cuda \
                --eval_model ${OUTPUT}/${item_model} \
                --output_dir ${OUTPUT}
        done
    done

    ```
    
    Please refer to [train.sh](train.sh) for training commands on other datasets.

2. For the pretraining result, first use the following command to pretrain model on the mixed dataset. Then use the following command to fine-tune on mixed RefCOCO series datasets. 
    ```
    NGPU=8
    PORT=25449
    MODEL_NAME=PLVL
    BACKBONE=ViTDet_Dec
    
    # mix pretrain
    DATASET=mixed_pretrain
    OUTPUT=outputs/${DATASET}/PLVL_2
    python -m torch.distributed.launch --nproc_per_node=$NGPU --master_port $PORT --use_env train.py \
        --batch_size 20 \
        --device cuda \
        --loss_alpha 0.5 \
        --epochs 20 \
        --aug_scale --aug_translate --aug_crop \
        --is_rec --ca_block_indexes 2 5 8 11 \
        --model_name $MODEL_NAME \
        --backbone $BACKBONE \
        --dataset $DATASET \
        --output_dir $OUTPUT
    
    # coco fine-tune
    DATASET=mixed_coco
    OUTPUT=outputs/${DATASET}/PLVL
    python -m torch.distributed.launch --nproc_per_node=$NGPU --master_port $PORT --use_env train.py \
        --batch_size 20 \
        --device cuda \
        --loss_alpha 0.05 \
        --epochs 150 \
        --aug_scale --aug_translate --aug_crop \
        --is_res --is_rec  --ca_block_indexes 2 5 8 11 \
        --model_name $MODEL_NAME \
        --backbone $BACKBONE \
        --dataset $DATASET \
        --output_dir $OUTPUT \
        --pretrain outputs/mixed_pretrain/PLVL/checkpoint.pth
    ```
    Please refer to [train_mix.sh](train_mix.sh) for training commands on other datasets.

### Our checkpoints

Our checkpoints are available at [百度网盘](https://pan.baidu.com/s/1ebbYBOxJZDojEAz9bjiieQ?pwd=gw75).

## Acknowledgement

Our model is related to [EEVG](https://github.com/chenwei746/EEVG), [UVLTrack](https://github.com/OpenSpaceAI/UVLTrack), [ViTDet](https://github.com/ViTAE-Transformer/ViTDet). Thanks for their great work!

## Citation
If our work is useful for your research, please consider cite:
```
@misc{wang2025progressivelanguageguidedvisuallearning,
      title={Progressive Language-guided Visual Learning for Multi-Task Visual Grounding}, 
      author={Jingchao Wang and Hong Wang and Wenlong Zhang and Kunhua Ji and Dingjiang Huang and Yefeng Zheng},
      year={2025},
      eprint={2504.16145},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.16145}, 
}
```

[//]: # (## Star History)

[//]: # ()
[//]: # ([![Star History Chart]&#40;https://api.star-history.com/svg?repos=jcwang0602/PLVL&type=Date&#41;]&#40;https://star-history.com/#linhuixiao/HiVG&Date&#41;)