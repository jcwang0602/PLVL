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
# for unc and unc+
declare -a items_dataset=("val" "testA" "testB")
# for gref_umd
# declare -a items_dataset=("val" "test")

for item_model in "${items_model[@]}"; do
    for item_datatset in "${items_dataset[@]}"; do
        python -m torch.distributed.launch --nproc_per_node=${NGPU} --master_port ${PORT} --use_env eval.py \
            --batch_size 80 \
            --is_res --is_rec \
            --ca_block_indexes 2 5 8 11 \
            --model_name ${MODEL_NAME} \
            --backbone ${BACKBONE} \
            --dataset ${DATASET} \
            --eval_set ${item_datatset} \
            --device cuda \
            --eval_model ${OUTPUT}/${item_model} \
            --output_dir ${OUTPUT}
    done
done


