export export CUDA_VISIBLE_DEVICES=3,4,5,6
export MODEL_DIR="/home/lgd/common/ComfyUI/models/checkpoints/stable-diffusion-v1.5-no-safetensor/"
export OUTPUT_DIR="/home/lgd/e_commerce_sd/outputs/mask_controlnet"

accelerate launch --mixed_precision="fp16" --multi_gpu "/home/lgd/e_commerce_sd/tools/train/train_controlnet.py" \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir "/home/lgd/e_commerce_sd/data/mask_controlnet/mask.py" \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "/home/lgd/e_commerce_sd/data/datasets/test_hard/label/R2403010_0070185059_000000011713270255.jpg" "/home/lgd/e_commerce_sd/data/datasets/test_hard/label/R6095003_0070946670_000000011400706518.jpg" \
 --validation_prompt "flowers" "A cow is in the grass" \
 --train_batch_size=48 \
 --gradient_accumulation_steps=1 \
 --mixed_precision="fp16" \
 --tracker_project_name="controlnet-demo" \
 --report_to=wandb
 # --dataset_name=fusing/fill50k \