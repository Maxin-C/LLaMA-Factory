CUDA_VISIBLE_DEVICES= python src/export_model.py \
    --model_name_or_path /mnt/pvc-data.common/ChenZikang/huggingface/Qwen/Qwen2.5-7B-Instruct \
    --adapter_name_or_path /mnt/pvc-data.common/ChenZikang/codes/LLaMA-Factory/saves/Qwen1.5-7B/lora/train_usmle  \
    --template default \
    --finetuning_type lora \
    --export_dir /mnt/pvc-data.common/ChenZikang/huggingface/Qwen/Qwen2.5-7B-Instruct-USMLE \
    --export_size 2 \
    --export_legacy_format False