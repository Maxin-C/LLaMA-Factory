formatted_datetime=$(date +'%Y-%m-%d %H:%M:%S')
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
accelerate launch --config_file config.yaml src/train_bash.py \
    --ddp_timeout 180000000 \
    --stage sft \
    --do_train \
    --model_name_or_path /root/lanyun-tmp/huggingface/Qwen/Qwen2.5-7B-Instruct \
    --dataset huatuo_qa_train \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir saves/Qwen2.5-7B-Instruct/lora/train_$formatted_datetime \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16