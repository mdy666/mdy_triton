set -f
DATA_PATHS="/mnt/workspace/mdy/data/train_data/intruct_v1.json /mnt/workspace/mdy/data/train_data/choose_qa/* /mnt/workspace/mdy/data/train_data/low_math/*"
# DATA_PATHS="/mnt/workspace/mdy/data/train_data/choose_qa/* "
MODEl_PATH=/mnt/workspace/mdy/models/Llama-3.2-1B-Instruct
# MODEl_PATH=/mnt/workspace/mdy/models/Qwen2.5-0.5B
MICRO_BATCH_SIZE=16
GLOBAL_BATCH_SIZE=256
OUTPUT_DIR=train_model/logs/llama3-1B-with-kernel-unsloth-loss-pretrain
OUTPUT_DIR=train_model/logs/qwen2-0.5B-with-kernel-unsloth-loss-pretrain
MAX_SEQ_LEN=2048
MAX_STEPS=2000

N_GPUS=8
torchrun --nproc_per_node=$N_GPUS train.py $@ \
    --model_path $MODEl_PATH \
    --data_paths $DATA_PATHS \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --output_dir $OUTPUT_DIR \
    --max_seq_len $MAX_SEQ_LEN \
    --max_steps $MAX_STEPS

# 可选参数  --pretrain        不mask user，需要读取的是messages
#          --deepspeed      开启zero2，具体可以自己去配置
#          --replace_kernel 是否替换算子

# 启动命令一般就是 bash train.sh --deepspeed --replace_kernel


# python train.py $@ \
#     --model_path $MODEl_PATH \
#     --data_paths $DATA_PATHS \
#     --micro_batch_size $MICRO_BATCH_SIZE \
#     --global_batch_size $GLOBAL_BATCH_SIZE \
#     --output_dir $OUTPUT_DIR \
#     --max_seq_len $MAX_SEQ_LEN \
#     --max_steps $MAX_STEPS
