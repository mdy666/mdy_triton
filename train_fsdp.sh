set -f
DATA_PATHS="/mnt/workspace/mdy/data/train_data/intruct_v1.json /mnt/workspace/mdy/data/train_data/choose_qa/* /mnt/workspace/mdy/data/train_data/low_math/*"
DATA_PATHS="/mnt/workspace/mdy/data/train_data/intruct_v1.json"

MODEl_PATH=/mnt/workspace/mdy/models/Qwen2.5-7B
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=32
OUTPUT_DIR=train_model/logs/qwen2-7B-with-kernel
MAX_SEQ_LEN=2048
MAX_STEPS=1000

N_GPUS=8
torchrun --nproc_per_node=$N_GPUS train_fsdp.py $@ \
    --model_path $MODEl_PATH \
    --data_paths $DATA_PATHS \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --output_dir $OUTPUT_DIR \
    --max_seq_len $MAX_SEQ_LEN \
    --max_steps $MAX_STEPS


# python train.py $@ \
#     --model_path $MODEl_PATH \
#     --data_paths $DATA_PATHS \
#     --micro_batch_size $MICRO_BATCH_SIZE \
#     --global_batch_size $GLOBAL_BATCH_SIZE \
#     --output_dir $OUTPUT_DIR \
#     --max_seq_len $MAX_SEQ_LEN \
#     --max_steps $MAX_STEPS
    