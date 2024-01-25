# ${변수} 정의
DATA_DIR="/data/ephemeral/home/level2-cv-datacentric-cv-05/data/medical"
OUTPUT_DIR="/data/ephemeral/home/level2-cv-datacentric-cv-05/code/pths"
RESIZE=1024
CROP=1024
BATCH=8
EPOCH=100
# accuracy logging
START=0 # 30
INTERVAL=5 # 10

# run with args
python train.py \
--data_dir ${DATA_DIR} \
--model_dir ${OUTPUT_DIR} \
--image_size ${RESIZE} \
--input_size ${CROP} \
--batch_size ${BATCH} \
--max_epoch ${EPOCH} \
--val_start ${START} \
--val_interval ${INTERVAL}
