GPU_NUM=8
torchrun --nproc_per_node=$GPU_NUM --standalone celebv_multitalk.py \
    --ckpt_dir /home/schu23/.cache/weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir '/home/schu23/.cache/weights/chinese-wav2vec2-base' \
    --dit_fsdp --t5_fsdp \
    --ulysses_size=$GPU_NUM \
    --input_json examples/single_example_1.json \
    --sample_steps 40 \
    --mode clip \
    --max_frame_num 125 \
    --use_teacache \
    --save_file test_out_multitalk_14B_480P
