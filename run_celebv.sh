python celebv_multitalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --sample_steps 40 \
    --mode clip \
    --max_frame_num 125 \
    --num_persistent_param_in_dit 0 \
    --use_teacache \
    --save_file test_out_multitalk_14B_480P
