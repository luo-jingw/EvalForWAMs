LOG='test'
CFG='config.yaml'
PROMPT_PATH='visualization/samples_16.txt'
GPU_ID=0

#CUDA_VISIBLE_DEVICES=$GPU_ID python get_calib_data.py --quant-config "./configs/${CFG}" --log "./logs/${LOG}"  --prompt $PROMPT_PATH

#CUDA_VISIBLE_DEVICES=$GPU_ID python ptq.py --quant-config "./configs/${CFG}" --log "./logs/${LOG}"
CUDA_VISIBLE_DEVICES=$GPU_ID python quant_inference.py --quant-config "./configs/${CFG}" --log "./logs/${LOG}"