CFG_NAME="quarot"
EXP_NAME="quarot_test"

#python get_calib_data.py \
	#--log "./logs/${EXP_NAME}" \

# PTQ command
#python ptq.py \
	#--image-size 256\
	#--seed 1 \
	#--ptq-config "./configs/${CFG_NAME}.yaml"\
	#--log "./logs/${EXP_NAME}" \

python quant_inference.py \
	--image-size 256\
	--seed 1 \
	--ptq-config "./configs/${CFG_NAME}.yaml"\
	--log "./logs/${EXP_NAME}" \

