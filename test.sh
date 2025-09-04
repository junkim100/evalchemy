CUDA_VISIBLE_DEVICES=7 python -m eval.eval --model vllm \
	    --model_args pretrained=Qwen/Qwen3-0.6B,gpu_memory_utilization=0.4 \
		--apply_chat_template True \
		--tasks CreativeWriting \
		--device cuda:0 \
		--batch_size auto \
		--debug \
		--output_path logs > log 2>&1 &


    
