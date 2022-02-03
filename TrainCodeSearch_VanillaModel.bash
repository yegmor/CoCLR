model=./model/search_codebert
CUDA_VISIBLE_DEVICES="0" python3 ./code_search/run_siamese_test.py \
		--model_type roberta \
		--do_train \
		--do_eval \
		--eval_all_checkpoints \
        --data_dir ./data/search/ \
		--train_data_file cosqa-retrieval-train-19604.json \
		--eval_data_file cosqa-retrieval-dev-500.json \
		--retrieval_code_base code_idx_map.txt \
		--code_type code \
		--max_seq_length 200 \
		--per_gpu_train_batch_size 8 \
		--per_gpu_retrieval_batch_size 67 \
		--learning_rate 1e-6 \
		--num_train_epochs 10 \
		--gradient_accumulation_steps 1 \
		--evaluate_during_training \
		--checkpoint_path ./model/codesearchnet-checkpoint \
        --output_dir ${model} \
        --encoder_name_or_path microsoft/codebert-base \
        2>&1 | tee ./log/search-train-codebert.log
