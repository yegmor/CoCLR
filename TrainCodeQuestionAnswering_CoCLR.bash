qra=switch
model=./model/qa_codebert_${qra}
CUDA_VISIBLE_DEVICES="0" python3 ./code_qa/run_siamese_test.py \
		--model_type roberta \
		--augment \
		--do_train \
		--do_eval \
		--eval_all_checkpoints \
		--data_dir ./data/qa/ \
		--train_data_file cosqa-train-qra-${qra}-29707.json \
		--eval_data_file cosqa-dev.json \
		--max_seq_length 200 \
		--per_gpu_train_batch_size 8 \
		--per_gpu_eval_batch_size 16 \
		--learning_rate 1e-5 \
		--warmup_steps 1000 \
		--num_train_epochs 10 \
		--gradient_accumulation_steps 1 \
		--evaluate_during_training \
		--checkpoint_path ./model/codesearchnet-checkpoint \
		--output_dir ${model} \
		--encoder_name_or_path microsoft/codebert-base \
        2>&1 | tee ./log/qa-train-codebert-coclr-${qra}.log