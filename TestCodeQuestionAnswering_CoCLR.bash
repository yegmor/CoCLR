qra=switch
model=./model/qa_codebert_${qra}
CUDA_VISIBLE_DEVICES="0" python3 ./code_qa/run_siamese_test.py \
		--model_type roberta  \
		--augment \
		--do_test \
		--data_dir ./data/qa \
		--test_data_file test_webquery.json \
		--max_seq_length 200 \
		--per_gpu_eval_batch_size 2 \
		--output_dir ${model}/checkpoint-best-aver/ \
		--encoder_name_or_path microsoft/codebert-base \
		--pred_model_dir ${model}/checkpoint-best-aver/ \
		--test_predictions_output ${model}/webquery_predictions.txt \
		2>&1| tee ./log/qa-test-codebert-coclr-${qra}.log