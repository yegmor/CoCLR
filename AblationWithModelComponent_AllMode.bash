cd ./data/search/
python3 split_code_for_retrieval.py

cd ../../

qra=switch
model=./model/search_codebert_${qra}

for mode in header_only doc_only body_only no_header no_doc no_body
do
	echo $mode
	CUDA_VISIBLE_DEVICES="0" python3 ./code_search/run_siamese_test.py \
			--model_type roberta \
			--do_retrieval \
			--data_dir ./data/search/ablation_test_code_component/${mode} \
			--test_data_file cosqa-retrieval-test-500.json \
			--retrieval_code_base code_idx_map.txt \
			--code_type code \
			--max_seq_length 200 \
			--per_gpu_retrieval_batch_size 67 \
			--output_dir ${model}/checkpoint-best-mrr/ \
			--encoder_name_or_path microsoft/codebert-base \
			--pred_model_dir ${model}/checkpoint-best-mrr \
			--retrieval_predictions_output ${model}/retrieval_outputs.txt \
			2>&1 | tee ./log/search-test-ablation-codebert-coclr-${qra}-${mode}.log
done