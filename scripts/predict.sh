random_string=hdDJr
echo "随机字符串: $random_string"
CUDA_VISIBLE_DEVICES="7" python -u /share/yanzhongxiang/transformers/examples/pytorch/text-classification/run_glue.py \
--model_name_or_path /share/yanzhongxiang/models/bert-1012-2merge-hdDJr/checkpoint-2940 \
--task_name newsflash/twitter \
--train_file /share/yanzhongxiang/newsflash_classification/data/2merge_train.jsonl \
--validation_file /share/yanzhongxiang/newsflash_classification/data/aih_title_val.jsonl \
--test_file /share/yanzhongxiang/newsflash_classification/data/online_twitter_predict_1013.jsonl \
--output_dir /share/yanzhongxiang/models/bert-1012-2merge-$random_string \
--fp16 \
--overwrite_cache \
--do_predict \
--max_seq_length 512 \
--per_device_train_batch_size 128  \
--per_device_eval_batch_size 128 \
--learning_rate 2e-5 \
--num_train_epochs 3 \
--seed 42  \
--save_strategy epoch \
--evaluation_strategy epoch \
--save_total_limit 3 \