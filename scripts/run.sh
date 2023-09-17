 CUDA_VISIBLE_DEVICES="0" proxychains4 -q python /share/yanzhongxiang/transformers/examples/pytorch/text-classification/run_glue.py \
 --model_name_or_path /share/yanzhongxiang/models/bert-base-uncased \
 --task_name newsflash/twitter \
 --train_file /share/yanzhongxiang/generate/twitter/twitter.gpt.fixed.jsonl  \
 --output_dir /share/yanzhongxiang/models/bert-0918 \
 --fp16 --overwrite_cache --do_train  --overwrite_output_dir --do_eval \
 --max_seq_length 128 \
 --per_device_train_batch_size 32  \
 --learning_rate 2e-5 \
 --num_train_epochs 5