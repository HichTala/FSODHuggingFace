Command to launch training

```bash
python run_object_detection.py \
    --model_name_or_path facebook/detr-resnet-50 \
    --dataset_name HichTala/dior \
    --do_train true \
    --do_eval true \
    --output_dir path/to/output_dir \
    --num_train_epochs 100 \
    --image_square_size 800 \
    --fp16 true \
    --learning_rate 5e-6 \
    --weight_decay 1e-4 \
    --dataloader_num_workers 2 \
    --dataloader_prefetch_factor 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --remove_unused_columns false \
    --eval_do_concat_batches false \
    --ignore_mismatched_sizes true \
    --metric_for_best_model eval_map \
    --greater_is_better true \
    --load_best_model_at_end true \
    --logging_strategy epoch \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 2 \
    --push_to_hub false \
    --seed 1337
    --report_to wandb
```
