export TASK_NAME=cola


models=(orionweller/enc-17m_de_decay-hf orionweller/enc-17m_de_decay_rope-hf)
# 39.4 vs 35.4
models=(orionweller/enc-150m_prolong_decay_mix_original-hf)

# models=("orionweller/enc-150m_de_decay-hf")

lr=1e-5
wd=8e-5
epochs=12

for model in ${models[@]}; do
  echo "Running with model: $model"
  file_safe_model=$(echo $model | tr '/' '_')
  python run_glue.py \
  --model_name_or_path $model \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --learning_rate $lr \
  --weight_decay $wd \
  --num_train_epochs $epochs \
  --output_dir tmp_part_glue/$TASK_NAME/ \
  --overwrite_output_dir \
  --save_strategy epoch \
  --max_grad_norm 10.0 \
  --evaluation_strategy epoch \
  --metric_for_best_model eval_matthews_correlation \
  --greater_is_better True \
  --report_to wandb \
  --run_name $model-${TASK_NAME}-${lr}-${wd}-${epochs} \
  --load_best_model_at_end True 
done