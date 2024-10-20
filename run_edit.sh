
model_name=gpt2-xl
output_dir_prefix=/output/edited_llm_for_evaluation
editing_method=MEMIT
editing_dataset=cf
log=logs/${model_name}_${editing_method}_${editing_dataset}.log.txt
file_name=gpt2-xl
edit_nums=10

CUDA_VISIBLE_DEVICES=5 \
python edit.py \
    --model_name ${model_name} \
    --editing_method ${editing_method} \
    --editing_dataset ${editing_dataset} \
    --is_sequential_editing True \
    --hparams_file hparams/${editing_method}/${file_name}.yaml \
    --output_dir ${output_dir_prefix}/${model_name}_${editing_method}_${editing_dataset}_eval_${edit_nums} \
    --editing_dataset_size ${edit_nums}


