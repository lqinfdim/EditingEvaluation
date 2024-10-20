
model_name=gpt2-xl
output_dir_prefix=/output/edited_llm_for_evaluation
editing_dataset=cf
log=logs/${model_name}_${editing_method}_${editing_dataset}.log.txt
file_name=gpt2-xl

for editing_method in MEMIT ROME KN PMET
do
    for edit_nums in 500 1000
    do
        echo "Editing with  ${editing_method} and ${edit_nums} samples"

        CUDA_VISIBLE_DEVICES=5 \
        python edit.py \
            --model_name ${model_name} \
            --editing_method ${editing_method} \
            --editing_dataset ${editing_dataset} \
            --is_sequential_editing True \
            --hparams_file hparams/${editing_method}/${file_name}.yaml \
            --output_dir ${output_dir_prefix}/${file_name}_${editing_method}_${editing_dataset}_eval_${edit_nums} \
            --editing_dataset_size ${edit_nums}
    done
done