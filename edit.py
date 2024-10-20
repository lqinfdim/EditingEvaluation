import os
import torch
import transformers
import fastedit
import argparse
import json
import time
import shutil
import numpy as np
import pandas as pd
from transformers import PreTrainedModel

from fastedit import (
    FTHyperParams, 
    IKEHyperParams, 
    KNHyperParams, 
    MEMITHyperParams, 
    ROMEHyperParams, 
    LoRAHyperParams,
    MENDHyperParams,
    SERACHparams,
    PMETHyperParams,
    MALMENHyperParams,
    GraceHyperParams,
    MELOHyperParams
    )
from fastedit import BaseEditor
from fastedit import ZsreDataset, CounterFactDataset
from fastedit.trainer import EditTrainer
from datasets import load_dataset
from tqdm import tqdm

DATASET_MAPPING = {
    "zsre" : ZsreDataset,
    "cf" : CounterFactDataset,
}

METHOD_MAPPING = {
    "FT" : FTHyperParams,
    "IKE" : IKEHyperParams,
    "MEMIT" : MEMITHyperParams,
    "ROME" : ROMEHyperParams,
    "KN" : KNHyperParams,
    "PMET" : PMETHyperParams,
    "MEND" : MENDHyperParams,
    "MALMEN": MALMENHyperParams,
    "LORA" : LoRAHyperParams,
    "SERAC": SERACHparams,
    "GRACE" : GraceHyperParams,
    "MELO":MELOHyperParams
}

DATASET_PATH = {
    "cf" : ("/dataset/counterfact/counterfact-train.json", "/dataset/counterfact/counterfact-val.json"),
    "zsre" : ("/dataset/zsre/zsre_mend_train.json", "/dataset/zsre/zsre_mend_eval.json"),
}

def get_method_hyperparameter_class(method_name : str):
    return METHOD_MAPPING[method_name]

def get_dataset_class(dataset_name : str):
    return DATASET_MAPPING[dataset_name]

def get_mend_dataset_path(ds_name):
    return DATASET_PATH[ds_name]
    
def create_or_clear_dir(path_name):
    if os.path.exists(path_name):
        shutil.rmtree(path_name)
        
    os.makedirs(path_name)
    os.makedirs(os.path.join(path_name, 'result'))
    os.makedirs(os.path.join(path_name, "final_model"))   



def editing_parse_arg():
    parser = argparse.ArgumentParser("hyperparameters for model editing")
    
    parser.add_argument("--model_name", type=str, default=None, required=True)
    parser.add_argument("--editing_method", type=str, default=None, required=True)
    parser.add_argument("--hparams_file", type=str, default=None, required=True)
    parser.add_argument("--is_sequential_editing", type=bool, default=True)
    parser.add_argument("--data_dir", type=str, default="/dataset")
    parser.add_argument("--output_dir", type=str, default=None, required=True)
    parser.add_argument("--editing_dataset", type=str, default=None, required=True, choices=['zsre', "cf"])
    parser.add_argument("--editing_dataset_size", type=int, default=None)
    
    
    args = parser.parse_args()
    return args
    
    
def main():
    
    args = editing_parse_arg()
    dataset_cls = get_dataset_class(args.editing_dataset)
    editor_hyperparameter_class = get_method_hyperparameter_class(args.editing_method)  
    edit_ds = dataset_cls(size=args.editing_dataset_size)
    print(f"[INFO] The editing method is {args.editing_method} . \n")
    print(f"[INFO] The size of editing dataset is {len(edit_ds)} . \n")
    hparams = editor_hyperparameter_class.from_hparams(args.hparams_file)
    roll_back_weight = not args.is_sequential_editing
    
    
    if not roll_back_weight:
        print("[INFO] Performing Sequential Editing. \n")
    else:
        print("[INFO] Performing Single Editing. \n")
        
    print(f'[INFO] Editing {hparams.model_name} Model. \n')
    
    

        
    if args.editing_method in ["MEND", "SERAC"]:
                
            if (not hasattr(hparams, "archive") and args.editing_method == "IKE") or (hparams.archive is None) :
                print(f"Pre-training for {args.editing_method}")
                train_path, eval_path = get_mend_dataset_path(args.editing_dataset)
                train_ds = dataset_cls(train_path, config=hparams)
                eval_ds = dataset_cls(eval_path, config=hparams)
                trainer = EditTrainer(
                    config=hparams,
                    train_set=train_ds,
                    val_set=eval_ds
                )
                trainer.run()
                
   
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        edit_ds,
        keep_original_weight=roll_back_weight
    )
    
    
    if isinstance(edited_model, PreTrainedModel):
        returned_model = edited_model
    elif hasattr(edited_model, "model"):
        returned_model = edited_model.model
    else:
        raise NotImplementedError
    
    returned_model = returned_model.cpu()
    returned_tokenizer = editor.tok
    torch.cuda.empty_cache()
    
    create_or_clear_dir(args.output_dir)

    json.dump(metrics, open(os.path.join(args.output_dir, "result" , f'{args.model_name}_{args.editing_method}_results.json'), 'w'), indent=4)
    
    if args.editing_method not in ["GRACE", "SERAC"]:
        returned_model.save_pretrained(os.path.join(args.output_dir, "final_model"))
        returned_tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    print("[INFO] Editing Finish, the edited model is saved at ", args.output_dir)

if __name__ == "__main__":
    main()