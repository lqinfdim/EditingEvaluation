# EditingEvaluation

This is the official code for NeurIPS24 Paper "Should We Really Edit Language Models? On the Evaluation of Edited Language Models."

## Create Conda Environment

```
conda env create -f env.yml

```

and then

```
conda activate editingeval

```

## Config 

### Dataset
Before running editing, you must download the dataset and set the path in the edit.py (line 54-55)

### Cache

Setting KV and STAT cache dir in global.yml

### Output Dir

Setting the output dir in run_edit.sh or run_edit_batch.sh

## Running Experiments

After activate the editingeval environment, you can run edit with

```
bash run_edit.sh
```