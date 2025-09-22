![image](https://github.com/runtsang/MEPT/blob/master/imgs/mept.jpg)

**Official implementation of EMNLP "MEPT: Mixture of Expert Prompt Tuning as a Manifold Mapper"**

**Contact me:** [runjia.tech](https://runjia.tech/) | rz4545@rit.edu | runjia@msn.com

[Paper](https://arxiv.org/abs/2509.00996) | [Homepage](https://runjia.tech/emnlp_mept/)

# 📣News

(👉Under construction! There are several redundancies in the current version, and the commands/instructions are not perfectly ready for formal release. I will gradually update it! Please stay tuned.)

**2024/09/14:** Our [homepage](https://runjia.tech/emnlp_mept/) is available now (slides and video are on the way)! Check it out to see more details.

**2025/09/13:** Our code is publicly available now! Thank you for your attention and patience!

## Install required modules 

This codebase is implemented on Python 3.9.18. 
Run 

```
pip install -r requirements.txt
```
to download all required modules.

Attention: we are using the local transformer library (version 4.43.2) instead of the latest release.

In `./transformers`

Run

```
pip install -e .
```

## Downloading the Dataset
Download the SuperGLUE datasets by
```
python data/superglue/get_huggingface_superglue.py
```
or use your custom dataset. In that case, you need to create your custom `Dataset` class for your dataset in `src/dataset.py` and apply mandatory changes such as importing your dataset or modifying the training script.

## Training
![image](https://github.com/runtsang/MEPT/blob/master/imgs/overview.jpg)

Then, you can execute `scripts/train.py` with training arguments as follows

```
python scripts/train.py \
--lr {lr} \
--batch_size {bs}  \
--epoch {epoch} \
--max_length 512 \
# t5-base, t5-large, meta-llama/Llama-3.2-1B
--model_name_or_path {name} \
--tokenizer_name_or_path {name2} \
--warmup_ratio 0.06 \
--method prompt-expert \
# '"[\'cb\', \'copa\', \'rte\', \'boolq\', \'wic\', \'multirc\']"' for mixed training
# 'boolq', 'cb', 'copa', 'multirc', 'rte', 'wic' for vanilla training
--dataset_name {dataset} \ 
--num_virtual_tokens {tokens} \
--num_virtual_tokens_full {tokens_full} \
--perturb_router True \
--topk {topk} \
--comment "{comment}" \
--losstrack {track} \
--gumbel {gumbel} \
--layers "{layers}" \
--txt {txt}
```

### Arguments
- `method`: The training method
  - `full`: Full model fine-tuning
  - `prompt-tuning`: Directly fine-tuning the soft prompts (from [Lester et al., 2021](https://aclanthology.org/2021.emnlp-main.243/))
  - `p-tuning`: Utilizing a reparameterization model on the soft prompts (from [Liu et al, 2021](https://arxiv.org/abs/2103.10385))
  - `prompt-routing`: Use [SMoP](https://github.com/jyjohnchoi/SMoP) for training
  - `prompt-expert`: Use MEPT for training
  
- `num_virtual_tokens`: The number of the soft prompt tokens attached to the input instance. No impact when the training method is `full`
- `num_virtual_tokens_full`: The total number of soft prompt tokens used during training. For `prompt-routing`, this is different from 'num_virtual_tokens', while it is the same on other methods.
  - For example, if you want to use [SMoP](https://github.com/jyjohnchoi/SMoP) with 4 soft prompts of length 5, you need to set `num_virtual_tokens` as 5 and `num_virtual_tokens_full` as 20.

- `perturb_router`: If True, scaled Gaussian noise is applied during training.

- `topk`: Number of soft prompt tokens to route each input instance. If larger than 2, the weighted sum of multiple soft prompts is applied.

- `shared_expert`: Indicates whether a shared expert is used in each layer.

- `shared_expert_ratio`: Specifies the number of shared experts per layer. The default value is 1.
- `layers`: Defines the layers where prompts are inserted. The default applies to all layers.


## Citation
```
@inproceedings{zeng2025mept,
  title={MEPT: Mixture of Expert Prompt Tuning as a Manifold Mapper},
  author={Zeng, Runjia and Sun, Guangyan and Wang, Qifan and Geng, Tong and Dianat, Sohail and Han, Xiaotian and Rao, Raghuveer and Zhang, Xueling and Han, Cheng and Huang, Lifu and others},
  booktitle={EMNLP},
  year={2025}
}
```

## Acknowledgments
The documentation above and code are copied and modified  from [SMoP](https://github.com/jyjohnchoi/SMoP). Thanks for their effort.

Our implementation is largely based on the [HuggingFace PEFT](https://github.com/huggingface/peft) library.
