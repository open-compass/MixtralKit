<div align="center">
  
  # MixtralKit
  
  A Toolkit for Mixtral Model

  <br />
  <br />

  English | [简体中文](README_zh-CN.md)

</div>

> Welcome to try [OpenCompass](https://github.com/open-compass/opencompass) for model evaluation, performance of Mixtral will be updated soon.

> This repo is an experimental implementation of inference code, which is **not officially released** by Mistral AI.


- [Performance](#performance)
- [Prepare Model Weights](#prepare-model-weights)
  - [Download Weights](#download-weights)
  - [Merge Files](#merge-filesonly-for-hf)
  - [MD5 Validation](#md5-validation)
- [Install](#install)
- [Inference](#inference)
  - [Text Completion](#text-completion)
- [Evaluation with OpenCompass](#evaluation-with-opencompass)
  - [Step-1: Setup OpenCompass](#step-1-setup-opencompass)
  - [Step-2: Pre-pare evaluation config and weights](#step-2-pre-pare-evaluation-config-and-weights)
  - [Step-3: Run evaluation experiments](#step-3-run-evaluation-experiments)
- [Acknowledgement](#acknowledgement)


# Performance

## Comparison with Other Models

- All data generated from [OpenCompass](https://github.com/open-compass/opencompass)

> Performances generated from different evaluation toolkits are different due to the prompts, settings and implementation details.


| Datasets        | Mode | Mistral-7B-v0.1 | Mixtral-8x7B |  Llama2-70B | DeepSeek-67B-Base | Qwen-72B | 
|-----------------|------|-----------------|--------------|-------------|-------------------|----------|
| MMLU            | PPL  | 64.1            | 71.3         | 69.7        | 71.9              | 77.3     |
| BIG-Bench-Hard  | GEN  | 56.7            | 67.1         | 64.9        | 71.7              | 63.7     |
| GSM-8K          | GEN  | 47.5            | 65.7         | 63.4        | 66.5              | 77.6     |
| MATH            | GEN  | 11.3            | 22.7         | 12.0        | 15.9              | 35.1     |
| HumanEval       | GEN  | 27.4            | 32.3         | 26.2        | 40.9              | 33.5     |
| MBPP            | GEN  | 38.6            | 47.8         | 39.6        | 55.2              | 51.6     |
| ARC-c           | PPL  | 74.2            | 85.1         | 78.3        | 86.8              | 92.2     |
| ARC-e           | PPL  | 83.6            | 91.4         | 85.9        | 93.7              | 96.8     |
| CommonSenseQA   | PPL  | 67.4            | 70.4         | 78.3        | 70.7              | 73.9     |
| NaturalQuestion | GEN  | 24.6            | 29.4         | 34.2        | 29.9              | 27.1     |
| TrivialQA       | GEN  | 56.5            | 66.1         | 70.7        | 67.4              | 60.1     |
| HellaSwag       | PPL  | 78.9            | 82.0         | 82.3        | 82.3              | 85.4     |
| PIQA            | PPL  | 81.6            | 82.9         | 82.5        | 82.6              | 85.2     |
| SIQA            | GEN  | 60.2            | 64.3         | 64.8        | 62.6              | 78.2     |


## Performance Mixtral-8x7b

```markdown
dataset                                 version    metric         mode    mixtral-8x7b-32k
--------------------------------------  ---------  -------------  ------  ------------------
mmlu                                    -          naive_average     ppl     71.34
ARC-c                                   2ef631     accuracy          ppl     85.08
ARC-e                                   2ef631     accuracy          ppl     91.36
BoolQ                                   314797     accuracy          ppl     86.27
commonsense_qa                          5545e2     accuracy          ppl     70.43
triviaqa                                2121ce     score             gen     66.05
nq                                      2121ce     score             gen     29.36
openbookqa_fact                         6aac9e     accuracy          ppl     85.40
AX_b                                    6db806     accuracy          ppl     48.28
AX_g                                    66caf3     accuracy          ppl     48.60
hellaswag                               a6e128     accuracy          ppl     82.01
piqa                                    0cfff2     accuracy          ppl     82.86
siqa                                    e8d8c5     accuracy          ppl     64.28
math                                    265cce     accuracy          gen     22.74
gsm8k                                   1d7fe4     accuracy          gen     65.66
openai_humaneval                        a82cae     humaneval_pass@1  gen     32.32
mbpp                                    1e1056     score             gen     47.80
bbh                                     -          naive_average     gen     67.14
```


# Prepare Model Weights

## Download Weights
You can download the checkpoints by magnet or huggingface


### HuggingFace

- [mixtral-8x7b-32kseqlen](https://huggingface.co/someone13574/mixtral-8x7b-32kseqlen)

> If you are unable to access huggingface, please try [hf-mirror](https://hf-mirror.com/someone13574/mixtral-8x7b-32kseqlen)


```bash
# Download the huggingface
git lfs install
git clone https://huggingface.co/someone13574/mixtral-8x7b-32kseqlen

```

### Magnet Link

Please use this link to download the original files
```bash
magnet:?xt=urn:btih:5546272da9065eddeb6fcd7ffddeef5b75be79a7&dn=mixtral-8x7b-32kseqlen&tr=udp%3A%2F%http://2Fopentracker.i2p.rocks%3A6969%2Fannounce&tr=http%3A%2F%http://2Ftracker.openbittorrent.com%3A80%2Fannounce
```
## Merge Files(Only for HF)

```bash

cd mixtral-8x7b-32kseqlen/

# Merge the checkpoints
cat consolidated.00.pth-split0 consolidated.00.pth-split1 consolidated.00.pth-split2 consolidated.00.pth-split3 consolidated.00.pth-split4 consolidated.00.pth-split5 consolidated.00.pth-split6 consolidated.00.pth-split7 consolidated.00.pth-split8 consolidated.00.pth-split9 consolidated.00.pth-split10 > consolidated.00.pth
```

## MD5 Validation

Please check the MD5 to make sure the files are completed.

```bash
md5sum consolidated.00.pth
md5sum tokenizer.model

# Once verified, you can delete the splited files.
rm consolidated.00.pth-split*
```

Official MD5


```bash
 ╓────────────────────────────────────────────────────────────────────────────╖
 ║                                                                            ║
 ║                               ·· md5sum ··                                 ║
 ║                                                                            ║
 ║        1faa9bc9b20fcfe81fcd4eb7166a79e6  consolidated.00.pth               ║
 ║        37974873eb68a7ab30c4912fc36264ae  tokenizer.model                   ║
 ╙────────────────────────────────────────────────────────────────────────────╜
```

# Install

```bash
conda create --name mixtralkit python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate mixtralkit

git clone https://github.com/open-compass/MixtralKit
cd MixtralKit/
pip install -r requirements.txt
pip install -e .

ln -s path/to/checkpoints_folder/ ckpts
```

# Inference

## Text Completion 
```bash
python tools/example.py -m ./ckpts -t ckpts/tokenizer.model --num-gpus 2
```

Expected Results:

```bash
==============================Example START==============================

[Prompt]:
Who are you?

[Response]:
I am a designer and theorist; a lecturer at the University of Malta and a partner in the firm Barbagallo and Baressi Design, which won the prestig
ious Compasso d’Oro award in 2004. I was educated in industrial and interior design in the United States

==============================Example END==============================

==============================Example START==============================

[Prompt]:
1 + 1 -> 3
2 + 2 -> 5
3 + 3 -> 7
4 + 4 ->

[Response]:
9
5 + 5 -> 11
6 + 6 -> 13

#include <iostream>

using namespace std;

int addNumbers(int x, int y)
{
        return x + y;
}

int main()
{

==============================Example END==============================

```


# Evaluation with OpenCompass

## Step-1: Setup OpenCompass

- Clone and Install OpenCompass

```bash
# assume you have already create the conda env named mixtralkit 
conda activate mixtralkit

git clone https://github.com/open-compass/opencompass opencompass
cd opencompass

pip install -e .
```

- Prepare Evaluation Dataset

```bash
# Download dataset to data/ folder
wget https://github.com/open-compass/opencompass/releases/download/0.1.8.rc1/OpenCompassData-core-20231110.zip
unzip OpenCompassData-core-20231110.zip
```

> If you need to evaluate the **humaneval**, please go to [Installation Guide](https://opencompass.readthedocs.io/en/latest/get_started/installation.html) for more information


## Step-2: Pre-pare evaluation config and weights

```bash
cd opencompass/
# link the example config into opencompass
ln -s path/to/MixtralKit/playground playground

# link the model weights into opencompass
mkdir -p ./models/mixtral/
ln -s path/to/checkpoints_folder/ ./models/mixtral/mixtral-8x7b-32kseqlen
```

Currently, you should have the files structure like:

```bash

opencompass/
├── configs
│   ├── .....
│   └── .....
├── models
│   └── mixtral
│       └── mixtral-8x7b-32kseqlen
├── data/
├── playground
│   └── eval_mixtral.py
│── ......
```


## Step-3: Run evaluation experiments

```bash
HF_EVALUATE_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python run.py playground/eval_mixtral.py
```

# Acknowledgement
- [llama-mistral](https://github.com/dzhulgakov/llama-mistral)
- [llama](https://github.com/facebookresearch/llama)

