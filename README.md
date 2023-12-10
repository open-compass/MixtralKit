<div align="center">
  
  <font size=6>MixtralKit</font>
  
        A Toolkit for Mixtral Model

  <br />
  <br />

        English | [简体中文](README_zh-CN.md)

</div>

> Welcome to try [OpenCompass](https://github.com/open-compass/opencompass) for model evaluation, performance of Mixtral will be updated soon.

> This repo is an experimental implementation of inference code, which is **not officially released** by Mistral AI.

# Performance


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

# Example

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

