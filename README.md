# MixtralKit

English | [简体中文](README_zh-CN.md)

A Toolkit for Mixtral Model

> Welcome to try [OpenCompass](https://github.com/open-compass/opencompass) for model evaluation, performance of Mixtral will be updated soon.

> This repo is an experimental implementation of inference code, which is **not officially released** by Mistral AI.

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
git clone https://github.com/open-compass/MixtralKit
cd MixtralKit/
pip install -r requirements.txt
pip install -e .

ln -s path/to/checkpoints ckpts
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

==============================Example START==============================

[Prompt]:
请问你是什么模型？

[Response]:
“先知”姚振杰“心灵信使”“八球先生”吗？

Now, this should work. We are going to change the default agent role so that now the user “agent” will be able to send emails on behalf of gmail or any other

==============================Example END==============================
```


# Evaluation with OpenCompass

Coming Soon....

# Acknowledgement
- [llama-mistral](https://github.com/dzhulgakov/llama-mistral)
- [llama](https://github.com/facebookresearch/llama)

