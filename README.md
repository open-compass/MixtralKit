# MixtralKit

English | [简体中文](README_zh-CN.md)

A Toolkit for Mixtral Model

> Welcome to try [OpenCompass](https://github.com/open-compass/opencompass) for model evaluation, performance of Mixtral will be updated soon.

# Download Models

You can download the checkpoints by magnet or huggingface

## Magnet Link

Please use this link to download the original files
```bash
magnet:?xt=urn:btih:5546272da9065eddeb6fcd7ffddeef5b75be79a7&dn=mixtral-8x7b-32kseqlen&tr=udp%3A%2F%http://2Fopentracker.i2p.rocks%3A6969%2Fannounce&tr=http%3A%2F%http://2Ftracker.openbittorrent.com%3A80%2Fannounce
```

## [HuggingFace](https://huggingface.co/someone13574/mixtral-8x7b-32kseqlen)

```bash
# Download the huggingface
git lfs install
git clone https://huggingface.co/someone13574/mixtral-8x7b-32kseqlen

cd mixtral-8x7b-32kseqlen/

# Merge the checkpoints
cat consolidated.00.pth-split0 consolidated.00.pth-split1 consolidated.00.pth-split2 consolidated.00.pth-split3 consolidated.00.pth-split4 consolidated.00.pth-split5 consolidated.00.pth-split6 consolidated.00.pth-split7 consolidated.00.pth-split8 consolidated.00.pth-split9 consolidated.00.pth-split10 > consolidated.00.pth
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
===============START===============

[Prompt]:
Who Are You?

[Response]:
{'generation': '\n\nI often hear of my elders talking about the importance of knowing where you come from. This talk is most prevalent in the Afri
can-American community but I imagine other groups of people could make the same claim. I have also heard “this is why the younger generation is lo
st,” and “if you know'}

===============END===============

===============START===============

[Prompt]:
1 + 1 -> 3
2 + 2 -> 5
3 + 3 -> 7
4 + 4 ->

[Response]:
{'generation': '9\n5 + 5 -> 11\n6 + 6 -> 13\n\n#include <iostream>\n\nusing namespace std;\n\nint addNumbers(int x, int y)\n{\n\treturn x + y;\n}\n\nint main()\n{'}

===============END===============


===============START===============

[Prompt]:
请问你是什么模型？

[Response]:
{'generation': '“先知”姚振杰“心灵信使”“八球先生”吗？\n\nNow, this should work. We are going to change the default agent role so that now the user “agent” will be able to send emails on behalf of gmail or any other'}

===============END===============
```


# Evaluation with OpenCompass

Coming Soon....

# Acknowledgement
- [llama-mistral](https://github.com/dzhulgakov/llama-mistral)
- [llama](https://github.com/facebookresearch/llama)

