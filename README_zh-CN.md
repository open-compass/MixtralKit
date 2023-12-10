<div align="center">
  
  <font size=6>MixtralKit</font>
  
  Mixtral 模型工具箱

  <br />
  <br />

  [English](/README.md) | 简体中文

</div>




> 欢迎试用 [OpenCompass](https://github.com/open-compass/opencompass) 进行模型评测，Mixtral模型性能将会很快更新。

> 本仓库仅提供实验性质的推理代码，非Mistral AI官方提供的示例代码。


# 性能

马上更新

# 准备模型权重

## 下载模型权重

你可以通过使用磁力链接(迅雷)或使用HuggingFace进行下载

### HuggingFace

社区用户提供的HF文件切分版：[HuggingFace仓库](https://huggingface.co/someone13574/mixtral-8x7b-32kseqlen)

```bash
# Download the huggingface
git lfs install
git clone https://huggingface.co/someone13574/mixtral-8x7b-32kseqlen

```
> 用户如果无法访问huggingface, 可以使用[国内镜像](https://hf-mirror.com/someone13574/mixtral-8x7b-32kseqlen)

```bash
# Download the huggingface
git lfs install
git clone https://hf-mirror.com/someone13574/mixtral-8x7b-32kseqlen
```

### Magnet Link

Please use this link to download the original files
```bash
magnet:?xt=urn:btih:5546272da9065eddeb6fcd7ffddeef5b75be79a7&dn=mixtral-8x7b-32kseqlen&tr=udp%3A%2F%http://2Fopentracker.i2p.rocks%3A6969%2Fannounce&tr=http%3A%2F%http://2Ftracker.openbittorrent.com%3A80%2Fannounce
```

## 文件拼接(仅HF格式需要)

```bash
cd mixtral-8x7b-32kseqlen/

# Merge the checkpoints
cat consolidated.00.pth-split0 consolidated.00.pth-split1 consolidated.00.pth-split2 consolidated.00.pth-split3 consolidated.00.pth-split4 consolidated.00.pth-split5 consolidated.00.pth-split6 consolidated.00.pth-split7 consolidated.00.pth-split8 consolidated.00.pth-split9 consolidated.00.pth-split10 > consolidated.00.pth
```


## 文件校验

请在使用文件前，进行md5校验，保证文件在下载过程中并未损坏
```bash
md5sum consolidated.00.pth
md5sum tokenizer.model

# 如果完成校验，可删除slit文件
rm consolidated.00.pth-split*
```

官方校验值

```bash
 ╓────────────────────────────────────────────────────────────────────────────╖
 ║                                                                            ║
 ║                               ·· md5sum ··                                 ║
 ║                                                                            ║
 ║        1faa9bc9b20fcfe81fcd4eb7166a79e6  consolidated.00.pth               ║
 ║        37974873eb68a7ab30c4912fc36264ae  tokenizer.model                   ║
 ╙────────────────────────────────────────────────────────────────────────────╜
```

# 安装

```bash
git clone https://github.com/open-compass/MixtralKit
cd MixtralKit/
pip install -r requirements.txt
pip install -e .

ln -s path/to/checkpoints ckpts
```

# 示例

## 文本补全

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


# 使用OpenCompass评测

## 第一步: 配置OpenCompass

- 克隆和安装 OpenCompass

```bash
# assume you have already create the conda env named mixtralkit 
conda activate mixtralkit

git clone https://github.com/open-compass/opencompass opencompass
cd opencompass

pip install -e .
```

- 准备评测数据集

```bash
# Download dataset to data/ folder
wget https://github.com/open-compass/opencompass/releases/download/0.1.8.rc1/OpenCompassData-core-20231110.zip
unzip OpenCompassData-core-20231110.zip
```

> If you need to evaluate the **humaneval**, please go to [Installation Guide](https://opencompass.readthedocs.io/en/latest/get_started/installation.html) for more information


## 第二步: 准备评测配置文件和数据集

```bash
cd opencompass/
# link the example config into opencompass
ln -s path/to/MixtralKit/playground playground

# link the model weights into opencompass
mkdir -p ./models/mixtral/
ln -s path/to/checkpoints_folder/ ./models/mixtral/mixtral-8x7b-32kseqlen
```

现在文件结构应该如下所示

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


## 第三步：执行评测

```bash
HF_EVALUATE_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python run.py playground/eval_mixtral.py

# 请编辑playground/eval_mixtral.py来配置希望评测的数据集

```

# 致谢
- [llama-mistral](https://github.com/dzhulgakov/llama-mistral)
- [llama](https://github.com/facebookresearch/llama)

