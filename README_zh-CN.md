<div align="center">
  <img src="https://github.com/open-compass/MixtralKit/assets/7881589/149f8930-3a34-49b6-b27d-79dc192aeac7" width="500px"/>

  # MixtralKit
  
  Mixtral 模型工具箱

  <a href="#-性能">📊性能 </a> •
  <a href="#-社区项目">✨社区项目 </a> •
  <a href="#-模型架构">📖模型架构 </a> •
  <a href="#-模型权重">📂模型权重 </a> •
  <a href="#-安装">🔨安装 </a> •
  <a href="#-推理">🚀推理 </a> •
  <a href="#-致谢">🤝致谢 </a>

  <br />
  <br />

  [English](/README.md) | 简体中文

</div>


> [!重要]
> <div align="center">
> <b>
> 📢欢迎试用 <a href="https://github.com/open-compass/opencompass">OpenCompass</a> 进行模型评测 📢
> </b>
> <br>
> <b>
> 🤗 欢迎将你的Mixtral相关的项目添加到README </a>!
> </b>
> <br>
> <b>
> 🙏 本仓库仅提供**实验性质**的推理代码
> </b>
> </div>




# 📊 性能


- 所有数据来源自[OpenCompass](https://github.com/open-compass/opencompass)

> 由于不同评测框架在提示词，评测设定和实现细节上均有所不同，所以请勿直接对比不同框架获得的评测结果。

## 性能对比


| 数据集        | Mode | Mistral-7B-v0.1 | Mixtral-8x7B(MoE) |  Llama2-70B | DeepSeek-67B-Base | Qwen-72B | 
|-----------------|------|-----------------|--------------|-------------|-------------------|----------|
| 激活参数   |  -   |      7B         |     12B      |     70B     |       67B         |   72B    |
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


## Mixtral-8x7b 性能

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
# ✨ 社区项目

## 博客
- [MoE Blog from HuggingFace](https://huggingface.co/blog/moe)
- [Enhanced MoE Parallelism, Open-source MoE Model Training Can Be 9 Times More Efficient](https://www.hpc-ai.tech/blog/enhanced-moe-parallelism-open-source-moe-model-training-can-be-9-times-more-efficient)

## 论文

|  题目  |   会议/期刊  |   日期   |   代码   |   示例   |
|:--------|:--------:|:--------:|:--------:|:--------:|
|[Mixture-of-Experts Meets Instruction Tuning:A Winning Combination for Large Language Models](https://arxiv.org/abs/2305.14705)           | Arxiv       | 23.05 | | 
|[MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841)                                         | Arxiv       | 22.11 | [megablocks](https://github.com/stanford-futuredata/megablocks) | |
|[ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906)                                        | Arxiv       | 22.02 |
|[Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)          | Arxiv       | 21.01 |
|[GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/abs/2112.06905)                                    | ICML 2022   | 21.12 |
|[GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)                      | Arxiv       | 20.06 |
|[Learning Factored Representations in a Deep Mixture of Experts](https://arxiv.org/abs/1312.4314)                                         | Arxiv       | 13.12 |
|[FastMoE: A Fast Mixture-of-Expert Training System](https://arxiv.org/abs/2103.13262)   | Arxiv | 21.03 | [FastMoE](https://github.com/laekov/FastMoE)|
|[FasterMoE: Modeling and Optimizing Training of Large-scale Dynamic Pre-trained Models](https://dl.acm.org/doi/10.1145/3503221.3508418)   | ACM SIGPLAN PPoPP 2022 | 22.03 | [FasterMoE](https://github.com/laekov/FastMoE)|
|[SmartMoE: Efficiently Training Sparsely-Activated Models through Combining Offline and Online Parallelization](https://www.usenix.org/conference/atc23/presentation/zhai)   | USENIX ATC 2023 | 22.03 | [SmartMoE](https://github.com/zms1999/SmartMoE)|
|[Adaptive Mixture of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)                                                  | Neural Computation | 1991 |

## 评测
- [x] 评测工具 [OpenCompass](https://github.com/open-compass/opencompass)

## 训练
- Megablocks: https://github.com/stanford-futuredata/megablocks
- FairSeq: https://github.com/facebookresearch/fairseq/tree/main/examples/moe_lm
- OpenMoE: https://github.com/XueFuzhao/OpenMoE
- ColossalAI MoE: https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/openmoe
- FastMoE(FasterMoE): https://github.com/laekov/FastMoE
- SmartMoE: https://github.com/zms1999/SmartMoE

## 微调
- [x] 使用XTuner微调Mixtral-8x7B 方案(全参数/QLoRA): [XTuner](https://github.com/InternLM/xtuner/tree/main/xtuner/configs/mixtral) 
- [x] 微调模型Mixtral-8x7B(DiscoResearch): [DiscoLM-mixtral-8x7b-v2](https://huggingface.co/DiscoResearch/DiscoLM-mixtral-8x7b-v2)

## 部署

TBD

# 📖 模型架构

> Mixtral-8x7B-32K MoE模型主要由32个相同的MoEtransformer block组成。MoEtransformer block与普通的transformer block的最大差别在于其FFN层替换为了**MoE FFN**层。在MoE FFN层，tensor首先会经过一个gate layer计算每个expert的得分，并根据expert得分从8个expert中挑出top-k个expert，将tensor经过这top-k个expert的输出后聚合起来，从而得到MoE FFN层的最终输出，其中的每个expert由3个Linear层组成。值得注意的是，mixtral MoE的所有Norm Layer也采用了和LLama一样的RMSNorm，而在attention layer中，mixtral MoE的QKV矩阵中的Q矩阵shape为(4096,4096)，K和V矩阵shape则为(4096,1024)。

模型结构图如下:

<div align="center">
  <img src="https://github.com/open-compass/MixtralKit/assets/7881589/0bd59661-4799-4e39-8a92-95fd559679e9" width="800px"/>
</div>


# 📂 模型权重

## HuggingFace 格式

- [官方基座模型 Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [官方对话模型 Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)

## 原生格式

你可以通过使用磁力链接(迅雷)或使用HuggingFace进行下载

### 使用HF下载

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

# Merge Files(Only for HF)
cd mixtral-8x7b-32kseqlen/

# Merge the checkpoints
cat consolidated.00.pth-split0 consolidated.00.pth-split1 consolidated.00.pth-split2 consolidated.00.pth-split3 consolidated.00.pth-split4 consolidated.00.pth-split5 consolidated.00.pth-split6 consolidated.00.pth-split7 consolidated.00.pth-split8 consolidated.00.pth-split9 consolidated.00.pth-split10 > consolidated.00.pth
```

### 使用磁力链下载

```bash
magnet:?xt=urn:btih:5546272da9065eddeb6fcd7ffddeef5b75be79a7&dn=mixtral-8x7b-32kseqlen&tr=udp%3A%2F%http://2Fopentracker.i2p.rocks%3A6969%2Fannounce&tr=http%3A%2F%http://2Ftracker.openbittorrent.com%3A80%2Fannounce
```


### 文件校验

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

# 🔨 安装

```bash
git clone https://github.com/open-compass/MixtralKit
cd MixtralKit/
pip install -r requirements.txt
pip install -e .

ln -s path/to/checkpoints ckpts
```

# 🚀 推理

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


# 🏗️ 评测

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

# 🤝 致谢
- [llama-mistral](https://github.com/dzhulgakov/llama-mistral)
- [llama](https://github.com/facebookresearch/llama)

# 🖊️ 引用


```latex
@misc{2023opencompass,
    title={OpenCompass: A Universal Evaluation Platform for Foundation Models},
    author={OpenCompass Contributors},
    howpublished = {\url{https://github.com/open-compass/opencompass}},
    year={2023}
}
```
