<div align="center">
  <img src="https://github.com/open-compass/MixtralKit/assets/7881589/149f8930-3a34-49b6-b27d-79dc192aeac7" width="500px"/>
  
  # MixtralKit

  A Toolkit for Mixtral Model

  <a href="#-performance">📊Performance </a> •
  <a href="#-resources">✨Resources </a> •
  <a href="#-model-architecture">📖Architecture </a> •
  <a href="#-model-weights">📂Weights </a> •
  <a href="#-install"> 🔨 Install </a> •
  <a href="#-inference">🚀Inference </a> •
  <a href="#-acknowledgement">🤝 Acknowledgement </a>

  <br />
  <br />

  English | [简体中文](README_zh-CN.md)

</div>


> [!Important]
> <div align="center">
> <b>
> 📢 Welcome to try <a href="https://github.com/open-compass/opencompass">OpenCompass</a> for model evaluation 📢
> </b>
> <br>
> <b>
> 🤗 Request for update your mixtral-related projects is open</a>!
> </b>
> <br>
> <b>
> 🙏 This repo is an **experimental** implementation of inference code.
> </b>
> </div>





# 📊 Performance

## Comparison with Other Models

- All data generated from [OpenCompass](https://github.com/open-compass/opencompass)

> Performances generated from different evaluation toolkits are different due to the prompts, settings and implementation details.



| Datasets        | Mode | Mistral-7B-v0.1 | Mixtral-8x7B(MoE) |  Llama2-70B | DeepSeek-67B-Base | Qwen-72B | 
|-----------------|------|-----------------|--------------|-------------|-------------------|----------|
| Active Params   |  -   |      7B         |     12B      |     70B     |       67B         |   72B    |
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

# ✨ Resources

## Blog
- [MoE Blog from Hugging Face](https://huggingface.co/blog/moe)
- [Enhanced MoE Parallelism, Open-source MoE Model Training Can Be 9 Times More Efficient](https://www.hpc-ai.tech/blog/enhanced-moe-parallelism-open-source-moe-model-training-can-be-9-times-more-efficient)

## Papers

|  Title  |   Venue  |   Date   |   Code   |   Demo   |
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

## Evaluation

- [x] Evaluation Toolkit: [OpenCompass](https://github.com/open-compass/opencompass)

## Training
- Megablocks: https://github.com/stanford-futuredata/megablocks
- FairSeq: https://github.com/facebookresearch/fairseq/tree/main/examples/moe_lm
- OpenMoE: https://github.com/XueFuzhao/OpenMoE
- ColossalAI MoE: https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/openmoe
- FastMoE(FasterMoE): https://github.com/laekov/FastMoE
- SmartMoE: https://github.com/zms1999/SmartMoE

## Fine-tuning

- [x] Finetuning script (Full-parameters or QLoRA) from [XTuner](https://github.com/InternLM/xtuner/tree/main/xtuner/configs/mixtral) 
- [x] Finetuned Mixtral-8x7B from DiscoResearch: [DiscoLM-mixtral-8x7b-v2](https://huggingface.co/DiscoResearch/DiscoLM-mixtral-8x7b-v2)

## Deployment

- [x] [Inference with vLLM](https://github.com/vllm-project/vllm)

# 📖 Model Architecture

>  The Mixtral-8x7B-32K MoE model is mainly composed of 32 identical MoEtransformer blocks. The main difference between the MoEtransformer block and the ordinary transformer block is that the FFN layer is replaced by the **MoE FFN** layer. In the MoE FFN layer, the tensor first goes through a gate layer to calculate the scores of each expert, and then selects the top-k experts from the 8 experts based on the expert scores. The tensor is aggregated through the outputs of the top-k experts, thereby obtaining the final output of the MoE FFN layer. Each expert consists of 3 linear layers. It is worth noting that all Norm Layers of Mixtral MoE also use RMSNorm, which is the same as LLama. In the attention layer, the QKV matrix in the Mixtral MoE has a Q matrix shape of (4096,4096) and K and V matrix shapes of (4096,1024).

We plot the architecture as the following:

<div align="center">
  <img src="https://github.com/open-compass/MixtralKit/assets/7881589/0bd59661-4799-4e39-8a92-95fd559679e9" width="800px"/>
</div>

# 📂 Model Weights

## Hugging Face Format

- [Official Base Model](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [Official Chat Model](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)

## Raw Format

You can download the checkpoints by magnet or Hugging Face

### Download via HF

- [mixtral-8x7b-32kseqlen](https://huggingface.co/someone13574/mixtral-8x7b-32kseqlen)

> If you are unable to access Hugging Face, please try [hf-mirror](https://hf-mirror.com/someone13574/mixtral-8x7b-32kseqlen)


```bash
# Download the Hugging Face
git lfs install
git clone https://huggingface.co/someone13574/mixtral-8x7b-32kseqlen

# Merge Files(Only for HF)
cd mixtral-8x7b-32kseqlen/

# Merge the checkpoints
cat consolidated.00.pth-split0 consolidated.00.pth-split1 consolidated.00.pth-split2 consolidated.00.pth-split3 consolidated.00.pth-split4 consolidated.00.pth-split5 consolidated.00.pth-split6 consolidated.00.pth-split7 consolidated.00.pth-split8 consolidated.00.pth-split9 consolidated.00.pth-split10 > consolidated.00.pth
```

### Download via Magnet Link

Please use this link to download the original files
```bash
magnet:?xt=urn:btih:5546272da9065eddeb6fcd7ffddeef5b75be79a7&dn=mixtral-8x7b-32kseqlen&tr=udp%3A%2F%http://2Fopentracker.i2p.rocks%3A6969%2Fannounce&tr=http%3A%2F%http://2Ftracker.openbittorrent.com%3A80%2Fannounce
```
### MD5 Validation

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

# 🔨 Install

```bash
conda create --name mixtralkit python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate mixtralkit

git clone https://github.com/open-compass/MixtralKit
cd MixtralKit/
pip install -r requirements.txt
pip install -e .

ln -s path/to/checkpoints_folder/ ckpts
```

# 🚀 Inference

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


# 🏗️ Evaluation

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

# 🤝 Acknowledgement

- [llama-mistral](https://github.com/dzhulgakov/llama-mistral)
- [llama](https://github.com/facebookresearch/llama)

# 🖊️ Citation


```latex
@misc{2023opencompass,
    title={OpenCompass: A Universal Evaluation Platform for Foundation Models},
    author={OpenCompass Contributors},
    howpublished = {\url{https://github.com/open-compass/opencompass}},
    year={2023}
}
```
