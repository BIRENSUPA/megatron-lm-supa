---
license: other
license_name: tongyi-qianwen
license_link: https://huggingface.co/Qwen/Qwen2-72B/blob/main/LICENSE
language:
- en
pipeline_tag: text-generation
tags:
- pretrained
---

# Qwen2-72B

## Introduction

Qwen2 is the new series of Qwen large language models. For Qwen2, we release a number of base language models and instruction-tuned language models ranging from 0.5 to 72 billion parameters, including a Mixture-of-Experts model. This repo contains the 72B Qwen2 base language model.

Compared with the state-of-the-art opensource language models, including the previous released Qwen1.5, Qwen2 has generally surpassed most opensource models and demonstrated competitiveness against proprietary models across a series of benchmarks targeting for language understanding, language generation, multilingual capability, coding, mathematics, reasoning, etc.

For more details, please refer to our [blog](https://qwenlm.github.io/blog/qwen2/), [GitHub](https://github.com/QwenLM/Qwen2), and [Documentation](https://qwen.readthedocs.io/en/latest/).
<br>


## Model Details
Qwen2 is a language model series including decoder language models of different model sizes. For each size, we release the base language model and the aligned chat model. It is based on the Transformer architecture with SwiGLU activation, attention QKV bias, group query attention, etc. Additionally, we have an improved tokenizer adaptive to multiple natural languages and codes.

## Requirements
The code of Qwen2 has been in the latest Hugging face transformers and we advise you to install `transformers>=4.37.0`, or you might encounter the following error:
```
KeyError: 'qwen2'
```


## Usage

We do not advise you to use base language models for text generation. Instead, you can apply post-training, e.g., SFT, RLHF, continued pretraining, etc., on this model.

You can use the following code for text completion:
```python
from modelscope import AutoModelForCausalLM, AutoTokenizer

device = "cuda"  # the device to load the model onto
model_name_or_path = "qwen/Qwen2-72B"
model = AutoModelForCausalLM.from_pretrained(
  model_name_or_path,
  torch_dtype="auto",
  device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

prefix = "北京是中国的首都"
model_inputs = tokenizer([prefix], return_tensors="pt").to(device)

generated_ids = model.generate(
  model_inputs.input_ids,
  max_new_tokens=400,
  repetition_penalty=1.15
)
generated_ids = [
  output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```


## Performance

The evaluation of base models mainly focuses on the model performance of natural language understanding, general question answering, coding, mathematics, scientific knowledge, reasoning, multilingual capability, etc. 

The datasets for evaluation include: 
 
**English Tasks**: MMLU (5-shot), MMLU-Pro (5-shot), GPQA (5shot), Theorem QA (5-shot), BBH (3-shot), HellaSwag (10-shot), Winogrande (5-shot), TruthfulQA (0-shot), ARC-C (25-shot)
 
**Coding Tasks**: EvalPlus (0-shot) (HumanEval, MBPP, HumanEval+, MBPP+), MultiPL-E (0-shot) (Python, C++, JAVA, PHP, TypeScript, C#, Bash, JavaScript)
  
**Math Tasks**: GSM8K (4-shot), MATH (4-shot)
 
**Chinese Tasks**: C-Eval (5-shot), CMMLU (5-shot)
 
**Multilingual Tasks**: Multi-Exam (M3Exam 5-shot, IndoMMLU 3-shot, ruMMLU 5-shot, mMMLU 5-shot), Multi-Understanding (BELEBELE 5-shot, XCOPA 5-shot, XWinograd 5-shot, XStoryCloze 0-shot, PAWS-X 5-shot), Multi-Mathematics (MGSM 8-shot), Multi-Translation (Flores-101 5-shot)
 
#### Qwen2-72B performance
|  Datasets  | DeepSeek-V2 | Mixtral-8x22B   |   Llama-3-70B  |   Qwen1.5-72B  |   Qwen1.5-110B  |  **Qwen2-72B**  |
| :--------| :---------: | :------------: | :------------: | :------------: | :------------: |:------------: |
|Architecture | MoE | MoE | Dense | Dense | Dense | Dense |
|#Activated Params | 21B | 39B | 70B | 72B | 110B | 72B |
|#Params | 236B | 140B | 70B | 72B | 110B   | 72B|
|   ***English***  |    |    |   |    |	    |	    |
|MMLU |78.5 | 77.8  | 79.5 | 77.5 | 80.4 |  **84.2**  |
|MMLU-Pro | - | 49.5  | 52.8 | 45.8 | 49.4 |  **55.6**  |
|GPQA | -| 34.3  | 36.3 | 36.3 | 35.9 |  **37.9**  |
|Theorem QA | -| 35.9  | 32.3 | 29.3 | 34.9 |  **43.1**  |
|BBH  | 78.9 |78.9   | 81.0 | 65.5 | 74.8 |  **82.4**  |
|HellaSwag  | 87.8 | **88.7**   | 88.0 |  86.0 | 87.5 | 87.6 |
|WindoGrande  | 84.8|85.0  |  **85.3**  |  83.0 | 83.5 |  85.1 |
|ARC-C  | 70.0| **70.7**   | 68.8 | 65.9 | 69.6 |  68.9 |
|TruthfulQA  | 42.2 | 51.0  | 45.6 |  **59.6**  | 49.6 | 54.8 |
|   ***Coding***  |    |    |   |    |	    |	    |
|HumanEval | 45.7 | 46.3  | 48.2 | 46.3 | 54.3 |  **64.6**   |
|MBPP |73.9 | 71.7  | 70.4 | 66.9 | 70.9 |  **76.9**   |
|EvalPlus | 55.0 | 54.1  | 54.8 | 52.9 | 57.7 |  **65.4**   |
|MultiPL-E |44.4 | 46.7  | 46.3 | 41.8 | 52.7 |  **59.6**   |
|   ***Mathematics***  |    |    |   |    |	    |	    |
|GSM8K | 79.2 | 83.7   | 83.0 | 79.5 | 85.4 |  **89.5**  |
|MATH  | 43.6 | 41.7  | 42.5 | 34.1 | 49.6 |  **51.1**  |
|   ***Chinese***  |    |    |   |    |	    |	    |
|C-Eval | 81.7 | 54.6    |  65.2 |  84.1 | 89.1 |   **91.0**  |
|CMMLU   | 84.0 | 53.4  | 67.2 | 83.5 | 88.3 |  **90.1**  |
|   ***Multilingual***  |    |    |   |    |	    |	    |
|Mulit-Exam   | 67.5 | 63.5 |   70.0    |  66.4 |  75.6 |   **76.6**  |
|Multi-Understanding | 77.0 |  77.7    |  79.9 |  78.2 | 78.2 |   **80.7**  |
|Multi-Mathematics |  58.8 | 62.9    |  67.1 |  61.7 | 64.4 |   **76.0**  |
|Multi-Translation |   36.0 | 23.3    |   **38.0**  |  35.6 | 36.2 |  37.8 |

## Citation

If you find our work helpful, feel free to give us a cite.

```
@article{qwen2,
  title={Qwen2 Technical Report},
  year={2024}
}
```

