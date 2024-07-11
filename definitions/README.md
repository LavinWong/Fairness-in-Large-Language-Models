# Fairness Definitions in Language Models Explained

This is the artifact for the paper [Fairness Definitions in Language Models Explained](). This artifact supplies the tools and implementation guidelines to reproduce and apply fairness definitions discussed in our paper. 

Authors: Thang Viet Doan, Zhibo Chu, Zichong Wang and Wenbin Zhang

## Installation

Install required packages/libraries:

```shell script
$ pip install -r requirements.txt
```
For [GPT](https://openai.com/api/) and [Llama](https://www.together.ai/), access keys are required for API requests. Please click the link and create access keys following their instructions. After obtaining the access credentials, fill them in  `api_key.py`.

```shell script
OPENAI_KEY = "you openai key" # for OpenAI GPT
TOGETHERAPI_KEY = "your togetherapi key" # for Llama2 
```

## Run Experiments

This section is organized according to the section in our paper. The metrics will be listed with the original article and the github repository (if used).  

### Fairness definitions for medium-sized language models

**Intrinsic bias** 

* Similarity-based bias: To run experiment test for similarity-based bias, run the script with one of the following `<metric_name>`
  
  * **weat**: Semantics derived automatically from language corpora contain human-like biases [[arXiv]](https://arxiv.org/abs/1608.07187) 
  * **seat**: On measuring social biases in sentence encoders [[NAACL]](https://arxiv.org/abs/1903.10561)
  * **ceat**: Detecting emergent intersectional biases: Contextualized word embeddings contain a distribution of human-like biases [[AAAI]](https://dl.acm.org/doi/abs/10.1145/3461702.3462536)
  
```shell script
$ python main.py medium intrinsic similarity <metric_name>
```

* Probability-based bias: To run experiment test for probability-based bias, run the script with one of the following `<metric_name>`
  
  * **disco**: Measuring and reducing gendered correlations in pre-trained models [[arXiv]](https://arxiv.org/abs/2010.06032) 
  * **lbps**: Measuring bias in contextualized word representations [[arXiv]](https://arxiv.org/abs/1906.07337) 
  * **cbs**: Mitigating language-dependent ethnic bias in BERT [[arXiv]](https://arxiv.org/abs/2109.05704) 
  * **ppl**: Masked language model scoring [[arXiv]](https://arxiv.org/abs/1910.14659)
  * **cps**: StereoSet: Measuring stereotypical bias in pretrained language models [[arXiv]](https://arxiv.org/abs/2004.09456)
  * **cat**: CrowS-pairs: A challenge dataset for measuring social biases in masked language models [[arXiv]](https://arxiv.org/abs/2010.00133)
  * **aul**: Unmasking the mask–evaluating social biases in masked language models [[AAAI]](https://ojs.aaai.org/index.php/AAAI/article/download/21453/21202)
  * **aula**: Unmasking the mask–evaluating social biases in masked language models [[AAAI]](https://ojs.aaai.org/index.php/AAAI/article/download/21453/21202)
    
```shell script
$ python main.py medium intrinsic probability <metric_name>
```

**Extrinsic bias**

* Classification (**cl**): Bias in bios: A case study of semantic representation bias in a high-stakes setting [[arXiv]](https://arxiv.org/pdf/1901.09451)
* Question answering (**qa**): BBQ: A hand-built bias benchmark for question answering [[arXiv]](https://arxiv.org/pdf/2110.08193)

```shell script
$ python main.py medium extrinsic <task_name>
```

### Fairness definitions for large-sized language models

* Demographic Representation (**dr**)
  * **exp1**: Language models are few-shot learners [[NeurIPS]](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)
  * **exp2**: Understanding stereotypes in language models: Towards robust measurement and zero-shot debiasing [[arXiv]](https://arxiv.org/pdf/2212.10678)
  * **exp3**: Holistic evaluation of language models [[arXiv]](https://arxiv.org/pdf/2211.09110)
    
* Stereotypical Association (**sa**)
  * **exp1**: Language models are few-shot learners [[NeurIPS]](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)
  * **exp2**: Holistic evaluation of language models [[arXiv]](https://arxiv.org/pdf/2211.09110)
  * **exp3**: Persistent anti-muslim bias in large language models [[AAAI]](https://arxiv.org/pdf/2101.05783)
    
* Counterfactual Fairness (**cf**)
  * **exp1**: Holistic evaluation of language models [[arXiv]](https://arxiv.org/pdf/2211.09110)
  * **exp2**: Fairness of chatgpt [[arXiv]](https://arxiv.org/pdf/2305.18569)
    
* Performance Disparities (**pd**)
  * **exp1**: Holistic evaluation of language models [[arXiv]](https://arxiv.org/pdf/2211.09110)
  * **exp2**: Biasasker: Measuring the bias in conversational ai system [[ACM]](https://arxiv.org/pdf/2305.12434)
  * **exp3**: Is chatgpt fair for recommendation? evaluating fairness in large language model recommendation [[ACM]](https://arxiv.org/pdf/2305.07609)
    
```shell script
$ python main.py large <strategy_name> <experiment_name>
```
