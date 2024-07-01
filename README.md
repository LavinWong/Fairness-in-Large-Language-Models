# Fairness in Large Language Models

This ongoing endeavor aims to consolidate interesting efforts in the area of fairness in large language models, as outlined in the proposed taxonomy.

**Introduction to LLMs:** [History, Development, and Principles of Large Language Models-An Introductory Survey
](https://arxiv.org/abs/2402.06853) Zhibo Chu, Shiwen Ni, Zichong Wang, Xi Feng, Min Yang, Wenbin Zhang

**Fairness in LLMs:** [Fairness in Large Language Models: A Taxonomic Survey](https://arxiv.org/abs/2404.01349) Zhibo Chu, Zichong Wang, and Wenbin Zhang.

**Fairness Definition in LLMs:** [Fairness Definitions in Language Models Explained] Thang Viet Doan, Zhibo Chu, Zichong Wang, Wenbin Zhang


Email:zb.chu2001@gmail.com - Zhibo Chu

![Fairness in Large Language Models](https://github.com/super-hash/Fairness-in-Large-Language-Models/blob/main/Fairness%20in%20Large%20Language%20Models.png)
## Surveys
+ Fairness in Large Language Models: A Taxonomic Survey [[arXiv]](https://arxiv.org/abs/2404.01349)
+ Bias and Fairness in Large Language Models: A Survey [[arXiv]](https://arxiv.org/abs/2309.00770)
+ A survey on fairness in large language models [[arXiv]](https://arxiv.org/abs/2308.10149)

## Quantifying Bias in LLMs
### Embbedding-based Metrics
+ Semantics derived automatically from language corpora contain human-like biases (WEAT) [[arXiv]](https://arxiv.org/abs/1608.07187)
+ On measuring social biases in sentence encoders (SEAT) [[NAACL]](https://arxiv.org/abs/1903.10561)
+ Detecting emergent intersectional biases: Contextualized word embeddings contain a distribution of human-like biases [[AAAI]](https://dl.acm.org/doi/abs/10.1145/3461702.3462536)
### Probability-based Metrics
+ Measuring and reducing gendered correlations in pre-trained models (DisCo) [[arXiv]](https://arxiv.org/abs/2010.06032)
+ Measuring Bias in Contextualized Word Representations [[ACL]] (https://aclanthology.org/W19-3823/)
+ CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models [[ACL]] (https://aclanthology.org/2020.emnlp-main.154/)
### Generation-based Metrics
+ Identifying and reducing gender bias in word-level language models [[ACL]](https://aclanthology.org/N19-3002.pdf)
## Mitigating Bias in LLMs
### Pre-processing
+ Measuring and reducing gendered correlations in pre-trained models [[arXiv]](https://arxiv.org/abs/2010.06032)
+ Counterfactual Data Augmentation for Mitigating Gender Stereotypes in Languages with Rich Morphology [[ACL]](https://aclanthology.org/P19-1161/)
+ Deep Learning on a Healthy Data Diet: Finding Important Examples for Fairness [[AAAI]](https://arxiv.org/abs/2211.11109)
### In-training
+ Enhancing Model Robustness and Fairness with Causality: A Regularization Approach [[ACL]](https://aclanthology.org/2021.cinlp-1.3/)
+ Never Too Late to Learn: Regularizing Gender Bias in Coreference Resolution [[WSDM]](https://dl.acm.org/doi/abs/10.1145/3539597.3570473)
+ Sustainable Modular Debiasing of Language Models [[ACL]](https://aclanthology.org/2021.findings-emnlp.411.pdf)
+ Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection [[ACL]](https://aclanthology.org/2020.acl-main.647/)
### Intra-processing
+ Debiasing algorithm through model adaptation [[ICLR]](https://arxiv.org/abs/2310.18913)
+ DUnE: Dataset for Unified Editing [[EMNLP]] (https://arxiv.org/abs/2311.16087)
+ Reducing Sentiment Bias in Language Models via Counterfactual Evaluation [[EMNLP]] (https://aclanthology.org/2020.findings-emnlp.7.pdf)
### Post-processing
+ Evaluating Gender Bias in Large Language Models via Chain-of-Thought Prompting [[arXiv]](https://arxiv.org/abs/2401.15585)
+ Queer People are People First: Deconstructing Sexual Identity Stereotypes in Large Language Models [[ACL]] (https://arxiv.org/abs/2307.00101)
+ Text Style Transfer for Bias Mitigation using Masked Language Modeling [[NAACL]] (https://aclanthology.org/2022.naacl-srw.21/)
## Datasets
+ [BOLD](https://github.com/amazon-science/bold)
+ [BBQ](https://github.com/nyu-mll/BBQ)
+ [CrowS-Pairs](https://github.com/nyu-mll/crows-pairs/)
+ [StereoSet](https://github.com/moinnadeem/stereoset)
+ [WinoBias](https://github.com/uclanlp/corefBias)
+ [WinoBias+](https://github.com/vnmssnhv/NeuTralRewriter)
+ [WinoGender](https://github.com/rudinger/winogender-schemas)
+ [WinoQueer](https://github.com/katyfelkner/winoqueer)

## Citation

Pre-print: https://arxiv.org/abs/2404.01349

If you find that our survey helps your research, please consider citing it:
```
@article{chu2024fairness,
  title={Fairness in Large Language Models: A Taxonomic Survey},
  author={Chu, Zhibo and Wang, Zichong and Zhang, Wenbin},
  journal={arXiv preprint arXiv:2404.01349},
  year={2024}
}
```
