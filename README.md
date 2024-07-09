# Fairness in Large Language Models

This ongoing project aims to consolidate interesting efforts in the field of fairness in Large Language Models (LLMs), drawing on the proposed taxonomy and surveys dedicated to various aspects of fairness in LLMs.



**Tutorial:** [Fairness in Large Language Models in Three Hours]()<br>
Thang Viet Doan, Zichong Wang, Minh Nhat Nguyen and Wenbin Zhang<br>
*Proceedings of the ACM International Conference on Information and Knowledge Management (CIKM), Boise, USA, 2024*



**Fairness in LLMs:** [Fairness in Large Language Models: A Taxonomic Survey](https://arxiv.org/abs/2404.01349)<br>
Zhibo Chu, Zichong Wang and Wenbin Zhang<br>
*ACM SIGKDD Explorations Newsletter, 2024*

**Introduction to LLMs:** [History, Development, and Principles of Large Language Models-An Introductory Survey
](https://arxiv.org/abs/2402.06853)<br>
Zhibo Chu, Shiwen Ni, Zichong Wang, Xi Feng, Min Yang and Wenbin Zhang

**Fairness Definitions:** [Fairness Definitions in Language Models Explained]()<br>
Thang Viet Doan, Zhibo Chu, Zichong Wang and Wenbin Zhang


Email: thang.dv509@gmail.com - Thang Doan<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ziwang@fiu.edu - Zichong Wang<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;zb.chu2001@gmail.com - Zhibo Chu

![Fairness in Large Language Models](https://github.com/super-hash/Fairness-in-Large-Language-Models/blob/main/Fairness%20in%20Large%20Language%20Models.png)


## Quantifying Bias in LLMs
### Embbedding-based Metrics
+ Semantics derived automatically from language corpora contain human-like biases (WEAT) [[arXiv]](https://arxiv.org/abs/1608.07187)
+ On measuring social biases in sentence encoders (SEAT) [[NAACL]](https://arxiv.org/abs/1903.10561)
+ Detecting emergent intersectional biases: Contextualized word embeddings contain a distribution of human-like biases [[AAAI]](https://dl.acm.org/doi/abs/10.1145/3461702.3462536)
### Probability-based Metrics
+ Measuring and reducing gendered correlations in pre-trained models (DisCo) [[arXiv]](https://arxiv.org/abs/2010.06032)
+ Measuring Bias in Contextualized Word Representations [[ACL]](https://aclanthology.org/W19-3823/)
+ CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models [[ACL]](https://aclanthology.org/2020.emnlp-main.154/)
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
+ DUnE: Dataset for Unified Editing [[EMNLP]](https://arxiv.org/abs/2311.16087)
+ Reducing Sentiment Bias in Language Models via Counterfactual Evaluation [[EMNLP]](https://aclanthology.org/2020.findings-emnlp.7.pdf)
### Post-processing
+ Evaluating Gender Bias in Large Language Models via Chain-of-Thought Prompting [[arXiv]](https://arxiv.org/abs/2401.15585)
+ Queer People are People First: Deconstructing Sexual Identity Stereotypes in Large Language Models [[ACL]](https://arxiv.org/abs/2307.00101)
+ Text Style Transfer for Bias Mitigation using Masked Language Modeling [[NAACL]](https://aclanthology.org/2022.naacl-srw.21/)
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

Our survey paper "Fairness in Large Language Models: A Taxonomic Survey" has been accepted by ACM SIGKDD Explorations Newsletter and released on arxiv: [![PDF](https://img.shields.io/badge/PDF-Download-red)](https://arxiv.org/abs/2404.01349)

If you find that our survey helps your research, we would appreciate citations to the following paper:
```
@article{chu2024fairness,
  title={Fairness in Large Language Models: A Taxonomic Survey},
  author={Chu, Zhibo and Wang, Zichong and Zhang, Wenbin},
  journal={ACM SIGKDD Explorations Newsletter},
  year={2024}
}
```
or:
```
\bibitem{chu2024fairness}
Chu, Z., Wang, Z., Zhang, W.: Fairness in large language models: A taxonomic
  survey. ACM SIGKDD Explorations Newsletter  (2024)
```
