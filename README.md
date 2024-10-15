# Fairness in Large Language Models

This ongoing project aims to consolidate interesting efforts in the field of fairness in Large Language Models (LLMs), drawing on the proposed taxonomy and surveys dedicated to various aspects of fairness in LLMs.

**Disclaimer: We may have missed some relevant papers in the list. If you have suggestions or want to add papers, please submit a pull request or email us—your contributions are greatly appreciated!**

**Tutorial:** [Fairness in Large Language Models in Three Hours](https://fairness-in-llms.github.io)<br>
Thang Viet Doan, Zichong Wang, Nhat Hoang and Wenbin Zhang<br>
*Proceedings of the ACM International Conference on Information and Knowledge Management (CIKM), Boise, USA, 2024*

**Fairness in LLMs:** [Fairness in Large Language Models: A Taxonomic Survey](https://dl.acm.org/doi/abs/10.1145/3682112.3682117)<br>
Zhibo Chu, Zichong Wang and Wenbin Zhang<br>
*ACM SIGKDD Explorations Newsletter, 2024*

**Introduction to LLMs:** [History, Development, and Principles of Large Language Models-An Introductory Survey
](https://arxiv.org/abs/2402.06853)<br>
Zichong Wang, Zhibo Chu, Thang Viet Doan, Shiwen Ni, Min Yang and Wenbin Zhang<br>
*AI and Ethics, 2024*

**Fairness Definitions in LLMs:** [Fairness Definitions in Language Models Explained](https://arxiv.org/abs/2407.18454)<br>
Thang Viet Doan, Zhibo Chu, Zichong Wang and Wenbin Zhang

**Datasets for fairness in LLMs:** [Datasets for Fairness in Language Models: An In-Depth Survey]()<br>
Thang Viet Doan, Nhat Hoang, Zichong Wang and Wenbin Zhang


Email: thang.dv509@gmail.com - Thang Viet Doan<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ziwang@fiu.edu - Zichong Wang<br>

## Mitigating Bias in LLMs ([Link](https://dl.acm.org/doi/abs/10.1145/3682112.3682117) to the paper)
![Fairness in Large Language Models](https://github.com/super-hash/Fairness-in-Large-Language-Models/blob/main/images/Fairness%20in%20Large%20Language%20Models.png)
### Pre-processing
+ Measuring and reducing gendered correlations in pre-trained models [[arXiv]](https://arxiv.org/abs/2010.06032)
+ Counterfactual Data Augmentation for Mitigating Gender Stereotypes in Languages with Rich Morphology [[ACL]](https://aclanthology.org/P19-1161/)
+ Deep Learning on a Healthy Data Diet: Finding Important Examples for Fairness [[AAAI]](https://arxiv.org/abs/2211.11109)
+ Improving Gender Fairness of Pre-Trained Language Models without Catastrophic Forgetting.[[ACL]](https://aclanthology.org/2023.acl-short.108/)
+ Data-Centric Explainable Debiasing for Improving Fairness in Pre-trained Language Models. [[ACL]](https://aclanthology.org/2024.findings-acl.226/)
+ 
### In-training
+ Enhancing Model Robustness and Fairness with Causality: A Regularization Approach [[ACL]](https://aclanthology.org/2021.cinlp-1.3/)
+ Never Too Late to Learn: Regularizing Gender Bias in Coreference Resolution [[WSDM]](https://dl.acm.org/doi/abs/10.1145/3539597.3570473)
+ Sustainable Modular Debiasing of Language Models [[ACL]](https://aclanthology.org/2021.findings-emnlp.411.pdf)
+ Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection [[ACL]](https://aclanthology.org/2020.acl-main.647/)
+ Does Gender Matter? Towards Fairness in Dialogue Systems [[COLING]](https://aclanthology.org/2020.coling-main.390/)
+ Debiasing pretrained text encoders by paying attention to paying attention [[EMNLP]](https://hal.science/hal-03919992/document)
+ Reducing Gender Bias in Word-Level Language Models with a Gender-Equalizing Loss Function [[ACL]](https://arxiv.org/pdf/1905.12801)
+ FineDeb: A Debiasing Framework for Language Models [[arXiv]](https://amulyayadav.github.io/AI4SG2023/images/24.pdf)
### Intra-processing
+ Debiasing algorithm through model adaptation [[ICLR]](https://arxiv.org/abs/2310.18913)
+ DUnE: Dataset for Unified Editing [[EMNLP]](https://arxiv.org/abs/2311.16087)
+ Reducing Sentiment Bias in Language Models via Counterfactual Evaluation [[EMNLP]](https://aclanthology.org/2020.findings-emnlp.7.pdf)
+ Using In-Context Learning to Improve Dialogue Safety [[EMNLP]](https://arxiv.org/pdf/2302.00871)
+ DExperts: Decoding-Time Controlled Text Generation with Experts and Anti-Experts [[ACL]](https://arxiv.org/pdf/2105.03023)
### Post-processing
+ Evaluating Gender Bias in Large Language Models via Chain-of-Thought Prompting [[arXiv]](https://arxiv.org/abs/2401.15585)
+ Queer People are People First: Deconstructing Sexual Identity Stereotypes in Large Language Models [[ACL]](https://arxiv.org/abs/2307.00101)
+ Text Style Transfer for Bias Mitigation using Masked Language Modeling [[NAACL]](https://aclanthology.org/2022.naacl-srw.21/)
+ They, them, theirs: Rewriting with gender-neutral english [[arXiv]](https://arxiv.org/pdf/2102.06788)

## Fairness Definitions in LMs ([Link](https://arxiv.org/pdf/2407.18454) to the paper)
![Fairness Definitions in LMs](https://github.com/super-hash/Fairness-in-Large-Language-Models/blob/main/images/Fairness%20Definitions%20in%20LMs.png)
### Fairness definitions for medium-sized LMs
**Intrinsic bias**: also known as upstream bias or representational bias, refers to the inherent biases present in the output representation generated by a medium-sized LM
* Similarity-based bias
  * Semantics derived automatically from language corpora contain human-like biases [[arXiv]](https://arxiv.org/abs/1608.07187) 
  * On measuring social biases in sentence encoders [[NAACL]](https://arxiv.org/abs/1903.10561)
  * Detecting emergent intersectional biases: Contextualized word embeddings contain a distribution of human-like biases [[AAAI]](https://dl.acm.org/doi/abs/10.1145/3461702.3462536)
  
* Probability-based bias
  * Measuring and reducing gendered correlations in pre-trained models [[arXiv]](https://arxiv.org/abs/2010.06032) 
  * Measuring bias in contextualized word representations [[arXiv]](https://arxiv.org/abs/1906.07337) 
  * Mitigating language-dependent ethnic bias in BERT [[arXiv]](https://arxiv.org/abs/2109.05704) 
  * Masked language model scoring [[arXiv]](https://arxiv.org/abs/1910.14659)
  * StereoSet: Measuring stereotypical bias in pretrained language models [[arXiv]](https://arxiv.org/abs/2004.09456)
  * CrowS-pairs: A challenge dataset for measuring social biases in masked language models [[arXiv]](https://arxiv.org/abs/2010.00133)
  * Unmasking the mask–evaluating social biases in masked language models [[AAAI]](https://ojs.aaai.org/index.php/AAAI/article/download/21453/21202)
  * Pro-Woman, Anti-Man? Identifying Gender Bias in Stance Detection [[ACL]](https://aclanthology.org/2024.findings-acl.192.pdf)
  
**Extrinsic bias**: refers to the disparity in a LM’s performance across different downstream tasks, also known as downstream bias or prediction bias
* Classification
  * Bias in bios: A case study of semantic representation bias in a high-stakes setting [[arXiv]](https://arxiv.org/pdf/1901.09451)

* Natural Language Inference
  * On measuring and mitigating biased inferences of word embeddings [[AAAI]](https://ojs.aaai.org/index.php/AAAI/article/view/6267)
  * On measuring social biases in prompt-based multi-task learning [[arXiv]](https://arxiv.org/pdf/2110.08193)

* Question answering
  * BBQ: A hand-built bias benchmark for question answering [[arXiv]](https://arxiv.org/pdf/2110.08193)
 
* Recommender Systems
  *  Up5: Unbiased foundation model for fairness-aware recommendation [[arXiv]](https://arxiv.org/pdf/2305.12090)

### Fairness definitions for large-sized LMs
Fairness in these models is evaluated using specific strategies designed to quantify it.

* **Demographic Representation**: The systematic discrepancy in the frequency of mentions of different demographic groups within the generated text.
  * Language models are few-shot learners [[NeurIPS]](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)
  * Understanding stereotypes in language models: Towards robust measurement and zero-shot debiasing [[arXiv]](https://arxiv.org/pdf/2212.10678)
  * Holistic evaluation of language models [[arXiv]](https://arxiv.org/pdf/2211.09110)
    
* **Stereotypical Association**: The systematic discrepancy in the model’s associations between demographic groups and specific stereotypes, which reflects societal prejudice
  * Language models are few-shot learners [[NeurIPS]](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)
  * Holistic evaluation of language models [[arXiv]](https://arxiv.org/pdf/2211.09110)
  * Persistent anti-muslim bias in large language models [[AAAI]](https://arxiv.org/pdf/2101.05783)
    
* **Counterfactual Fairnes**: The model’s sensitivity to demographic-specific terms, measuring how changes to these terms affect its output
  * Holistic evaluation of language models [[arXiv]](https://arxiv.org/pdf/2211.09110)
  * Fairness of chatgpt [[arXiv]](https://arxiv.org/pdf/2305.18569)
    
* **Performance Disparities**: The systematic variation in accuracy or other performance metrics when the model is applied to tasks involving different demographic groups
  * Holistic evaluation of language models [[arXiv]](https://arxiv.org/pdf/2211.09110)
  * Biasasker: Measuring the bias in conversational ai system [[ACM]](https://arxiv.org/pdf/2305.12434)
  * Is chatgpt fair for recommendation? evaluating fairness in large language model recommendation [[ACM]](https://arxiv.org/pdf/2305.07609)

## Datasets
![Screenshot 2024-10-07 at 1 52 35 PM](https://github.com/user-attachments/assets/37895985-ccf6-4183-a035-d3238d5eac47)

+ [WinoBias](https://github.com/uclanlp/corefBias)
+ [WinoBias+](https://github.com/vnmssnhv/NeuTralRewriter)
+ [WinoGender](https://github.com/rudinger/winogender-schemas)
+ [WinoQueer](https://github.com/katyfelkner/winoqueer)
+ [BEC-Pro](https://github.com/marionbartl/gender-bias-BERT)
+ [BUG](https://github.com/SLAB-NLP/BUG)
+ [GAP](https://github.com/google-research-datasets/gap-coreference)
+ [StereoSet](https://github.com/moinnadeem/stereoset)
+ [HONEST](https://github.com/MilaNLProc/honest)
+ [Bias-NLI](https://github.com/sunipa/On-Measuring-and-Mitigating-Biased-Inferences-of-Word-Embeddings)
+ [CrowS-Pairs](https://github.com/nyu-mll/crows-pairs/)
+ [EEC](https://www.svkir.com/resources.html#EEC)
+ [PANDA](https://github.com/facebookresearch/ResponsibleNLP)
+ [RedditBias](https://github.com/umanlp/RedditBias)
+ [TrustGPT](https://github.com/HowieHwong/TrustGPT)
+ [FairPrism](https://github.com/microsoft/FairPrism)
+ [BOLD](https://github.com/amazon-science/bold)
+ [RealToxicityPrompts](https://allenai.org/data/real-toxicity-prompts)
+ [HolisticBias](https://github.com/facebookresearch/ResponsibleNLP)
+ [BBQ](https://github.com/nyu-mll/BBQ)
+ [UnQover](https://github.com/allenai/unqover)
+ [CEB](https://github.com/SongW-SW/CEB)


## Citation
### Fairness in Large Language Models: A Taxonomic Survey [![PDF](https://img.shields.io/badge/PDF-Download-red)](https://dl.acm.org/doi/abs/10.1145/3682112.3682117)

If you find that our taxonomic survey helps your research, we would appreciate citations to the following paper:
```
@article{chu2024fairness,
  title={Fairness in Large Language Models: A Taxonomic Survey},
  author={Chu, Zhibo and Wang, Zichong and Zhang, Wenbin},
  journal={ACM SIGKDD Explorations Newsletter},
  volume={26},
  number={1},
  pages={34--48},
  year={2024},
  publisher={ACM New York, NY, USA}
}
```

### History, Development, and Principles of Large Language Models-An Introductory Survey [![PDF](https://img.shields.io/badge/PDF-Download-red)](https://arxiv.org/pdf/2402.06853)

If you find that our introduction survey helps your research, we would appreciate citations to the following paper:
```
@article{wang2024history,
  title={History, Development, and Principles of Large Language Models: An Introductory Survey},
  author={Wang, Zichong and Chu, Zhibo and Doan, Thang Viet and Ni, Shiwen and Yang, Min and Zhang, Wenbin},
  journal={AI and Ethics},
  year={2024},
  publisher={Springer}
}

```

### Fairness Definitions in Language Models Explained [![PDF](https://img.shields.io/badge/PDF-Download-red)](https://arxiv.org/abs/2407.18454)

If you find that our definition survey helps your research, we would appreciate citations to the following paper:
```
@misc{doan2024fairnessdefinitionslanguagemodels,
      title={Fairness Definitions in Language Models Explained}, 
      author={Thang Viet Doan and Zhibo Chu and Zichong Wang and Wenbin Zhang},
      year={2024},
      eprint={2407.18454},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.18454}, 
}


