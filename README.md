# Causality-in-CV

关于机器学习中因果相关的论文清单。其中部分内容参考了Awesome-Causality-in-CV。一些解释内容参考[笔记](https://blog.csdn.net/qq_40943760/article/details/123373859)


## Contents
1. Type of Causality in CV
2. Before 2020
3. 2020 Venues
4. 2021 Venues
5. 2022 Venues
6. 2023 Venues
7. Arxiv

## 1. Type of Causality in CV

| Type | `IT` | `CF` | `CR` | `Else` |
| :---: | :---: | :---: | :---: | :---: |
| Explanation | Intervention | Counterfactual | Causal Representation | Other Types |

> **Causal Representation**
> 主要参考[《Towards Casual Representation Learning》](https://ieeexplore.ieee.org/abstract/document/9363924) \\
> 现实应用中，许多关键问题都可以归结为OOD(out-of-distribution)问题。因为统计学习模型需要独立同分布(iid)假设，若测试数据与训练数据来自不同的分布，统计学习模型往往会出错。在很多情况下，iid的假设是不成立的，而因果推断所研究的正是这样的情形：如何学习一个可以在不同分布下工作、蕴含因果机制的因果模型(Causal Model)，并使用因果模型进行干预或反事实推断。因此，人工智能和因果关系的一个核心问题是因果表征学习，即从低级观察（low-level observations）中发现高级因果变量（high-level causal variables）。
> 很自然地想到将因果推断的优点结合到机器学习中，然而现实没有这么容易。因果模型往往处理的是结构化数据（行数据，可以用二维表结构来逻辑表达实现的数据），并不能处理机器学习中常见的高维的低层次的原始数据，例如图像。为此，回到最初的问题，因果表征即可理解为可以用于因果模型的表征，因果表征学习即为将图像这样的原始数据转化为可用于因果模型的结构化变量。因果表征学习就是连接因果科学与机器学习的桥梁，解决这一相关问题，就可以将因果推断与机器学习结合起来，构建下一代更强大的AI。





## 2. Before 2020

| Title | Venue | Type | Code |
| :--- | :---: | :---: | :---: |
| [Counterfactual Visual Explanations](https://arxiv.org/abs/1904.07451) | ICML2019 | CF | - |
| [CausalGAN: Learning Causal Implicit Generative Models with Adversarial Training](https://arxiv.org/abs/1709.02023) | ICLR2018 | 0 | [Pytorch(Author)](https://github.com/mkocaoglu/CausalGAN) |
| [Discovering causal signals in images](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lopez-Paz_Discovering_Causal_Signals_CVPR_2017_paper.pdf) | CVPR2017 | 0 | [TensorFlow(3rd)](https://github.com/kyrs/NCC-experiments) | 




## 3. 2020 Venues

| Title | Venue | Type | Code |
| :--- | :---: | :---: | :---: |
| [Counterfactual Contrastive Learning for Weakly-Supervised Vision-Language Grounding](https://papers.nips.cc/paper/2020/file/d27b95cac4c27feb850aaa4070cc4675-Paper.pdf) | NeurIPS | CF | - |
| [A causal view of compositional zero-shot recognition](https://arxiv.org/abs/2006.14610) | NeurIPS | IT | - |
| [Counterfactual Vision-and-Language Navigation: Unravelling the Unseen](https://papers.nips.cc/paper/2020/hash/39016cfe079db1bfb359ca72fcba3fd8-Abstract.html) | NeurIPS | CF | - |
| [Causal Intervention for Weakly-Supervised Semantic Segmentation](https://arxiv.org/abs/2009.12547) | NeurIPS | IT | [PyTorch(Author)](https://github.com/ZHANGDONG-NJUST/CONTA) |
| [Interventional Few-Shot Learning](http://arxiv.org/abs/2009.13000) | NeurIPS | IT | [PyTorch(Author)](https://github.com/yue-zhongqi/ifsl) |
| [Long-Tailed Classification by Keeping the Good and Removing the Bad Momentum Causal Effect](https://arxiv.org/abs/2009.12991) | NeurIPS | CF | [PyTorch(Author)](https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch) |
| [Traffic Accident Benchmark for Causality Recognition](https://arxiv.org/abs/1911.07308) | ECCV | 0 | [PyTorch(Author)](https://github.com/tackgeun/CausalityInTrafficAccident) |
| [Towards causal benchmarking of bias in face analysis algorithms](https://arxiv.org/abs/2007.06570) | ECCV | 0 | [PyTorch(Author)](https://github.com/balakg/transects-eccv2020) |
| [Learning What Makes a Difference from Counterfactual Examples and Gradient Supervision](https://arxiv.org/abs/2004.09034) | ECCV | CF | - |
| [Counterfactual Vision-and-Language Navigation via Adversarial Path Sampling](https://arxiv.org/abs/1911.07308) | ECCV | CF | - |
| [Visual Commonsense R-CNN](https://arxiv.org/abs/2002.12204) | CVPR | CR/IT | [PyTorch(Author)](https://github.com/Wangt-CN/VC-R-CNN) |
| [Unbiased scene graph generation from biased training](https://arxiv.org/abs/2002.11949) | CVPR | CF | [PyTorch(Author)](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) |
| [Two causal principles for improving visual dialog](https://arxiv.org/abs/1911.10496) | CVPR | IT | [PyTorch(Author)](https://github.com/simpleshinobu/visdial-principles) |
| [Counterfactual samples synthesizing for robust visual question answering](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Counterfactual_Samples_Synthesizing_for_Robust_Visual_Question_Answering_CVPR_2020_paper.pdf) | CVPR | CF | [PyTorch(Author)](https://github.com/yanxinzju/CSS-VQA) |
| [Towards Causal VQA: Revealing and Reducing Spurious Correlations by Invariant and Covariant Semantic Editing](https://arxiv.org/abs/1912.07538) | CVPR | CF | [PyTorch(Author)](https://github.com/AgarwalVedika/CausalVQA) |
| [Counterfactuals uncover the modular structure of deep generative models](https://arxiv.org/abs/1812.03253) | ICLR | CF | - |
| [Exploratory Not Explanatory: Counterfactual Analysis of Saliency Maps for Deep Reinforcement Learning](https://arxiv.org/abs/1912.05743) | ICLR | CF | - |



## 4. 2021 Venues

| Title | Venue | Type | Code |
| :--- | :---: | :---: | :---: |
| [Transporting Causal Mechanisms for Unsupervised Domain Adaptation](https://arxiv.org/abs/2107.11055) | ICCV | 0 | [PyTorch(Author)](https://github.com/yue-zhongqi/tcm) |
| [Causal Attention for Unbiased Visual Recognition](https://arxiv.org/abs/2108.08782) | ICCV | IT | [PyTorch(Author)](https://github.com/Wangt-CN/CaaM) |
| [Learning Causal Representation for Training Cross-Domain Pose Estimator via Generative Interventions]() | ICCV | IT/CR | - |
| [Human Trajectory Prediction via Counterfactual Analysis](https://arxiv.org/abs/2107.14202) | ICCV | CF | [PyTorch(Author)](https://github.com/CHENGY12/CausalHTP) |
| [Counterfactual Attention Learning for Fine-Grained Visual Categorization and Re-Identification](https://arxiv.org/abs/2108.08728) | ICCV | CF | [PyTorch(Author)](https://github.com/raoyongming/cal) |
| [Counterfactual VQA: A Cause-Effect Look at Language Bias](https://arxiv.org/abs/2006.04315) | CVPR | CF | [PyTorch(Author)](https://github.com/yuleiniu/cfvqa) |
| [Counterfactual Zero-Shot and Open-Set Visual Recognition](https://arxiv.org/abs/2103.00887) | CVPR | CF | [PyTorch(Author)](https://github.com/yue-zhongqi/gcm-cf) |
| [Distilling Causal Effect of Data in Class-Incremental Learning](https://arxiv.org/pdf/2103.01737) | CVPR | CF | [PyTorch(Author)](https://github.com/JoyHuYY1412/DDE_CIL) |
| [Causal Attention for Vision-Language Tasks](https://arxiv.org/abs/2103.03493) | CVPR | IT | [PyTorch(Author)](https://github.com/yangxuntu/lxmertcatt) |
| [The Blessings of Unlabeled Background in Untrimmed Videos](https://arxiv.org/abs/2103.13183) | CVPR | CF | - |
| [Affect2MM: Affective Analysis of Multimedia Content Using Emotion Causality](https://arxiv.org/abs/2103.06541) | CVPR | 0 | [PyTorch(Author)](https://gamma.umd.edu/researchdirections/affectivecomputing/affect2mm/) |
| [CausalVAE: Disentangled Representation Learning via Neural Structural Causal Models](https://arxiv.org/abs/2004.08697) | CVPR | CR | - |
| [Generative Interventions for Causal Learning](https://arxiv.org/abs/2012.12265) | CVPR | CR | [PyTorch(Author)](https://github.com/cvlab-columbia/GenInt) |
| [ACRE: Abstract Causal REasoning Beyond Covariation](https://arxiv.org/abs/2103.14232) | CVPR | 0 | [PyTorch(Author)](https://github.com/WellyZhang/ACRE) |
| [Towards Robust Classification Model by Counterfactual and Invariant Data Generation](https://arxiv.org/abs/2106.01127) | CVPR | CF | [PyTorch(Author)](https://github.com/WellyZhang/ACRE) |
| [Interventional Video Grounding With Dual Contrastive Learning](https://arxiv.org/abs/2106.11013) | CVPR | IT | - |
| [Representation Learning via Invariant Causal Mechanisms](https://arxiv.org/abs/2010.07922) | ICLR | CR | - |
| [Counterfactual Generative Networks](https://arxiv.org/abs/2101.06046) | ICLR | CF | [PyTorch(Author)](https://github.com/autonomousvision/counterfactual_generative_networks) |
|[Counterfactual Fairness with Disentangled Causal Effect Variational Autoencoder](https://ojs.aaai.org/index.php/AAAI/article/view/16990)| AAAI | CF | - |


## 5. 2022 Venues

| Title | Venue | Type | Code |
| :--- | :---: | :---: | :---: |
|[Causal Representation Learning for Out-of-Distribution Recommendation](https://dl.acm.org/doi/abs/10.1145/3485447.3512251)| WWW | CR/IT | - |
| [Counterfactual Cycle-Consistent Learning for Instruction Following and Generation in Vision-Language Navigation](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Counterfactual_CycleConsistent_Learning_for_Instruction_Following_and_Generation_in_Vision-Language_CVPR_2022_paper.html)| CVPR | CF | [Author](https://github.com/HanqingWangAI/CCC-VLN) |
| [Debiased Learning from Naturally Imbalanced Pseudo-Labels](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Debiased_Learning_From_Naturally_Imbalanced_Pseudo-Labels_CVPR_2022_paper.html)| CVPR | CF | [Author](https://github.com/frank-xwang/debiased-pseudo-labeling) |
| [Evaluating and Mitigating Bias in Image Classifiers: A Causal Perspective Using Counterfactuals](https://openaccess.thecvf.com/content/WACV2022/html/Dash_Evaluating_and_Mitigating_Bias_in_Image_Classifiers_A_Causal_Perspective_WACV_2022_paper.html) | WACV | CF | - |
|[Invariant Grounding for Video Question Answering](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Invariant_Grounding_for_Video_Question_Answering_CVPR_2022_paper.html)| CVPR | IT | [Author](https://github.com/yl3800/IGV ) |
| [Learning and Evaluating Graph Neural Network Explanations based on Counterfactual and Factual Reasoning](https://dl.acm.org/doi/abs/10.1145/3485447.3511948)| WWW | CF | - |
|[Learning disentangled representations in the imaging domain](https://www.sciencedirect.com/science/article/pii/S1361841522001633)| Medical Image Analysis | 0 | [Author](https://github.com/vios-s/disentanglement_tutorial) |
|[Weakly supervised causal representation learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/fa567e2b2c870f8f09a87b6e73370869-Abstract-Conference.html)| NeurIPS | CR | - |
| [Weakly Supervised Disentangled Generative Causal Representation Learning](https://dl.acm.org/doi/abs/10.5555/3586589.3586830) | JMLR | CR | [Author](https://github.com/xwshen51/DEAR) |



### 2022 other papers with lower citations

> | Title | Venue | Type | Code |
> | :--- | :---: | :---: | :---: |
> |[Diffusion Visual Counterfactual Explanations](https://proceedings.neurips.cc/paper_files/paper/2022/hash/025f7165a452e7d0b57f1397fed3b0fd-Abstract-Conference.html)| NeurIPS | CF | [Author](https://github.com/valentyn1boreiko/DVCEs) |
> | [Show, Deconfound and Tell: Image Captioning with Causal Inference](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Show_Deconfound_and_Tell_Image_Captioning_With_Causal_Inference_CVPR_2022_paper.html) | CVPR | IT | [Author](https://github.com/CUMTGG/CIIC) |
> |[The Role of Deconfounding in Meta-learning](https://proceedings.mlr.press/v162/jiang22a.html) | ICML | 0 | - |


## 6. 2023 Venues

因为时间因素，该部分的一些论文的引用量可能不高。

| Title | Venue | Type | Code |
| :--- | :---: | :---: | :---: |
| [Context De-confounded Emotion Recognition](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Context_De-Confounded_Emotion_Recognition_CVPR_2023_paper.pdf) | CVPR | 0 | - |
| [Counterfactual Samples Synthesizing and Training for Robust Visual Question Answering](https://ieeexplore.ieee.org/abstract/document/10164132) | Pattern Analysis and Machine Intelligence | CF | [Author]( https://github.com/hengyuan-hu/bottom-up-attention-vqa) | 
| [Cross-Modal Causal Relational Reasoning for Event-Level Visual Question Answering](https://ieeexplore.ieee.org/abstract/document/10146482) | Pattern Analysis and Machine Intelligence | IT | [Autthor](https://github.com/HCPLab-SYSU/CMCIR) |



## 7. Arxiv

| Title | Venue | Type | Code |
| :--- | :---: | :---: | :---: |
| [Learning Causal Semantic Representation for Out-of-Distribution Prediction](https://arxiv.org/abs/2011.01681) | Arxiv | CR | - |
| [ECINN: Efficient Counterfactuals from Invertible Neural Networks](https://arxiv.org/abs/2103.13701) | Arxiv | CF | - |
| [A Structural Causal Model for MR Images of Multiple Sclerosis](https://arxiv.org/abs/2103.03158) | Arxiv | CF | - |
| [Counterfactual Variable Control for Robust and Interpretable Question Answering](https://arxiv.org/abs/2010.05581) | Arxiv | CF | [PyTorch(Author)](https://github.com/PluviophileYU/CVC-QA) |
| [Deconfounded Image Captioning: A Causal Retrospect](https://arxiv.org/abs/2003.03923) | Arxiv | IT | - |
| [Latent Causal Invariant Model](https://arxiv.org/abs/2011.02203) | Arxiv | 0 | - |
|[Causal Induction From Visual Obervations For Goal Directed Tasks](https://arxiv.org/abs/1910.01751)| Arxiv | 0 | - |
| [Diffusion Causal Models for Counterfactual Estimation](https://arxiv.org/abs/2202.10166) | Arxiv | CF | [Author](https://github.com/vios-s/Diff-SCM) |
|[Explaining Classifiers with Causal Concept Effect (CaCE)](https://arxiv.org/abs/1907.07165)| Arxiv | 0 | - |


## 8. Related Repo

- [Awesome-Causality-in-CV](https://github.com/Wangt-CN/Awesome-Causality-in-CV)
