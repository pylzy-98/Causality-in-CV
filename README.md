# Causality-in-ML

关于机器学习中因果相关的论文清单。其中部分内容参考了Awesome-Causality-in-CV。


## Contents
1. Type of Causality in ML
2. Before 2020
3. 2020 Venues
4. 2021 Venues
5. 2022 Venues
6. 2023 Venues
7. Arxiv

## 1. Type of Vausality in ML

| Type | `IT` | `CF` | `CR` | `Else` |
| :---: | :---: | :---: | :---: | :---: |
| Explanation | Intervention | Counterfactual | Causal Representation | Other Types |

## 2. Before 2020

| Title | Venue | Type | Code |
| :--- | :---: | :---: | :---: |
| [Counterfactual Visual Explanations](https://arxiv.org/abs/1904.07451) | ICML2019 | CF | - |
| [CausalGAN: Learning Causal Implicit Generative Models with Adversarial Training](https://arxiv.org/abs/1912.07538) | ICLR2018 | 0 | [Pytorch(Author)](https://github.com/mkocaoglu/CausalGAN) |
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
| [Counterfactuals uncover the modular structure of deep generative models](https://arxiv.org/abs/1912.07538) | ICLR | CF | - |
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

## 5. 2022 Venues

## 6. 2023 Venues


## 7. Arxiv

| Title | Venue | Type | Code |
| :--- | :---: | :---: | :---: |
| [Learning Causal Semantic Representation for Out-of-Distribution Prediction](https://arxiv.org/abs/2011.01681) | Arxiv | CR | - |
| [ECINN: Efficient Counterfactuals from Invertible Neural Networks](https://arxiv.org/abs/2103.13701) | Arxiv | CF | - |
| [A Structural Causal Model for MR Images of Multiple Sclerosis](https://arxiv.org/abs/2103.03158) | Arxiv | CF | - |
| [Counterfactual Variable Control for Robust and Interpretable Question Answering](https://arxiv.org/abs/2010.05581) | Arxiv | CF | [PyTorch(Author)](https://github.com/PluviophileYU/CVC-QA) |
| [Deconfounded Image Captioning: A Causal Retrospect](https://arxiv.org/abs/2003.03923) | Arxiv | IT | - |
| [Latent Causal Invariant Model](https://arxiv.org/abs/2011.02203) | Arxiv | 0 | - |


## 8. Related Repo

- [Awesome-Causality-in-CV](https://github.com/Wangt-CN/Awesome-Causality-in-CV)
