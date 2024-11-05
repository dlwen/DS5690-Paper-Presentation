# Multimodal Chain-of-Thought Reasoning in Language Models
Zhang, Z., Zhang, A., Li, M., Zhao, H., & Smola, A. (2024). Multimodal Chain-of-Thought Reasoning in Language Models. Transactions on Machine Learning Research

Author:  
Zhuosheng Zhang  zhangzs@sjtu.edu.cn  
Aston Zhang  az@astonzhang.com  
Mu Li  muli@cs.cmu.edu  
Hai Zhao  zhaohai@cs.sjtu.edu.cn  
George Karypis  gkarypis@amazon.com  
Alex Smola  alex@smola.org

Link to the paper: https://arxiv.org/abs/2302.00923  
Link to the paper repo: [GitHub - amazon-science/mm-cot: Official implementation for “Multimodal Chain-of-Thought Reasoning in Language Models” (stay tuned and more will be updated)](https://github.com/amazon-science/mm-cot)

<br>

## Overview
Large Language Models (LLMs)  have demonstrated impressive capabilities in complex reasoning tasks through chain-of-thought (CoT) prompting. The development of CoT prompting allows models to break down complex problems into intermediate steps, similar to human thought processes. However, the CoT in only on text data. The integration of multiple modalities, particularly combining visual and textual information in reasoning tasks, remains a significant challenge. Therefore, the author propose Multimodal-CoT that incorporates language (text) and vision (images) modalities into a two-stage framework that separates rationale generation and answer inference. In this way, answer inference can leverage better generated rationales that are based on multimodal information. With Multimodal- CoT, their model under 1 billion parameters achieves state-of-the-art performance on the ScienceQA benchmark. The analysis indicates that Multimodal-CoT offers the advantages of mitigating hallucination and enhancing convergence speed.
<img width="800" alt="Screen Shot 2024-11-05 at 02 03 56" src="https://github.com/user-attachments/assets/a2e59508-3cc2-42f8-97a0-049adad397e6">


<br>


## Backgrounds
### What is CoT?
Chain of Thought (CoT) is a prompting technique that encourages models to break down complex problems into smaller, logical steps before going to a final answer, similar to how humans think through problems. It was first introduced in 2022 through paper  [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903.pdf). By explicitly showing its reasoning process, the model is more likely to arrive at accurate conclusions and can help users understand how it reached its answer. This approach is particularly effective for tasks involving math, logic, or complex reasoning. Existing studies related to CoT reasoning are largely isolated in the language modality (Wang et al., 2022c; Zhou et al., 2022; Lu et al., 2022b; Fu et al., 2022), with little consideration of multimodal scenarios. 
<img width="875" alt="Screen Shot 2024-11-02 at 22 58 16" src="https://github.com/user-attachments/assets/ec4b00d6-15b5-4448-b72f-a8b407b6f730">


### One-Stage CoT
**Input**  
Q: the concatenation of tokens of the question text  
C: the context text  
M: multiple options 

**Output**  
 (i) QCM→A：No-CoT which predicts the answer directly  
 (ii) QCM→RA: Reasoning where answer inference is conditioned to the rationale  
(iii) QCM→AR: Explanation where the rationale is used for explaining the answer inference 

**Question 1**  
Which would have the higher accuracy?  
  
  **Answer**  
<img width="431" alt="Screen Shot 2024-11-02 at 23 57 19" src="https://github.com/user-attachments/assets/2e9f2e5d-4a93-47cf-abcd-291b21942c81">
* A ↓12.31% accuracy decrease (81.63%→69.32%) if the model predicts rationales before answers (QCM→RA). The results imply that the rationales might not necessarily contribute to predicting the right answer. 
* **Hallucinated Rationales**: models make up false information that isn’t supported by the input data.

  <br>


## PROBLEM & MOTIVATION:
* Current CoT methods are limited to text only
* Real-world reasoning often requires multiple modalities (text + images)
* How to enable effective reasoning across modalities?

<br>

## Challenge  
**Architectural Challenges:**  
* How to effectively combine visual and textual information in a way that preserves the benefits of CoT reasoning  
* Designing a system that can handle both modalities without losing the coherence of the reasoning process  

**Model Size:**  
* Most successful CoT implementations rely on very large language models (>100B parameters). Smaller models (<1B parameters) tend to generate hallucinated rationales  
* As observed by [Wei et al. (2022b)](https://arxiv.org/abs/2201.11903) , models under 100 billion parameters tend to produce illogical CoT that leads to wrong answers. In other words, it might be harder for 1B-models to generate effective CoT than directly generating the answer. It becomes even more challenging in a multimodal setting where answering the question also requires understanding the multimodal inputs  

**Quality of Reasoning:**  
* Ensuring that visual information genuinely contributes to the reasoning process  
* Preventing hallucination in multimodal contexts  
* Maintaining coherent reasoning chains across modalities  

<br>

## KEY INNOVATION:  
* Two-stage framework separating:  
	1 Rationale Generation  
	2 Answer Inference  
* Achieves SOTA with smaller model (<1B parameters)  
* Reduces hallucination and improves convergence speed  


## Method: Two-Stage framework
### Model General Architecture
CoT problem into two stages:
Rationale Generation (using both text + image)
Answer Inference (using rationale + original inputs)

![](Screen%20Shot%202024-11-02%20at%2023.33.08.png)

In the first stage, the model takes both language input (question, context, and options) and vision input (image features extracted by ViT) to generate a reasoned rationale. This stage allows the model to form a coherent understanding of the problem by integrating information from both modalities. The vision features are processed through a ViT encoder and fused with the language features using an attention-based mechanism, enabling the model to generate rationales that incorporate both textual and visual context.

In the second stage, the generated rationale is appended to the original language input, creating an enhanced context for answer inference. This stage takes this combined input along with the original vision features to produce the final answer. The key innovation here is that the answer inference can use better-generated rationales that are grounded in multimodal information, reducing hallucination and improving accuracy. 

Each stage is trained independently but shares the same model architecture, using a T5 encoder-decoder backbone enhanced with vision processing capabilities. This separation of tasks allows each stage to specialize in its task while maintaining the benefits of multimodal reasoning throughout the process. Experimental results show that this two-stage approach not only achieves state-of-the-art performance but does so with a relatively small model of less than 1 billion parameters, making it both effective and practical for real-world applications.


### Multimodality Contributes to Reduce Hallucination 
One of the key challenges in language models is hallucination mistakes in rationale generation. In the original problem space, 56% of model errors were due to hallucination, with the remaining 44% due to other types of mistakes. This high rate of hallucination is a significant problem that needed to be addressed.

The authors demonstrate how their multimodal approach helps solve this problem. They explore two different methods to incorporate visual information into their model: using image captions and using direct vision features. 

**Question 2: Which would method can more effectively reduce hallucination, Caption or Vision Features?**

**Caption**
In the caption approach, they first convert the image into text descriptions and append these captions to the input. However, this led to minimal improvements - the rationale generation quality increased slightly from 90.73% to 90.88% RougeL score, and answer accuracy improved marginally from 78.57% to 79.37%. This suggests that converting visual information into text loses important details and does not fully solve the hallucination problem.

**Vision Features**
Their vision features approach, which directly processes the image using a Vision Transformer (ViT) and integrates these features with the text through an attention mechanism, showed much more substantial improvements. This method increased rationale generation quality from 90.73% to 93.46% RougeL score and significantly boosted answer accuracy from 78.57% to 85.31%.  The stark difference between the caption approach (+0.80% improvement) and the vision features approach (+6.74% improvement) demonstrates that direct visual feature processing is much more effective at grounding the model’s reasoning in actual evidence.

![](Screen%20Shot%202024-11-03%20at%2001.18.28.png)

With those effective rationales, the phenomenon of hallucination is mitigated — 60.7% hallucination mistakes have been corrected while only 29.3% remained unresolved. Vision features are indeed beneficial for generating effective rationales and contributing to accurate answer inference. The significant improvement in error correction demonstrates the power of combining multiple modalities in LLM reasoning.
![](Screen%20Shot%202024-11-03%20at%2001.20.13.png)


## Pseudocode
![](Screen%20Shot%202024-11-04%20at%2023.50.20.png)
![](Screen%20Shot%202024-11-04%20at%2023.48.26.png)



## Experiments
The paper adopts the T5 encoder-decoder architecture under Base (200M) and large (700M) settings in framework. The vision features are obtained by the frozen ViT-large encoder.  The experiments fine-tune the models up to 20 epochs, with a learning rate of 5e-5. The maximum input sequence length is 512. The batch size is 8. The experiments are run on 8 NVIDIA Tesla V100 32G GPUs. 

### Dataset
**ScienceQA**
![](Screen%20Shot%202024-11-04%20at%2018.53.09.png)
Baseline Models:
i) Visual question answering (VQA) models
(ii) LMs
(iii) Fine-tuned large vision-language model
Chameleon, LLaMA-Adapter, LLaVA, and InstructBLIP are concurrent works released several months after this paper
Mutimodal-CoTLarge achieves substantial performance gains over the prior best model in publications (86.54%→90.45%) 

**A-OKVQA**
Multimoal-CoTBase also has the best accuracy over all chosen model on A-OKVQA dataset.
![](7F7B4FD2-EF89-456E-BA82-82C59FBB6400.png)



**Multimodality Boosts Convergence**
![](Screen%20Shot%202024-11-04%20at%2018.59.41.png)



One-stage: the No-CoT baseline QCM→A input-output format
Two-stage: Two-stage framework. 

The two-stage framework starts with better accuracy compared to the simpler one-stage methods that produce answers without reasoning steps. However, when vision features are not included, the two-stage model’s performance doesn’t improve much over time because it generates poor-quality explanations. On the other hand, when vision features are added, the model creates better reasoning steps that lead to more accurate answers in our Multimodal-CoT approach.


## Critical Analysis
### Architecture vs. Data Contribution:
* While the authors emphasize their two-stage architecture, the results suggest that the major improvement comes from the addition of visual features rather than the architectural innovation itself. According to the graph above, both one-stage and two-stage models show significant improvement when visual features are added, with the one-stage multimodal approach achieving similar performance to the two-stage baseline. This suggests that the rich information provided by visual features is the primary driver of improvement, rather than the proposed two-stage framework. 
* The authors could have strengthened their architectural claims by conducting experiments that isolated the benefits of their two-stage approach from the inherent advantages of having additional visual information.
### Unexplored Alternative Modal Interaction Patterns
* The paper is the lack of exploration of alternative data flow designs. The current architecture follows a fixed pattern where visual features are computed and fused with language data, and then this fused information is used again with vision data in the second stage. However, the authors did not investigate other potential arrangements of modality interaction. 
* They could have experimented with: 
Stage 1: Input {Xlanguage, Xvision1} → Generate R
Stage 2: Input {Xlanguage, Xvision2} → Generate Answer (where Xvision2 incorporates R with original visual features)


# Impact
## What is the impact of the work?
* First formal approach to extend Chain-of-Thought to multimodal scenarios
* Shows smaller models can achieve strong reasoning with multimodal input
* Provides practical solution to reduce hallucination in language models

## How does/did the work change the landscape of AI?
* Bridges the gap between language reasoning and visual understanding
* Demonstrates that combining modalities can improve reliability of AI reasoning
* Makes multimodal reasoning possible without massive models or computational resources

## Importance of the work
* Shows how adding images helps AI reason better and make fewer mistakes.
* Achieves great results with smaller, cheaper models anyone can use.
* Opens new possibilities for adding other modalities of information (like audio or sensor data) to AI reasoning systems.

## Intersection with Other Work
### Past:
Builds on Chain-of-Thought prompting research
Extends visual-language models to include reasoning capabilities
### Present:
Provides framework for combining multiple modalities in AI reasoning
Shows way to reduce hallucination through visual feature extraction
### Future:
Framework can potentially be extended to other modalities beyond vision (audio, sensor data, etc.)
Two-stage approach could be adapted for other types of reasoning tasks




# Reference
* Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, and Denny Zhou. Rationale-augmented ensembles in language models. _ArXiv preprint_, abs/2207.00747, 2022c. 
* Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Olivier Bousquet, Quoc Le, and Ed Chi. Least-to-most prompting enables complex reasoning in large language models. _ArXiv preprint_, abs/2205.10625, 2022. 
* Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning. _ArXiv preprint_, abs/2209.14610, 2022b. 
* Yao Fu, Hao Peng, Ashish Sabharwal, Peter Clark, and Tushar Khot. Complexity-based prompting for multi-step reasoning. _ArXiv preprint_, abs/2210.00720, 2022. 
