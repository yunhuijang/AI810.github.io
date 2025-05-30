---
layout: home
title: AI810 Blog Post (20255347)
permalink: /
---

# From Sequences to Structures: The Rise of Language Models in Protein Prediction


Recent advances in large language models (LLMs) have revolutionized fields such as machine translation, arithmetic reasoning, and code generation, where AI now competes or even surpasses human performance—raising discussions about the future of replacing human jobs. Beyond replacing human tasks, LLMs are also being applied to domains where human capability is limited, such as biology. In particular, protein science has undergone a transformative shift, with deep learning models, such as AlphaFold [3], solving the once-intractable problem of protein structure prediction. While AlphaFold captured global attention for its performance, it relies on computationally intensive multiple sequence alignments (MSAs), which require identifying and aligning related sequences. To overcome this limitation, a new class of models known as Evolutionary Scale Modeling (ESM) has emerged, leveraging LLMs trained directly on raw protein sequences without MSAs. This blog post introduces the ESM series, with a focus on the recent developments in ESMFold [2] and ESM-3 [1].


## Overview: New Paradigm of Protein Structure Prediction with LLMs

Traditional protein structure prediction relied heavily on physics-based simulations, which is very slow. A breakthrough came with AlphaFold2, which leverages multiple sequence alignments (MSAs) and attention-based neural networks to infer 3D structure from amino acid sequences. However, generating MSAs is computationally expensive, as it involves searching massive databases to identify related sequences and aligning them to identify conserved subsequences.

<img src="https://hackmd.io/_uploads/r1XUeSHMgg.png" alt="MSA: computationally expensive alignment to identify conserved mutations" width="400px">
*MSA: computationally expensive alignment to identify conserved mutations*

In contrast, the protein language models including the ESM series bypass this step by treating protein sequences as a language. These models are trained on millions of protein sequences and learn rich contextual embeddings directly from sequence data. Notably, they capture structural constraints, folding patterns, and even functioning, which is analogous to how GPT learns grammar and semantics from text.

<img src="https://hackmd.io/_uploads/ryhSLBrzll.png" alt="Protein language models" width="400px">
*Protein language models*

⸻


## ESMFold: Scaling Language Models for Structure

### Overview

ESMFold demonstrates that full atomic-level protein structures can be directly inferred from primary protein sequences using LLMs. Its core idea lies in leveraging the power of LLMs-scaling up to 15B parameters—to predict structures without relying on MSAs. By eliminating the need for MSA, ESMFold significantly accelerates the structure prediction process while maintaining high precision. This efficiency opens the door to practical large-scale applications, such as high-throughput metagenomic analysis.

### Architecture

ESMFold consists of three components: ESM-2, folding trunk, and structure module. Specifically, ESM-2 learns the meaningful embeddings of protein sequences with masked language modeling (MLM). Then, the folding trunk is trained for end-to-end single-sequence structure prediction. Next, the output of the folding block is passed to an equivariant transformer structure module. Finally, the recycling is performed before outputting a final atomic-level structure. 

![image](https://hackmd.io/_uploads/r1IYPrrzle.png)
*ESMFold model architecture*

#### ESM-2

First, ESM-2 is trained solely on protein sequences using a masked language modeling (MLM) [4] objective. In detail, MLM randomly masks a subset of amino acids in the input sequence and trains the model to predict these masked ones based on the surrounding context. This enables the model to capture the inter-dependencies of amino acids, which ensures learning the meaningful representations of protein sequences.

#### Folding trunk

Next, the learned representation is passed to a folding trunk. The trunk begins with a series of folding blocks. Each folding block alternates between updating a sequence representation and a pairwise representation. The alternating updates between the sequence and pairwise representations enable the model to iteratively refine both local (residue-wise) and global (pairwise) structure features.

![image](https://hackmd.io/_uploads/S1BTcLrMxe.png)
*Folding block architecture*

#### Structure module

Finally, the updated representations are passed to an equivariant transformer module, which predicts the 3D coordinates of atoms. This ensures that the physical consistency of the predicted structure is preserved.


In contrast to AlphaFold, which relies on heavy use of MSA-based templates, ESM-2's structure module is MSA-free and more modular, enabling faster predictions while retaining competitive accuracy.

### Experiments and results

Here, the experiments validate that ESMFold is not only fast and scalable but also biologically grounded and accurate, offering practical utility.


#### Structure prediction accuracy

ESMFold produces accurate atomic-resolution predictions, achieving performance comparable to RoseTTAFold on the CAMEO benchmark. Notably, when MSAs are omitted in AlphaFold [3] and RoseTTAFold [6], their accuracy significantly declines. This highlights the robustness of ESMFold's MSA-free architecture and validates the effectiveness of language model-based representations in capturing structural information directly from sequence alone. 



![image](https://hackmd.io/_uploads/HJwYsEEzge.png)
*High structure prediction accuracy of ESMFold*

![image](https://hackmd.io/_uploads/SypA6IHMxl.png)
*Successful structure prediction example*

#### Speed

By removing the computationally expensive MSAs, ESMFold achieves up to a 60$\times$ speedup in inference time. Additionally, it eliminates the need for the time-consuming sequence search step, which alone can take over 10 minutes per protein. This dramatic speed advantage enables ESMFold to scale structure prediction to massive metagenomic datasets.

#### Scaling

For the scaling behavior of ESMFold, there exists consistent improvement as model capacity increases. In detail, with larger models, ESM-2 achieves lower RMSD and higher pLDDT confidence scores, which suggests better sequence understanding. Notably, the 15B parameter model delivers the most accurate prediction (RMSD = 2.6, pLDDT = 75.6), demonstrating a clear scaling advantage in both structure and sequence modeling. 
![image](https://hackmd.io/_uploads/SyBysVEzgl.png)


### Conclusion

In conclusion, ESMFold made a pivotal step in protein structure prediction by demonstrating that atomic-resolution structures can emerge directly from language models trained solely on sequence data. By eliminating the need for MSAs, ESMFold achieves a remarkable acceleration—up to 60$\times$ faster inference—while maintaining high accuracy. This speed and scalability unlock new possibilities for exploring metagenomic datasets at unprecedented scale, enabling the structural annotation of hundreds of millions of previously unseen proteins.


## ESM-3: Towards Unified Multi-modal Protein Prediction

### Overview

The goal of ESM-3 is to take a significant step beyond ESM-2 by exploring the power of multi-modality in protein representation learning. Specifically, ESM-3 introduces a unified pretraining objective that integrates three modalities: structure, sequence, and function. Instead of treating each modality in isolation or fusing them only at downstream tasks, ESM-3 incorporates all of them during the pretraining stage, enabling synergistic learning from the start. Crucially, this is achieved through a shared tokenization scheme, allowing the model to process data in a consistent format. This unified architecture aims to produce general-purpose protein representations that can power a wide range of tasks.

![image](https://hackmd.io/_uploads/HJpbuDSzge.png)
*Multi-modality of ESM-3*

### Architecture

Models such as ESM-2 remain constrained by the limitations of the masked language modeling (MLM) objective, which frames learning as a "fill-in-the-blank" task. This is problematic in the context of proteins, where proteins do not evolve to complete missing residues but rather fold into three-dimensional structures governed by complex structural and functional constraints. Therefore, the MLM objective may fail to capture the underlying biological principles that drive protein behavior. ESM-3 takes a substantial step forward by realigning the training objective with biological reality, which shifts from a pure language-based framework to a multi-modal generative architecture that jointly models sequence, structure, and function in a unified manner.

#### Multi-modal MLM with consistent tokenization

ESM-3 retains the principle of MLM: learning to generate or reconstruct missing information. However, it significantly extends this objective by moving beyond single-modality inputs. Unlike previous models that operate solely on protein sequences, ESM-3 jointly learns from sequence, structure, and function, enabling richer representations.

![image](https://hackmd.io/_uploads/rJBFhvSMxg.png)
*ESM-3 architecture*

Importantly, this is not a late-stage fusion of modalities; rather, all three domains—sequence, 3D structures, and functions—are integrated from the outset of pretraining through a unified tokenization scheme. In particular, structural information is encoded by discretizing the local atomic environment surrounding each amino acid into tokens, enabling the model to represent the structural context in a format compatible with language modeling. This consistent and modality-aware tokenization allows ESM-3 to jointly reason over sequence patterns, structural conformations, and functional roles within a single framework.

![image](https://hackmd.io/_uploads/SJrd0vBGxx.png)
*Structure tokenization of ESM-3*

This early fusion allows the model to align cross-modal signals during training, promoting synergistic learning across molecular modalities.


#### Geometric attention


To effectively process 3D structural information, ESM-3 incorporates geometric attention. Standard attention layers are replaced with modules that are SE(3)-invariant, a critical requirement for protein modeling. Specifically, it combines distance-based attention, which captures how far residues are from each other, with direction-based attention, which encodes the relative orientations between residues. This dual-geometric attention allows the model to faithfully represent interactions essential for folding and function.

#### Alignment

However, the standard MLM objective may not align with the biological reality of protein evolution, as proteins do not evolve to complete artificially masked residues or structures. To address this, ESM-3 introduces a pairwise alignment-based fine-tuning strategy. Specifically, the model is further trained using aligned pairs of proteins. During this stage, the model is trained to prioritize higher-score proteins rather than lower-score proteins. By leveraging alignment, ESM-3 refines its representations to better reflect the functional perspective of proteins.

![image](https://hackmd.io/_uploads/BkjQnmNGex.png)
*Computation of alignment*

### Experiments

Here, the experiments validate that ESM-3 is accurate and adaptable for various prompts including sequences, structures, and functions, offering its practical utility.

#### Basic unconditional generation

ESM-3 demonstrates strong unconditional generative capabilities, capable of producing novel protein structures that broadly cover the structural landscape observed in nature. Proteins generated by ESM-3 (98B) populate a similar distribution as real proteins from the Protein Data Bank (PDB).

![image](https://hackmd.io/_uploads/HJCABmNfxx.png)
*Distribution of unconditional protein generation*

#### Conditional generation


ESM-3 accepts diverse prompts as input due to its multi-modal architecture, enabling it to generate proteins conditioned on sequence, structure, or function. In this example, ESM-3 is prompted with structural constraints, specifically symmetry, and successfully generates proteins exhibiting the desired symmetrical folds. This demonstrates the model's ability to follow structural prompts and produce proper outputs. Such controllable generation highlights the potential of ESM-3 as a flexible tool for protein design.

![image](https://hackmd.io/_uploads/B1dILXVzxx.png)
*Prompting for symmetric proteins*

#### Conditional generation with different combinations of prompts

Moreover, ESM-3 demonstrates the ability to generate proteins conditionally by combining multiple types of prompts. As shown in the figure, ESM-3 can successfully integrate structural folds like alpha/beta hydrolase and functional sites such as motifs and binding sites. The resulting proteins not only adopt the correct overall fold but also position the functional motifs in accurate positions. This highlights ESM-3's potential for protein design, where users can guide the generative model to design multi-objective desirable proteins by simply specifying structural and functional components as prompts.

![image](https://hackmd.io/_uploads/Sku3OQNfex.png)
*Conditional generation with multiple prompts*

#### Scaling

For the scaling behavior of ESM-3, there exists consistent improvement as model capacity increases. In detail, with larger models (1.4B, 7B, and 98B), ESM-3 achieves lower negative log-likelihood (NLL). 

![image](https://hackmd.io/_uploads/ByzVL9LMel.png)
*Scaling of ESM-3*

### Conclusion

In conclusion, ESM-3 marks a significant advancement in protein modeling, moving beyond single-modality approaches. By unifying sequence, structure, and function through a generative multimodal masked language modeling framework, ESM-3 not only learns a richer representation of proteins but also demonstrates controllable generation capabilities. Its geometric attention mechanisms and unified tokenization enable accurate 3D structure reasoning, and its alignment-based fine-tuning reveals remarkable improvements in motif design and functional protein generation.

## Conclusion

The emergence of LLMs in the protein domain marks a paradigm shift in how we understand proteins. Moving beyond methods that rely on MSAs, the ESM series demonstrates that powerful, scalable models can learn biological insights directly from sequence data. ESMFold showed that atomic-resolution structure prediction is achievable without MSAs, unlocking speed and scalability for massive datasets. ESM-3 extends this vision further, unifying sequence, structure, and function into a single generative, multimodal model capable of controllable protein design. Together, these advancements suggest that LLMs are not only tools for understanding biology but also accelerators in scientific discovery that might go beyond human knowledge in the future.


## References
[1] Hayes, T., et al. Simulating 500 million years of evolution with a language model. Science, 2025.

[2] Lin, Z., et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science, 2023.

[3] Jumper, J., et al. Highly accurate protein structure prediction with AlphaFold. Nature, 2021.

[4] Devlin, J., et al. BERT: Pre-training of deep bidirectional transformers for language understanding. NAACL-HLT, 2019.

[5] https://blog.ml6.eu/unlocking-the-secrets-of-life-ai-protein-models-demystified-f286b222d571

[6] Baek, M., et al. Accurate prediction of protein structures and interactions using a three-track neural network. Science, 2021.