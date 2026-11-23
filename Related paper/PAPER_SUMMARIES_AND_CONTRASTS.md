# In-Depth Paper Summaries & Direct Architectural Contrasts

This document provides structured, one-page technical breakdowns for the 15 most critical related papers across the computer vision literature. Each entry outlines the paper's core methodology, its fundamental failure mode or bottleneck in fine-grained athletic video analysis, and how **CMAT-HOI** directly improves upon or resolves the issue.

---

## 1. Video Action Recognition & Fine-Grained Sports Benchmarks

### 1.1 FineGym: A Hierarchical Video Dataset for Fine-grained Action Understanding (CVPR 2020, Oral)
- **Authors**: Dian Shao, Yue Zhao, Bo Dai, Dahua Lin
- **Core Problem**: Traditional video benchmarks (e.g., UCF101, Kinetics) focus on coarse action categories (e.g., swimming vs. cycling). Real-world athletic analysis demands discriminating between subtle sub-actions and routines within the same sporting event.
- **Key Methodology**: Introduced a three-level hierarchical taxonomy (Events $\to$ Sets $\to$ Elements) on gymnastics videos, providing precise temporal boundaries for elementary movements (e.g., distinguishing between vault salto variations with different rotation degrees).
- **Limitation / Gap**: While FineGym demonstrated the necessity of fine-grained modeling, the authors relied predominantly on standard 2D/3D CNN baselines (TSN, TSM, I3D), which struggle to isolate the athlete's physical kinematics from visually cluttered gymnasium apparatus backgrounds.
- **How CMAT-HOI Contrasts & Improves**: CMAT-HOI leverages explicit skeletal pose and equipment cross-attention to resolve fine-grained distinctions without relying on high-capacity 3D frame volumes, attaining 90.19% Top-1 accuracy across 82 distinct sports categories.

---

### 1.2 An Overview of Human Action Recognition in Sports Based on Computer Vision (IEEE Access, 2022)
- **Authors**: Kristina Host, Miran Pobar, Marina Ivašić-Kos
- **Core Problem**: Comprehensive systematization of computer vision algorithms applied to sports analytics across individual, dual, and team sports.
- **Key Methodology**: Categorized sports actions based on complexity, movement velocity, and athlete-environment interactions. Evaluated state-of-the-art pipelines across broadcast and field footage.
- **Limitation / Gap**: The survey emphasized that unimodal systems encounter an impasse: RGB appearance suffers from camera motion and lighting, while skeleton models suffer from equipment blindness. The authors identified multimodal joint-gear integration as an open research frontier.
- **How CMAT-HOI Contrasts & Improves**: CMAT-HOI directly fulfills this identified frontier by implementing the first unified transformer combining ResNet-50 visual features, Keypoint R-CNN skeletons, Faster R-CNN equipment boxes, and pairwise geometric HOI vectors into a single 5.2M parameter model.

---

### 1.3 Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset (CVPR 2017)
- **Authors**: Joao Carreira, Andrew Zisserman
- **Core Problem**: Extending successful 2D ImageNet convolutional filters to 3D spatio-temporal video volumes.
- **Key Methodology**: Introduced the Two-Stream Inflated 3D ConvNet (I3D), inflating 2D convolutional filters into 3D kernels ($N \times N \to N \times N \times N$) to learn continuous spatio-temporal features from RGB and optical flow.
- **Limitation / Gap**: I3D possesses severe *scene bias*: the model frequently classifies sports actions based on dominant background textures (e.g., tennis court clay, swimming pool water, grassy soccer fields) rather than the athlete's actual physical posture. Furthermore, optical flow computation is computationally prohibitive ($>100\times$ slower than real-time).
- **How CMAT-HOI Contrasts & Improves**: CMAT-HOI dispenses with heavy optical flow and 3D convolutions. It computes 1st-order kinematic velocities ($\Delta p_t$) directly on skeletal joints in $O(1)$ time, and uses GMFU gating to suppress static background textures during rapid athletic gestures.

---

### 1.4 SlowFast Networks for Video Recognition (ICCV 2019)
- **Authors**: Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, Kaiming He
- **Core Problem**: Balancing spatial high-resolution appearance with high-temporal-frequency motion without astronomical FLOP counts.
- **Key Methodology**: Dual-pathway 3D CNN: A *Slow pathway* with high spatial resolution and low frame rate ($T/\tau$) to capture static scene details, paired with a *Fast pathway* with low spatial resolution and high frame rate ($T$) to capture rapid motion dynamics, fused via lateral connections.
- **Limitation / Gap**: SlowFast still treats the video volume as an ungrounded pixel grid. It has no structural knowledge of human anatomy or athletic apparatus. In fine-grained sports (e.g., distinguishing a forehand slice from a topspin strike), pixel differences are subtle, leading to misclassifications when camera angles change.
- **How CMAT-HOI Contrasts & Improves**: CMAT-HOI explicitly anchors features to Keypoint R-CNN skeletal coordinates and Faster R-CNN sports apparatus boxes, providing structural invariance across varying broadcast camera angles.

---

### 1.5 Video Swin Transformer (CVPR 2022)
- **Authors**: Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, Han Hu
- **Core Problem**: Adapting Vision Transformers to long video sequences while avoiding quadratic self-attention complexity $O(T \cdot H \cdot W)^2$.
- **Key Methodology**: Extended the Swin Transformer's shifted window self-attention mechanism to 3D video spatio-temporal patches, restricting attention computation to local $P \times M \times M$ non-overlapping windows.
- **Limitation / Gap**: Video Swin requires massive pre-training datasets (Kinetics-400 / Kinetics-600) and substantial GPU compute ($>88$M parameters for Swin-B). Moreover, fine-grained sports apparatus (e.g., golf ball, tennis racket grip) are easily diluted or blurred within large spatio-temporal patch tokens.
- **How CMAT-HOI Contrasts & Improves**: CMAT-HOI isolates equipment as an explicit dedicated stream ($X_O \in \mathbb{R}^{T \times 7}$), guaranteeing that small sports gear is never lost inside visual patch pooling, all within a compact 5.2M parameter budget.

---

## 2. Skeleton-Based Action Recognition & Graph Neural Networks

### 2.1 Spatial Temporal Graph Convolutional Networks (ST-GCN) (AAAI 2018)
- **Authors**: Sijie Yan, Yuanjun Xiong, Dahua Lin
- **Core Problem**: Traditional skeleton action recognition flattened joint coordinates into vector sequences, discarding the natural anatomical graph structure of the human body.
- **Key Methodology**: Constructed a spatiotemporal graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ where joints are nodes and physical bones are edges. Applied spatial graph convolutions across human limbs and 1D temporal convolutions along the time dimension.
- **Limitation / Gap**: **Object Blindness**: ST-GCN operates entirely in a vacuum without environmental or gear awareness. Actions defined by external tools (e.g., *CricketShot* vs. *BaseballPitch*, or *TennisSwing* vs. *BadmintonSmash*) have virtually identical joint coordinate trajectories and are frequently confused.
- **How CMAT-HOI Contrasts & Improves**: CMAT-HOI directly incorporates the sports apparatus via the `Faster R-CNN` stream and binds it to the skeleton using mutual Multi-Head Cross-Attention ($h=4$), boosting accuracy from 52.15% (skeleton-only baseline) to 90.19%.

---

### 2.2 Two-Stream Adaptive Graph Convolutional Networks (2s-AGCN) (CVPR 2019)
- **Authors**: Lei Shi, Yifan Zhang, Jian Cheng, Hanqing Lu
- **Core Problem**: Fixed, hand-crafted skeleton adjacency matrices in ST-GCN prevent the network from discovering relationships between non-physically connected joints (e.g., hand-to-foot coordination in athletic jumping).
- **Key Methodology**: Formulated an adaptive graph convolutional layer where the adjacency matrix is parameterized as $A = A_k + B_k + C_k$, combining fixed physical topology ($A_k$), data-dependent learned global connections ($B_k$), and sample-specific attention graphs ($C_k$). Combined joint and bone streams into a two-stream model.
- **Limitation / Gap**: While adaptive topologies capture inter-joint coordination, the model remains incapable of modeling non-human physical objects. In equipment-heavy sports, the hand's relationship to the ball or racket is far more informative than hand-to-foot coordination.
- **How CMAT-HOI Contrasts & Improves**: CMAT-HOI models the interaction between limbs and sports equipment through dynamic cross-attention and pairwise geometric distances ($X_H \in \mathbb{R}^{T \times 16}$), explicitly capturing the hand-to-apparatus vector.

---

### 2.3 PoseConv3D: Revisiting 3D CNNs for Skeleton-based Action Recognition (CVPR 2022)
- **Authors**: Haodong Duan, Yue Zhao, Kai Chen, Dahua Lin, Bo Dai
- **Core Problem**: GCNs suffer from sensitivity to coordinate noise/jitter, struggle to scale to multi-person scenarios, and cannot easily fuse with standard 3D visual CNN architectures.
- **Key Methodology**: Converted 2D skeleton keypoint sequences into continuous 3D heatmap volumes (stacking Gaussian joint heatmaps across time) and processed them using standard 3D CNN backbones.
- **Limitation / Gap**: Rasterizing keypoints into 3D heatmaps inflates memory consumption significantly. Furthermore, sports equipment (which often consists of thin, metallic, or fast-moving apparatus) cannot be naturally represented as human anatomical heatmaps.
- **How CMAT-HOI Contrasts & Improves**: CMAT-HOI keeps skeletal keypoints as lightweight coordinate tensors ($85$-d) and equipment as bounding parameters ($7$-d), achieving 16.28 batches/sec evaluation latency on a single workstation GPU without 3D heatmap rasterization.

---

## 3. Human–Object Interaction (HOI) & Spatio-Temporal Scene Graphs

### 3.1 QPIC: Query-Based Pairwise Human-Object Interaction Detection (CVPR 2021)
- **Authors**: Masato Tamura, Hiroki Ohashi, Tomoaki Yoshinaga
- **Core Problem**: Prior two-stage HOI detectors detected humans and objects separately and evaluated all exhaustive $O(N \times M)$ pairs, leading to severe combinatorial explosion and false positives in crowded scenes.
- **Key Methodology**: Single-stage Transformer framework using learned interaction queries to directly predict human-object bounding box pairs and interaction action categories simultaneously via bipartite matching loss.
- **Limitation / Gap**: QPIC is strictly an **image-level detector**. It evaluates static 2D frames and has zero temporal aggregation capability. It cannot distinguish between dynamic action phases (e.g., preparation vs. ball impact vs. follow-through).
- **How CMAT-HOI Contrasts & Improves**: CMAT-HOI extends HOI principles to video sequences by feeding enhanced cross-attention representations into a 3-layer Spatio-Temporal Transformer Encoder, modeling how limb-equipment relationships evolve across $T=16$ equidistant frames.

---

### 3.2 HOTR: End-to-End Human-Object Interaction Detection with Transformers (CVPR 2021)
- **Authors**: Bumsoo Kim, Junhyun Lee, Jaewoo Kang, Eunsung Kim, Hyunwoo J. Choi
- **Core Problem**: Eliminating hand-crafted post-processing (e.g., non-maximum suppression and pairwise heuristic matching) in HOI detection.
- **Key Methodology**: Employed dual transformer decoders (Object Detection Query & Interaction Detection Query) with a semantic pointer matching scheme to link human, object, and verb predictions in single static images.
- **Limitation / Gap**: Like QPIC, HOTR operates on static images (e.g., HICO-DET and V-COCO). In athletic competitions, an instantaneous snapshot of an athlete holding a racket could be a serve, a return, or a celebration; temporal context is indispensable.
- **How CMAT-HOI Contrasts & Improves**: CMAT-HOI couples frame-wise equipment interactions with 1st-order temporal displacement velocities ($\Delta p_t$) and multi-scale temporal pooling, achieving 90.19% Top-1 accuracy on complex dynamic video clips.

---

### 3.3 Action Genome: Actions as Compositions of Spatio-Temporal Scene Graphs (CVPR 2020)
- **Authors**: Jingwei Ji, Ranjay Krishna, Li Fei-Fei, Juan Carlos Niebles
- **Core Problem**: Actions are traditionally represented as monolithic global labels, ignoring the rich underlying sequence of contact, spatial, and attention relationships between people and objects.
- **Key Methodology**: Introduced the Action Genome benchmark, annotating spatio-temporal scene graphs (human-object bounding box trajectories and time-varying relational edges) across Charades videos.
- **Limitation / Gap**: Action Genome scene graph generation relies on heavy dense graph parsing, which is computationally prohibitive for fast real-time inference and focused on coarse indoor household activities (e.g., sitting on a chair, holding a cup) rather than high-velocity sports.
- **How CMAT-HOI Contrasts & Improves**: CMAT-HOI condenses sports interaction into an ultra-compact 16-dimensional geometric HOI vector (wrist/ankle-to-apparatus Euclidean distance, orientation angles, and rates of change), providing scene-graph-level geometric reasoning at negligible computational overhead.

---

## 4. Multimodal Fusion & Representation Collapse

### 4.1 What Makes Training Multi-modal Networks Hard? (NeurIPS 2020)
- **Authors**: Weiyao Wang, Du Tran, Matt Feiszli
- **Core Problem**: Counter-intuitively, jointly training multimodal networks often yields *worse* accuracy than training the best unimodal baseline alone.
- **Key Methodology**: Identified two core mathematical causes: (1) increased capacity causes greater overfitting; (2) different modalities generalize and overfit at vastly different rates under a single global gradient update ("greedy learning"). Proposed *Gradient-Blending* to balance loss updates.
- **Limitation / Gap**: While Gradient-Blending addresses optimization dynamics across visual and audio streams, it requires computing empirical Overfitting-to-Generalization Ratios (OGR) via expensive auxiliary validation evaluations during training.
- **How CMAT-HOI Contrasts & Improves**: CMAT-HOI prevents dominant modality collapse structurally through **Dimension-Aligned Modality Stems**: projecting all 4 streams ($2048$-d, $85$-d, $7$-d, $16$-d) into an identical $d=256$ representation space before fusion, ensuring equalized initial gradient flow without auxiliary OGR computations.

---

### 4.2 Attention Bottlenecks for Multimodal Fusion (NeurIPS 2021)
- **Authors**: Arsha Nagrani, Shan Yang, Anurag Arnab, Aren Jansen, Cordelia Schmid, Chen Sun
- **Core Problem**: Dense all-to-all cross-modal attention across unaligned video and audio tokens results in quadratic complexity $O(N^2)$ and introduces noisy cross-modal correlations.
- **Key Methodology**: Introduced the Multimodal Bottleneck Transformer (MBT), routing cross-modal communication through a small set of shared, learnable bottleneck latent tokens ($B \ll N$).
- **Limitation / Gap**: MBT is designed for general video-audio classification, where tokens are ungrounded spatial/spectral patches. In athletic activities, interactions are physically grounded between specific human effectors (wrists/hands) and sports equipment.
- **How CMAT-HOI Contrasts & Improves**: Rather than general bottleneck tokens, CMAT-HOI uses physically grounded query-key-value cross-attention between skeletal keypoints and equipment coordinates, directly filtering out irrelevant scene objects.

---

### 4.3 Multimodal Transformer for Unaligned Multimodal Language Sequences (ACL 2019)
- **Authors**: Yao-Hung Hubert Tsai, Shaojie Bai, Paul Pu Liang, J. Zico Kolter, Louis-Philippe Morency, Ruslan Salakhutdinov
- **Core Problem**: Multimodal time-series (language, vision, audio) operate at different sampling frequencies and temporal alignments.
- **Key Methodology**: Proposed MulT, using directional cross-modal attention blocks where one modality's sequence queries the sequence of another modality ($Q=M_1, K=M_2, V=M_2$), aligning unaligned features without explicit temporal warping.
- **Limitation / Gap**: MulT was designed for multimodal sentiment analysis (CMU-MOSEI). Applying pairwise cross-attention across all possible modality combinations causes quadratic scaling in the number of modalities ($M \times (M-1)$ cross-attention blocks).
- **How CMAT-HOI Contrasts & Improves**: CMAT-HOI tactically restricts bi-directional cross-attention exclusively to the interacting pair (**Pose $\leftrightarrow$ Equipment**), using the lightweight GMFU unit to fuse visual context and geometric kinematics, keeping total parameters at only 5.2M.

---

### 4.4 Gated Multimodal Units for Information Fusion (ICLR Workshop 2017)
- **Authors**: John Arevalo, Thamar Solorio, Manuel Montes-y-Gómez, Fabio A. González
- **Core Problem**: Linear combination or static concatenation of multimodal representations cannot adjust when one modality is corrupted, occluded, or uninformative in certain samples.
- **Key Methodology**: Formulated the Gated Multimodal Unit (GMU), using a sigmoid gating neuron per modality to dynamically scale feature vectors based on sample context.
- **Limitation / Gap**: GMU applied independent sigmoid gates without a cross-modality competitive constraint, allowing activations to grow unstably across deep architectures. Furthermore, it operated on static vectors rather than temporal video sequences.
- **How CMAT-HOI Contrasts & Improves**: CMAT-HOI’s GMFU incorporates a frame-wise **Softmax constraint** ($\sum_{m=1}^4 g_t^{(m)} = 1$) over temporal sequences, ensuring stable convex combination and allowing visual background cues to be dynamically suppressed when kinematic signals dominate.
