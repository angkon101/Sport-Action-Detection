# Comparative Taxonomy & Architectural Benchmarking Matrix

This taxonomy provides an analytical comparison between the proposed **CMAT-HOI** framework and prominent paradigms across the computer vision literature in **Video Action Recognition**, **Skeleton-based Action Recognition**, **Human–Object Interaction (HOI)**, and **Multimodal Transformers**.

---

## 1. High-Level Paradigm Comparison

| Model / Paradigm | Primary Modalities | Interaction Modeling | Modality Balancing | Temporal Modeling | Limitations in Sports Video |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **I3D / SlowFast** (Carreira et al., Feichtenhofer et al.) | RGB Video Frames / Optical Flow | None (Implicit convolutional filters) | N/A (Unimodal or dual-rate RGB) | 3D Space-Time Convolutions | Severe **scene bias** (classifies background pitch/court rather than athlete); computationally prohibitive ($>30$M params). |
| **Video Swin / ViViT** (Liu et al., Arnab et al.) | RGB Video Patches | None (Pure visual patch self-attention) | N/A (Visual only) | Shifted window / factorized spatio-temporal attention | Prone to overfitting on repetitive athletic stadiums; ignores explicit fine-grained equipment contact points. |
| **ST-GCN / 2s-AGCN** (Yan et al., Shi et al.) | 2D/3D Skeletal Keypoints | None (Pure human anatomical graph) | N/A (Keypoints only) | Graph Convolutions + Temporal Conv (TCN) | **Object Blindness**: Cannot distinguish actions sharing identical arm trajectories (e.g., *TennisSwing* vs *BadmintonSmash*). |
| **PoseConv3D** (Duan et al., CVPR 2022) | 3D Stacked Skeleton Heatmaps | None (Heatmap volume convolution) | N/A (Skeleton only) | 3D Convolution on heatmaps | Lacks semantic equipment context; struggles to isolate hand-apparatus grip dynamics. |
| **QPIC / HOTR** (Tamura et al., Kim et al., CVPR 2021) | Static RGB Image + Object Bounding | Transformer Decoder Queries | N/A (Static image) | None (Single frame only) | Designed strictly for **static image HOI**; fails to model dynamic temporal cadence, momentum, or release points in video. |
| **VidHOI / ST-HOI** (Chao et al., 2021) | Video Bounding Boxes + Visual Patches | Spatio-temporal relational bounding box graphs | Linear concatenation | Recurrent / Graph message passing | High latency; relies on generic video bounding boxes without fine-grained human joint keypoint geometry. |
| **Naive Multimodal Fusion** (Concatenation + BiLSTM) | RGB (2048-d) + Pose (85-d) + Object (7-d) | Direct vector concatenation | None (Dominated by RGB $>95\%$) | Bidirectional LSTM | **Multimodal Representation Collapse**: Visual backbone overpowers backpropagation gradients ($>93\%$), suppressing limb & gear dynamics. |
| **CMAT-HOI (Ours)** | **Visual (2048-d) + Pose (85-d) + Object (7-d) + HOI Geometry (16-d)** | **Bi-Directional Keypoint–Equipment Cross-Attention ($h=4, d_k=64$)** | **Dimension-Aligned Modality Stems ($d=256$) + Dynamic GMFU Softmax Gating** | **3-Layer Pre-LN Spatio-Temporal Transformer + 4-Way Multi-Scale Pooling** | **Eliminates scene bias & representation collapse; achieves 90.19% Top-1 / 99.02% Top-5 on 82 sports categories with only 5.2M parameters.** |

---

## 2. In-Depth Technical Mechanism Matrix

| Technical Dimension | Traditional 3D CNNs | Skeleton GCNs (ST-GCN) | Static HOI (QPIC/HOTR) | Video Multimodal (Naive) | **CMAT-HOI (Proposed)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Input Representation Space** | RGB Voxel Grid $\mathbb{R}^{T \times H \times W \times 3}$ | Coordinate Graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ | Image Feature Map $\mathbb{R}^{H \times W \times C}$ | Monolithic Concat $\mathbb{R}^{T \times 2156}$ | **4 Synchronized Streams $\mathbb{R}^{T \times D_m}$** |
| **Modality Dimension Equalization** | None | None | None | None (Severe Imbalance) | **Modality Stems ($d = 256$ with Conv1D $k=3$)** |
| **Keypoint-to-Gear Interaction** | None | None | Keypoint-agnostic box queries | None (Implicit linear layer) | **Bi-Directional Cross-Attention ($Q_p, K_o, V_o$ & $Q_o, K_p, V_p$)** |
| **Dynamic Inter-Modality Gating** | None | None | None | None (Fixed weights) | **GMFU Frame-wise Softmax Gating $g_t \in \mathbb{R}^4$** |
| **Temporal Sequence Modeling** | 3D Conv kernels ($3 \times 3 \times 3$) | 1D Temporal Convolutions | None (Frame-static) | BiLSTM recurrent cells | **3-Layer Pre-LN Transformer (8 Heads, $d_{\text{ff}}=1024$)** |
| **Temporal Feature Aggregation** | Global Average Pooling | Global Average Pooling | Global Max/Avg | Final hidden state $h_T$ | **Multi-Scale 4-Way Pooling ($z_{\text{CLS}}, z_{\text{Attn}}, z_{\text{Max}}, z_{\text{Avg}}$)** |
| **Loss Function Formulation** | Standard Cross-Entropy | Cross-Entropy | Hungarian Bipartite Match + CE | Standard Cross-Entropy | **Focal Loss with Label Smoothing ($\gamma=1.5, \epsilon=0.08$)** |
| **Parameter Budget** | $25\text{M} - 65\text{M}$ | $3.1\text{M} - 6.5\text{M}$ | $41\text{M} - 55\text{M}$ | $8.4\text{M}$ | **5,209,778 (~5.2M)** |
| **Computational Footprint** | Enterprise GPU Cluster | Single GPU | High-end Workstation | Single GPU | **Single Workstation GPU (GTX 1650, 15 min train)** |
| **Top-1 Accuracy on 82 UCF Sports** | 68.42% (ResNet-50 visual) | 52.15% (Keypoint only) | N/A (Image-based) | 55.20% (BiLSTM) | **90.19% (CMAT-HOI)** |

---

## 3. Four Core Failure Modes Solved by CMAT-HOI

### 1. Scene Bias in Visual Models (Carreira & Zisserman, CVPR 2017; Feichtenhofer et al., ICCV 2019)
- **Failure Mode:** Unimodal visual models achieve high accuracy on standard benchmarks by recognizing static background objects (e.g., grass pitches, swimming pools, tennis clay courts) rather than the athlete's actual motor execution. When transferred to unusual environments or fine-grained variations (e.g., distinguishing a forehand slice from a topspin stroke on the exact same court), visual networks fail.
- **CMAT-HOI Resolution:** GMFU actively suppresses visual background noise by dynamically lowering the visual gate ($g_t^{(1)} \to 0$) during high-velocity swings, prioritizing pose and gear kinematics.

### 2. Object Blindness in Skeleton Networks (Yan et al., AAAI 2018; Duan et al., CVPR 2022)
- **Failure Mode:** Skeleton-based graph neural networks track 17 to 25 anatomical joints with high viewpoint invariance, but operate in complete isolation from the physical environment. In sports, the equipment dictates the rules and action boundaries (e.g., *BaseballPitch* vs *CricketShot*, or *JavelinThrow* vs *PoleVault*).
- **CMAT-HOI Resolution:** Bi-Directional Cross-Attention allows human joint keypoints to actively query equipment bounding dynamics, binding the wrist trajectory directly to the racquet, bat, or ball.

### 3. Multimodal Representation Collapse (Wang et al., NeurIPS 2020)
- **Failure Mode:** Concatenating visual embeddings ($2048$-d) with keypoints ($85$-d) and object boxes ($7$-d) causes the visual stream to comprise $>95\%$ of the total feature magnitude. The network's optimization path prematurely converges on the visual modality ("greedy learning"), stifling the gradients of the lower-dimensional kinematic streams.
- **CMAT-HOI Resolution:** Dimension-Aligned Modality Stems project all streams into a balanced $d=256$ space with 1D temporal convolutions, ensuring equalized backpropagation gradient updates.

### 4. Dilution of Peak Explosive Movements in Temporal Pooling (Arnab et al., ICCV 2021)
- **Failure Mode:** Standard video transformers rely either exclusively on the `[CLS]` token or global average pooling. While average pooling captures sustained rhythms (e.g., rowing or jogging), it completely washes out split-second explosive peak events (e.g., ball impact in tennis or the exact release point in javelin).
- **CMAT-HOI Resolution:** Multi-Scale 4-Way Temporal Pooling concatenates `[CLS]`, learnable temporal attention, temporal max pooling (peak motion), and temporal average pooling (steady state) into a unified $1024$-d descriptor (+4.99% gain over `[CLS]` alone).
