\section{Methodology}

This section details the proposed Cross-Modal Attention Transformer for Human--Object Interaction (CMAT-HOI). The architecture processes synchronized visual appearance, human joint kinematics, sports equipment bounding dynamics, and geometric interaction features across uniform video temporal windows. Fig.~\ref{fig:pipeline} illustrates the overall pipeline.

\begin{figure*}[t]
\centering
\includegraphics[width=0.96\textwidth]{figures/system_architecture_diagram.png}
\caption{Overall architecture of CMAT-HOI. Input video frames are decomposed into four synchronized streams: visual appearance ($2048$-d), skeletal pose ($85$-d), sports equipment ($7$-d), and geometric HOI ($16$-d). Modality stems project each stream to a uniform dimension ($d=256$) with local temporal convolutions. A bi-directional cross-attention module models limb-equipment interactions. The Gated Multimodal Fusion Unit adaptively weights stream contributions before sequence encoding through a 3-layer Transformer. A 4-way multi-scale temporal pooling head aggregates token representations into a $1024$-d vector for final 82-class action classification.}
\label{fig:pipeline}
\end{figure*}

\subsection{Feature Representation and Stream Extraction}
Let $\mathcal{V} = \{v_i, y_i\}_{i=1}^N$ be a video dataset with ground-truth class labels $y_i \in \{1, \dots, C\}$ over $C = 82$ sports categories. Each video clip is sampled into $T = 16$ equidistant frames. For each frame $t \in \{1, \dots, T\}$, four feature representations are extracted:

\begin{enumerate}
    \item \textbf{Visual Stream} ($X_V \in \mathbb{R}^{T \times 2048}$): Global average-pooled representations from the final convolutional block of a ResNet-50 backbone~\cite{he2016deep} pre-trained on ImageNet.
    \item \textbf{Pose Stream} ($X_P \in \mathbb{R}^{T \times 85}$): Human skeletal coordinates extracted via Keypoint R-CNN~\cite{he2017mask}. For 17 anatomical joints, this stream encodes spatial coordinates and detection confidence $(x_j, y_j, c_j)$ yielding 51 dimensions, concatenated with 34-dimensional inter-frame joint velocity vectors $(\Delta x_j, \Delta y_j)$.
    \item \textbf{Object Stream} ($X_O \in \mathbb{R}^{T \times 7}$): Sports apparatus detections extracted with Faster R-CNN~\cite{ren2015faster} filtered across sports equipment categories (e.g., bats, racquets, balls, skis). Each detection is parameterized as $[x_1, y_1, x_2, y_2, s, c, a]$, denoting normalized box coordinates, confidence score, category identifier, and relative area. When multiple objects appear, the detection with highest confidence is retained. Zero-padding is applied if no equipment is detected.
    \item \textbf{Geometric HOI Stream} ($X_H \in \mathbb{R}^{T \times 16}$): Explicit interaction geometry between human effectors (left/right wrists and left/right ankles) and the equipment centroid. Features include normalized Euclidean distances, spatial orientation angles, and frame-to-frame distance differentials.
\end{enumerate}

\subsection{Dimension-Aligned Modality Stems}
\textbf{Motivation:} Concatenating raw multimodal vectors directly creates an extreme dimensional imbalance: the visual vector ($2048$-d) comprises over $95\%$ of the total dimension ($2156$-d), while pose ($85$-d) and object ($7$-d) features contribute less than $5\%$. Backpropagation gradients become dominated by visual features, which causes the network to overfit to background stadium textures rather than athlete movements.

\textbf{Module Design:} To equalize stream representations, each raw feature tensor $X_m$ for modality $m \in \{V, P, O, H\}$ is mapped into a shared latent dimension $d = 256$ through a dedicated modality stem:
\begin{equation}
\tilde{X}_m = \text{LayerNorm}(W_m X_m + b_m), \quad W_m \in \mathbb{R}^{d \times D_m}
\end{equation}
To model local temporal dependencies across adjacent frames prior to global attention, a 1D temporal convolution is applied over the frame sequence with a residual connection:
\begin{equation}
H_m = \text{Dropout}\left(\text{GELU}\left(\tilde{X}_m + \text{Conv1D}_{k=3}(\tilde{X}_m)\right), p=0.2\right)
\end{equation}
where $k=3$ is the kernel size with padding of 1 to preserve sequence length $T=16$.

\textbf{Technical Advantages:} The stems equalize gradient magnitudes across all modalities. The 1D temporal convolution introduces a localized temporal receptive field, smoothing high-frequency noise from frame-level pose jitter before self-attention.

\subsection{Bi-Directional Cross-Modal Attention}
\textbf{Motivation:} Distinguishing fine-grained sports actions requires understanding joint articulation with respect to equipment (for example, wrist angle relative to a cricket bat versus a baseball bat). Unimodal processing analyzes joints and objects in isolation, missing the structural contact between athlete and gear.

\textbf{Module Design:} Let $P \in \mathbb{R}^{T \times d}$ and $O \in \mathbb{R}^{T \times d}$ denote the aligned pose and object feature sequences ($d = 256$). A bi-directional cross-attention module calculates mutual queries, keys, and values between the two modalities using multi-head attention:
\begin{equation}
\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
\end{equation}
\begin{equation}
\text{head}_i = \text{Softmax}\left(\frac{(Q W_i^Q)(K W_i^K)^\top}{\sqrt{d_k}}\right) (V W_i^V)
\end{equation}
where $h = 4$ heads and $d_k = d/h = 64$.

In the forward path, human pose queries the object features to identify equipment context:
\begin{equation}
P_{\text{enh}} = \text{LayerNorm}\left(P + \text{Dropout}(\text{MHA}(Q=P, K=O, V=O))\right)
\end{equation}
Concurrently, object features query the human pose to associate gear motion with limb articulation:
\begin{equation}
O_{\text{enh}} = \text{LayerNorm}\left(O + \text{Dropout}(\text{MHA}(Q=O, K=P, V=P))\right)
\end{equation}

\textbf{Technical Advantages:} Mutual cross-attention filters out background objects that do not correlate with body keypoints. The enhanced representations $P_{\text{enh}}$ and $O_{\text{enh}}$ encode interactive kinematics without increasing the sequence length.

\subsection{Gated Multimodal Fusion Unit (GMFU)}
\textbf{Motivation:} Different sports depend on different modalities. Activities like diving or skiing present strong visual scene cues, whereas weightlifting or bodyweight squats depend almost entirely on skeleton articulation. Fixed summation or static concatenation cannot adjust to these varying modality reliabilities.

\textbf{Module Design:} The Gated Multimodal Fusion Unit dynamically computes frame-level gating coefficients across the active modalities. At frame $t$, the four streams are concatenated into a joint vector:
\begin{equation}
U_t = [V_t; \; P_{\text{enh}, t}; \; O_{\text{enh}, t}; \; H_t] \in \mathbb{R}^{1024}
\end{equation}
A two-layer multi-layer perceptron computes normalized gating weights via softmax:
\begin{equation}
s_t = W_2 \cdot \text{GELU}(W_1 U_t + b_1) + b_2, \quad W_1 \in \mathbb{R}^{256 \times 1024}, W_2 \in \mathbb{R}^{4 \times 256}
\end{equation}
\begin{equation}
g_t = \text{Softmax}(s_t) = [g_t^{(1)}, g_t^{(2)}, g_t^{(3)}, g_t^{(4)}]^\top, \quad \sum_{m=1}^4 g_t^{(m)} = 1
\end{equation}
The fused frame embedding $F_t \in \mathbb{R}^{256}$ is obtained by taking the gated convex combination of the streams:
\begin{equation}
F_t = \text{LayerNorm}\left(g_t^{(1)} V_t + g_t^{(2)} P_{\text{enh}, t} + g_t^{(3)} O_{\text{enh}, t} + g_t^{(4)} H_t\right)
\end{equation}

\textbf{Technical Advantages:} GMFU suppresses visual background noise by lowering $g_t^{(1)}$ when kinematic cues ($g_t^{(2)}$) and interaction dynamics ($g_t^{(3)}$) provide more reliable discriminative evidence. The softmax constraint keeps the fused activations within a stable numerical range.

\subsection{Spatio-Temporal Transformer Encoder}
\textbf{Motivation:} Athletic actions involve temporal structure: preparatory movements, contact or strike points, and follow-through. Standard recurrent networks suffer from gradient degradation over longer sequences and execute sequentially. Self-attention layers compute direct pairwise dependencies across all frames simultaneously.

\textbf{Module Design:} A learnable classification token $z_{[\text{CLS}]} \in \mathbb{R}^{1 \times 256}$ is prepended to the fused temporal sequence $\{F_1, \dots, F_T\}$. Fixed sinusoidal temporal positional encodings $E_{\text{pos}} \in \mathbb{R}^{(T+1) \times 256}$ are added to retain temporal ordering:
\begin{equation}
Z_0 = [z_{[\text{CLS}]}; \; F_1; \; \dots; \; F_T] + E_{\text{pos}}
\end{equation}
The sequence is processed through $L = 3$ Transformer encoder layers. Each layer uses Pre-LayerNorm multi-head self-attention with $h = 8$ heads and a feedforward network with expansion dimension $d_{\text{ff}} = 1024$:
\begin{equation}
\hat{Z}_l = \text{MHA}(\text{LayerNorm}(Z_{l-1})) + Z_{l-1}
\end{equation}
\begin{equation}
Z_l = \text{FFN}(\text{LayerNorm}(\hat{Z}_l)) + \hat{Z}_l, \quad l \in \{1, 2, 3\}
\end{equation}
where $\text{FFN}(x) = W_b \cdot \text{GELU}(W_a x + b_a) + b_b$.

\textbf{Technical Advantages:} Pre-LayerNorm configuration avoids vanishing gradients and stabilizes optimization. The self-attention mechanism captures temporal associations between non-adjacent phases of an action, such as backswing and ball contact.

\subsection{Multi-Scale 4-Way Temporal Pooling Head}
\textbf{Motivation:} Relying solely on the $[\text{CLS}]$ token or on global average pooling risks discarding crucial temporal cues. Sports actions include both brief explosive moments (e.g., the release in javelin throw) that require peak activation pooling, and extended repetitive cadences (e.g., rowing) that benefit from uniform temporal averaging.

\textbf{Module Design:} Four complementary pooling operations are extracted from the final encoder output $Z_L \in \mathbb{R}^{(T+1) \times 256}$:
\begin{enumerate}
    \item \textbf{Classification Token:} Global sequence summary from the prepended token:
    \begin{equation}
    z_{\text{CLS}} = Z_L[0] \in \mathbb{R}^{256}
    \end{equation}
    \item \textbf{Attention Pooling:} Weighted summation across frame tokens $t \in \{1, \dots, T\}$ guided by a learnable scoring vector $w \in \mathbb{R}^{128}$:
    \begin{equation}
    \alpha_t = \frac{\exp(w^\top \tanh(W_\alpha Z_L[t]))}{\sum_{j=1}^T \exp(w^\top \tanh(W_\alpha Z_L[j]))}, \quad z_{\text{Attn}} = \sum_{t=1}^T \alpha_t Z_L[t]
    \end{equation}
    \item \textbf{Temporal Max Pooling:} Captures peak kinematic activations across the duration:
    \begin{equation}
    z_{\text{Max}} = \max_{t \in \{1, \dots, T\}} Z_L[t] \in \mathbb{R}^{256}
    \end{equation}
    \item \textbf{Temporal Average Pooling:} Captures steady baseline motion trends:
    \begin{equation}
    z_{\text{Avg}} = \frac{1}{T} \sum_{t=1}^T Z_L[t] \in \mathbb{R}^{256}
    \end{equation}
\end{enumerate}

These four representations are concatenated into a $1024$-dimensional feature vector:
\begin{equation}
Z_{\text{video}} = [z_{\text{CLS}}; \; z_{\text{Attn}}; \; z_{\text{Max}}; \; z_{\text{Avg}}] \in \mathbb{R}^{1024}
\end{equation}
The action probability distribution $\hat{y} \in \mathbb{R}^{82}$ is produced by an MLP classifier with LayerNorm, GELU activations, and dropout ($p=0.3$):
\begin{equation}
\hat{y} = \text{Softmax}(W_4 \cdot \text{GELU}(\text{LayerNorm}(W_3 Z_{\text{video}} + b_3)) + b_4)
\end{equation}
where $W_3 \in \mathbb{R}^{512 \times 1024}$ and $W_4 \in \mathbb{R}^{82 \times 512}$.

\textbf{Technical Advantages:} Combining these four statistics prevents information loss from localized peak motions while preserving global temporal context, yielding a $+4.99\%$ accuracy improvement over using $[\text{CLS}]$ alone.

\subsection{Optimization Objective and Regularization}
\textbf{Motivation:} Fine-grained sports datasets contain class imbalance and classes with high visual similarity. Standard cross-entropy loss assigns equal gradient weight to easily classified samples, which slows parameter updates on hard, ambiguous classes.

\textbf{Module Design:} The network is trained with Focal Loss~\cite{lin2017focal} combined with label smoothing ($\epsilon = 0.08$)~\cite{szegedy2016rethinking}:
\begin{equation}
\mathcal{L}_{\text{total}} = -\frac{1}{B} \sum_{i=1}^B \sum_{c=1}^C \alpha_c (1 - p_{i, c})^\gamma y_{i, c}^{\text{smooth}} \log(p_{i, c})
\end{equation}
where $B$ is the batch size, $\gamma = 1.5$ is the focusing parameter that downweights well-classified examples, $\alpha_c = 0.08$ is the weighting factor, and $y_{i, c}^{\text{smooth}}$ represents the smoothed target:
\begin{equation}
y_{i, c}^{\text{smooth}} = (1 - \epsilon) y_{i, c} + \frac{\epsilon}{C}
\end{equation}

During training, three feature-space data augmentations are applied:
\begin{itemize}
    \item \textbf{Multimodal Mixup:} Interpolates pairs of feature representations and targets with mixing coefficient $\lambda \sim \text{Beta}(0.4, 0.4)$ with probability $0.5$~\cite{zhang2017mixup}.
    \item \textbf{Kinematic Jitter:} Adds zero-mean Gaussian noise $\mathcal{N}(0, 0.008^2)$ to pose and object coordinates to simulate joint tracking noise.
    \item \textbf{Temporal Frame Dropout:} Randomly masks 1 to 2 frames per sequence ($p=0.25$) using nearest-neighbor interpolation to simulate occlusions.
\end{itemize}

\textbf{Technical Advantages:} The focal weighting prevents gradient saturation from common action classes, while label smoothing reduces overconfident predictions on visually similar sports activities.
