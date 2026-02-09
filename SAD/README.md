# HPFML-IC$^2$--ASTER
### Constrained-Hypernetwork Personalized Federated Meta-Learning for EEG Vigilance Detection with Efficient Transformer Pruning
#### *By: Van-Duc Khuat∗1, Yue Cao∗2, and Wansu Lim∗1
##### 1.Department of Electrical and Computer Engineering, Sungkyunkwan, University, Republic of Korea, Suwon. 
###### 2.School of Cyber Science and Engineering and Shenzhen Research Institute, Wuhan University, Wuhan, 430072, China. 
## Abstract
![HPFML-IC$^2$--ASTER](imgs/DARTS_MultiLevel_Pruning.png)
Electroencephalography (EEG)-based vigilance detection is critical for safe autonomous driving, yet reliable learning remains challenging due to privacy-sensitive signals, limited labels, and severe heterogeneity across subjects, sessions, and acquisition devices. Federated learning preserves privacy by keeping EEG data on-device, but a single global model often underfits hard clients under pronounced non-IID distributions. To address this challenge, we propose HPFML, a personalized federated meta-learning framework that restricts personalization to the classifier head via a constrained hypernetwork, enabling fast and stable adaptation under heterogeneous EEG clients. The server learns an EEG-aware Transformer meta-backbone as a shared initialization, capturing transferable and noise-robust representations through EEG-specific spatial--temporal inductive biases and temporal attention pooling. Conditioned on a learnable client embedding, the hypernetwork generates a lightweight head, while a scheduled head-mixing scheme provides principled control over the personalization strength. Each client performs bi-level local meta-learning with inner-loop adaptation on a support split and outer-loop optimization on a query split. The server aggregates client updates via FedAvg and meta-supervises the hypernetwork by distilling clients' inner-adapted heads, achieving personalization without sharing raw EEG. To further analyze and refine the learned representations, we propose IC$^2$--ASTER, a capacity-aware pruning criterion remove redundant Transformer feed-forward network (FFN) neurons. The pruned backbone is retrained under HPFML, preserving personalization behavior and representation transferability while revealing FFN redundancy under heterogeneous EEG clients. Experiments on SAD, SEED-VIG, and SEED-VRW demonstrate consistent improvements over recent state-of-the-art methods, achieving up to 95.34\% ACC and 93.29\% Macro-F1. IC$^2$--ASTER further reduces the backbone to 0.141--0.168M parameters ($\approx$18--20\% reduction) without performance degradation, facilitating efficient on-device inference.


We used three public datasets in this study:
- [SAD](Cao, Z., Chuang, CH., King, JK. et al. Multi-channel EEG recordings during a sustained-attention driving task. \emph{Sci Data} 6, 19 (2019).)
- [SEED-VIG](ZHENG, Wei-Long; LU, Bao-Liang. A multimodal approach to estimating vigilance using EEG and forehead EOG. \emph{Journal of neural engineering}, 2017, 14.2: 026017.)
- [EED-VRW](Y. Luo, W. Liu, H. Li, Y. Lu, and B.-L. Lu, “A cross-scenario and cross-subject domain adaptation method for driving fatigue detection,” \emph{Journal of neural engineering}, vol. 21, no. 4, Aug. 2024, Art. no.046004.)

## At present, we only provide results on the SAD dataset; the complete code and full training outputs for both stages will be released upon paper acceptance.

After downloading the datasets, you can prepare them as follows:
```
python Data_Preprocessing.py

```

Running python Data_Preprocessing.py generates the preprocessed dataset cache file: SAD_cache_paper_augSelect_hp0.5_lp50.0_fs250_t-1.0_2.0_aug1x1.npz.
The resulting cache file is available at: https://drive.google.com/drive/folders/15HGTmJDijXHBPVg4aSFKJXH6l8fraoxy?usp=drive_link

## Training HPFML
```
- HPFML1.ipynb
```
Training HPFML follows the same procedure, and the resulting outputs are stored at: (https://drive.google.com/drive/folders/15HGTmJDijXHBPVg4aSFKJXH6l8fraoxy?usp=drive_link).
## Training HPFML-IC$^2$--ASTER
```
- IC2–ASTER1.ipynb
```
Training IC2–ASTER1.ipynb  follows the same procedure, and the resulting outputs are stored at: (https://drive.google.com/drive/folders/15HGTmJDijXHBPVg4aSFKJXH6l8fraoxy?usp=drive_link).


## The SEED-VIG and SEED-VRW datasets, along with the corresponding results and training code, will be fully released upon paper acceptance.

Prerequisites:

-The proposed framework comprises two sequential stages. Both stages were implemented in PyTorch (v2.4.0) on the NVIDIA GPU Cloud (NGC) platform, using CUDA 12.6 for GPU acceleration, and executed on a high-performance GPU supercomputing environment.

---

D. Implementation details

a) Stage 1 - HPFML:

• Data split and local meta-learning setup: For all datasets,
each client (subject or acquisition device) is treated
independently. The local dataset of client i is first split
into a training set and a held-out test set with a ratio of
70%/30%. The training portion is further divided into
a support set Si and a query set Qi using a 60%/40%
split. This hierarchical split is consistent with the bi-level
formulation in method, where the support set is used for
inner-loop adaptation and the query set is used for outer-
loop optimization and validation.

• PFML configuration: The PFML procedure is conducted
for T = 250 communication rounds on SEED-VIG and
SEED-VRW, and for T = 500 communication rounds on
SAD. At each round, clients perform K = 5 inner-loop
update steps on Si with learning rate α = 10−3, followed
by M = 1 outer-loop update step on Qi with learning
rate β = 10−3, as described in (16) and (18). Mini-batch
stochastic optimization is used with a batch size of 64
for both inner and outer updates.

• Personalization ratio scheduling: The head-mixing ratio
rt is gradually increased following the schedule in (14),
with an initial ratio r0 and a maximum personalization
cap set to r = 0.6. This schedule prevents prema-
ture over-personalization in early rounds while allowing
stronger client-specific adaptation as training progresses.
• Hypernetwork configuration: Client embeddings have
dimension de = 16 and are mapped to personalized
classifier heads through a constrained hypernetwork with
a hidden dimension of 128. On the server side, the
hypernetwork parameters and the embedding table are
optimized using λ = 1, learning rates ηϕ = 5 × 10−4 and
ηE = 5 × 10−4, respectively, following the update rule
in (20) and (21).

b) Stage 2 - HPFML-IC2–ASTER: The IC2–ASTER
pruning parameters are set to ρ = 0.5, σ = 0.5, α = 1.0,
k = 5, and B = 32 as described in (28), (29), (31), and (33).
After offline IC2–ASTER pruning, the pruned backbone is
retrained using the same HPFML procedure as Stage 1. Due to
the reduced model capacity and faster convergence after prun-
ing, fewer communication rounds are required. Specifically,
HPFML retraining is conducted for T = 100 communication
rounds on SAD, and for T = 50 communication rounds on
SEED-VIG and SEED-VRW, while keeping the remaining
optimization hyperparameters the same as in Stage 1.

## Contact
Van-Duc Khuat  
Department of Electrical and Computer Engineering,  
Sungkyunkwan University, Suwon, Republic of Korea  
Email: duckv@g.skku.edu

Wansu Lim*  
Department of Electrical and Computer Engineering,  
Sungkyunkwan University, Suwon, Republic of Korea  
*Corresponding author- Email: wansu.lim@skku.edu

