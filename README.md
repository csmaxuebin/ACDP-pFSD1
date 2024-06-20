# This code is the source code implementation for the paper "ACDP-pFSD: A Personalized Federated Self-Decoupled Distillation Framework with User-level Differential Privacy based Adaptive Clipping"！
# Abstract
![输入图片说明](https://github.com/csmaxuebin/ACDP-pFSD1/blob/main/picture/4.png)
Federated learning faces two major challenges: data heterogeneity and privacy leakage risks. In response to these problems, this paper introduces a personalized federated distillation method with user-level differential privacy, named ACDP-pFSD. This method provides personalized models for each client
while considering the performance of the global model and ensuring user-level differential privacy. Specifically, the method saves the personalized models locally from previous training. Through self-decoupled knowledge distillation, it learns personalized knowledge into the new local models to train models that match the client’s data distribution. Moreover, to mitigate the negative impact of user-level differential privacy on model performance, adaptive model update clipping is used to balance privacy and utility. Finally, extensive simulation experiments validate the effectiveness of ACDP-pFSD. The algorithm not only shows improved accuracy in both personalized and global models but also achieves a good balance between model performance and privacy protection.

# Experimental Environment

```
- cuda 11.6.1
- dp-accounting 0.4.2
- h5py 3.8.0
- keras 2.12.0
- matplotlib 3.7.1
- numpy 1.23.1
- oauthlib 3.2.2
- opacus 1.3.0
- python 3.9.16
- scikit-learn 0.23.1
- scipy 1.10.1
- six 1.16.0
- torch 1.13.1
- torchaudio 0.13.1
- torchsummary 1.5.1
- torchvision 0.14.1 
```

## Datasets
```
- EMNIST
- SVHN
- CIFAR10 
```
## Experimental Setup
**Hyperparameters:**
-  𝑄 = 5 local training iterations per round 
-  𝑇 = 100 rounds of communication
- batch size = 64.

**Algorithms Compared:** 
-  pFSD: A federated learning method without differential privacy.
-  DP-pFSD: A fixed clipping threshold
- AQC: A adaptive clipping, applied within the same personalized settings as ACDP-pFSD.
## Python Files
-   **update.py**:  This code is for client training and updating related functions
-   **ours_global.py**: This code implements a federated learning framework, which is the main function of the code.
-   **ours_global_DP.py**: This code implements a federated learning differential privacy framework, which protects the privacy of the model and serves as the main function of the code
-   **DP_clip.py**: This code implements an adaptive clippping method for model updates

## Experimental Results
![输入图片说明](https://github.com/csmaxuebin/ACDP-pFSD1/blob/main/picture/1.png)Table 3 shows the accuracy of ACDP-pFSD algorithm across all three datasets.
![输入图片说明](https://github.com/csmaxuebin/ACDP-pFSD1/blob/main/picture/2.png)
Fig. 3 shaows the accuracy analysis at the Same Level of Privacy Protection.
![输入图片说明](https://github.com/csmaxuebin/ACDP-pFSD1/blob/main/picture/3.png)
Fig. 4 shows the performance of personalized and global models across all algorithms within the heterogeneous settings 𝑠1 and 𝑠2.

## Update log

```
- {24.06.17} Uploaded overall framework code and readme file
```



