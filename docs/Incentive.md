# Photo AI Recovery Subnet Incentive Mechanism

This document covers the current state of SN***'s incentive mechanism. 
1. [Overview](#overview)
2. [Rewards](#rewards)
3. [Scores](#scores)
4. [Weights](#weights)
5. [Incentive](#incentives)

## Overview

Miner reward is based on the evaluation of the first task's effectiveness—photo upscaling. These scores are used by validators to set weights for miners, which determine their reward distribution, incentivizing high-quality and consistent performance.


## Rewards

$$
reward = 0.1 \cdot PSNR + 0.1 \cdot SSIM + 0.8 \cdot (1 - LPIPS)
$$

The miner's reward is determined using a weighted combination of three evaluation metrics:  

- **PSNR (Peak Signal-to-Noise Ratio)** – Measures the ratio between the original and restored image's signal strength. Higher values indicate better reconstruction quality.  
- **SSIM (Structural Similarity Index Measure)** – Evaluates structural and perceptual similarities between images. A higher SSIM score means the restored image is closer to the original.  
- **LPIPS (Learned Perceptual Image Patch Similarity)** – A deep-learning-based metric that assesses perceptual differences between images. Lower values indicate higher quality.



## Scores

Validators set weights based on historical miner performances, tracked by their score vector. 

For each challenge *t*, a validator will randomly sample 50 miners, send them an image/video, and compute their reward as described above. These reward values are then used to update the validator's score vector *V* using an exponential moving average (EMA) with *&alpha;* = 0.05. 

$$
score = 0.05 \cdot reward + 0.95 \cdot score
$$

A low *&alpha;* value places emphasis on a miner's historical performance, adding additional smoothing to avoid having a single prediction cause significant score fluctuations.


## Weights

Validators set weights around once per tempo (360 blocks) by sending a normalized score vector to the Bittensor blockchain (in `UINT16` representation).

Weight normalization by L1 norm:

$$w = \frac{\text{V}}{\lVert\text{V}\rVert_1}$$


## Incentives

The [Yuma Consensus algorithm](https://docs.bittensor.com/yuma-consensus) translates the weight matrix *W* into incentives for the subnet miners and dividends for the subnet validators

Specifically, for each miner *j*, incentive is a function of rank *R*:

$$I_j = \frac{R_j}{\sum_k R_k}$$

where rank *R* is *W* (a matrix of validator weight vectors) weighted by validator stake vector *S*. 

$$R_k = \sum_i S_i \cdot W_{ik}$$




