# <h1 align="center"> Subnet *** Photo AI Recovery 🗣️</h1> 

- [ Subnet \*\*\* Photo AI Recovery 🗣️](#-subnet--photo-ai-recovery-️)
  - [Resources](#resources)
  - [Introduction](#introduction)
  - [Roadmap](#roadmap)
    - [Q3 2025:](#q3-2025)
    - [Q4 2025:](#q4-2025)
    - [Q1 2026:](#q1-2026)
    - [Q2 2026+:](#q2-2026)
  - [List of Miner Tasks](#list-of-miner-tasks)
    - [Task 1: Upscaling](#task-1-upscaling)
    - [Task 2: Denoising  # TO DO](#task-2-denoising---to-do)
    - [Task 3: Inpainting  # TO DO](#task-3-inpainting---to-do)
    - [Task 4: Watermark Removal  # TO DO](#task-4-watermark-removal---to-do)
  - [Validating and Mining](#validating-and-mining)
    - [Validator](#validator)
    - [Miner](#miner)
  - [Reward Mechanism](#reward-mechanism)

## Resources

- wandb https://wandb.ai/neman-team-ai/sn***-validator
- hugging face repo https://huggingface.co/datasets/NemanTeam/mit-adobe-5k


## Introduction

In today's world, visual content plays a key role, and image quality directly affects how information is perceived. Modern algorithms for denoising, inpainting, artifact removal, and watermark removal are actively used in various fields — from professional photography to processing user-generated content in mobile applications. Despite the successes of existing solutions, they are unable to adequately address the challenge of restoring image quality when significant changes occur, such as increasing resolution by 6 times or more. The Bittensor Photo AI Recovery subnet aims to solve the problem of photo recovery through a decentralized competition system and commercial service.


## Roadmap

### Q3 2025:
  - Subnet code stabilization; improvements to task generation and scoring.
  - Monitoring and analysis for detection of inconsistencies or unfair practices.
### Q4 2025:
  - Expansion to additional tasks (denoising, inpainting, artifact removal, watermark removal).
  - Launch of the MVP commercial web service.
### Q1 2026:
  - Public testing of the commercial platform; onboarding of early users.
### Q2 2026+:
  - Full commercial launch; release of APIs for partners and enterprise integrations.



## List of Miner Tasks

### Task 1: Upscaling  
Miners must increase the resolution of an image by **6×** while improving its quality.  

- **Validators generate a synthetic task** by combining several photos (through augmentation and overlay) to create a unique sample.  
- The **sample is then downscaled** by a factor of 6 and sent to a Miner.  
- **Miners apply their upscaling algorithms**, striving to restore the image as close as possible to the original.  
- **Validators assess the results** using objective metrics such as PSNR, SSIM, and LPIPS.  

### Task 2: Denoising  # TO DO
Miners must **remove noise** from an image while preserving its details and overall quality.  

### Task 3: Inpainting  # TO DO
A portion of the image is **missing**, and miners must reconstruct it in a way that seamlessly blends with the original content.  

### Task 4: Watermark Removal  # TO DO
Miners must **remove watermarks** from an image while maintaining the integrity and quality of the surrounding content.  


## Validating and Mining

### Validator

Validators are responsible for analyzing and verifying data submitted by miners.

If you are interested in validating, follow this [guide](docs/VALIDATOR.md).

Validators create [tasks](#tasks) every 5 minutes.

### Miner

Miners in the Bittensor subnet are responsible for solving validators' tasks and providing computational resources.

In the baseline solution, LLMs are used to identify and score each article.

If you are interested in mining, follow this [guide](docs/MINER.md).

## Reward Mechanism

If you are interested in reward model, follow this [Incentive](docs/Incentive.md).
