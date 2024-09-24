# Awesome-LLM-for-Autonomous-Driving-Resources
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)![GitHub stars](https://img.shields.io/github/stars/Thinklab-SJTU/Awesome-LLM4AD?color=yellow) ![GitHub forks](https://img.shields.io/github/forks/Thinklab-SJTU/Awesome-LLM4AD?color=9cf) [![GitHub license](https://img.shields.io/github/license/Thinklab-SJTU/Awesome-LLM4AD)](https://github.com/Thinklab-SJTU/Awesome-LLM4AD/blob/main/LICENSE)

This is a collection of research papers about **LLM-for-Autonomous-Driving(LLM4AD)**.
And the repository will be continuously updated to track the frontier of LLM4AD. *Maintained by SJTU-ReThinklab.*

Welcome to follow and star! If you find any related materials could be helpful, feel free to contact us (yangzhenjie@sjtu.edu.cn or jiaxiaosong@sjtu.edu.cn) or make a PR.

## Citation
Our survey paper is at https://arxiv.org/abs/2311.01043 which includes more detailed discussions and will be continuously updated. 
**The latest version was updated on August 12, 2024.**

If you find our repo is helpful, please consider cite it.
```BibTeX
@misc{yang2023survey,
      title={LLM4Drive: A Survey of Large Language Models for Autonomous Driving}, 
      author={Zhenjie Yang and Xiaosong Jia and Hongyang Li and Junchi Yan},
      year={2023},
      eprint={2311.01043},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```


## Table of Contents
- [Awesome LLM-for-Autonomous-Driving(LLM4AD)](#awesome-llm-for-autonomous-driving-resources)
  - [Table of Contents](#table-of-contents)
  - [Overview of LLM4AD](#overview-of-llm4ad)
  - [ICLR 2024 Under Review](#iclr-2024-under-review)
  - [Papers](#papers)
  - [Datasets](#datasets)
  - [Citation](#citation)
  - [License](#license)

## Overview of LLM4AD
LLM-for-Autonomous-Driving (LLM4AD) refers to the application of Large Language Models(LLMs) in autonomous driving. We divide existing works based on the perspective of applying LLMs: planning, perception, question answering, and generation. 

![image info](./assets/llm4adpipeline.png)

## Motivation of LLM4AD
The orange circle represents the ideal level of driving competence, akin to that possessed by an experienced human driver. There are two main methods to acquire such proficiency: one, through learning-based techniques within simulated environments; and two, by learning from offline data through similar methodologies. It’s important to note that due to discrepancies between simulations and the real-world, these two domains are not fully the same, i.e. sim2real gap. Concurrently, offline data serves as a subset of real-world data since it’s collected directly from actual surroundings. However, it is difficult to fully cover the distribution as well due to the notorious long-tailed nature of autonomous driving tasks. The final goal of autonomous driving is to elevate driving abilities from a basic green stage to a more advanced blue level through extensive data collection and deep learning.

![image info](./assets/whyllmenhance.png)

## ICLR 2024 Open Review
<details close>
<summary>Toggle</summary>

```
format:
- [title](paper link) [links]
  - task
  - keyword
  - code or project page
  - datasets or environment or simulator
  - summary
```
- [Large Language Models as Decision Makers for Autonomous Driving](https://openreview.net/forum?id=NkYCuGM7E2)
  - Keywords: Large language model, Autonomous driving
  - [Previous summary](#LanguageMPC)

- [DriveGPT4: Interpretable End-to-end Autonomous Driving via Large Language Model](https://openreview.net/forum?id=DUkYDXqxKp)
  - Keywords: Interpretable autonomous driving, large language model, robotics, computer vision
  - [Previous summary](#DriveGPT4)

- [BEV-CLIP: Multi-modal BEV Retrieval Methodology for Complex Scene in Autonomous Driving](https://openreview.net/forum?id=wlqkRFRkYc)
  - Keywords: Autonomous Driving, BEV, Retrieval, Multi-modal, LLM, prompt learning
  - Task: Contrastive learning, Retrieval tasks
  - Datasets: [nuScenes](https://www.nuscenes.org/nuscenes)
  - Summary:
    - Propose a multimodal retrieval method powered by LLM and knowledge graph to achieve contrastive learning between text description and BEV feature retrieval for autonomous driving.

- [GPT-Driver: Learning to Drive with GPT](https://openreview.net/forum?id=SXMTK2eltf)
  - Keywords: Motion Planning, Autonomous Driving, Large Language Models (LLMs), GPT
  - [Previous summary](#GPT-Driver)
  
- [Radar Spectra-language Model for Automotive Scene Parsing](https://openreview.net/forum?id=bdJaYLiOxi)
  - Keywords: radar spectra, radar perception, radar object detection, free space segmentation, autonomous driving, radar classification
  - Task: Detection
  - Datasets: 
    - [RADIal](https://github.com/valeoai/RADIal), [CRUW](https://www.cruwdataset.org/), [nuScenes](https://www.nuscenes.org/nuscenes)
    - For RADIal and CRUW, both images and ground truth labels are used. From nuScenes, only images are taken.
    - Random captions for a frame from CRUW dataset based on ground truth object positions and pseudo ground-truth classes. (not open)
  - Summary:
    - Conduct a benchmark comparison of off-the-shelf vision-language models (VLMs) for classification in automotive scenes.
    - Propose to fine-tune a large VLM specially for automated driving scenes.
  
- [GeoDiffusion: Text-Prompted Geometric Control for Object Detection Data Generation](https://openreview.net/forum?id=xBfQZWeDRH)
  - Keywords: diffusion model, controllable generation, object detection, autonomous driving
  - Task: Detection, Data generation
  - Datasets: [nuScenes](https://www.nuscenes.org/nuscenes), which consists of 60K training samples and 15K validation samples with high-quality bounding box annotations from 10 semantic classes.
  - Summary:
    - propose GEODIFFUSION, an embarrassing simple framework to integrate geometric controls into pre-trained diffusion models for detection data generation via text prompts.
   
- [SPOT: Scalable 3D Pre-training via Occupancy Prediction for Autonomous Driving](https://openreview.net/forum?id=9zEBK3E9bX)
  - Keywords: 3D pre-training, object detection, autonomous driving
  - Task: Detection
  - Summary:
    - propose SPOT a scalable 3D pre-training paradigm for LiDAR pretraining.

- [3D DENSE CAPTIONING BEYOND NOUNS: A MIDDLEWARE FOR AUTONOMOUS DRIVING](https://openreview.net/forum?id=8T7m27VC3S)
  - Keywords: Autonomous Driving, Dense Captioning, Foundation model
  - Task: Caption， Dataset Construction
  - Datasets: [nuScenes](https://www.nuscenes.org/nuscenes)
  - Summary:
    - Design a scalable rule-based auto-labelling methodology to generate 3D dense captioning.
    - Construct a large-scale dataset nuDesign based upon nuScenes, which consists of an unprecedented number of 2300k sentences.
</details>


## Papers
<details open>
<summary>Toggle</summary>

```
format:
- [title](paper link) [links]
  - author1, author2, and author3...
  - publisher
  - task
  - keyword
  - code or project page
  - datasets or environment or simulator
  - publish date
  - summary
  - metrics
```
- [LeGEND: A Top-Down Approach to Scenario Generation of Autonomous Driving Systems Assisted by Large Language Models](https://arxiv.org/abs/2409.10066)
  - Shuncheng Tang, Zhenya Zhang, Jixiang Zhou, Lei Lei, Yuan Zhou, Yinxing Xue **ASE 2024**
  - Publisher: University of Science and Technology of China, Kyushu University, Zhejiang Sci-Tech University
  - Task: Generation
  - Publish Date: 2024.09.16
  - Code: [LeGEND](https://github.com/MayDGT/LeGEND)
  - Summary:
    - LeGEND, a top-down scenario generation approach that can achieve both criticality and diversity of scenarios.
    - Devise a two-stage transformation, by using an intermediate language, from accident reports to logical scenarios; so, LeGEND involves two LLMs, each in charge of one different stage.
    - Implement LeGEND and demonstrate its effectiveness on Apollo, and we detect 11 types of critical concrete scenarios that reflect different aspects of system defects.

- [MiniDrive: More Efficient Vision-Language Models with Multi-Level 2D Features as Text Tokens for Autonomous Driving](https://arxiv.org/abs/2409.07267)
  - Enming Zhang, Xingyuan Dai, Yisheng Lv, Qinghai Miao
  - Publisher: University of Chinese Academy of Sciences, CASIA
  - Task: QA
  - Publish Date: 2024.09.14
  - Code: [MiniDrive](https://github.com/EMZucas/minidrive)
  - Summary:
    - MiniDrive addresses the challenges of efficient deployment and real-time response in VLMs for autonomous driving systems. It can be fully trained simultaneously on an
    RTX 4090 GPU with 24GB of memory.
    - Feature Engineering Mixture of Experts (FE-MoE) addresses the challenge of efficiently encoding 2D features from multiple perspectives into text token embeddings, effectively reducing the number of visual feature tokens and minimizing feature redundancy.
    - Dynamic Instruction Adapter through a residual structure, which addresses the problem of fixed visual tokens for the same image before being input into the language model.

- [OccLLaMA: An Occupancy-Language-Action Generative World Model for Autonomous Driving](https://arxiv.org/abs/2409.03272)
  - Julong Wei, Shanshuai Yuan, Pengfei Li, Qingda Hu, Zhongxue Gan, Wenchao Ding
  - Publisher: Fudan University, Tsinghua University
  - Task: Perception(Occ) + Reasoning
  - Publish Date: 2024.09.05
  - Summary:
    - OccLLaMA, a unified 3D occupancy-language-action generative world model, which unifies VLA-related tasks including but not limited to scene understanding, planning, and 4D occupancy forecasting.
    - A novel scene tokenizer(VQVAE-like architecture) that efficiently discretize and reconstruct Occ scenes, considering sparsity and classes imbalance.

- [ContextVLM: Zero-Shot and Few-Shot Context Understanding for Autonomous Driving using Vision Language Models](https://arxiv.org/abs/2409.00301)
  - Shounak Sural, Naren, Ragunathan Rajkumar **ITSC 2024**
  - Publisher: Carnegie Mellon University
  - Task: Context Recognition
  - Code: [ContextVLM](https://github.com/ssuralcmu/ContextVLM)
  - Publish Date: 2024.08.30
  - Summary:
    - DrivingContexts, a large publicly-available datasetwith a combination of hand-annotated and machine annnotated labels to improve VLMs for better context recognition.
    - ContextVLM uses vision-language models to detect contexts using zero- and few-shot approaches.

- [DriveGenVLM: Real-world Video Generation for Vision Language Model based Autonomous Driving](https://arxiv.org/abs/2408.16647)
  - Yongjie Fu, Anmol Jain, Xuan Di, Xu Chen, Zhaobin Mo **IAVVC 2024**
  - Publisher: Columbia University
  - Task: Generation
  - Dataset: [Waymo open dataset](https://waymo.com/open/)
  - Publish Date: 2024.08.29
  - Summary:
    - DriveGenVLM employ a video generation framework based on Denoising Diffusion Probabilistic Models to create realistic video sequences that mimic real-world dynamics.
    - The videos generated are then evaluated for their suitability in Visual Language Models (VLMs) using a pre-trained model called Efficient In-context Learning on Egocentric Videos (EILEV). 

- [Edge-Cloud Collaborative Motion Planning for Autonomous Driving with Large Language Models](https://arxiv.org/abs/2408.09972)
  - Jiao Chen, Suyan Dai, Fangfang Chen, Zuohong Lv, Jianhua Tang
  - Publisher: South China University of Technology, Pazhou Lab
  - Task: Planning + QA
  - Project Page: [EC-Drive](https://sites.google.com/view/ec-drive)
  - Publish Date: 2024.08.19
  - Summary:
    - EC-Drive, a novel edge-cloud collaborative autonomous driving system.

- [V2X-VLM: End-to-End V2X Cooperative Autonomous Driving Through Large Vision-Language Models](https://arxiv.org/abs/2408.09251)
  - Junwei You, Haotian Shi, Zhuoyu Jiang, Zilin Huang, Rui Gan, Keshu Wu, Xi Cheng, Xiaopeng Li, Bin Ran
  - Publisher: University of Wisconsin-Madison, Nanyang Technological University, Texas A&M University, Cornell University
  - Task: Planning
  - Projcet Page: [V2X-VLM](https://zilin-huang.github.io/V2X-VLM-website/)
  - Code: [V2X-VLM](https://github.com/zilin-huang/V2X-VLM)
  - Dataset: [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X)
  - Publish Date: 2024.08.09
  - Summary:
    - V2X-VLM, a large vision-language model empowered E2E VICAD framework, which improves the ability of autonomous vehicles to navigate complex traffic scenarios through advanced multimodal understanding and decision-making.
    - A contrastive learning technique is employed to refine the model’s ability to distinguish between relevant and irrelevant features, which ensures that the model learns robust and discriminative representations of specific driving environments, leading to improved accuracy in trajectory planning in V2X cooperation scenarios.

- [VLM-MPC: Vision Language Foundation Model (VLM)-Guided Model Predictive Controller (MPC) for Autonomous Driving](https://arxiv.org/abs/2408.04821)
  - Keke Long, Haotian Shi, Jiaxi Liu, Xiaopeng Li
  - Publisher: University of Wisconsin-Madison
  - Task: Planning
  - Publish Date: 2024.08.04
  - Summary:
    - It proposed a closed-loop autonomous driving controller that applies VLMs for high-level vehicle control. 
    - The upper-level VLM uses the vehicle's front camera images, textual scenario description, and experience memory as inputs to generate control parameters needed by the lower-level MPC. 
    - The lower-level MPC utilizes these parameters, considering vehicle dynamics with engine lag, to achieve realistic vehicle behavior and provide state feedback to the upper level. 
    - This asynchronous two-layer structure addresses the current issue of slow VLM response speeds.

- [SimpleLLM4AD: An End-to-End Vision-Language Model with Graph Visual Question Answering for Autonomous Driving](https://arxiv.org/abs/2407.21293)
  - Peiru Zheng, Yun Zhao, Zhan Gong, Hong Zhu, Shaohua Wu **IEIT Systems**
  - Publisher: University of Wisconsin-Madison
  - Task: QA
  - Publish Date: 2024.07.31
  - Summary:
    - SimpleLLM4AD reimagines the traditional autonomous driving pipeline by structuring the task into four interconnected stages: perception, prediction, planning, and behavior. 
    - Each stage is framed as a series of visual question answering (VQA) pairs, which are interlinked to form a Graph VQA (GVQA). This graph-based structure allows the system to reason about each VQA pair systematically, ensuring a coherent flow of information and decision-making from perception to action.

- [Testing Large Language Models on Driving Theory Knowledge and Skills for Connected Autonomous Vehicles](https://arxiv.org/abs/2407.17211)
  - Zuoyin Tang, Jianhua He, Dashuai Pei, Kezhong Liu, Tao Gao
  - Publisher: Aston University, Essex University, Wuhan University of Technology, Chang’An University
  - Task: Evaluation
  - Publish Date: 2024.07.24
  - Data: [UK Driving Theory Test Practice Questions and Answers](https://www.drivinginstructorwebsites.co.uk/uk-driving-theory-test-practice-questions-and-answers)
  - Summary:
    - Design and run driving theory tests for several proprietary LLM models (OpenAI GPT models, Baidu Ernie and Ali QWen) and open-source LLM models (Tsinghua MiniCPM-2B and MiniCPM-Llama3-V2.5) with more than 500 multiple-choices theory test questions.

- [KoMA: Knowledge-driven Multi-agent Framework for Autonomous Driving with Large Language Models](https://arxiv.org/abs/2407.14239)
  - Kemou Jiang, Xuan Cai, Zhiyong Cui, Aoyong Li, Yilong Ren, Haiyang Yu, Hao Yang, Daocheng Fu, Licheng Wen, Pinlong Cai
  - Publisher: Beihang University, Johns Hopkins University, Shanghai Artificial Intelligence Laboratory
  - Task: Multi Agent Planning
  - Env: [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv)
  - Project Page: [KoMA](https://jkmhhh.github.io/KoMA/)
  - Publish Date: 2024.07.19
  - Summary:
    - Introduce a knowledge-driven autonomous driving framework KoMA that incorporates multiple agents empowered by LLMs, comprising five integral modules: Environment, Multi-agent Interaction, Multi-step Planning, Shared Memory, and Ranking-based Reflection. 

- [WOMD-Reasoning: A Large-Scale Language Dataset for Interaction and Driving Intentions Reasoning](https://arxiv.org/abs/2407.04281)
  - Yiheng Li, Chongjian Ge, Chenran Li, Chenfeng Xu, Masayoshi Tomizuka, Chen Tang, Mingyu Ding, Wei Zhan
  - Publisher: UC Berkeley, UT Austin
  - Task: Dataset + Reasoning
  - Publish Date: 2024.07.05
  - Datasets: [WOMD-Reasoning](https://waymo.com/open/download)
  - Summary:
    - WOMD-Reasoning, a language dataset centered on interaction descriptions and reasoning. It provides extensive insights into critical but previously overlooked interactions induced by traffic rules and human intentions.
    - Develop an automatic language labeling pipeline, leveraging a rule-based translator to interpret motion data into language descriptions, and a set of manual prompts for ChatGPT to generate Q&A pairs. 

- [Asynchronous Large Language Model Enhanced Planner for Autonomous Driving](https://arxiv.org/abs/2406.14556)
  - Yuan Chen, Zi-han Ding, Ziqin Wang, Yan Wang, Lijun Zhang, Si Liu  **ECCV 2024**
  - Publisher: Beihang University, Tsinghua University
  - Task: Planning
  - Publish Date: 2024.06.20
  - Code: [AsyncDriver](https://github.com/memberRE/AsyncDriver)
  - Datasets: [nuPlan Closed-Loop Reactive Hard20](https://www.nuscenes.org/nuplan)
  - Summary:
    - AsyncDriver, a novel asynchronous LLM-enhanced framework, in which the inference frequency of LLM is controllable and can be decoupled from that of the real-time planner.
    - Adaptive Injection Block, which is model-agnostic and can easily integrate scene-associated instruction features into any transformer based
    real-time planner, enhancing its ability in comprehending and following series of language-based routing instructions.
    - Compared with existing methods, our approach demonstrates superior closedloop evaluation performance in nuPlan’s challenging scenarios.

- [A Superalignment Framework in Autonomous Driving with Large Language Models](https://arxiv.org/abs/2406.05651)
  - Xiangrui Kong, Thomas Braunl, Marco Fahmi, Yue Wang
  - Publisher: University of Western Australia, Queensland Government, Brisbane, Queensland University of Technology
  - Task: QA
  - Publish Date: 2024.06.09
  - Summary
    - Propose a secure interaction framework for LLMs to effectively audit data interacting with cloud-based LLMs.
    - Analyze 11 autonomous driving methods based on large language models, including driving safety, token usage, privacy, and consistency with human values.
    - Evaluate the effectiveness of driving prompts in the nuScenesQA dataset and compare different results between gpt-35-turbo and llama2-70b LLM backbones.

- [PlanAgent: A Multi-modal Large Language Agent for Closed-loop Vehicle Motion Planning](https://arxiv.org/abs/2406.01587)
  - Yupeng Zheng, Zebin Xing, Qichao Zhang, Bu Jin, Pengfei Li, Yuhang Zheng, Zhongpu Xia, Kun Zhan, Xianpeng Lang, Yaran Chen, Dongbin Zhao
  - Publisher: Chinese Academy of Sciences, Beijing University of Posts and Telecommunications, Beihang University, Tsinghua University, Li Auto
  - Task: Planning
  - Publish Date: 2024.06.04
  - Summary:
    - PlanAgent is the first closed-loop mid-to-mid(use bev, no raw sensor) autonomous driving planning agent system based on a Multi-modal Large Language Model.
    - Propose an efficient Environment Transformation module that extracts multi-modal information inputs with lanegraph representation.
    - Design a Reasoning Engine module that introduces a hierarchical chain-of-thought (CoT) to instruct MLLM to generate planner code and a Reflection module that combines simulation and scoring to filter out unreasonable proposals generated by the MLLM.

- [ChatScene: Knowledge-Enabled Safety-Critical Scenario Generation for Autonomous Vehicles](https://arxiv.org/abs/2405.14062)
  - Jiawei Zhang, Chejian Xu, Bo Li **CVPR 2024**
  - Publisher: UIUC, UChicago
  - Task: Scenario Generation
  - Env: [Carla](https://github.com/carla-simulator)
  - Code: [ChatScene](https://github.com/javyduck/ChatScene)
  - Publish Date: 2024.05.22
  - Summary:
    - ChatScene, a novel LLM-based agent capable of generating safety-critical scenarios by first providing textual descriptions and then carefully transforming them into executable simulations in CARLA via Scenic programming language.
    - An expansive retrieval database of Scenic code snippets has been developed. It catalogs diverse adversarial behaviors and traffic configurations, utilizing the rich knowledge stored in LLMs, which significantly augments the variety and critical nature of the driving scenarios generated.

- [Probing Multimodal LLMs as World Models for Driving](https://arxiv.org/abs/2405.05956)
  - Shiva Sreeram, Tsun-Hsuan Wang, Alaa Maalouf, Guy Rosman, Sertac Karaman, Daniela Rus
  - Publisher: MIT CSAIL, TRI, MIT LID
  - Task: Benchmark & Evaluation
  - Code: [DriveSim](https://github.com/sreeramsa/DriveSim)
  - Publish Date: 2024.05.09
  - Summary:
    - A comprehensive experimental study to evaluate the capability of different MLLMs to reason/understand scenarios involving closed-loop driving and making decisions.
    - DriveSim, a specialized simulator designed to generate a diverse array of driving scenarios, thereby providing a platform to test and evaluate/benchmark the capabilities of MLLMs in understanding and reasoning about real-world driving scenes from a fixed in-car camera perspective, the same as the drive viewpoint.

- [OmniDrive: A Holistic LLM-Agent Framework for Autonomous Driving with 3D Perception Reasoning and Planning](https://arxiv.org/abs/2405.01533)
  - Shihao Wang, Zhiding Yu, Xiaohui Jiang, Shiyi Lan, Min Shi, Nadine Chang, Jan Kautz, Ying Li, Jose M. Alvarez
  - Publisher: Beijing Inst of Tech, NVIDIA, Huazhong Univ of Sci and Tech
  - Task: Benchmark & Planning
  - Publisher Data: 2024.05.02
  - Code: [OmniDrive](https://github.com/NVlabs/OmniDrive)
  - Summary:
    - OmniDrive, a holistic framework for strong alignment between agent models and 3D driving tasks.
    - Propose a new benchmark with comprehensive visual question-answering (VQA) tasks, including scene description, traffic regulation, 3D grounding, counterfactual reasoning, decision making and planning.

- [REvolve: Reward Evolution with Large Language Models for Autonomous Driving](https://arxiv.org/abs/2406.01309)
  - Rishi Hazra, Alkis Sygkounas, Andreas Persson, Amy Loutfi, Pedro Zuidberg Dos Martires
  - Publisher: Centre for Applied Autonomous Sensor Systems (AASS), Örebro University, Swede
  - Task: Reward Generation
  - Env: [AirSim](https://github.com/microsoft/AirSim?tab=readme-ov-file)
  - Project Page: [REvolve](https://rishihazra.github.io/REvolve/)
  - Publish Date: 2024.04.09
  - Summary:
    - Reward Evolve (REvolve), a novel evolutionary framework using LLMs, specifically GPT-4, to output reward functions (as executable Python codes) for AD and evolve them based on human feedback.

- [AGENTSCODRIVER: Large Language Model Empowered Collaborative Driving with Lifelong Learning](https://arxiv.org/pdf/2404.06345.pdf)
  - Senkang Hu, Zhengru Fang, Zihan Fang, Xianhao Chen, Yuguang Fang
  - Publisher: City University of Hong Kong, The University of Hong Kong
  - Task: Planning(Multiple vehicles collaborative)
  - Publish Date: 2024.04.09
  - Env: [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv)
  - Summary:
    - AGENTSCODRIVER, an LLM-powered multi-vehicle collaborative driving framework with lifelong learning, which allows different driving agents to communicate with each other and collaboratively drive in complex traffic scenarios.
    - It features reasoning engine, cognitive memory, reinforcement reflection, and communication module. 

- [Multi-Frame, Lightweight & Efficient Vision-Language Models for Question Answering in Autonomous Driving](https://arxiv.org/abs/2403.19838)
  - Akshay Gopalkrishnan, Ross Greer, Mohan Trivedi
  - Publisher: UCSD
  - Task: QA
  - Publish Date: 2024.03.28
  - Code: [official](https://github.com/akshaygopalkr/EM-VLM4AD)
  - Datasets: [DriveLM](https://github.com/OpenDriveLab/DriveLM)
  - Summary:
    - EM-VLM4AD, an efficient, lightweight, multi-frame vision language model which performs Visual Question Answering for autonomous driving.
    - EM-VLM4AD requires at least 10 times less memory and floating point operations, while also achieving higher BLEU-4, METEOR, CIDEr, and ROGUE scores than the existing baseline on the DriveLM dataset.

- [LC-LLM: Explainable Lane-Change Intention and Trajectory Predictions with Large Language Models](https://arxiv.org/abs/2403.18344)
  - Mingxing Peng, Xusen Guo, Xianda Chen, Meixin Zhu, Kehua Chen, Hao (Frank) Yang, Xuesong Wang, Yinhai Wang
  - Publisher: The Hong Kong University of Science and Technology, Johns Hopkins University, Tongji University, STAR Lab
  - Task: Trajectory Prediction
  - Publish Date: 2024.03.27
  - Datasets: [highD](https://levelxdata.com/highd-dataset/)
  - Summary:
    - LC-LLM, the first Large Language Model for lane change prediction. It leverages the powerful capabilities of LLMs to understand complex interactive scenarios, enhancing the performance of lane change prediction.
    - LC-LLM achieves explainable predictions. It not only predicts lane change intentions and trajectories but also generates explanations for the prediction results.

- [AIDE: An Automatic Data Engine for Object Detection in Autonomous Driving](https://arxiv.org/abs/2403.17373)
  - Mingfu Liang, Jong-Chyi Su, Samuel Schulter, Sparsh Garg, Shiyu Zhao, Ying Wu, Manmohan Chandraker
  - Publisher: Northwestern University, NEC Laboratories America, Rutgers University, UC San Diego
  - Publish Date: 2024.03.26
  - Task: Object Detection
  - Datasets: [Mapillary](https://www.mapillary.com/dataset/vistas), [Cityscapes](https://www.cityscapes-dataset.com/), [nuImages](https://www.nuscenes.org/nuimages), [BDD100k](https://www.vis.xyz/bdd100k/), [Waymo](https://waymo.com/open/), [KITTI](https://www.cvlibs.net/datasets/kitti/)
  - Summary:
    - An Automatic Data Engine (AIDE) that can automatically identify the issues, efficiently curate data, improve the model using auto-labeling, and verify the model through generated diverse scenarios.

- [Engineering Safety Requirements for Autonomous Driving with Large Language Models](https://arxiv.org/abs/2403.16289)
  - Ali Nouri, Beatriz Cabrero-Daniel, Fredrik Törner, Hȧkan Sivencrona, Christian Berger
  - Publisher: Chalmers University of Technology, University of Gothenburg, Volvo Cars, Zenseact, University of Gothenburg
  - Task: QA
  - Publish Date: 2024.03.24
  - Summary:
    - Propose a prototype of a pipeline of prompts and LLMs that receives an item definition and outputs solutions in the form of safety requirements.

- [LeGo-Drive: Language-enhanced Goal-oriented Closed-Loop End-to-End Autonomous Driving](https://arxiv.org/abs/2403.20116)
  - Pranjal Paul, Anant Garg, Tushar Choudhary, Arun Kumar Singh, K. Madhava Krishna
  - Publisher: The International Institute of Information Technology, Hyderabad, University of Tartu, Estonia
  - Project Page: [LeGo-Drive](https://reachpranjal.github.io/lego-drive/)
  - Code: [LeGo-Drive](https://github.com/reachpranjal/lego-drive)
  - Env: [Carla](https://github.com/carla-simulator)
  - Task: Trajectory Prediction
  - Publish Date: 2024.03.20
  - Summary:
    - A novel planning-guided end-to-end LLM-based goal point navigation solution that predicts and improves the desired state by dynamically interacting with the
environment and generating a collision-free trajectory.

- [Hybrid Reasoning Based on Large Language Models for Autonomous Car Driving](https://arxiv.org/abs/2402.13602v3)
  - Mehdi Azarafza, Mojtaba Nayyeri, Charles Steinmetz, Steffen Staab, Achim Rettberg
  - Publisher: Univ. of Applied Science Hamm-Lippstadt, University of Stuttgart
  - Publish Date: 2024.03.18
  - Task: Reasoning
  - Env: [Carla](https://github.com/carla-simulator)
  - Summary:
    - Combining arithmetic and commonsense elements, utilize the objects detected by YOLOv8.
    - Regarding the "location of the object," "speed of our car," "distance to the object," and "our car’s direction" are fed into the large language model for mathematical calculations within CARLA.
    - After formulating these calculations based on overcoming weather conditions, precise control values for brake and speed are generated.

- [Large Language Models Powered Context-aware Motion Prediction](https://arxiv.org/pdf/2403.11057.pdf)
  - Xiaoji Zheng, Lixiu Wu, Zhijie Yan, Yuanrong Tang, Hao Zhao, Chen Zhong, Bokui Chen, Jiangtao Gong
  - Publisher: Tsinghua University
  - Task: Motion Prediction
  - Publish Data: 2024.03.17
  - Dataset: [WOMD](https://github.com/waymo-research/waymo-open-dataset)
  - Summary:
    - Design and conduct prompt engineering to enable an unfine-tuned GPT4-V to comprehend complex traffic scenarios.
    - Introduced a novel approach that combines the context information outputted by GPT4-V with [MTR](https://arxiv.org/abs/2209.13508).

- [Generalized Predictive Model for Autonomous Driving](https://arxiv.org/abs/2403.09630)
  - Jiazhi Yang, Shenyuan Gao, Yihang Qiu, Li Chen, Tianyu Li, Bo Dai, Kashyap Chitta, Penghao Wu, Jia Zeng, Ping Luo, Jun Zhang, Andreas Geiger, Yu Qiao, Hongyang Li **ECCV 2024**
  - Publisher:  OpenDriveLab and Shanghai AI Lab, Hong Kong University of Science and Technology, University of Hong Kong, University of Tubingen, Tubingen AI Center
  - Task: Datasets + Generation
  - Code: [DriveAGI](https://github.com/OpenDriveLab/DriveAGI.)
  - Publish Date: 2024.03.14
  - Summary:
    - Introduce the first large-scale video prediction model in the autonomous driving discipline.
    - The resultant dataset accumulates over 2000 hours of driving videos, spanning areas all over the world with diverse weather conditions and traffic scenarios.
    - GenAD, inheriting the merits from recent latent diffusion models, handles the challenging dynamics in driving scenes with novel temporal reasoning blocks.

- [LLM-Assisted Light: Leveraging Large Language Model Capabilities for Human-Mimetic Traffic Signal Control in Complex Urban Environments](https://arxiv.org/abs/2403.08337)
  - Maonan Wang, Aoyu Pang, Yuheng Kan, Man-On Pun, Chung Shue Chen, Bo Huang
  - Publisher: The Chinese University of Hong Kong, Shanghai AI Laboratory, SenseTime Group Limited, Nokia Bell Labs
  - Publish Date: 2024.03.13
  - Task: Generation
  - Code: [LLM-Assisted-Light](https://github.com/Traffic-Alpha/LLM-Assisted-Light)
  - Summary:
    - LA-Light, a hybrid TSC framework that integrates the human-mimetic reasoning capabilities of LLMs, enabling the signal control algorithm to interpret and respond to complex traffic scenarios with the nuanced judgment typical of human cognition.
    - A closed-loop traffic signal control system has been developed, integrating LLMs with a comprehensive suite of interoperable tools.

- [DriveDreamer-2: LLM-Enhanced World Models for Diverse Driving Video Generation](https://arxiv.org/abs/2403.06845)
  - Guosheng Zhao, Xiaofeng Wang, Zheng Zhu, Xinze Chen, Guan Huang, Xiaoyi Bao, Xingang Wang
  - Publisher: Institute of Automation, Chinese Academy of Sciences, GigaAI
  - Publish Date: 2024.03.11
  - Task: Generation
  - Project: [DriveDreamer-2](https://drivedreamer2.github.io)
  - Datasets: [nuScenes](https://www.nuscenes.org/nuscenes)
  - Summary:
    - DriveDreamer-2, which builds upon the framework of [DriveDreamer](#DriveDreamer) and incorporates a Large Language Model (LLM) to generate user-defined driving videos.
    - UniMVM(Unified Multi-View Model) enhances temporal and spatial coherence in the generated driving videos.
    - HDMap generator ensure the background elements do not conflict with the foreground trajectories.
    - Utilize the constructed text-to-script dataset to finetune GPT-3.5 into an LLM with specialized trajectory generation knowledge.

- [Editable Scene Simulation for Autonomous Driving via Collaborative LLM-Agents](https://arxiv.org/abs/2402.05746)
  - Yuxi Wei, Zi Wang, Yifan Lu, Chenxin Xu, Changxing Liu, Hao Zhao, Siheng Chen, Yanfeng Wang
  - Publisher: Shanghai Jiao Tong University, Shanghai AI Laboratory, Carnegie Mellon University, Tsinghua University
  - Publish Date: 2024.03.11
  - Task: Generation
  - Code: [ChatSim](https://github.com/yifanlu0227/ChatSim)
  - Datasets: [Waymo](https://waymo.com/open/)
  - Summary:
    - ChatSim, the first system that enables editable photo-realistic 3D driving scene simulations via natural language commands with external digital assets.
    - McNeRF, a novel neural radiance field method that incorporates multi-camera inputs, offering a broader scene rendering. It helps generate photo-realistic outcomes.
    - McLight, a novel multicamera lighting estimation that blends skydome and surrounding lighting. It makes external digital assets with their realistic textures and materials.
  
- [Embodied Understanding of Driving Scenarios](https://arxiv.org/abs/2403.04593)
  - Yunsong Zhou, Linyan Huang, Qingwen Bu, Jia Zeng, Tianyu Li, Hang Qiu, Hongzi Zhu, Minyi Guo, Yu Qiao, Hongyang Li **ECCV 2024**
  - Shanghai AI Lab, Shanghai Jiao Tong University, University of California, Riverside
  - Publish Date: 2024.03.07
  - Task: Benchmark & Scene Understanding
  - Code: [ELM](https://github.com/OpenDriveLab/ELM)
  - Summary:
    - ELM is an embodied language model for understanding the long-horizon driving scenarios in space and time. 

- [DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models](https://arxiv.org/abs/2402.12289) 
  - Xiaoyu Tian, Junru Gu, Bailin Li, Yicheng Liu, Chenxu Hu, Yang Wang, Kun Zhan, Peng Jia, Xianpeng Lang, Hang Zhao
  - Publisher: IIIS, Tsinghua University, Li Auto
  - Publish Date: 2024.02.25
  - Task: + Planning
  - Project: [DriveVLM](https://tsinghua-mars-lab.github.io/DriveVLM/)
  - Datasets: [nuScenes](https://www.nuscenes.org/nuscenes), SUP-AD
  - Summary:
    - DriveVLM, a novel autonomous driving system that leverages VLMs for effective scene understanding and planning.
    - DriveVLM-Dual, a hybrid system that incorporates DriveVLM and a traditional autonomous pipeline. 

- [GenAD: Generative End-to-End Autonomous Driving](https://arxiv.org/abs/2402.11502)
  - Wenzhao Zheng, Ruiqi Song, Xianda Guo, Long Chen  **ECCV 2024**
  - University of California, Berkeley, Waytous, Institute of Automation, Chinese Academy of Sciences
  - Publish Date: 2024.02.20
  - Task: Generation
  - Code: [GenAD](https://github.com/wzzheng/GenAD)
  - Datasets: [nuScenes](https://www.nuscenes.org/nuscenes)
  - Summary:
    - GenAD models autonomous driving as a trajectory generation problem to unleash the full potential of endto-end methods. 
    - Propose an instance-centric scene tokenizer that first transforms the surrounding scenes into map-aware instance tokens.
    - Employ a variational autoencoder to learn the future trajectory distribution in a structural latent space for trajectory prior modeling and adopt a temporal model to capture the agent and ego movements in the latent space to generate more effective future trajectories. 

- [RAG-Driver: Generalisable Driving Explanations with Retrieval-Augmented In-Context Learning in Multi-Modal Large Language Model](https://arxiv.org/abs/2402.10828)
  - Jianhao Yuan, Shuyang Sun, Daniel Omeiza, Bo Zhao, Paul Newman, Lars Kunze, Matthew Gadd
  - Publisher: University of Oxford, Beijing Academy of Artificial Intelligence
  - Publish Date: 2024.02.16
  - Task: Explainable Driving
  - Project: [RAG-Driver](https://yuanjianhao508.github.io/RAG-Driver/)
  - Summary:
    - RAG-Driver is a Multi-Modal Large Language Model with Retrieval-augmented In-context Learning capacity designed for generalisable and explainable end-to-end driving with strong zero-shot generalisation capacity.
    - Achieve State-of-the-art action explanation and justification performance in both BDD-X (in-distribution) and SAX (out-distribution) benchmarks. 

- [Driving Everywhere with Large Language Model Policy Adaptation](https://arxiv.org/abs/2402.05932)
  - Boyi Li, Yue Wang, Jiageng Mao, Boris Ivanovic, Sushant Veer, Karen Leung, Marco Pavone **CVPR 2024**
  - Publisher: NVIDIA, University of Southern California, University of Washington, Stanford University
  - Publish Date: 2024.02.08
  - Task: Planning
  - Datasets: [nuScenes](https://www.nuscenes.org/nuscenes)
  - Project: [LLaDA](https://boyiliee.github.io/llada/)
  - Summary:
    - LLaDA is a training-free mechanism to assist human drivers and adapt autonomous driving policies to new environments.
    - Traffic Rule Extractor (TRE), which aims to organize and filter the inputs (initial plan+unique traffic code) and feed the output into the frozen LLMs to obtain the final new plan. 
    - LLaDA set GPT-4 as default LLM.

- [LimSim++](https://arxiv.org/abs/2402.01246)
  - Daocheng Fu, Wenjie Lei, Licheng Wen, Pinlong Cai, Song Mao, Min Dou, Botian Shi, Yu Qiao
  - Publisher: Shanghai Artificial Intelligence Laboratory, Zhejiang University
  - Publish Date: 2024.02.02
  - Project: [LimSim++](https://pjlab-adg.github.io/limsim_plus/)
  - Summary:
    - LimSim++, an extended version of LimSim designed for the application of (M)LLMs in autonomous driving.
    - Introduce a baseline (M)LLM-driven framework, systematically validated through quantitative experiments across diverse scenarios.

- [LangProp: A code optimization framework using Language Models applied to driving](https://openreview.net/forum?id=UgTrngiN16)
  - Shu Ishida, Gianluca Corrado, George Fedoseev, Hudson Yeo, Lloyd Russell, Jamie Shotton, João F. Henriques, Anthony Hu
  - Publisher: Wayve Technologies, Visual Geometry Group, University of Oxford
  - Publish Date: 2024.01.18
  - Task: Code generation, Planning
  - Code: [LangProp](https://github.com/shuishida/LangProp)
  - Env: [CARLA](https://github.com/carla-simulator)
  - Summary:
    - LangProp is a framework for iteratively optimizing code generated by large language models (LLMs) in a supervised/reinforcement learning setting.
    - Use LangProp in CARLA and generate driving code based on the state of the scene.

- [VLP: Vision Language Planning for Autonomous Driving](https://arxiv.org/abs/2401.05577)
  - Chenbin Pan, Burhaneddin Yaman, Tommaso Nesti, Abhirup Mallik, Alessandro G Allievi, Senem Velipasalar, Liu Ren **CVPR 2024**
  - Publisher: Syracuse University, Bosch Research North America & Bosch Center for Artificial Intelligence (BCAI)
  - Publish Date: 2024.01.14
  - Datasets: [nuScenes](https://www.nuscenes.org/nuscenes)
  - Summary:
    - Propose VLP, a Vision Language Planning model, which is composed of novel components ALP and SLP, aiming to improve the ADS from self-driving BEV reasoning and self-driving decision-making aspects, respectively.
    - ALP(agent-wise learning paradigm) aligns the produced BEV with a true bird’s-eye-view map.
    - SLP(selfdriving-car-centric learning paradigm) aligns the ego-vehicle query feature with the ego-vehicle textual planning feature.

- [DME-Driver: Integrating Human Decision Logic and 3D Scene Perception in Autonomous Driving](https://arxiv.org/abs/2401.03641)
  - Wencheng Han, Dongqian Guo, Cheng-Zhong Xu, and Jianbing Shen
  - Publisher: SKL-IOTSC, CIS, University of Macau
  - Publish Date: 2024.01.08
  - Summary:
    - DME-Driver = Decision-Maker + Executor + CL
    - Executor network which is based on UniAD incorporates textual information for the OccFormer and the Planning module.
    - Decision-Maker which is based on LLaVA process inputs from three different modalities: visual inputs from the current and previous scenes textual inputs in the form of prompts, and current status information detailing the vehicle’s operating state.
    - CL is a consistency loss mechanism, slightly reducing performance metrics but significantly enhancing decision alignment between Executor and Decision-Maker.

- [AccidentGPT: Accident Analysis and Prevention from V2X Environmental Perception with Multi-modal Large Model](https://arxiv.org/abs/2312.13156)
  - Lening Wang, Yilong Ren, Han Jiang, Pinlong Cai, Daocheng Fu, Tianqi Wang, Zhiyong Cui, Haiyang Yu, Xuesong Wang, Hanchu Zhou, Helai Huang, Yinhai Wang
  - Publisher: Beihang University, Shanghai Artificial Intelligence Laboratory, The University of Hong Kong, Zhongguancun Laboratory, Tongji University, Central South University, University of Washington, Seattle
  - Publish Date: 2023.12.29
  - Project page: [AccidentGPT](https://accidentgpt.github.io)
  - Summary:
    - AccidentGPT, a comprehensive accident analysis and prevention multi-modal large model.
    - Integrates multi-vehicle collaborative perception for enhanced environmental understanding and collision avoidance.
    - Offer advanced safety features such as proactive remote safety warnings and blind spot alerts.
    - Serve traffic police and management agencies by providing real-time intelligent analysis of traffic safety factors.

- [Holistic Autonomous Driving Understanding by Bird’s-Eye-View Injected Multi-Modal Large Models](https://arxiv.org/abs/2401.00988)
  - Xinpeng Ding, Jinahua Han, Hang Xu, Xiaodan Liang, Wei Zhang, Xiaomeng Li **CVPR 2024**
  - Publisher: Hong Kong University of Science and Technology, Huawei Noah’s Ark Lab, Sun Yat-Sen University
  - Publish Date: 2023.12.21
  - Task: Datasets + VQA
  - Code: [official](https://github.com/xmed-lab/NuInstruct)
  - Summary:
    - Introduce NuInstruct, a novel dataset with 91K multi-view video-QA pairs across 17 subtasks, which based on [nuScenes](https://www.nuscenes.org/nuscenes).
    - Propose BEV-InMLMM to integrate instructionaware BEV features with existing MLLMs, enhancing them with a full suite of information, including temporal, multi-view, and spatial details.

- [LLM-ASSIST: Enhancing Closed-Loop Planning with Language-Based Reasoning](https://arxiv.org/abs/2401.00125)
  - S P Sharan, Francesco Pittaluga, Vijay Kumar B G, Manmohan Chandraker
  - Publisher: UT Austin， NEC Labs America， UC San Diego
  - Publish Date: 2023.12.30
  - Task: Planning
  - Env/Datasets: nuPlan Closed-Loop Non-Reactive Challenge
  - Project: [LLM-ASSIST](https://llmassist.github.io/)
  - Summary:
    - LLM-Planner takes over scenarios that PDM-Closed cannot handle
    - Propose two LLM-based planners.
      - LLM-ASSIST(unc) considers the most unconstrained version of the planning problem, in which the LLM must directly return a safe future trajectory for the ego car. 
      - LLM-ASSIST(par) considers a parameterized version of the planning problem, in which the LLM must only return a set of parameters for a rule-based planner, PDM-Closed.

- [DriveLM: Driving with Graph Visual Question Answering](https://arxiv.org/pdf/2312.14150.pdf)
  - Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Ping Luo, Andreas Geiger, Hongyang Li **ECCV 2024**
  - Publisher: OpenDriveLab, University of Tübingen, Tübingen AI Center, University of Hong Kong
  - Code: [official](https://github.com/OpenDriveLab/DriveLM)
  - Publish Date: 2023.12.21
  - Summary:
    - DriveLM-Task
      - Graph VQA involves formulating P1-3(Perception, Prediction, Planning) reasoning as a series of questionanswer pairs (QAs) in a directed graph.
    - DriveLM-Data
      - DriveLM-Carla
        - Collect data using CARLA 0.9.14 in the Leaderboard 2.0 framework [17] with a privileged rule-based expert.
      - Drive-nuScenes
        - Selecting key frames from video clips, choosing key objects within these key frames, and subsequently annotating the frame-level P1−3 QAs for these key objects. A portion of the Perception QAs are generated from the nuScenes and [OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2) ground truth, while the remaining QAs are manually annotated.
    - DriveLM-Agent
      - DriveLMAgent is built upon a general vision-language model and can therefore exploit underlying knowledge gained during pre-training. 

- [LingoQA: Video Question Answering for Autonomous Driving](https://arxiv.org/abs/2312.14115)
  - Ana-Maria Marcu, Long Chen, Jan Hünermann, Alice Karnsund, Benoit Hanotte, Prajwal Chidananda, Saurabh Nair, Vijay Badrinarayanan, Alex Kendall, Jamie Shotton, Oleg Sinavski
  - Publisher: Wayve
  - Task: VQA + Evaluation/Datasets
  - Code: [official](https://github.com/wayveai/LingoQA)
  - Publish Date: 2023.12.21
  - Summary:
    - Introduce a novel benchmark for autonomous driving video QA using a learned text classifier for evaluation. 
    - Introduce a Video QA dataset of central London consisting of 419k samples with its free-form questions and answers.
    - Establish a new baseline based on Vicuna-1.5-7B for this field with an identified model combination.

- [DriveMLM: Aligning Multi-Modal Large Language Models with Behavioral Planning States for Autonomous Driving](https://arxiv.org/abs/2312.09245)
  - Wenhai Wang, Jiangwei Xie, ChuanYang Hu, Haoming Zou, Jianan Fan, Wenwen Tong, Yang Wen, Silei Wu, Hanming Deng, Zhiqi Li, Hao Tian, Lewei Lu, Xizhou Zhu, Xiaogang Wang, Yu Qiao, Jifeng Dai
  - Publisher: OpenGVLab, Shanghai AI Laboratory, The Chinese University of Hong Kong, SenseTime Research, Stanford University, Nanjing University, Tsinghua University
  - Task: Planning + Explanation
  - Code: [official](https://github.com/OpenGVLab/DriveMLM)
  - Env: [Carla](https://carla.org/)
  - Publish Date: 2023.12.14
  - Summary:
    - DriveMLM, the first LLM-based AD framework that can perform close-loop
autonomous driving in realistic simulators.
    - Design an MLLM planner for decision prediction, and develop a data engine that can effectively generate decision states and corresponding explanation annotation for model training and evaluation.
    - Achieve 76.1 DS, 0.955 MPI results on CARLA Town05 Long.

- [Large Language Models for Autonomous Driving: Real-World Experiments](https://arxiv.org/abs/2312.09397)
  - Can Cui, Yunsheng Ma, Xu Cao, Wenqian Ye, Yang Zhou, Kaizhao Liang, Jintai Chen, Juanwu Lu, Zichong Yang, Kuei-Da Liao, Tianren Gao, Erlong Li, Kun Tang, Zhipeng Cao, Tong Zhou, Ao Liu, Xinrui Yan, Shuqi Mei, Jianguo Cao, Ziran Wang, Chao Zheng
  - Publisher: Purdue University
  - Publish Date: 2023.12.14
  - Project: [official](https://www.youtube.com/playlist?list=PLgcRcf9w8BmJfZigDhk1SAfXV0FY65cO7)
  - Summary:
    - Introduce a Large Language Model (LLM)-based framework Talk-to-Drive (Talk2Drive) to process verbal commands from humans and make autonomous driving decisions with contextual information, satisfying their personalized preferences for safety, efficiency, and comfort.

- [LMDrive: Closed-Loop End-to-End Driving with Large Language Models](https://arxiv.org/abs/2312.07488)
  - Hao Shao, Yuxuan Hu, Letian Wang, Steven L. Waslander, Yu Liu, Hongsheng Li      **CVPR 2024**
  - Publisher: CUHK MMLab, SenseTime Research, CPII under InnoHK, University of Toronto, Shanghai Artificial Intelligence Laboratory
  - Task: Planning + Datasets
  - Code: [official](https://github.com/opendilab/LMDrive)
  - Env: [Carla](https://carla.org/)
  - Publish Date: 2023.12.12
  - Summary:
    - LMDrive, a novel end-to-end, closed-loop, languagebased autonomous driving framework.
    - Release 64K clips dataset, including navigation instruction, notice instructions, multi-modal multi-view sensor data, and control signals.
    - Present the benchmark LangAuto for evaluating the autonomous agents.

- [Evaluation of Large Language Models for Decision Making in Autonomous Driving](https://arxiv.org/pdf/2312.06351.pdf)
  - Kotaro Tanahashi, Yuichi Inoue, Yu Yamaguchi, Hidetatsu Yaginuma, Daiki Shiotsuka, Hiroyuki Shimatani, Kohei Iwamasa, Yoshiaki Inoue, Takafumi Yamaguchi, Koki Igari, Tsukasa Horinouchi, Kento Tokuhiro, Yugo Tokuchi, Shunsuke Aoki
  - Publisher: Turing Inc., Japan
  - Task: Evalution
  - Publish Date: 2023.12.11
  - Summary:
    - Evaluate the two core capabilities
      - the spatial awareness decision-making ability, that is, LLMs can accurately identify the spatial layout based on coordinate information;
      - the ability to follow traffic rules to ensure that LLMs Ability to strictly abide by traffic laws while driving

- [LaMPilot: An Open Benchmark Dataset for Autonomous Driving with Language Model Programs](https://arxiv.org/abs/2312.04372)
  - Yunsheng Ma, Can Cui, Xu Cao, Wenqian Ye, Peiran Liu, Juanwu Lu, Amr Abdelraouf, Rohit Gupta, Kyungtae Han, Aniket Bera, James M. Rehg, Ziran Wang
  - Publisher: Purdue University, University of Illinois Urbana-Champaign, University of Virginia, InfoTech Labs, Toyota Motor North American
  - Task: Benchmark
  - Publish Date: 2023.12.07
  - Summary:
    - LaMPilot is the first interactive environment and dataset designed for evaluating LLM-based agents in a driving context.
    - It contains 4.9K scenes and is specifically designed to evaluate command tracking tasks in autonomous driving.

- [Reason2Drive: Towards Interpretable and Chain-based Reasoning for Autonomous Driving](https://arxiv.org/abs/2312.03661)
  - Ming Nie, Renyuan Peng, Chunwei Wang, Xinyue Cai, Jianhua Han, Hang Xu, Li Zhang
  - Publisher: Fudan University, Huawei Noah’s Ark Lab
  - Task: VQA + Datasets
  - Code: [official](https://github.com/fudan-zvg/Reason2Drive)
  - Datasets:
    - [nuScenes](https://www.nuscenes.org/nuscenes)
    - [Waymo](https://waymo.com/open/)
    - [ONCE](https://once-for-auto-driving.github.io/index.html)
  - Publish Date: 2023.12.06
  - Summary:
    - Reason2Drive, a benchmark dataset with over 600K video-text pairs, aimed at facilitating the study of interpretable reasoning in complex driving.
    - Introduce a novel evaluation metric to assess chain-based reasoning performance in autonomous driving environments, and address the semantic ambiguities of existing metrics such as BLEU and CIDEr.
    - Introduce a straightforward yet effective framework that enhances existing VLMs with two new components: a prior tokenizer and an instructed vision decoder.

- [GPT-4 Enhanced Multimodal Grounding for Autonomous Driving: Leveraging Cross-Modal Attention with Large Language Models](https://arxiv.org/abs/2312.03543)
  - Haicheng Liao, Huanming Shen, Zhenning Li, Chengyue Wang, Guofa Li, Yiming Bie, Chengzhong Xu
  - Publisher: University of Macau, UESTC, Chongqing University, Jilin University
  - Task: Detection/Prediction
  - Code: [official](https://github.com/Petrichor625/Talk2car_CAVG)
  - Datasets:
    - [Talk2car](https://github.com/talk2car/Talk2Car)
  - Publish Date: 2023.12.06
  - Summaray:
    - Utilize five encoder Text, Image, Context, and Cross-Modal—with a Multimodal decoder to pridiction object bounding box.

- [Dolphins: Multimodal Language Model for Driving](https://arxiv.org/abs/2312.00438)
  - Yingzi Ma, Yulong Cao, Jiachen Sun, Marco Pavone, Chaowei Xiao **ECCV 2024**
  - Publisher: University of Wisconsin-Madison, NVIDIA, University of Michigan, Stanford University
  - Task: VQA
  - Project: [Dolphins](https://vlm-driver.github.io/)
  - Code: [Dolphins](https://github.com/vlm-driver/Dolphins)
  - Datasets: 
    - Image instruction-following dataset
      - [GQA](https://cs.stanford.edu/people/dorarad/gqa/about.html)
      - [MSCOCO](https://cocodataset.org/#home): [VQAv2](https://visualqa.org/), [OK-VQA](https://okvqa.allenai.org/), [TDIUC](https://kushalkafle.com/projects/tdiuc.html), [Visual Genome dataset](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html)
    - Video instruction-following dataset
      - [BDD-X](https://github.com/JinkyuKimUCB/BDD-X-dataset)
  - Publish Date: 2023.12.01
  - Summary:
    - Dolphins which is base on OpenFlamingo architecture is a VLM-based conversational driving assistant.
    - Devise grounded CoT (GCoT) instruction tuning and develop datasets.

- [Driving into the Future: Multiview Visual Forecasting and Planning with World Model for Autonomous Driving](https://arxiv.org/abs/2311.17918)
  - Yuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen, Zhaoxiang Zhang
  - Publisher: CASIA, CAIR, HKISI, CAS
  - Task: Generation
  - Project: [Drive-WM](https://drive-wm.github.io/)
  - Code: [Drive-WM](https://github.com/BraveGroup/Drive-WM)
  - Datasets: [nuScenes](https://www.nuscenes.org/nuscenes), [Waymo Open Dataset](https://waymo.com/open/)
  - Publish Date: 2023.11.29
  - Summary:
    - Drive-WM, a multiview world model, which is capable of generating high-quality, controllable, and consistent multiview videos in autonomous driving scenes.
    - The first to explore the potential application of the world model in end-to-end planning for autonomous driving.

- [Empowering Autonomous Driving with Large Language Models: A Safety Perspective](https://arxiv.org/abs/2312.00812)
  - Yixuan Wang, Ruochen Jiao, Chengtian Lang, Sinong Simon Zhan, Chao Huang, Zhaoran Wang, Zhuoran Yang, Qi Zhu
  - Publisher: Northwestern University, University of Liverpool, Yale University
  - Task: Planning
  - Env: [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv)
  - Code: [official](https://github.com/wangyixu14/llm_conditioned_mpc_ad)
  - Publish Date: 2023.11.28
  - Summary:
    - Deploys the LLM as an intelligent decision-maker in planning, incorporating safety verifiers for contextual safety learning to enhance overall AD performance and safety.

- [GPT-4V Takes the Wheel: Evaluating Promise and Challenges for Pedestrian Behavior Prediction](https://arxiv.org/abs/2311.14786)
  - Jia Huang, Peng Jiang, Alvika Gautam, Srikanth Saripalli
  - Publisher: Texas A&M University, College Station, USA
  - Task: Evaluation(Pedestrian Behavior Prediction)
  - Datasets: 
    - [JAAD](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/)
    - [PIE](https://data.nvision2.eecs.yorku.ca/PIE_dataset/)
    - [WiDEVIEW](https://github.com/unmannedlab/UWB_Dataset)
  - Summary:
    - Provides a comprehensive evaluation of the potential of GPT-4V for pedestrian behavior prediction in autonomous driving using publicly available datasets.
    - It still falls short of the state-of-the-art traditional domain-specific models.
    - While GPT-4V represents a considerable advancement in AI capabilities for pedestrian behavior prediction, ongoing development and refinement are necessary to fully harness its capabilities in practical applications.

- [ADriver-I: A General World Model for Autonomous Driving](https://arxiv.org/abs/2311.13549)
  - Fan Jia, Weixin Mao, Yingfei Liu, Yucheng Zhao, Yuqing Wen, Chi Zhang, Xiangyu Zhang, Tiancai Wang
  - Publisher: MEGVII Technology, Waseda University, University of Science and Technology of China, Mach Drive
  - Task: Generation + Planning
  - Datasets: [nuScenes](https://www.nuscenes.org/nuscenes), Largescale private datasets
  - Publish Date: 2023.11.22
  - Summary:
    - ADriver-I takes the vision-action pairs as inputs and autoregressively predicts the control signal of current frame. The generated control signals together with the historical vision-action pairs are further conditioned to predict the future frames. 
    - MLLM(Multimodal large language model)=[LLaVA-7B-1.5](https://github.com/haotian-liu/LLaVA), VDM(Video Diffusion Model)=[latent-diffusion](https://github.com/CompVis/latent-diffusion)
  - Metrics:
    - L1 error including speed and steer angle of current frame.
    - Quality of Generation: Frechet Inception Distance(FID), Frechet Video Distance(FVD).

- [A Language Agent for Autonomous Driving](https://arxiv.org/abs/2311.10813)
  - Jiageng Mao, Junjie Ye, Yuxi Qian, Marco Pavone, Yue Wang
  - University of Southern California, Stanford University, NVIDIA
  - Task: Generation + Planning
  - Project: [Agent-Driver](https://usc-gvl.github.io/Agent-Driver/)
  - Datasets: [nuScenes](https://www.nuscenes.org/nuscenes)
  - Publish Date: 2023.11.17
  - Summary:
    - Agent-Driver integrates a tool library for dynamic perception and prediction, a cognitive memory for human knowledge, and a reasoning engine that emulates human decision-making.
    - For motion planning, follow GPT-Driver(#GPT-Driver) and fine-tune the LLM with human driving trajectories in the nuScenes training set for one epoch. 
    - For neural modules, adopte the modules in [UniAD](https://arxiv.org/abs/2212.10156).
  - Metric:
    - L2 error (in meters) and collision rate (in percentage).

- [Human-Centric Autonomous Systems With LLMs for User Command Reasoning](https://arxiv.org/abs/2311.08206)
  - Yi Yang, Qingwen Zhang, Ci Li, Daniel Simões Marta, Nazre Batool, John Folkesson
  - Publisher: KTH Royal Institute of Technology, Scania AB
  - Task: QA
  - Code: [DriveCmd](https://github.com/KTH-RPL/DriveCmd_LLM)
  - Datasets: [UCU Dataset](https://github.com/LLVM-AD/ucu-dataset)
  - Publish Date: 2023.11.14
  - Summary:
    - Propose to leverage the reasoning capabilities of Large Language Models (LLMs) to infer system requirements from in-cabin users’ commands.
    - LLVM-AD Workshop @ WACV 2024 
  - Metric:
    - Accuracy at the question level(accuracy for each individual question).
    - Accuracy at the command level(accuracy is only acknowledged if all questions for a particular command are correctly identified).

- [On the Road with GPT-4V(ision): Early Explorations of Visual-Language Model on Autonomous Driving](https://arxiv.org/abs/2311.05332)
  - Licheng Wen, Xuemeng Yang, Daocheng Fu, Xiaofeng Wang, Pinlong Cai, Xin Li, Tao Ma, Yingxuan Li, Linran Xu, Dengke Shang, Zheng Zhu, Shaoyan Sun, Yeqi Bai, Xinyu Cai, Min Dou, Shuanglu Hu, Botian Shi
  - Publisher: Shanghai Artificial Intelligence Laboratory,  GigaAI, East China Normal University, The Chinese University of Hong Kong, WeRide.ai
  - Project: [official](https://github.com/PJLab-ADG/GPT4V-AD-Exploration)
  - Datasets:
    - Scenario Understanding: [nuScenes](https://www.nuscenes.org/nuscenes), [BDD-X](https://github.com/JinkyuKimUCB/BDD-X-dataset), [Carla](https://github.com/carla-simulator), [TSDD](http://www.nlpr.ia.ac.cn/pal/trafficdata/detection.html), [Waymo](https://arxiv.org/abs/1912.04838), [DAIR-V2X](https://thudair.baai.ac.cn/index), [CitySim](https://github.com/ozheng1993/UCF-SST-CitySim-Dataset).
    - Reasoning Capability: [nuScenes](https://www.nuscenes.org/nuscenes), [D2-city](https://arxiv.org/abs/1904.01975), [Carla](https://github.com/carla-simulator), [CODA](https://arxiv.org/abs/2203.07724) and the internet
    - Act as a driver: Real-world driving scenarios.
  - Publish Date: 2023.11.9 
  - Summary:
    - Conducted a comprehensive and multi-faceted evaluation of the GPT-4V in various autonomous driving scenarios.
    - Test the capabilities of GPT-4V in Scenario Understanding, Reasoning, Act as a driver.

- [ChatGPT as Your Vehicle Co-Pilot: An Initial Attempt](https://ieeexplore.ieee.org/document/10286969)
  - Shiyi Wang, Yuxuan Zhu, Zhiheng Li, Yutong Wang, Li Li, Zhengbing He
  - Publisher: Tsinghua University, Institute of Automation, Chinese Academy of Sciences, Massachusetts Institute of Technology
  - Task: Planning
  - Publish Date: 2023.10.17
  - Summary:
    - Design a universal framework that embeds LLMs as a vehicle "Co-Pilot" of driving, which can accomplish specific driving tasks with human intention satisfied based on the information provided.

- [MagicDrive: Street View Generation with Diverse 3D Geometry Control](https://arxiv.org/abs/2310.02601)
  - Ruiyuan Gao, Kai Chen, Enze Xie, Lanqing Hong, Zhenguo Li, Dit-Yan Yeung, Qiang Xu
  - Publisher: The Chinese University of Hong Kong, Hong Kong University of Science and Technology, Huawei Noah’s Ark Lab
  - Task: Generation
  - Project: [MagicDrive](https://gaoruiyuan.com/magicdrive/)
  - Code: [MagicDrive](https://github.com/cure-lab/MagicDrive)
  - Datasets: [nuScenes](https://www.nuscenes.org/nuscenes)
  - Publish Date: 2023.10.13
  - Summary:
    - MagicDrive generates highly realistic images, exploiting geometric information from 3D annotations by independently encoding road maps, object boxes, and camera parameters for precise, geometry-guided synthesis. This approach effectively solves the challenge of multi-camera view consistency.
    - It also faces huge challenges in some complex scenes, such as night views and unseen weather conditions.

- [Receive, Reason, and React: Drive as You Say with Large Language Models in Autonomous Vehicles](https://arxiv.org/abs/2310.08034)
  - Can Cui, Yunsheng Ma, Xu Cao, Wenqian Ye, Ziran Wang
  - Publisher: Purdue University,  University of Illinois Urbana-Champaign，University of Virginia，PediaMed.AI.
  - Task: Planning
  - Project: [video](https://www.youtube.com/playlist?list=PLgcRcf9w8BmLJi_fqTGq-7KCZsbpEIE4a)
  - Env: [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv)
  - Publish Date: 2023.10.12
  - Summary:
    - Utilize LLMs’ linguistic and contextual understanding abilities with specialized tools to integrate the language and reasoning capabilities of LLMs into autonomous vehicles.

- [DrivingDiffusion: Layout-Guided multi-view driving scene video generation with latent diffusion model](https://arxiv.org/abs/2310.07771)
  - Xiaofan Li, Yifu Zhang, Xiaoqing Ye
  - Publisher: Baidu Inc.
  - Task: Generation
  - Project: [official](https://drivingdiffusion.github.io/)
  - Datasets: [nuScenes](https://www.nuscenes.org/nuscenes)
  - Summary:
    - Address the new problem of multi-view video data generation from 3D layout in complex urban scenes.'
    - Propose a generative model DrivingDiffusion to ensure the cross-view, cross-frame consistency and the instance quality of the generated videos.
    - Achieve state-of-the-art video synthesis performance on nuScenes dataset.
  - Metrics:
    - Quality of Generation: Frechet Inception Distance(FID), Frechet Video Distance(FVD)
    - Segmentation Metrics: mIoU

- <a id="LanguageMPC"></a>[LanguageMPC: Large Language Models as Decision Makers for Autonomous Driving](https://arxiv.org/pdf/2310.03026)
  - Hao Sha, Yao Mu, Yuxuan Jiang, Li Chen, Chenfeng Xu, Ping Luo, Shengbo Eben Li, Masayoshi Tomizuka, Wei Zhan, Mingyu Ding
  - Publisher: Tsinghua University, The University of Hong Kong, University of California, Berkeley
  - Task: Planning/Control
  - Code: [official](https://sites.google.com/view/llm-mpc)
  - Env: 
    - [ComplexUrbanScenarios](https://github.com/liuyuqi123/ComplexUrbanScenarios)
    - [Carla](https://github.com/carla-simulator)
  - Publish Date: 2023.10.04
  - Summary:
    - Leverage LLMs to provide high-level decisions through chain-of-thought.
    - Convert high-level decisions into mathematical representations to guide the bottom-level controller(MPC).
    - Metrics: Number of failure/collision cases， Inefficiency，time, Penalty

- <a id="DrivingwithLLMs"></a>[Driving with LLMs: Fusing Object-Level Vector Modality for Explainable Autonomous Driving](https://browse.arxiv.org/abs/2310.01957)
  - Long Chen, Oleg Sinavski, Jan Hünermann, Alice Karnsund, Andrew James Willmott, Danny Birch, Daniel Maund, Jamie Shotton
  - Publisher: Wayve
  - Task: Planning + VQA
  - Code: [official](https://github.com/wayveai/Driving-with-LLMs)
  - Simulator: a custom-built realistic 2D simulator.(The simulator is not open source.)
  - Datasets: [Driving QA](https://github.com/wayveai/Driving-with-LLMs/tree/main/data), data collection using RL experts in simulator.
  - Publish Date: 2023.10.03
  - Summary:
    - Propose a unique object-level multimodal LLM architecture(Llama2+Lora), using only vectorized representations as input.
    - Develop a new dataset of 160k QA pairs derived from 10k driving scenarios(control commands collected by RL(PPO), QA pair generated by GPT-3.5)
    - Metrics: 
      - Accuracy of traffic light detection
      - MAE for traffic light distance prediction
      - MAE for acceleration
      - MAE for brake pressure
      - MAE for steering wheel angle

- [Talk2BEV: Language-enhanced Bird’s-eye View Maps for Autonomous Driving](https://arxiv.org/abs/2310.02251)
  - Vikrant Dewangan, Tushar Choudhary, Shivam Chandhok, Shubham Priyadarshan, Anushka Jain, Arun K. Singh, Siddharth Srivastava, Krishna Murthy Jatavallabhula, K. Madhava Krishna
  - Publisher: IIIT Hyderabad, University of British Columbia, University of Tartu, TensorTour Inc, MIT
  - Project Page: [official](https://llmbev.github.io/talk2bev/)
  - Code: [Talk2BEV](https://github.com/llmbev/talk2bev)
  - Publish Date: 2023.10.03
  - Summary:
    - Introduces Talk2BEV, a large visionlanguage model (LVLM) interface for bird’s-eye view (BEV) maps in autonomous driving contexts.
    - Does not require any training or finetuning, relying instead on pre-trained image-language models
    - Develop and release Talk2BEV-Bench, a benchmark encom- passing 1000 human-annotated BEV scenarios, with more than 20,000 questions and ground-truth responses from the NuScenes dataset.

- <a id="DriveGPT4"></a>[DriveGPT4: Interpretable End-to-end Autonomous Driving via Large Language Model](https://arxiv.org/abs/2310.01412)
  - Zhenhua Xu, Yujia Zhang, Enze Xie, Zhen Zhao, Yong Guo, Kenneth K. Y. Wong, Zhenguo Li, Hengshuang Zhao
  - Publisher: The University of Hong Kong, Zhejiang University, Huawei Noah’s Ark Lab, University of Sydney
  - Project Page: [official](https://tonyxuqaq.github.io/projects/DriveGPT4/)
  - Task: Planning/Control + VQA
  - Datasets: 
    - [BDD-X dataset](https://github.com/JinkyuKimUCB/BDD-X-dataset).
  - Publish Date: 2023.10.02
  - Summary:
    - Develop a new visual instruction tuning dataset(based on BDD-X) for interpretable AD assisted by ChatGPT/GPT4.
    - Present a novel multimodal LLM called DriveGPT4(Valley + LLaVA).
  - Metrics: 
    - BLEU4, CIDEr and METETOR, ChatGPT Score.
    - RMSE for control signal prediction.

- <a id="GPT-Driver"></a>[GPT-DRIVER: LEARNING TO DRIVE WITH GPT](https://browse.arxiv.org/abs/2310.01415v1)
  - Jiageng Mao, Yuxi Qian, Hang Zhao, Yue Wang
  - Publisher: University of Southern California, Tsinghua University
  - Task: Planning(Fine-tuning Pre-trained Model)
  - Project: [official](https://pointscoder.github.io/projects/gpt_driver/index.html)
  - Datasets: [nuScenes](https://www.nuscenes.org/nuscenes)
  - Code: [GPT-Driver](https://github.com/PointsCoder/GPT-Driver)
  - Publish Date: 2023.10.02
  - Summary:
    - Motion planning as a language modeling problem.
    - Align the output of the LLM with human driving behavior through fine-tuning strategies using the OpenAI fine-tuning API.
    - Leverage the LLM to generate driving trajectories.
  - Metrics:
    - L2 metric and Collision rate

- [GAIA-1: A Generative World Model for Autonomous Driving](https://arxiv.org/abs/2309.17080)
  - Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, Gianluca Corrado
  - Publisher: Wayve
  - Task: Generation
  - Datasets: 
    - Training dataset consists of 4,700 hours at 25Hz of proprietary driving data collected in London,
UK between 2019 and 2023. It corresponds to approximately 420M unique images. 
    - Validation dataset contains 400 hours of driving data from runs not included in the training set.
    - text coming from either online narration or offline metadata sources
  - Publish Date: 2023.09.29
  - Summary:
    - Introduce GAIA-1, a generative world model that leverages video(pre-trained DINO), text(T5-large), and action inputs to generate realistic driving scenarios.
    - Serve as a valuable neural simulator, allowing the generation of unlimited data.

- [DiLu: A Knowledge-Driven Approach to Autonomous Driving with Large Language Models](https://arxiv.org/abs/2309.16292)
  - Licheng Wen, Daocheng Fu, Xin Li, Xinyu Cai, Tao Ma, Pinlong Cai, Min Dou, Botian Shi, Liang He, Yu Qiao   **ICLR 2024**
  - Publisher: Shanghai AI Laboratory, East China Normal University, The Chinese University of Hong Kong
  - Publish Date: 2023.09.28
  - Task: Planning
  - Env: 
    - [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv)
    - [CitySim](https://github.com/ozheng1993/UCF-SST-CitySim-Dataset), a Drone-Based vehicle trajectory dataset.
  - Summary: 
    - Propose the DiLu framework, which combines a Reasoning and a Reflection module to enable the system to perform decision-making based on common-sense knowledge and evolve continuously.

- [SurrealDriver: Designing Generative Driver Agent Simulation Framework in Urban Contexts based on Large Language Model](https://arxiv.org/abs/2309.13193)
  - Ye Jin, Xiaoxi Shen, Huiling Peng, Xiaoan Liu, Jingli Qin, Jiayang Li, Jintao Xie, Peizhong Gao, Guyue Zhou, Jiangtao Gong
  - Keywords: human-AI interaction, driver model, agent, generative AI, large language model, simulation framework
  - Env: [CARLA](https://github.com/carla-simulator)
  - Publisher: Tsinghua University
  - Summary: Propose a generative driver agent simulation framework based on large language models (LLMs), capable of perceiving complex traffic scenarios and providing realistic driving maneuvers.

- [Drive as You Speak: Enabling Human-Like Interaction with Large Language Models in Autonomous Vehicles](https://arxiv.org/abs/2309.10228)
  - Can Cui, Yunsheng Ma, Xu Cao, Wenqian Ye, Ziran Wang
  - Publisher: Purdue University, PediaMed.AI Lab, University of Virginia
  - Task: Planning
  - Publish Date: 2023.09.18
  - Summary:
    - Provide a comprehensive framework for integrating Large Language Models (LLMs) into AD.

- <a id="DriveDreamer"></a>[DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving](https://arxiv.org/abs/2309.09777)
  - Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jiwen Lu  **ECCV 2024**
  - Publisher: GigaAI, Tsinghua University
  - Task: Generation
  - Project Page: [official](https://drivedreamer.github.io/)
  - Datasets: [nuScenes](https://www.nuscenes.org/nuscenes)
  - Publish Date: 2023.09.18
  - Summary:
    - Harness the powerful diffusion model to construct a comprehensive representation of the complex environment.
    - Generate future driving videos and driving policies by a multimodal(text, image, HDMap, Action, 3DBox) world model.

- [Can you text what is happening? Integrating pre-trained language encoders into trajectory prediction models for autonomous driving](https://arxiv.org/abs/2309.05282)
  - Ali Keysan, Andreas Look, Eitan Kosman, Gonca Gürsun, Jörg Wagner, Yu Yao, Barbara Rakitsch
  - Publisher: Bosch Center for Artificial Intelligence, University of Tubingen, 
  - Task: Prediction
  - Datasets: [nuScenes](https://www.nuscenes.org/nuscenes)
  - Publish Date: 2023.09.13
  - Summary:
    - Integrating pre-trained language models as textbased input encoders for the AD trajectory prediction task.
  - Metrics:
    - minimum Average Displacement Error (minADEk)
    - Final Displacement Error (minFDEk)
    - MissRate over 2 meters

- [TrafficGPT: Viewing, Processing and Interacting with Traffic Foundation Models](https://arxiv.org/abs/2309.06719)
  - Siyao Zhang, Daocheng Fu, Zhao Zhang, Bin Yu, Pinlong Cai
  - Publisher: Beihang University, Key Laboratory of Intelligent Transportation Technology and System,  Shanghai Artificial Intelligence Laboratory
  - Task: Planning
  - Code: [official](https://github.com/lijlansg/TrafficGPT.git)
  - Publish Date: 2023.09.13
  - Summary:
    - Present TrafficGPT—a fusion of ChatGPT and traffic foundation models. 
    - Bridges the critical gap between large language models and traffic foundation models by defining a series of prompts.
  
- [HiLM-D: Towards High-Resolution Understanding in Multimodal Large Language Models for Autonomous Driving](https://arxiv.org/abs/2309.05186)
  - Xinpeng Ding, Jianhua Han, Hang Xu, Wei Zhang, Xiaomeng Li
  - Publisher: The Hong Kong University of Science and Technology, Huawei Noah’s Ark Lab
  - Task: Detection + VQA
  - Datasets: [DRAMA](https://usa.honda-ri.com/drama)
  - Publish Date: 2023.09.11
  - Summary:
    - Propose HiLM-D (Towards High-Resolution Understanding in MLLMs for Autonomous Driving), an efficient method to incorporate HR information into MLLMs for the ROLISP task.
    - ROLISP that aims to identify, explain and localize the risk object for the ego-vehicle meanwhile predicting its intention and giving suggestions.
  - Metrics:
    - LLM metrics, BLEU4, CIDEr and METETOR, SPICE.
    - Detection metrics, mIoU, IoUs so on.

- <a id="LanguagePrompt"></a>[Language Prompt for Autonomous Driving](https://arxiv.org/abs/2309.04379)
  - Dongming Wu, Wencheng Han, Tiancai Wang, Yingfei Liu, Xiangyu Zhang, Jianbing Shen
  - Publisher: Beijing Institute of Technology, University of Macau, MEGVII Technology, Beijing Academy of Artificial Intelligence
  - Task: Tracking
  - Code: [official](https://github.com/wudongming97/Prompt4Driving)
  - Datasets: NuPrompt(not open), based on [nuScenes](https://www.nuscenes.org/nuscenes). 
  - Publish Date: 2023.09.08
  - Summary:
    - Propose a new large-scale language prompt set(based on nuScenes) for driving scenes, named NuPrompt(3D object-text pairs).
    - Propose an efficient prompt-based tracking model with prompt reasoning modification on PFTrack, called PromptTrack. 

- [MTD-GPT: A Multi-Task Decision-Making GPT Model for Autonomous Driving at Unsignalized Intersections](https://arxiv.org/abs/2307.16118)
  - Jiaqi Liu, Peng Hang, Xiao Qi, Jianqiang Wang, Jian Sun. *ITSC 2023*
  - Publisher: Tongji University, Tsinghua University
  - Task: Prediction
  - Env: [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv)
  - Publish Date: 2023.07.30
  - Summary:
    - Design a pipeline that leverages RL algorithms to train single-task decision-making experts and utilize expert data.
    - Propose the MTD-GPT model for multi-task(left-turn, straight-through, right-turn) decision-making of AV at unsignalized intersections.

- [Domain Knowledge Distillation from Large Language Model: An Empirical Study in the Autonomous Driving Domain](https://arxiv.org/abs/2307.11769)
  - Yun Tang, Antonio A. Bruto da Costa, Xizhe Zhang, Irvine Patrick, Siddartha Khastgir, Paul Jennings. *ITSC 2023*
  - Publisher: University of Warwick
  - Task: QA
  - Publish Date: 2023.07.17
  - Summary:
    - Develop a web-based distillation assistant enabling supervision and flexible intervention at runtime by prompt engineering and the LLM ChatGPT.

- [Drive Like a Human: Rethinking Autonomous Driving with Large Language Models](https://browse.arxiv.org/abs/2307.07162)
  - Daocheng Fu, Xin Li, Licheng Wen, Min Dou, Pinlong Cai, Botian Shi, Yu Qiao
  - Publisher: Shanghai AI Lab, East China Normal University
  - Task: Planning
  - Code: [official](https://github.com/PJLab-ADG/DriveLikeAHuman)
  - Env: [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv)
  - Publish Date: 2023.07.14
  - Summary:
    - Identify three key abilities: Reasoning, Interpretation and Memorization(accumulate experience and self-reflection).
    - Utilize LLM in AD as decision-making to solve long-tail corner cases and increase interpretability.
    - Verify interpretability in closed-loop offline data.

- [Language-Guided Traffic Simulation via Scene-Level Diffusion](https://arxiv.org/abs/2306.06344)
  - Ziyuan Zhong, Davis Rempe, Yuxiao Chen, Boris Ivanovic, Yulong Cao, Danfei Xu, Marco Pavone, Baishakhi Ray
  - Publisher: Columbia University, NVIDIA Research, Stanford University, Georgia Tech
  - Task: Diffusion
  - Publish Date: 2023.07.10
  - Summary: 
    - Present CTG++, a language-guided scene-level conditional diffusion model for realistic query-compliant traffic simulation. 
    - Leverage an LLM for translating a user query into a differentiable loss function and propose a scene-level conditional diffusion model (with a spatial-temporal transformer architecture) to translate the loss function into realistic, query compliant trajectories.

- [ADAPT: Action-aware Driving Caption Transformer](https://arxiv.org/abs/2302.00673)
  - Bu Jin, Xinyu Liu, Yupeng Zheng, Pengfei Li, Hao Zhao, Tong Zhang, Yuhang Zheng, Guyue Zhou, Jingjing Liu **ICRA 2023**
  - Publisher: Chinese Academy of Sciences, Tsinghua University, Peking University, Xidian University, Southern University of Science and Technology, Beihang University
  - Code: [ADAPT](https://github.com/jxbbb/ADAPT)
  - Datasets: [BDD-X dataset](https://github.com/JinkyuKimUCB/BDD-X-dataset)
  - Summary:
    - Propose ADAPT, a new end-to-end transformerbased action narration and reasoning framework for
self-driving vehicles.
    - propose a multi-task joint training framework that aligns both the driving action captioning task and the control signal prediction task.
</details>

## WorkShop
<details open>
<summary>Toggle</summary>

- [Large Language and Vision Models for Autonomous Driving(LLVM-AD) Workshop @ WACV 2024](https://llvm-ad.github.io/)
  - Publisher: Tencent Maps HD Map T.Lab, University of Illinois Urbana- Champaign, Purdue University, University of Virginia
  - Challenge 1: MAPLM: A Large-Scale Vision-Language Dataset for Map and Traffic Scene Understanding
    - Datasets: [Download](https://drive.google.com/drive/folders/1cqFjBH8MLeP6nKFM0l7oV-Srfke-Mx1R?usp=sharing)
    - Task: QA
    - Code: https://github.com/LLVM-AD/MAPLM
    - Description: MAPLM combines point cloud BEV (Bird's Eye View) and panoramic images to provide a rich collection of road scenario images. It includes multi-level scene description data, which helps models navigate through complex and diverse traffic environments.
    - Metric:
      - Frame-overall-accuracy (FRM): A frame is considered correct if all closed-choice questions about it are answered correctly.
      - Question-overall-accuracy (QNS): A question is considered correct if its answer is correct.
      - LAN: How many lanes in current road?
      - INT: Is there any road cross, intersection or lane change zone in the main road?
      - QLT: What is the point cloud data quality in current road area of this image?
      - SCN: What kind of road scene is it in the images? (SCN)    
  - Challenge 2: In-Cabin User Command Understanding (UCU)
    - Datasets: [Download](https://github.com/LLVM-AD/ucu-dataset/blob/main/ucu.csv)
    - Task: QA
    - Code: https://github.com/LLVM-AD/ucu-dataset
    - Description: 
      - This dataset focuses on understanding user commands in the context of autonomous vehicles. It contains 1,099 labeled commands. Each command is a sentence that describes a user’s request to the vehicle. 
    - Metric:
      - Command-level accuracy: A command is considered correctly understood if all eight answers are correct.
      - Question-level accuracy: Evaluation at the individual question level.
</details>

## Datasets
<details open>
<summary>Toggle</summary>

```
format:
- [title](dataset link) [links]
  - author1, author2, and author3...
  - keyword
  - experiment environments or tasks
```
- [Rank2Tell: A Multimodal Driving Dataset for Joint Importance Ranking and Reasoning](https://arxiv.org/abs/2309.06597)
  - Enna Sachdeva, Nakul Agarwal, Suhas Chundi, Sean Roelofs, Jiachen Li, Behzad Dariush, Chiho Choi, Mykel Kochenderfer
  - Publisher: Honda Research Institute, Stanford University
  - Publish Date: 2023.09.10
  - Summary:
    - A multi-modal ego-centric dataset for Ranking the importance level and Telling the reason for the importance. 
    - Introduce a joint model for joint importance level ranking and natural language captions generation to benchmark our dataset.

- [DriveLM: Drive on Language](https://github.com/OpenDriveLab/DriveLM)
  - Publisher: Sima, Chonghao and Renz, Katrin and Chitta, Kashyap and Chen, Li and Zhang, Hanxue and Xie, Chengen and Luo, Ping and Geiger, Andreas and Li, Hongyang **ECCV 2024**
  - Dataset: [DriveLM](https://github.com/OpenDriveLab/DriveLM/blob/main/docs/getting_started.md#download-data)
  - Publish Date: 2023.08
  - Summary:
    - Construct dataset based on the nuScenes dataset.
    - Perception questions require the model to recognize objects in the scene. 
    - Prediction questions ask the model to predict the future status of important objects in the scene. 
    - Planning questions prompt the model to give reasonable planning actions and avoid dangerous ones.

- [WEDGE: A multi-weather autonomous driving dataset built from generative vision-language models](https://browse.arxiv.org/abs/2305.07528)
  - Aboli Marathe, Deva Ramanan, Rahee Walambe, Ketan Kotecha. **CVPR 2023**
  - Publisher: Carnegie Mellon University, Symbiosis International University
  - Dataset: [WEDGE](https://github.com/Infernolia/WEDGE)
  - Publish Date: 2023.05.12
  - Summary:
    - A multi-weather autonomous driving dataset built from generative vision-language models.

- [NuScenes-QA: A Multi-modal Visual Question Answering Benchmark for Autonomous Driving Scenario](https://arxiv.org/abs/2305.14836)
  - Tianwen Qian, Jingjing Chen, Linhai Zhuo, Yang Jiao, Yu-Gang Jiang
  - Publisher: Fudan University
  - Dataset: [NuScenes-QA](https://github.com/qiantianwen/NuScenes-QA)
  - Summary:
    - NuScenes-QA provides 459,941 question-answer pairs based on the 34,149 visual scenes, with 376,604 questions from 28,130 scenes used for training, and 83,337 questions from 6,019 scenes used for testing, respectively.
    - The multi-view images and point clouds are first processed by the feature extraction backbone
to obtain BEV features.

- [DRAMA: Joint Risk Localization and Captioning in Driving](https://arxiv.org/abs/2209.10767)
  - Srikanth Malla, Chiho Choi, Isht Dwivedi, Joon Hee Choi, Jiachen Li
  - Publisher: 
  - Datasets: [DRAMA](https://usa.honda-ri.com/drama#Introduction)
  - Summary:
    - Introduce a novel dataset DRAMA that provides linguistic descriptions (with the focus on reasons) of driving risks associated with important objects and that can be used to evaluate a range of visual captioning capabilities in driving scenarios.

- [Language Prompt for Autonomous Driving](https://arxiv.org/abs/2309.04379)
  - Datasets: Nuprompt(Not open)
  - [Previous summary](#LanguagePrompt)

- [Driving with LLMs: Fusing Object-Level Vector Modality for Explainable Autonomous Driving](https://browse.arxiv.org/abs/2310.01957)
  - Datasets: [official](https://github.com/wayveai/Driving-with-LLMs/tree/main/data), data collection using RL experts in simulator.
  - [Previous summary](#DrivingwithLLMs)

- [Textual Explanations for Self-Driving Vehicles](https://arxiv.org/abs/1807.11546)
  - Jinkyu Kim, Anna Rohrbach, Trevor Darrell, John Canny, Zeynep Akata **ECCV 2018**.
  - Publisher: University of California, Berkeley, Saarland Informatics Campus, University of Amsterdam
  - [BDD-X dataset](https://github.com/JinkyuKimUCB/BDD-X-dataset)

- [Grounding Human-To-Vehicle Advice for Self-Driving Vehicles](https://arxiv.org/abs/1911.06978)
  - Jinkyu Kim, Teruhisa Misu, Yi-Ting Chen, Ashish Tawari, John Canny **CVPR 2019**
  - Publisher: UC Berkeley, Honda Research Institute USA, Inc.
  - [HAD dataset](https://usa.honda-ri.com/had)
</details>


## License

Awesome LLM for Autonomous Driving Resources is released under the Apache 2.0 license.
