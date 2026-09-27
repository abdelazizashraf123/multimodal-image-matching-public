# Paper list

## MINIMA: Modality Invariant Image Matching

* Link: https://arxiv.org/abs/2412.19412
* Summary: Proposes a unified framework for cross-view and cross-modality image matching by scaling training data rather than adding complex modules. They generate large synthetic multi-modal training pairs (MD-syn) from RGB data using generative models and show strong in-domain and zero-shot performance across many modality pairs.

## MatchAnything: Universal Cross-Modality Image Matching with Large-Scale Pre-Training

* Link: https://arxiv.org/abs/2501.07556
* Summary: Introduces a large-scale pretraining pipeline that creates synthetic cross-modality supervision to teach a matching network to focus on modality-invariant structures. The resulting single set of weights transfers well to many unseen cross-modality registration tasks.

## SuperPoint: Self-Supervised Interest Point Detection and Description

* Link: https://arxiv.org/abs/1712.07629
* Summary: Presents a self-supervised method that jointly predicts keypoints and descriptors with a fully-convolutional network operating on whole images. It uses “Homographic Adaptation” to boost repeatability and enable synthetic-to-real adaptation, yielding strong homography estimation performance.

## MultiPoint: Cross-spectral registration of thermal and optical aerial imagery

* Link: https://proceedings.mlr.press/v155/achermann21a.html
* Summary: Learns cross-spectral interest points and descriptors for thermal–optical aerial image registration without needing known relative viewpoints. It uses offline mutual-information alignment and a multispectral homographic adaptation procedure to generate repeatable training targets, then runs fast at test time.

## XPoint: A Self-Supervised Visual-State-Space based Architecture for Multispectral Image Registration

* Link: https://arxiv.org/abs/2411.07430
* Summary: Proposes a self-supervised, modular multispectral matching framework that can be quickly adapted/fine-tuned to new modality pairs. It combines a VMamba-based encoder with heads for keypoints/descriptors and homography regression, and uses self-supervision to produce pseudo-ground-truth keypoints.

## RDD: Robust Feature Detector and Descriptor using Deformable Transformer

* link: https://arxiv.org/abs/2505.08013
* Summary: Introduces a deformable-transformer-based keypoint detector/descriptor designed to better capture long-range context and geometric invariance via deformable attention. It also adds an Air-to-Ground training dataset and proposes challenging benchmarks, reporting strong sparse matching results and semi-dense capability.

## VD-Matcher: A Very Deep Local Feature Matcher with Weight Recycling and Keypoint Detection

* link : https://www.semanticscholar.org/paper/VD-Matcher%3A-A-Very-Deep-Local-Feature-Matcher-With-Dai-Zhou/aeb7335d55c9cd41fbdc2d72e029d612e4baf508
* Summary: Builds a very deep transformer matcher while keeping parameters manageable by reusing (“recycling”) parts of the weights across blocks with additional transformations to preserve expressiveness. It also adds a lightweight multi-scale keypoint module to reduce compute and improve global aggregation, reporting strong benchmark performance with fewer parameters.
