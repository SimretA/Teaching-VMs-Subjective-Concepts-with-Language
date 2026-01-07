# Teaching Vision Models Subjective Concepts with Language

Exploring whether natural-language feedback can adapt a frozen vision-language model (CLIP) to user-defined visual concepts at inference time—without fine-tuning.

## Problem

CLIP works well for generic queries ("a dog on a beach") but struggles with subjective, user-specific concepts ("a person looking guilty"). Traditional fine-tuning is expensive and requires labeled data.

**Our approach**: Instead of updating model weights, update the query embedding using natural language feedback.

```
q' = q + α·embed("positive attributes") - β·embed("negative attributes")
```


## Tasks

### Task 1: Baseline (Complete)
- Set up CLIP (ViT-B/32) with HuggingFace Transformers [1]
- Curated subsets from Flickr8k [7] (300 images) and CelebA [8] (500 faces)
- Computed and saved CLIP embeddings
- Baseline retrieval with Precision@k evaluation

**Key finding**: CLIP achieves ~84% on objective queries (glasses, smiling) but only ~30% on subjective queries (looking guilty).

### Task 2: Autoencoders (In Progress)
- Extract interpretable directions in CLIP embedding space using autoencoders [2], [3]
- Validate using CelebA attribute labels

### Task 3: Language Feedback Loop (Upcoming)
- Parse natural language critiques into positive/negative attributes
- Update query embeddings with semantic directions
- Measure improvement over baseline

### Task 4: Experiments & Results (Upcoming)
- Quantify improvement from language feedback
- Compare objective vs subjective queries

## Setup

```bash
# Install dependencies
pip install transformers torch pillow matplotlib numpy pandas tqdm

# Download datasets
# Flickr8k: https://www.kaggle.com/datasets/adityajn105/flickr8k/data
# CelebA: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
```

## Baseline Results

| Dataset | Query Type | Precision@10 |
|---------|------------|--------------|
| CelebA | Objective (glasses, smiling) | ~90-100% |
| CelebA | Subjective (looking guilty) | ~30% |
| Flickr8k | Mixed | ~53% |

## References

[1] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, "Learning Transferable Visual Models From Natural Language Supervision," in *Proc. Int. Conf. Machine Learning (ICML)*, 2021. [Online]. Available: https://arxiv.org/abs/2103.00020

[2] D. Bank, N. Koenigstein, and R. Giryes, "Autoencoders," *arXiv preprint arXiv:2003.05991*, 2020. [Online]. Available: https://arxiv.org/abs/2003.05991

[3] U. Michelucci, "An Introduction to Autoencoders," *arXiv preprint arXiv:2201.03898*, Jan. 2022. [Online]. Available: https://arxiv.org/abs/2201.03898

[4] TensorFlow, "Intro to Autoencoders," TensorFlow Tutorials, 2024. [Online]. Available: https://www.tensorflow.org/tutorials/generative/autoencoder

[5] Neuronpedia, "Gemma Scope: Steering Neural Networks," 2024. [Online]. Available: https://www.neuronpedia.org/gemma-scope#steer

[6] D. Duhaime, "Visualizing Latent Spaces," 2019. [Online]. Available: https://douglasduhaime.com/posts/visualizing-latent-spaces.html

[7] Distill, "Activation Atlas," *Distill*, 2019. [Online]. Available: https://distill.pub/2019/activation-atlas/

[8] A. JN, "Flickr8k Dataset," Kaggle, 2020. [Online]. Available: https://www.kaggle.com/datasets/adityajn105/flickr8k/data

[9] Z. Liu, P. Luo, X. Wang, and X. Tang, "Deep Learning Face Attributes in the Wild," in *Proc. Int. Conf. Computer Vision (ICCV)*, 2015. [Online]. Available: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

[10] HuggingFace, "CLIP Documentation," HuggingFace Transformers, 2024. [Online]. Available: https://huggingface.co/docs/transformers/en/model_doc/clip
