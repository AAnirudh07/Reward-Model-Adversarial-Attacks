# Reward-Model-Adversarial-Attacks

### Motivation
The motivation behind this preliminary analysis is to evaluate the robustness of reward models that score text-to-image model outputs based on human preference. It examines if small adversarial perturbations can cause a high-scoring image to drop in score and whether such perturbations transfer between different reward models. It also explores which diffusion models are more prone to these attacks and investigates the underlying factors contributing to their vulnerability.

### Method
This analysis required decisions regarding reward models, target models, datasets, and attack methods. These are described below.

#### Reward Models
Two reward models were used for this task:
- HPSv1[​1]: This fine-tuned CLIP[2] model scores images and associated text prompts on a 0–100 scale based on alignment with  human preferences.
- HPSv2[3​]: An enhanced iteration of HPSv1, also based on a fine-tuned CLIP architecture, trained on a more diverse set of images and text prompts.

Due to memory constraints on a T4 GPU in Google Colab, I used a compressed version of the HPSv2 model. This limitation might affect the model's performance slightly compared to its full-scale counterpart.

#### Target Models
This analysis was conducted on the Stable Diffusion family of models:
- **Stable Diffusion Models**: Used [stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), [stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base), and [stable-diffusion-3-medium-diffusers](stabilityai/stable-diffusion-3-medium-diffusers) using default inference settings. 
    - Due to memory constraints, half-precision (`torch.float16`)variants were used, and for stable-diffusion-3, CPU offloading and optimizations such as dropping the T5 text encoder—were necessary to support `1024x1024` outputs.

Additionally, the following models were tested but not used:
- [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell): The smallest Flux model was tested; however, it could not be loaded due to memory limitations.
- **VQ-Diffusion Model**: The [VQ-Diffusion](https://huggingface.co/microsoft/vq-diffusion-ithq/tree/main) model could not be loaded due to [missing files](https://huggingface.co/microsoft/vq-diffusion-ithq/tree/main) in its public repository.

#### Datasets
Two datasets were used for analysis, chosen to ensure that the prompts originate from varied sources (e.g., DiffusionDB[4] features stylistic prompts exclusively from Stable Diffusion) and cover multiple categories for a fine-grained evaluation.
- **HPS Benchmark**: Introduced in the HPSv2 paper[3], evaluates models on their ability to generate images across four distinct styles: Animation, Concept-art, Painting, and Photo. 
- **Drawbench**: Similar to the HPS benchmark, the Drawbench dataset[5] comprises 11 categories.

_NOTE:_ Due to the large size of these datasets, a subset was selected to ensure a balanced evaluation. For the HPS Benchmark, the first 10 samples from each category (10 samples × 4 categories = 40) were initially taken and ranked by the reward model, with the top 5 per category (5 × 4 = 20) subsequently chosen. Similarly, for Drawbench, the first 4 samples from each category (4 samples × 11 categories = 44) were ranked, and the top 2 per category (2 × 11 = 22) were selected.

This approach was implemented to prevent cherry-picking the best prompts and to validate how well the attacks perform on average. In hindsight, this may have limited the ability to achieve the most optimal results.


#### Attacks
The Gaussian Noise, Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD), and SPSA attacks were modified to work with reward models. 

Since the torchattacks library[6] is designed for classification tasks using losses like `CrossEntropyLoss`, adjustments were necessary for reward models that output a single continuous value from 0 to 100. The base `Attack` class was altered to support custom models classes. Although subclassing the torchattacks datasets was not strictly required, it was adopted to facilitate similar modifications for other attacks from a base implementation. (`WB` - White Box; `BB` - Black Box)
- **Gaussian Noise (BB)**: Required no major changes as it simply adds noise to the input.

- **FGSM (WB)**: This white-box attack adds perturbations in the direction of the gradient i.e. increasing loss. The loss was redefined as `–reward` so that the perturbations effectively decrease the reward. The raw reward was used directly as it provided a clearer gradient signal. Additionally, the implementation was modified to support batch processing, allowing for the averaging of reward scores across batches.
- **PGD (WB)**: The PGD attack was adapted in a manner similar to FGSM. Moreover, to address memory constraints, the dataset was not fully loaded into memory but processed in batches instead
- **SPSA (BB)**: The SPSA attack code was modified to support batch processing used the same reward-based modifications as FGSM and PGD.


### Analysis
The experiment results are analyzed in the following section.

### Main Results
This section 

### Repository Details

### References
1. Wu, X., Sun, K., Zhu, F., Zhao, R., & Li, H. (2023). Human Preference Score: Better Aligning Text-to-Image Models with Human Preference. ArXiv. https://arxiv.org/abs/2303.14420
2. Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning Transferable Visual Models From Natural Language Supervision. ArXiv. https://arxiv.org/abs/2103.00020
3. Wu, X., Hao, Y., Sun, K., Chen, Y., Zhu, F., Zhao, R., & Li, H. (2023). Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis. ArXiv. https://arxiv.org/abs/2306.09341
4. Wang, Z. J., Montoya, E., Munechika, D., Yang, H., Hoover, B., & Chau, D. H. (2022). DiffusionDB: A Large-scale Prompt Gallery Dataset for Text-to-Image Generative Models. ArXiv. https://arxiv.org/abs/2210.14896
5. Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E., Ghasemipour, S. K., Ayan, B. K., Mahdavi, S. S., Lopes, R. G., Salimans, T., Ho, J., Fleet, D. J., & Norouzi, M. (2022). Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding. ArXiv. https://arxiv.org/abs/2205.11487
6. https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html

