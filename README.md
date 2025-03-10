# 1. Motivation
The motivation behind this preliminary analysis is to evaluate the robustness of reward models that score text-to-image model outputs based on human preference. It examines if small adversarial perturbations can cause a high-scoring image to drop in score and whether such perturbations transfer between different reward models. It also explores which diffusion models are more prone to these attacks and investigates the underlying factors contributing to their vulnerability.

# 2. Method
This analysis required decisions regarding reward models, target models, datasets, and attack methods. These are described below.

#### 2.1 Reward Models
Two reward models were used for this task:
- HPSv1[​1]: This fine-tuned CLIP[2] model scores images and associated text prompts on a 0–100 scale based on alignment with  human preferences.
- HPSv2[3​]: An enhanced iteration of HPSv1, also based on a fine-tuned CLIP architecture, trained on a more diverse set of images and text prompts.

Due to memory constraints on a T4 GPU in Google Colab, a compressed version of the HPSv2 model was used. This limitation might affect the model's performance slightly compared to its full-scale counterpart.

#### 2.2 Target Models
This analysis was conducted on the Stable Diffusion family of models:
- **Stable Diffusion Models**: Used [stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), [stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base), and [stable-diffusion-3-medium-diffusers](stabilityai/stable-diffusion-3-medium-diffusers) using default inference settings. 
    - Due to memory constraints, half-precision (`torch.float16`)variants were used, and for stable-diffusion-3, CPU offloading and optimizations such as dropping the T5 text encoder—were necessary to support `1024x1024` outputs.

Additionally, the following models were tested but not used:
- [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell): The smallest Flux model was tested; however, it could not be loaded due to memory limitations.
- **VQ-Diffusion Model**: The [VQ-Diffusion](https://huggingface.co/microsoft/vq-diffusion-ithq/tree/main) model could not be loaded due to [missing files](https://huggingface.co/microsoft/vq-diffusion-ithq/tree/main) in its public repository.

#### 2.3 Datasets
Two datasets were used for analysis, chosen to ensure that the prompts originate from varied sources (e.g., DiffusionDB[4] features stylistic prompts exclusively from Stable Diffusion) and cover multiple categories for a fine-grained evaluation.
- **HPS Benchmark**: Introduced in the HPSv2 paper[3], evaluates models on their ability to generate images across four distinct styles: Anime, Concept-art, Painting, and Photo. 
- **Drawbench**: Similar to the HPS benchmark, the Drawbench dataset[5] comprises 11 categories.

_NOTE:_ Due to the large size of these datasets, a subset was selected to ensure a balanced evaluation. For the HPS Benchmark, the first 10 samples from each category (10 samples × 4 categories = 40) were initially taken and ranked by the reward model, with the top 5 per category (5 × 4 = 20) subsequently chosen. Similarly, for Drawbench, the first 4 samples from each category (4 samples × 11 categories = 44) were ranked, and the top 2 per category (2 × 11 = 22) were selected.

This approach was implemented to prevent cherry-picking the best prompts and to validate how well the attacks perform on average. In hindsight, this may have limited the ability to achieve the most optimal results.

#### 2.4 Attacks
The Gaussian Noise, FGSM, PGD, and SPSA attacks were modified to work with reward models. 

Since the torchattacks library[6] is designed for classification tasks using losses like `CrossEntropyLoss`, adjustments were necessary for reward models that output a single continuous value from 0 to 100. The base `Attack` class was altered to support custom models classes. Although subclassing the torchattacks datasets was not strictly required, it was adopted to facilitate similar modifications for other attacks from a base implementation. (`WB` - White Box; `BB` - Black Box)
- **Gaussian Noise (BB)**: Required no major changes as it simply adds noise to the input.
- **FGSM (WB)**: This white-box attack adds perturbations in the direction of the gradient i.e. increasing loss. The loss was redefined as `–reward` so that the perturbations effectively decrease the reward. The raw reward was used directly as it provided a clear gradient signal. Additionally, the implementation was modified to support batch processing to allow averaging of reward scores across batches.
- **PGD (WB)**: The PGD attack was adapted in a manner similar to FGSM. Moreover, to address memory constraints, the dataset was not fully loaded into memory but processed in batches instead
- **SPSA (BB)**: The SPSA attack code was modified to support batch processing used the same reward-based modifications as FGSM and PGD. Marginal loss was removed and the raw reward was used directly. This could have led to the degradation of the attack's effectiveness, making it only as effective as FGSM.


# 3. Analysis
The experiment results are analyzed in this section.

_Reward Model Analysis_
- The reward scores for the HPSv1 model were lower than the HPSv2 model. This could be attributed to HPSv1 being trained on data from DiscordChatExplorer[7], which consists exclusively of prompts and images generated by Stable Diffusion models. HPSv1 may be more attuned to the characteristics of Stable Diffusion outputs. HPSv2 was trained on a more diverse dataset of images from nine different models as well as realistic images.
- Despite its lower average reward score, the HPSv1 model demonstrated greater robustness to attacks, as indicated by the absolute drop in reward scores. This could be attributed to the same reasons as above.
- The Color category in the DrawBench dataset appears to be robust to perturbations. This could mean that reward models might not be effective in capturing subtle differences.

_Target Model Analysis_
While all adversarial images were scored lower than their original counterpart, there were some common themes across models:
- Stylized categories such as Concept Art, Photo, and Paintings (HPS), Gary Marcus et al. (DrawBench), were the least robust to perturbations compared to the baseline. This suggests that the Stable Diffusion models’ approach to generating stylistic images is highly nuanced, where even minor perturbations can significantly degrade quality due to mismatching styles.
- Categories such as Counting, Conflicting (DrawBench) were more robust to perturbations. For categories like Counting, this could mean that the target model is good at generating images that align with the prompt when there are features that are difficult to perturb, such as quantities.

_Dataset Analysis_
- In many instances in DrawBench, no samples from the Misspellings or Rare Words categories were selected. A manual analysis revealed that image quality was not always poor. This highlights that the reward model lacks robustness to slight variations in the prompt. It would be interesting to explore how text perturbations impact reward scores!

_Attack Analysis_
- Simply adding Gaussian noise lowered the reward score demonstrating that it is not difficult to degrade the reward model's performance. However, FGSM and SPSA performed similarly to random noise, indicating that taking a single step in the gradient direction or estimating gradients does not offer any significant advantage over random perturbations.
- The PGD attack proved to be the most effective across both models and datasets, with movement in the direction of decreasing reward consistently fooling the reward model.

_Transfer Test Analysis_
Adversarial images generated to fool one reward model were only partially transferable to another. While reward scores were lower, the decreases were similar across different attacks and did not show the same pronounced drop. For example, PGD produced the highest drop across models and datasets but its scores in the transfer attack were comparable to those from Gaussian noise. This suggests that the reward models simply view these perturbations as random noise.

Despite this, some categorical similarities emerged; notably, the Concept Art category consistently exhibited a significant drop in score in both the target and transfer target model.


# 4. Discussion
Some areas where this preliminary analysis could be improved:

# 5. Main Results
This section presents the numerical results of the experiments. Please see the note in the 'Method > Datasets' section for more information on prompt sampling. Additionally,
- Due to ranking and choosing top samples, some categories in the DrawBench dataset may not have results.
- As the official HPS Benchmark has average scores of ~29, a very low threshold of 15 was used. The results consider any drop in reward scores to be significant.

### Attack Results
This sub-section presents the results of the attack experiments. Each entry is rounded to 2 decimal places. The lowest category in each row is highlighted in bold.

#### Stable Diffusion 1
_HPSv1 Reward Model_
<details>
<summary>HPS Benchmark Dataset</summary>

| Attack       | Anime | Concept-art | Painting | Photo  | Overall |
|--------------|-------|-------------|----------|--------|---------|
| Original     | 21.64 | 21.50       | 21.63    | 21.16  | 21.49   |
| GN           | 19.91 | **18.31**   | 19.26    | 19.70  | 19.37   |
| FGSM (B>1)   | 19.85 | **18.17**   | 19.19    | 19.62  | 19.29   |
| FGSM (B=1)   | 19.85 | **18.17**   | 19.19    | 19.62  | 19.29   |
| PGD          | 17.26 | **15.90**   | 17.28    | 17.75  | 17.10   |
| SPSA         | 19.85 | **18.17**   | 19.18    | 19.64  | 19.29   |
</details>

<details>
<summary>DrawBench Dataset</summary>

| Attack         | Colors | Conflicting | Counting   | DALL-E | Descriptions | Gary Marcus et al. |
|----------------|--------|-------------|------------|--------|--------------|--------------------|
| Original       | 20.24  | 20.78       | 19.33 | 21.02  | 19.89        | 20.29              |
| GN             | 19.29  | 19.42       | 19.09     | 19.68  | 19.28        | 18.66              |
| FGSM (B>1)     | 19.22  | 19.38       | 19.00     | 19.68  | 19.25        | 18.77              |
| FGSM (B=1)     | 19.22  | 19.38       | 19.00     | 19.68  | 19.25        | 18.77              |
| PGD            | 18.30  | 17.16       | 16.81     | 18.38  | 17.41        | 16.18              |
| SPSA           | 19.23  | 19.38       | 19.00     | 19.67  | 19.27        | 18.73              |

| Attack         | Misspellings | Positional  | Rare Words | Reddit | Text  | Overall |
|----------------|--------------|-------------|------------|--------|-------|---------|
| Original       | 19.48        | 20.20       | NA         | 20.34  | 19.96 | 20.29   |
| GN             | 18.17        | **17.72**  | NA         | 18.82  | 19.66 | 19.03   |
| FGSM (B>1)     | 18.09        | **17.72**  | NA         | 18.79  | 19.60 | 19.00   |
| FGSM (B=1)     | 18.09        | **17.72**  | NA         | 18.79  | 19.60 | 19.00   |
| PGD            | 16.30        | **15.89**  | NA         | 16.50  | 17.36 | 17.13   |
| SPSA           | 18.08        | **17.72**  | NA         | 18.79  | 19.59 | 19.00   |
</details> 
<br />

_HPSv2 Reward Model_
<details>
<summary>HPS Benchmark Dataset</summary>

| Attack         | Anime  | Concept-art | Painting | Photo  | Overall |
|----------------|--------|-------------|----------|--------|---------|
| Original       | 29.10  | 28.89       | 29.23    |28.57  | 28.92    |
| GN             | 24.70  | **23.55**   | 24.06    | 24.34  | 24.33   |
| FGSM (B>1)     | 23.94  | **23.16**   | 23.36    | 23.52  | 23.60   |
| FGSM (B=1)     | 23.94  | **23.22**   | 23.37    | 23.54  | 23.61   |
| PGD            | 20.89  | 21.82       | 21.19    | **20.63**  | 20.95   |
| SPSA           | 24.69  | **23.58**   | 24.13    | 24.33  | 24.34   |
</details>

<details>
<summary>DrawBench Dataset</summary>

| Attack         | Colors | Conflicting | Counting | DALL-E | Descriptions | Gary Marcus et al. |
|----------------|--------|-------------|----------|--------|--------------|--------------------|
| Original       | 28.88  | 29.17       | 29.49    | 28.46  | NA           | 28.20              |
| GN             | 25.87  | 25.23       | 26.81    | 25.39  | NA           | **23.62**          |
| FGSM (B>1)     | 25.38  | 24.13       | 25.95    | 25.02  | NA           | **22.80**          |
| FGSM (B=1)     | 25.35  | 24.08       | 25.93    | 24.98  | NA           | **22.81**          |
| PGD            | 23.23  | 20.73       | 23.55    | 23.24  | NA           | **19.93**          |
| SPSA           | 25.93  | 25.22       | 26.66    | 25.39  | NA           | **23.90**          |

| Attack         | Misspellings | Positional | Rare Words | Reddit | Text  | Overall |
|----------------|--------------|------------|------------|--------|-------|---------|
| Original       | 27.56    | 29.92      | NA         | 28.86  | 29.75 | 29.06       |
| GN             | 25.83        | 25.39      | NA         | 25.27  | 26.00 | 25.63   |
| FGSM (B>1)     | 25.53        | 24.47      | NA         | 24.56  | 25.41 | 24.93   |
| FGSM (B=1)     | 25.53        | 24.49      | NA         | 24.56  | 25.41 | 24.92   |
| PGD            | 22.52        | 21.28      | NA         | 22.94  | 23.45 | 22.47   |
| SPSA           | 25.86        | 25.36      | NA         | 25.36  | 25.83 | 25.62   |
</details>

#### Stable Diffusion 2
_HPSv1 Reward Model_
<details>
<summary>HPS Benchmark Dataset</summary>

| Attack         | Anime | Concept-art | Painting | Photo  | Overall |
|----------------|-------|-------------|----------|--------|---------|
| Original       | 21.62 | 22.65       | 22.34    | 21.31  | 21.93   |
| GN             | 20.03 | **18.77**   | 19.61    | 19.68  | 19.53   |
| FGSM (B>1)     | 20.07 | **18.54**   | 19.60    | 19.65  | 19.47   |
| FGSM (B=1)     | 20.07 | **18.54**   | 19.60    | 19.65  | 19.47   |
| PGD            | 16.55 | **15.44**   | 17.11    | 17.26  | 16.60   |
| SPSA           | 20.07 | **18.56**   | 19.59    | 19.65  | 19.47   |
</details>

<details>
<summary>DrawBench Dataset</summary>

| Attack         | Colors | Conflicting | Counting | DALL-E | Descriptions | Gary Marcus et al. |
|----------------|--------|-------------|----------|--------|--------------|--------------------|
| Original       | 20.34  | 21.02       | NA       | 20.63  | 19.87        | 20.69              |
| GN             | 19.21  | 19.09       | NA       | 19.23  | 18.77        | **18.53**          |
| FGSM (B>1)     | 19.15  | 19.09       | NA       | 19.19  | 18.71        | **18.52**          |
| FGSM (B=1)     | 19.15  | 19.09       | NA       | 19.19  | 18.71        | **18.52**          |
| PGD            | 17.59  | **16.12**   | NA       | 17.39  | 17.28        | 16.20              |
| SPSA           | 19.15  | 19.11       | NA       | 19.21  | 18.71        | **18.52**          |

| Attack         | Misspellings | Positional | Rare Words | Reddit  | Text   | Overall |
|----------------|--------------|------------|------------|---------|--------|---------|
| Original       | NA           | NA         | NA         | 21.02   | 20.24  | 20.58   |
| GN             | NA           | NA         | NA         | 19.43   | 20.09  | 19.24   |
| FGSM (B>1)     | NA           | NA         | NA         | 19.33   | 20.06  | 19.20   |
| FGSM (B=1)     | NA           | NA         | NA         | 19.33   | 20.06  | 19.20   |
| PGD            | NA           | NA         | NA         | 16.89   | 18.28  | 17.13   |
| SPSA           | NA           | NA         | NA         | 19.35   | 20.05  | 19.21   |
</details>
<br />

_HPSv2 Reward Model_
<details>
<summary>HPS Benchmark Dataset</summary>

| Attack      | Anime  | Concept-art | Painting | Photo  | Overall |
|-------------|--------|-------------|----------|--------|---------|
| Original    | 28.86 | 29.14       | 29.33    | 29.34  | 29.14   |
| GN          | 25.22  | **23.56**  | 25.22    | 24.55  | 24.54   |
| FGSM (B>1)  | 24.28  | **22.95**  | 24.61    | 23.83  | 23.83   |
| FGSM (B=1)  | 24.23  | **22.93**  | 24.59    | 23.85  | 23.81   |
| PGD         | 20.90  | **20.49**  | 22.50    | 21.28  | 21.21   |
| SPSA        | 25.13  | **23.62**  | 25.27    | 24.63  | 24.56   |
</details>

<details>
<summary>DrawBench Dataset</summary>

| Attack         | Colors | Conflicting | Counting | DALL-E | Descriptions | Gary Marcus et al. |
|----------------|--------|-------------|----------|--------|--------------|--------------------|
| Original       | 30.66  | 29.58       | 29.50    | 29.13  | NA           | 28.32              |
| GN             | 26.16  | 25.40       | 26.64    | 26.43  | NA           | 24.40              |
| FGSM (B>1)     | 25.26  | 24.60       | 25.96    | 25.86  | NA           | 23.65              |
| FGSM (B=1)     | 25.23  | 24.58       | 25.96    | 25.82  | NA           | 23.61              |
| PGD            | 22.72  | 20.56       | 23.69    | 23.28  | NA           | 21.53              |
| SPSA           | 26.20  | 25.45       | 26.49    | 26.34  | NA           | 24.57              |

| Attack         | Misspellings | Positional | Rare Words | Reddit  | Text   | Overall |
|----------------|--------------|------------|------------|---------|--------|---------|
| Original       | NA           | 29.90      | NA         | 28.25 | 30.02  | 29.47      |
| GN             | NA           | 26.03      | NA         | **23.24** | 26.30  | 25.76   |
| FGSM (B>1)     | NA           | 25.12      | NA         | **22.17** | 25.95  | 25.03   |
| FGSM (B=1)     | NA           | 25.14      | NA         | **22.20** | 25.94  | 25.02   |
| PGD            | NA           | 23.19      | NA         | **18.38** | 23.74  | 22.38   |
| SPSA           | NA           | 26.13      | NA         | **23.24** | 26.35  | 25.76   |
</details>

#### Stable Diffusion 3
_HPSv1 Reward Model_
<details>
<summary>HPS Benchmark Dataset</summary>

| Attack         | Anime  | Concept-art | Painting | Photo  | Overall |
|----------------|--------|-------------|----------|--------|---------|
| Original       | 23.00  | 22.88       | 23.52    | 21.66  | 22.96   |
| GN             | 20.26  | **19.71**   | 19.84    | 20.61  | 20.00   |
| FGSM (B>1)     | 20.16  | **19.62**   | 19.76    | 20.54  | 19.91   |
| FGSM (B=1)     | 20.16  | **19.62**   | 19.76    | 20.54  | 19.91   |
| PGD            | 17.12  | **16.81**   | 16.81    | 18.95  | 17.12   |
| SPSA           | 20.19  | **19.61**   | 19.80    | 20.55  | 19.93   |
|
</details>

<details>
<summary>DrawBench Dataset</summary>

| Attack         | Colors | Conflicting | Counting | DALL-E | Descriptions | Gary Marcus et al. |
|----------------|--------|-------------|----------|--------|--------------|--------------------|
| Original       | 21.02      | 21.36   | NA       | 21.49  | NA           | 21.38              |
| GN             | 19.59  | 19.80       | NA       | 19.56  | NA           | 19.47              |
| FGSM (B>1)     | 19.52  | 19.75       | NA       | 19.50  | NA           | 19.58              |
| FGSM (B=1)     | 19.52  | 19.75       | NA       | 19.50  | NA           | 19.58              |
| PGD            | 17.94  | 16.48       | NA       | 17.13  | NA           | 16.92              |
| SPSA           | 19.55  | 19.77       | NA       | 19.55  | NA           | 19.55              |


| Attack         | Misspellings | Positional | Rare Words | Reddit | Text  | Overall |
|----------------|--------------|------------|------------|--------|-------|---------|
| Original       | NA           | 21.72      | NA         | 23.39  | 21.04 | 21.70   |
| GN             | NA           | **18.98** | NA         | 20.59  | 19.71 | 19.76   |
| FGSM (B>1)     | NA           | **18.89** | NA         | 20.61  | 19.82 | 19.76   |
| FGSM (B=1)     | NA           | **18.89** | NA         | 20.61  | 19.82 | 19.76   |
| PGD            | NA           | **15.67** | NA         | 16.75  | 17.80 | 16.99   |
| SPSA           | NA           | **18.91** | NA         | 20.58  | 19.82 | 19.77   |
</details>
<br />

_HPSv2 Reward Model_
<details>
<summary>HPS Benchmark Dataset</summary>

| Attack         | Anime  | Concept-art | Painting | Photo  | Overall |
|----------------|--------|-------------|----------|--------|---------|
| Original       | 30.25  | 29.95       | 30.85    | **29.26**  | 30.14   |
| GN             | 25.84  | 24.10       | 25.07    | **23.74**  | 24.67   |
| FGSM (B>1)     | 25.16  | 23.33       | 24.17    | **22.71**  | 23.82   |
| FGSM (B=1)     | 25.18  | 23.31       | 24.15    | **22.73**  | 23.82   |
| PGD            | 22.37  | 20.61       | 21.98    | **19.24**  | 21.10   |
| SPSA           | 25.78  | **24.07**   | 24.97    | 24.55  | 24.62   |
</details>

<details>
<summary>DrawBench Dataset</summary>

| Attack         | Colors | Conflicting | Counting | DALL‑E | Descriptions | Gary Marcus et al. |
|----------------|--------|-------------|----------|--------|--------------|--------------------|
| Original       | 29.66  | 31.36       | 29.88    | 30.48  | 29.50   | 31.72              |
| GN             | 26.47  | 26.15       | 27.32    | 25.90  | 25.73        | 25.95              |
| FGSM (B>1)     | 25.80  | 25.03       | 26.90    | 25.45  | 25.16        | 24.81              |
| FGSM (B=1)     | 19.52  | 19.75       | 26.88    | 19.50  | 25.13        | 19.58              |
| PGD            | 17.94  | **16.48**   | 24.69    | 17.13  | 23.47        | 16.92              |
| SPSA           | 26.34  | 26.09       | 27.36    | 25.69  | 25.52        | 25.95              |

| Attack         | Misspellings | Positional | Rare Words | Reddit   | Text   | Overall |
|----------------|--------------|------------|------------|----------|--------|---------|
| Original       | NA           | 30.68      | NA         | 30.81    | 30.50  | 30.48   |
| GN             | NA           | 26.61      | NA         | **25.33**| 26.13  | 26.18   |
| FGSM (B>1)     | NA           | 25.56      | NA         | **24.35**| 25.59  | 25.41   |
| FGSM (B=1)     | NA           | **18.89**  | NA         | 22.20    | 19.82  | 25.76   |
| PGD            | NA           | 22.00      | NA         | 21.42    | 23.09  | 22.83   |
| SPSA           | NA           | 26.42      | NA         | **25.38**| 26.15  | 26.11   |
</details>

### Transfer Tests Results
This sub-section presents the results of the transfer test experiments. Each entry is rounded to 2 decimal places. The lowest category in each row is highlighted in bold. Since the HPS benchmark dataset yielded better results than the DrawBench dataset, its adversarial images were selected for the transfer tests.

#### Stable Diffusion 1
<details>
<summary>HPSv1 -> HPSv2</summary>

| Attack         | Anime  | Concept-art | Painting | Photo  | Overall |
|----------------|--------|-------------|----------|--------|---------|
| Original       | 29.08  | 25.04       | 27.73    | 27.98  | 27.66   |
| GN             | 26.77  | **23.70**  | 24.91    | 24.53  | 25.13   |
| FGSM (B>1)     | 26.82  | **23.74**  | 25.01    | 24.60  | 25.19   |
| FGSM (B=1)     | 26.82  | **23.74**  | 25.01    | 24.60  | 25.19   |
| PGD            | 26.70  | **23.60**  | 24.92    | 24.54  | 25.10   |
| SPSA           | 26.82  | **23.75**  | 25.01    | 24.60  | 25.20   |
</details>

<details>
<summary>HPSv2 -> HPSv1</summary>

| Attack         | Anime  | Concept-art  | Painting | Photo  | Overall |
|----------------|--------|--------------|----------|--------|---------|
| Original       | 20.32  | 19.32   | 20.31    | 20.16  | 20.16   |
| GN             | 19.21  | **17.57**   | 19.04    | 18.81  | 18.87   |
| FGSM (B>1)     | 19.08  | **17.50**   | 18.95    | 18.76  | 18.78   |
| FGSM (B=1)     | 19.07  | **17.50**   | 18.96    | 18.76  | 18.78   |
| PGD            | 19.12  | **17.45**   | 18.93    | 18.73  | 18.78   |
| SPSA           | 19.19  | **17.48**   | 19.00    | 18.79  | 18.84   |
</details>

#### Stable Diffusion 2
<details>
<summary>HPSv1 -> HPSv2</summary>

| Attack         | Anime | Concept-art | Painting | Photo | Overall |
|----------------|-------|-------------|----------|-------|---------|
| Original       | 28.63 | 27.72  | 28.87    | 28.61 | 28.45   |
| GN             | 25.79 | **24.13**  | 26.84    | 25.18 | 25.40   |
| FGSM (B>1)     | 25.95 | **24.15**  | 26.85    | 25.33 | 25.49   |
| FGSM (B=1)     | 25.95 | **24.15**  | 26.85    | 25.33 | 25.49   |
| PGD            | 25.80 | **24.07**  | 26.76    | 25.21 | 25.38   |
| SPSA           | 25.95 | **24.16**  | 26.83    | 25.32 | 25.49   |
</details>

<details>
<summary>HPSv2 -> HPSv1</summary>

| Attack         | Anime  | Concept-art | Painting | Photo  | Overall |
|----------------|--------|-------------|----------|--------|---------|
| Original       | 20.22 | 21.38      | 20.98    | 20.81  | 20.91   |
| GN             | 19.83  | **18.45**  | 19.25    | 19.56  | 19.16   |
| FGSM (B>1)     | 19.80  | **18.30**  | 18.95    | 18.76  | 19.06   |
| FGSM (B=1)     | 19.78  | **18.31**  | 19.07    | 19.60  | 19.06   |
| PGD            | 19.73  | **18.28**  | 18.99    | 19.51  | 19.01   |
| SPSA           | 19.84  | **18.45**  | 19.16    | 19.67  | 19.16   |
</details>

#### Stable Diffusion 3
<details>
<summary>HPSv1 -> HPSv2</summary>

| Attack         | Anime | Concept-art | Painting | Photo  | Overall |
|----------------|-------|-------------|----------|--------|---------|
| Original       | 28.67 | 29.36       | 28.93    | 30.48  | 29.15   |
| GN             | 26.88 | 25.52       | 26.08    | **24.20** | 25.94   |
| FGSM (B>1)     | 26.88 | 25.52       | 26.08    | **24.20** | 25.94   |
| FGSM (B=1)     | 26.88 | 25.52       | 26.08    | **24.20** | 25.94   |
| PGD            | 26.78 | 25.30       | 25.98    | **24.18** | 25.80   |
| SPSA           | 26.87 | 25.49       | 26.10    | **24.26** | 25.93   |

</details>

<details>
<summary>HPSv2 -> HPSv1</summary>

| Attack         | Anime  | Concept-art | Painting | Photo  | Overall |
|----------------|--------|-------------|----------|--------|---------|
| Original       | 21.80  | 20.86 | 21.88    | 22.04  | 21.59   |
| GN             | 20.27  | **19.13**  | 19.16    | 19.98  | 19.54   |
| FGSM (B>1)     | 19.80  | **17.50**  | 19.07    | 18.76  | 19.06   |
| FGSM (B=1)     | 19.78  | **18.31**  | 19.07    | 18.76  | 19.06   |
| PGD            | 19.73  | **18.28**  | 18.99    | 19.51  | 19.01   |
| SPSA           | 20.27  | 19.19      | **19.16**| 19.93  | 19.55   |

</details>

# 6. Repository Details


# 7. References
1. Wu, X., Sun, K., Zhu, F., Zhao, R., & Li, H. (2023). Human Preference Score: Better Aligning Text-to-Image Models with Human Preference. ArXiv. https://arxiv.org/abs/2303.14420
2. Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning Transferable Visual Models From Natural Language Supervision. ArXiv. https://arxiv.org/abs/2103.00020
3. Wu, X., Hao, Y., Sun, K., Chen, Y., Zhu, F., Zhao, R., & Li, H. (2023). Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis. ArXiv. https://arxiv.org/abs/2306.09341
4. Wang, Z. J., Montoya, E., Munechika, D., Yang, H., Hoover, B., & Chau, D. H. (2022). DiffusionDB: A Large-scale Prompt Gallery Dataset for Text-to-Image Generative Models. ArXiv. https://arxiv.org/abs/2210.14896
5. Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E., Ghasemipour, S. K., Ayan, B. K., Mahdavi, S. S., Lopes, R. G., Salimans, T., Ho, J., Fleet, D. J., & Norouzi, M. (2022). Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding. ArXiv. https://arxiv.org/abs/2205.11487
6. https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html
7. Oleksii Holub. DiscordChatExporter. https://github.com/Tyrrrz/DiscordChatExporter, 2022
