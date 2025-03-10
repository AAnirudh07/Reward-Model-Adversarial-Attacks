### Notebooks Overview
Jupyter notebooks used for generating images, running adversarial attacks, and testing transferability of adversarial images to other reward models. The structure is organized by Stable Diffusion version, with each version containing three main steps:

1. 01_generate_images
    - Notebooks for generating images using two datasets/benchmarks:
        - HPS (HPSBench)
        - DrawBench

2. 02_run_attack
    - Notebooks for running adversarial attacks on the generated images:
        - HPSv1 and HPSv2 refer to different versions or configurations of the HPS benchmark.
        - Separate notebooks target images generated from HPSBench and DrawBench.
        - These notebooks apply various adversarial methods (e.g., FGSM, PGD, SPSA, or Gaussian noise) to degrade the reward model's scores.

3. 03_run_transfer_test (or 03_run_transfer_text in some versions)
    - Notebooks for testing how well adversarial images (crafted for one reward model) transfer to another.