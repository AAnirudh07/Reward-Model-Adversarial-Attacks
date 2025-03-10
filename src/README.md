This folder contains the source code for generating images, running adversarial attacks, and testing transferability between different reward models.

## Directories

- **attacks**  
  Contains various adversarial attack implementations (e.g., FGSM, PGD, SPSA).

- **datasets**  
  Includes utilities and loading scripts for handling datasets (e.g., HPSBench, DrawBench).

- **models**  
  Houses model definitions and configurations for Stable Diffusion and reward models.

## Python Files

- **args.py**  
  Defines command-line arguments for the scripts (e.g., attack types, model paths, dataset options).

- **generate_images.py**  
  Generates images from Stable Diffusion models based on specified datasets.

- **run_attack.py**  
  Applies adversarial attacks to the generated images to degrade reward model scores.

- **run_transfer_test.py**  
  Tests the transferability of adversarial images between different models.

- **utils.py**  
  Contains shared helper functions (e.g., memory clearing, reward score computation).
