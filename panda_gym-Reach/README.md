
## Panda-Gym Reach Experiment (panda_gym-Reach)

### Overview
The Panda-Gym Reach experiment focuses on training a World Models-based agent to perform reaching tasks in a simulated robotic environment. Similar to the Car Racing experiment, it employs the Nouveau VAE, MDN-RNN, and a linear Controller (C).

### Steps to Run the Experiment

1. **Data Generation:**
   - Execute the data generation script to create a dataset of random rollouts:
     ```bash
     python data/generation_script.py --rollouts 1000 --rootdir datasets/panda_data --threads 8
     ```

2. **Training the Nouveau VAE:**
   - Train the NVAE using the `trainvae.py` script:
     ```bash
     python trainvae.py --logs exp_dir
     ```

3. **Training the MDN-RNN:**
   - Ensure the NVAE has been trained in the same `exp_dir`.
   - Train the MDN-RNN using the `trainmdrnn.py` script:
     ```bash
     python trainmdrnn.py --logs exp_dir
     ```

4. **Training and Testing the Controller:**
   - Train the linear controller using MLP and CMA-ES with the `traincontroller.py` script:
     ```bash
     python traincontroller.py --logs exp_dir --n-samples 4 --pop-size 4 --target-return 950 --display
     ```
   - Test the obtained policy with:
     ```bash
     python test_controller.py --logs exp_dir
     ```

