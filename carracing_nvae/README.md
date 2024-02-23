## Car Racing Experiment (carracing_nvae)

### Overview
The Car Racing experiment involves training a World Models-based agent to navigate and learn optimal actions in the OpenAI Gym's CarRacing-v2 environment. The agent employs a deep hierarchical variational autoencoder called Nouveau VAE (NVAE) for enhanced image compression, a Mixture-Density Recurrent Network (MDN-RNN) to predict future states, and a linear Controller (C) trained using Covariance-Matrix Adaptation Evolution-Strategy (CMA-ES).

### Additional Dream Car Racing Environment

In addition to the primary CarRacing-v2 environment, an experimentation with a dream car racing environment has been conducted. This dream environment is constructed utilizing the learned environment model and policy from the car racing-v2 task.

### Construction of Dream Car Racing-v2 Environment

The dream car racing-v2 environment is constructed differently from the primary environment to offer an alternative simulation for exploring the agent's performance under varied conditions. The following highlights the unique steps for the dream environment:

1. **Data Generation for Dream Environment:**
   - Sample 10,000 rollouts using a random policy in the dream car racing-v2 environment.
   - Record observations during interactions to understand the environment's dynamics.

2. **Training Vision Component (V) for Dream Environment:**
   - Train the visual component (V) specifically for the dream environment to encode frames into a latent vector (z) with a size of 32.

3. **Training Memory Component (M) for Dream Environment:**
   - Train the memory component (M) using the pre-trained data and observed random actions in the dream environment to create a mixture of Gaussians representing the environment dynamics.

4. **Evolution of Controller (C) for Dream Environment:**
   - Evolve the controller (C) specific to the dream environment to maximize the expected cumulative reward of a rollout using the Covariance-Matrix Adaptation Evolution-Strategy (CMA-ES) approach.

### Steps Common to Both Environments

1. **Data Generation (Common):**
   - Execute the data generation script to create a dataset of random rollouts.
     ```bash
     python data/generation_script.py --rollouts 10000 --rootdir datasets/carracing --threads 8
     ```

2. **Training the Nouveau VAE (Common):**
   - Train the NVAE using the `trainvae.py` script:
     ```bash
     python trainvae.py --logs exp_dir
     ```

3. **Training the MDN-RNN (Common):**
   - Ensure the NVAE has been trained in the same `exp_dir`.
   - Train the MDN-RNN using the `trainmdrnn.py` script:
     ```bash
     python trainmdrnn.py --logs exp_dir
     ```

4. **Training and Testing the Controller (Common):**
   - Train the linear controller using CMA-ES with the `traincontroller.py` script:
     ```bash
     python traincontroller.py --logs exp_dir --n-samples 4 --pop-size 4 --target-return 950 --display
     ```
   - Test the obtained policy with:
     ```bash
     python test_controller.py --logs exp_dir
     ```
