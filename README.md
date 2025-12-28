# **DQL-Navigator – Reinforcement Learning Race Car Simulator**
[https://prezi.com/p/edit/fuouykm4zmyg/](https://prezi.com/view/QArtwAtCwH5BxmTu6jHg/?referral_token=EpVnqflnB3FN)

DQL-Navigator explores how autonomous agents learn to drive in the **Gymnasium CarRacing-v3** environment. We evaluate two reinforcement learning approaches—**Tabular Q-Learning** and **Deep Q-Networks (DQN)**—to understand how agents learn optimal navigation, control, and decision-making strategies through trial and error.  
We begin with a discretized Q-Learning agent, then scale to a CNN-based DQN capable of handling high-dimensional visual input.

---

## **Features**
- **Gymnasium CarRacing-v3 Environment**  
  Continuous-control 2D racing environment adapted and optimized for RL experimentation.

- **Tabular Q-Learning Agent**  
  Classical Q-Learning with state discretization and epsilon-greedy exploration.

- **DQN Agent**  
  CNN-based Deep Q-Network with experience replay and target network stabilization.

- **Training Analytics**  
  Logging and plotting of reward trends, episode length, and epsilon decay.

- **Model Saving & Evaluation**  
  Save and reload Q-tables and DQN checkpoints for continued training or evaluation.

---

## **Output**

### **Training Progress**
<p align="center" style="font-size: 20px; font-weight: bold;">
  FROM RANDOM ACTIONS  
  ⟶  
  LEARNED Q-POLICY
</p>

<p align="center">
  <img src="resources/500 GIF.gif" width="450" />
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="resources/1.gif" width="350" />
</p>

---

## **Setup Instructions**

1. **Clone the repository**
  ```bash
  git clone https://github.com/gyuszix/DQL-Navigator.git
  cd DQL-Navigator
  ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Install extra packages for rendering**
   ```bash
   sudo apt-get install ffmpeg xvfb
   ```


## Project Structure
```
├── DQN
│   ├── DeepQ.py
│   ├── ImageProcessing.py
│   ├── checkpoints
│   │   ├── checkpoint_ep1000.pth
│   │   ├── checkpoint_ep1500.pth
│   │   └── checkpoint_ep500.pth
│   ├── main.py
│   ├── plots
│   │   ├── dqn_carracing.pth
│   │   ├── episode_rewards.png
│   │   ├── epsilon_decay.png
│   │   ├── training_metrics.pkl
│   │   └── unique_states.png
│   └── replay.py
├── FAI final prez.pdf
├── Prev_Codebase
│   ├── main.py
│   ├── q_learning.py
│   ├── q_table.pkl
│   └── test_policy.py
├── Q_Learning
│   ├── main.py
│   ├── metrics.csv
│   ├── qlearning.py
│   ├── requirements.txt
│   └── test_policy.py
├── README.md
├── requirements.txt
└── resources
    └── test.md

```
## Training

### 1. Q-Learning
Run the Q-learning agent:
```bash
python Q_Learning/main.py
```

Adjust hyperparameters such as learning rate (`alpha`), discount factor (`gamma`), or epsilon decay inside the script.

### 2. DQN
Run the DQN training:
```bash
python DQN/main.py
```
### 3. Visualizing the results
To visualize the model run the test policy file in each folder, for example:
```bash
python Q_Learning/test_policy.py

```
