# DisCorrGAN: Correlation-Aware GAN for Multivariate Time Series Generation

## 📋 Project Overview

DisCorrGAN is a GAN-based model that generates high-quality synthetic data while preserving correlation structures in multivariate financial time series data. This project combines Temporal Convolutional Network (TCN) with Self-Attention mechanisms to learn and reproduce the complex correlation structures of real financial data.

## 🎯 Key Features

- **Correlation Preservation**: Explicitly learns inter-asset correlations to maintain statistical properties of real data
- **TCN + Self-Attention Architecture**: Effectively models long-term dependencies in time series data
- **WGAN-GP Training**: Employs Wasserstein Loss and Gradient Penalty for stable GAN training
- **Portfolio Strategy Evaluation**: Implements various trading strategies to validate the practicality of generated data
- **Multivariate Financial Data**: Utilizes 6 major financial indices (DJI, Nasdaq, JPM, Hang Seng, Gold, WTI)

## 🏗️ Project Structure

```
DisCorrGAN/
├── configs/
│   └── config.yaml              # Model configuration file
├── data/
│   ├── data_crawling.py         # Financial data crawling script
│   ├── indices.csv             # Collected financial index data
│   ├── ref_log_return.pkl      # Reference log return data
│   └── ref_price.pkl           # Reference price data
├── src/
│   ├── baselines/
│   │   ├── networks/
│   │   │   ├── generators.py    # TCN-based generator networks
│   │   │   ├── discriminators.py # TCN-based discriminator networks
│   │   │   └── tcn.py          # Temporal Convolutional Network implementation
│   │   └── trainer.py          # GAN training logic
│   ├── evaluation/
│   │   ├── eval_gen_quality/
│   │   │   └── gen_quality.py  # Generation quality evaluation
│   │   └── eval_portfolio/
│   │       ├── strategies.py   # Trading strategy implementations
│   │       ├── loss.py         # Portfolio loss functions
│   │       └── summary.py      # Evaluation result summaries
│   ├── preprocess/
│   │   ├── gaussianize.py      # Data normalization
│   │   └── __init__.py
│   └── utils.py                # Utility functions
├── outputs/                    # Generated outputs (graphs, statistics)
├── results/                    # Trained model repository
├── etc/                       # Experimental notebooks
└── wandb/                     # Weights & Biases logs
```

## 🚀 Installation and Usage

### 1. Environment Setup

```bash
# Clone the project
git clone [repository-url]
cd DisCorrGAN

# Create and activate virtual environment
conda create -n discorrgan python=3.8
conda activate discorrgan

# Install required packages
pip install torch torchvision torchaudio
pip install wandb yfinance pandas numpy scikit-learn
pip install ml-collections pyyaml tqdm
```

### 2. Data Preparation

```bash
# Collect financial data
cd data
python data_crawling.py
```

### 3. Model Training

```bash
# Start training with default configuration
python -m src.baselines.trainer

# Or modify config file first
python -m src.baselines.trainer --config configs/config.yaml
```

### 4. Result Evaluation

```bash
# Evaluate generation quality
python -m src.evaluation.eval_gen_quality.gen_quality

# Evaluate portfolio strategies
python -m src.evaluation.eval_portfolio.summary
```

## ⚙️ Key Configuration Parameters

You can adjust the following parameters in `configs/config.yaml`:

```yaml
# Data settings
n_vars: 6                    # Number of variables (financial indices)
n_steps: 256                 # Time series length

# Training settings
batch_size: 256              # Batch size
n_epochs: 200                # Number of epochs
lr_G: 0.0002                 # Generator learning rate
lr_D: 0.0001                 # Discriminator learning rate

# Model settings
hidden_dim: 48               # Hidden layer dimension
noise_dim: 3                 # Noise dimension
corr_loss_type: 'l1'         # Correlation loss function ('l1', 'l2', 'fro')
corr_weight: 1.0             # Correlation loss weight
rampup_epochs: 40            # Correlation loss rampup period
```

## 🧠 Model Architecture

### Generator
- **TCN-based**: Temporal Convolutional Network for learning time series patterns
- **Self-Attention**: Attention mechanism for modeling long-term dependencies
- **Multi-head Attention**: 4 heads to capture diverse temporal relationships

### Discriminator
- **TCN + Self-Attention**: Similar architecture to the generator
- **WGAN-GP**: Wasserstein Loss and Gradient Penalty for stable training
- **Spectral Normalization**: Enhanced training stability

### Correlation Loss Function
```python
def correlation_loss_pair(fake_i, fake_j, real_i, real_j, corr_loss_type):
    # Calculate correlation coefficient differences between generated and real data
    # Compute loss using L1, L2, or Frobenius norm
```

## 📊 Evaluation Methods

### 1. Generation Quality Assessment
- **Statistical Similarity**: Comparison of distributions, autocorrelation functions, and correlation coefficients
- **Visual Analysis**: Time series plots, distribution histograms, correlation heatmaps

### 2. Portfolio Strategy Evaluation
- **Equal Weight Portfolio**: Buy & Hold strategy
- **Mean Reversion Strategy**: Moving average-based contrarian strategy
- **Trend Following Strategy**: Short/long-term moving average crossover strategy
- **Volatility Trading Strategy**: Volatility threshold-based trading

## 📈 Key Results

- **Correlation Preservation**: Correlation coefficient error < 5% between real and generated data
- **Statistical Similarity**: High similarity achieved in distributions and autocorrelation functions
- **Practical Validation**: Similar performance to real data across various trading strategies

## 🔬 Experiments and Analysis

The project includes various experimental settings:

- **Seed-based Experiments**: Experiments with multiple seed values for reproducibility
- **Ablation Study**: Experiments removing correlation loss and Self-Attention components
- **Hyperparameter Tuning**: Optimization of learning rates, loss weights, and model architecture

## 📝 References

- Wasserstein GAN with Gradient Penalty
- Temporal Convolutional Networks for Action Segmentation
- Attention Is All You Need
- Multivariate Time Series Generation with Generative Adversarial Networks

## 🤝 Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is distributed under the MIT License. See `LICENSE` file for more details.

## 📧 Contact

If you have any questions about the project, please create an issue.

---

**DisCorrGAN** - A New Paradigm for Multivariate Financial Time Series Generation
