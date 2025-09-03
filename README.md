<p align="center">
  <img src="https://raw.githubusercontent.com/ufuk-cakir/ufuk-cakir.github.io/7d1d9f2b3a8b7b3db0e6ecca77fbd7b332cd6c84/assets/images/eso%20logo.png" alt="ESO logo" width="200"/>
</p>

<h1 align="center">ESO: Evolutionary Spectrogram Optimisation 🧬🔊</h1>

<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT" />
  </a>
  <a href="https://github.com/ufuk-cakir/eso/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/ufuk-cakir/eso/ci.yml?branch=main" alt="GitHub Workflow Status" />
  </a>
  <a href="https://eso.readthedocs.io/">
    <img src="https://readthedocs.org/projects/eso/badge/" alt="Documentation Status" />
  </a>
  <a href="https://codecov.io/gh/ufuk-cakir/eso">
    <img src="https://codecov.io/gh/ufuk-cakir/eso/branch/main/graph/badge.svg" alt="codecov" />
  </a>
</p>
Optimising spectrogram inputs for efficient passive acoustic monitoring using genetic algorithms. Based on our paper presented at the ICLR 2024 ML4RS Workshop.

---

### The Challenge 🤔

Traditional bioacoustic monitoring with Convolutional Neural Networks (CNNs) often requires:
*   Large, complex models.
*   Computationally expensive preprocessing (like filtering and downsampling).
*   This makes real-time analysis on low-resource edge devices difficult. 🐢⏳

### Our Solution: ESO ✨

ESO (Evolutionary Spectrogram Optimisation) tackles this by using a **Genetic Algorithm (GA)** to intelligently select the most informative frequency bands directly from the original spectrogram.

Instead of processing the *entire* spectrogram, ESO:
1.  🧬 **Encodes** horizontal spectrogram bands (defined by position `Pₜ` and height `h`) as "genes".
2.  📜 **Combines** genes into "chromosomes", representing a specific selection of bands.
3.  💪 **Evolves** a population of chromosomes using selection, crossover, and mutation.
4.  📈 **Optimizes** for a fitness function balancing classification performance (F1-score) and model simplicity (trainable parameters).
5.  📉 **Outputs** the best chromosome, which defines narrow bands to be extracted and stacked, creating a highly compressed input for a much simpler CNN.

[Figure gene chromosome.pdf](https://github.com/user-attachments/files/22117192/Figure.gene.chromosome.pdf)


<p align="center">
  <img src="images/gene-chromosome.pdf" alt="Figure 1 from paper showing ESO concept" width="600"/>
  <br/><em>Concept from Figure 1: Genes define bands, chromosomes collect genes, bands are stacked for the CNN.</em>
</p>

### Key Benefits & Features 🚀

*   ✅ **Drastically Reduced Model Size:** ~91% fewer trainable parameters compared to the baseline.
*   ✅ **Faster Inference:** ~70% reduction in processing time for raw audio.
*   ✅ **Efficient:** Minimizes need for heavy preprocessing like downsampling.
*   ✅ **Effective:** Achieves comparable performance with only a minor (~4%) trade-off in F1-score.
*   🐍 Usable as a **Python package**.
*   🖥️ Includes an easy-to-use **Graphical User Interface (GUI)**.
*   📊 **TensorBoard** integration for monitoring training progress.

### Getting Started 🛠️

1. **Clone the repository**
    ```bash
    git clone https://github.com/ufuk-cakir/ESO.git
    cd ESO
    ```

2. **Set up a virtual environment** (recommended)
    ```bash
    python -m venv venv
    # On Linux/macOS
    source venv/bin/activate
    # On Windows
    venv\Scripts\activate
    ```

3. **Install PyTorch** based on your system configuration  
   (see [PyTorch](https://pytorch.org/get-started/locally/) to choose the correct version for your machine)
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cu126
    ```

4. **Install other dependencies**
    ```bash
    pip install -r requirements.txt
    ```

### Running ESO 🏃

*   **Using the GUI:**
    ```bash
    python path/to/your/repository/eso_app.py 
    ```
    The GUI provides options to select data, configure hyperparameters, run ESO, and monitor progress (including TensorBoard).

*   **As a Python Package:**
    Import the necessary modules from the `eso` package in your Python scripts. (Refer to the documentation or example scripts within the repository for specific usage details).

### Citation ✍️

If you use ESO in your research or work, please cite our paper:

```bibtex
@inproceedings{cakir2024eso,
  title={{ESO: Evolutionary Spectrogram Optimisation for Passive Acoustic Monitoring}},
  author={Ufuk {\c{C}}ak{\i}r and Lor{\`e}ne Jeantet and Joel Lontsi and Emmanuel Dufourq},
  booktitle={ICLR 2024 Machine Learning for Remote Sensing (ML4RS) Workshop},
  year={2024},
  url={https://ml-for-rs.github.io/iclr2024/camera_ready/papers/51.pdf}
}
