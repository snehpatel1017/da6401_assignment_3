# Hindi Transliteration using Sequence-to-Sequence Models

This repository contains implementations of sequence-to-sequence models for transliterating text from Latin script to Devanagari (Hindi) script. The project includes both a basic Seq2Seq model and an enhanced attention-based model.

## 📋 Project Overview

Transliteration is the process of converting text from one script to another while preserving pronunciation. This project focuses on Latin (Roman) to Devanagari script transliteration using neural sequence-to-sequence architectures.

## 🗂️ Repository Structure

```
A_3/
├── _pycache_/
├── predictions_attention/    # Predictions from attention model
├── predictions_vanilla/      # Predictions from basic model
├── attention_grid.py         # Code for attention visualization
├── Basic_Model.py            # Implementation of basic Seq2Seq model
├── connectivity_grid.py      # Connectivity visualization between input/output
├── final_notebook.ipynb      # Jupyter notebook with complete implementation
├── Model_with_Attention.py   # Implementation of attention-based Seq2Seq model
├── prediction_attention.py   # Prediction handling for attention model
├── prediction_vanilla.py     # Prediction handling for basic model
├── run_basic_model.py        # Script to train and evaluate basic model
├── run_model_with_attention.py  # Script to train and evaluate attention model
└── sample                    # Sample data file
```

## 🚀 Installation

1. Clone the repository:
```
git clone https://github.com/snehpatel1017/da6401_assignment_3.git
```

2. Install required dependencies:
```
pip install torch matplotlib seaborn numpy wandb
```

# 🔤 Setting Up Devanagari Font Support

For proper visualization of Devanagari script in plots, you need to install and configure a font that supports Devanagari characters:

## 1. **Download and Install the Noto Sans Devanagari Font**:

```bash
# Download the font
wget -O NotoSansDevanagari-Regular.ttf "https://github.com/googlefonts/noto-fonts/blob/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf?raw=true"

# Create .fonts directory if it doesn't exist
mkdir -p ~/.fonts

# Move font to the fonts directory
mv NotoSansDevanagari-Regular.ttf ~/.fonts/
```

## 2. **Register the Font with Matplotlib**: 

Add the following code to your scripts or notebook before creating any plots:

```python
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# Register font with matplotlib
font_dirs = ['~/.fonts']
font_files = fm.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    fm.fontManager.addfont(font_file)

# Set font family with fallback to DejaVu Sans for English text
plt.rcParams['font.family'] = ['Noto Sans Devanagari', 'DejaVu Sans']
```

## 3. **Restart Kernel/Python Session**: 

After installing the font, restart your kernel or Python session for the changes to take effect.

## 4. **Verify Font Installation** (optional):

```python
# Check if font is correctly installed
available_fonts = [f.name for f in fm.fontManager.ttflist]
if 'Noto Sans Devanagari' in available_fonts:
    print("Devanagari font installed successfully!")
else:
    print("Devanagari font not found. Visualizations may not display Hindi characters correctly.")
```

**Note**: If you're running this in a Jupyter notebook or Colab, you may need to repeat the font installation steps each time you start a new session.

## 🔧 Usage

Running the Basic Seq2Seq Model
```bash
python -m A_3.run_basic_model
```

Running the Attention-Based Model
```bash
python -m A_3.run_model_with_attention
```

## 📊 Models and Performance

1. **Basic Sequence-to-Sequence Model**
   * Architecture: Encoder-Decoder with GRU/LSTM units
   * Implementation: `Basic_Model.py`
   * Word-level accuracy on test data: **37.75%**

2. **Attention-Based Sequence-to-Sequence Model**
   * Architecture: Enhanced with attention mechanism
   * Implementation: `Model_with_Attention.py`
   * Word-level accuracy on test data: **38.39%**

The attention mechanism provides a **0.64%** improvement over the basic model, demonstrating its effectiveness in handling the complexities of transliteration tasks.

## 📈 Visualizations

The project includes several visualization methods for analyzing model behavior:

1. **Attention Grid**: Shows attention weight heatmaps for multiple examples
   * Implementation: `attention_grid.py`
   * Usage: Visualize which input characters influence each output character

2. **Connectivity Visualization**: Maps relationships between input and output characters
   * Implementation: `connectivity_grid.py`
   * Usage: See which input character has the strongest influence on each output character

3. **Prediction Analysis**: Comparison of predictions vs. actual translations
   * Available for both models in their respective prediction folders

## 🧪 Key Features

* Character-level processing for precise transliteration
* Attention mechanism for improved sequence alignment
* Visualization tools for model interpretability
* Word-level  accuracy evaluation
* Experiment logging and analysis with Weights & Biases

## 📝 Results and Analysis

The attention-based model shows superior performance compared to the basic sequence-to-sequence model. Key observations include:
* Better handling of longer and more complex words
* Improved character alignment between scripts
* More consistent preservation of phonetic information

This project was created as part of Assignment 3 for a sequence-to-sequence modeling course.
