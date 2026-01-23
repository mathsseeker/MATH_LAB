# Getting Started with MATH_LAB

This guide will help you set up your environment for the Mathematics Laboratory course.

## Step 1: Install Python

### Windows
1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run the installer
3. **Important**: Check "Add Python to PATH" during installation
4. Verify installation:
   ```bash
   python --version
   ```

### macOS
1. Install using Homebrew (recommended):
   ```bash
   brew install python3
   ```
2. Or download from [python.org](https://www.python.org/downloads/)

### Linux
Python is usually pre-installed. If not:
```bash
sudo apt-get update
sudo apt-get install python3 python3-pip
```

## Step 2: Set Up Virtual Environment (Recommended)

Using a virtual environment keeps your project dependencies isolated.

### Create Virtual Environment
```bash
# Navigate to the MATH_LAB directory
cd MATH_LAB

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

## Step 3: Install Required Packages

With your virtual environment activated:

```bash
pip install -r requirements.txt
```

This will install:
- NumPy (numerical computing)
- SciPy (scientific computing)
- Pandas (data analysis)
- Matplotlib (plotting)
- Seaborn (statistical visualization)
- Jupyter (interactive notebooks)
- SymPy (symbolic mathematics)
- scikit-learn (machine learning)

## Step 4: Verify Installation

Create a test file `test_setup.py`:

```python
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt

print("NumPy version:", np.__version__)
print("SciPy version:", scipy.__version__)
print("Pandas version:", pd.__version__)

# Test basic functionality
x = np.array([1, 2, 3, 4, 5])
print("\nNumPy array:", x)
print("Mean:", np.mean(x))

print("\nAll packages imported successfully!")
```

Run it:
```bash
python test_setup.py
```

## Step 5: Launch Jupyter Notebook (Optional)

Jupyter notebooks provide an interactive environment:

```bash
jupyter notebook
```

This will open in your web browser. You can create new notebooks to work on labs.

## IDE Recommendations

### Visual Studio Code
- Install from [code.visualstudio.com](https://code.visualstudio.com/)
- Install Python extension
- Features: debugging, linting, IntelliSense

### PyCharm
- Community edition is free
- Full-featured Python IDE
- Great for larger projects

### Jupyter Lab
- Web-based interface
- Good for data analysis and visualization
- Install with: `pip install jupyterlab`
- Run with: `jupyter lab`

## Repository Structure

After setup, your directory should look like:

```
MATH_LAB/
├── venv/                    # Virtual environment (don't commit)
├── labs/                    # Lab exercises
├── assignments/             # Homework assignments
├── examples/                # Code examples
├── resources/               # Reference materials
├── data/                    # Datasets
├── requirements.txt         # Package dependencies
├── .gitignore              # Git ignore file
├── README.md               # Main documentation
└── SYLLABUS.md             # Course syllabus
```

## Running Example Code

Try running an example:

```bash
cd examples
python example1_matrix_operations.py
```

This will demonstrate various matrix operations and create visualizations.

## Common Issues and Solutions

### Issue: "pip not found"
**Solution**: 
```bash
python -m pip install --upgrade pip
```

### Issue: "Permission denied" on macOS/Linux
**Solution**: Use virtual environment or:
```bash
pip install --user -r requirements.txt
```

### Issue: Import errors after installation
**Solution**: 
1. Make sure virtual environment is activated
2. Reinstall packages:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

### Issue: Matplotlib plots not showing
**Solution**: Add this to your script:
```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
```

## Working with Data Files

The `data/` directory contains sample datasets. To load them:

```python
import pandas as pd

# Load production data
df = pd.read_csv('data/production_data.csv')
print(df.head())
```

## Git Workflow

### Clone the repository
```bash
git clone https://github.com/mathsseeker/MATH_LAB.git
cd MATH_LAB
```

### Keep your repository updated
```bash
git pull origin main
```

### Submit your work
1. Create a branch for your work:
   ```bash
   git checkout -b assignment1-yourname
   ```

2. Add your files:
   ```bash
   git add your_file.py
   git commit -m "Complete assignment 1"
   ```

3. Push to GitHub:
   ```bash
   git push origin assignment1-yourname
   ```

## Tips for Success

1. **Practice regularly**: Run examples and modify them
2. **Read documentation**: NumPy and SciPy docs are excellent
3. **Use version control**: Commit your work frequently
4. **Ask for help**: Create issues in the repository
5. **Test your code**: Verify results with simple cases
6. **Comment your code**: Future you will thank you

## Getting Help

- **Documentation issues**: Check package documentation
- **Concept questions**: Review syllabus and resources
- **Technical problems**: Create an issue in the repository
- **General discussion**: Use course discussion forum

## Next Steps

1. Review the [README.md](../README.md) for course overview
2. Read the [SYLLABUS.md](../SYLLABUS.md) for detailed schedule
3. Check out [Lab 1](../labs/lab1_intro_to_python.md) to start
4. Explore [examples](../examples/) to see concepts in action
5. Review the [Quick Reference](../resources/quick_reference.md) as needed

---

**Welcome to MATH_LAB! Happy learning! 🎓**
