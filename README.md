# iris-flower-machine-learning
Statistical classification of Iris flowers using machine learning with Python

https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
https://medium.com/gft-engineering/start-to-learn-machine-learning-with-the-iris-flower-classification-challenge-4859a920e5e3


## Localhost Setup

### Install Python and pip

In Ubuntu:

```bash
# Upgrade apt
sudo apt update
sudo apt -y upgrade

# Install python, pip, virtual env, and others
python3 -V
sudo apt install -y python3 python3-pip python3-venv
sudo apt install -y build-essential libssl-dev libffi-dev python3-dev
```

### Setup Virtual Environment

```bash
# Clone this repo
git clone git@github.com:bill-c-martin/iris-flower-machine-learning.git

# Enter repo and setup a virtual env
cd iris-flower-machine-learning
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Test Python and pip are working:
python -V
pip -V

# Install packages through pip:
python -m pip install numpy scipy matplotlib ipython jupyter pandas sympy nose sklearn
```

To exit the virtual environment:

```bash
deactivate
```

### Check Packages

To check the pip packages installed in the previous step, run this from the repo root:

```bash
source venv/bin/activate
python check-packages.py
```

It should print out something similar to:

```bash
Python: 3.8.10 (default, Nov 26 2021, 20:14:08)
[GCC 9.3.0]
scipy: 1.8.0
numpy: 1.22.2
matplotlib: 3.5.1
pandas: 1.4.1
sklearn: 1.0.2
```