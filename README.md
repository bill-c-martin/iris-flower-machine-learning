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

# Install python, pip, and others
python3 -V
sudo apt install -y python3-pip
sudo apt install -y build-essential libssl-dev libffi-dev python3-dev

# Install python virtual environment
sudo apt install -y python3-venv
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

