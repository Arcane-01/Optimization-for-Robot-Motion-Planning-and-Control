# Optimization for Robot Motion Planning and Control

Repository associated with the course **Optimization for Robot Motion Planning and Control (LOTI.05.095)**

---

## Installation Guide

### 1. Create a Conda Environment (Python 3.10.19)

```bash
conda create -n irsim_env python=3.10.19 -y
conda activate irsim_env
```

---

### 2. Install IR-Sim

```bash
pip install ir-sim[all]
```

---

### 3. Install Open3D

```bash
pip install open3d
```

---

### 4. Install JAX (Match Your CUDA Version)

Install JAX depending on your system configuration.
---

## Running the Script

```bash
python3 test.py
```
---

## Note

* You can use different world configuration YAML files. Modify the environment inside `test.py`:

	```python
	irsim.make('obstacle_world.yaml')
	```

Change `'obstacle_world.yaml'` to any other available world configuration file.

* Implement your planner inside `planner.py`.
