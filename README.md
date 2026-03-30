# OpenEnv Project Setup Guide

This guide will help you set up and run the OpenEnv evaluation environment locally.

---

## 1. Install Anaconda & Create Environment

Make sure Anaconda is installed.

Verify:

```
conda --version
```

Create and activate environment:

```
conda create -n openenv python=3.10 -y
conda activate openenv
```

You should see:

```
(openenv)
```

---

## 2. Install Dependencies

Install required Python packages:

```
pip install fastapi uvicorn
pip install requests
```

---

## 3. VS Code Setup (Important)

Open the project folder in VS Code.

Then:

* Press `Ctrl + Shift + P`
* Search: `Python: Select Interpreter`
* Select:

  ```
  Python 3.10 (openenv)
  ```

---

## 4. Run Locally (Without Docker)

Start the API server:

```
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

In another terminal, run:

### Validator

```
python validator.py
```

Expected output:

```
All checks passed
```

### Baseline Agent

```
python baseline/run.py
```

---

## 5. Install Docker

Install Docker Desktop and restart your system.

Verify installation:

```
docker --version
docker run hello-world
```

---

## 6. Build Docker Image

From the project root directory:

```
docker build -t openenv .
```

---

## 7. Run Container

```
docker run -p 7860:7860 openenv
```

---

## 8. Validate Docker Setup

In a new terminal:

```
python validator.py
```

Expected:

```
All checks passed
```

---

## Notes

* Always activate the environment before running anything:

  ```
  conda activate openenv
  ```

* Do not use Git Bash for setup; prefer:

  * Command Prompt
  * PowerShell

* Ensure Docker is running before building.


## Status Check

If everything works:

* API runs ✔
* Validator passes ✔
* Baseline runs ✔
* Docker builds ✔
* Docker validation passes ✔

You are ready to proceed.
