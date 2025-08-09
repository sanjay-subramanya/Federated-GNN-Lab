# Federated GNN Lab

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A playground for training and analyzing simple Graph Neural Networks in a federated learning environment using the TCGA-BRCA dataset. With this interactive laboratory, you can dissect the decision-making processes of each of the individual models real time, aided by visualization tools.

![Federated Learning](https://img.shields.io/badge/Federated%20Learning-Enabled-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-blue)
![Next.js](https://img.shields.io/badge/Next.js-15.4+-blue)

## 📊 Features

- **Federated Learning Simulation**: Track training across multiple clients
- **Dataset Overview**: Analyze the dataset using violin plots
- **Patient-level Predictions**: Make predictions on individual patient data
- **Client Divergence Analysis**: Visualize how client models differ from the global model
- **Feature Importance**: Identify which features contribute most to predictions
- **UMAP Visualization**: Explore high-dimensional data clustering in 2D space


## 📋 Prerequisites

- Python 3.10+
- Node.js 18+ and npm
- Git

## 🚀 Getting Started

### Clone the Repository

```bash
git clone https://github.com/sanjay-subramanya/Federated-GNN-Lab.git
cd Federated-GNN-Lab
```

### Install Dependencies

1. **Backend Dependencies**

```bash
pip install -r backend/requirements.txt
```

2. **Frontend Dependencies**

```bash
cd frontend
npm install
cd ..
```

## 🔧 Running the Application

### Backend Server (API Endpoints)

1. Navigate to the backend directory:

```bash
cd backend
```

2. Start the FastAPI server:

```bash
uvicorn router:app --reload
```

The server will start at port 8000 with API documentation available at http://127.0.0.1:8000/docs.


### Frontend Visualizer

1. Navigate to the frontend directory:

```bash
cd frontend
```

2. Start the development server:

```bash
npm run dev
```

The frontend will be available at http://localhost:3000. Ensure the backend server first is running to enable all features.

### Standalone Training Mode

To run FL training without the frontend interface:

```bash
cd backend
python -m backend.main
```

#### Configuration Options

The application supports two federated learning modes, which can be configured in `backend/config/settings.py`:

- **Flower Simulation**: Set `flower_simulation = True`  to use the Flower (flwr) framework
- **Manual Simulation**: Set `flower_simulation = False` for a simplified implementation

