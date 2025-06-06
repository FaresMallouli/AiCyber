# SENTINEL-AI: Real-Time Network Threat Detection & Explainability Framework


This repository contains the real-time inference component of the **SENTINEL-AI** project, submitted for The Fursah AI Competition - Tunisia Edition. SENTINEL-AI is a multi-perspective AI framework designed for holistic cyber threat demystification and real-time interception.

## 📊 Dataset Information

This project was trained using the **Network Intrusion Detection Dataset** available on Kaggle:
- **Dataset Link:** [https://www.kaggle.com/datasets/bcccdatasets/network-intrusion-detection](https://www.kaggle.com/datasets/bcccdatasets/network-intrusion-detection)
- The dataset contains comprehensive network traffic data for training cybersecurity threat detection models

## 🚀 Features

The `inference.py` script leverages pre-trained models to analyze live network traffic (or PCAP files) using NFStream, providing classifications for:

1. **Threat Type** (e.g., Benign, DDoS, Botnet, Web Attacks)
2. **Operator Type** (Human vs. Bot)
3. **Source Category** (e.g., ISP/Residential, Cloud, VPN/Proxy)

It also provides real-time SHAP explanations for the Threat Type and Operator Type predictions.

## 📁 Directory Structure

For the `inference.py` script to run correctly, your project should have the following directory structure:

```
Your_Project_Root/
├── inference.py                    # The main real-time inference script
├── models/                         # Directory for all trained model artifacts
│   ├── main_threat_detector/       # Artifacts for the Main Threat Detector
│   │   ├── main_detector_rf_nfstream_ctgan_FSTrue_RSSMOTE_RUS.joblib
│   │   ├── web_detector_rf_nfstream_ctgan_FSTrue_RSSMOTE_RUS.joblib
│   │   ├── label_encoder_rf_nfstream_ctgan_FSTrue_RSSMOTE_RUS.joblib
│   │   ├── main_model_scaler_rf_nfstream_ctgan_FSTrue_RSSMOTE_RUS.joblib
│   │   └── selected_features_main_model_rf_nfstream_ctgan_FSTrue_RSSMOTE_RUS.joblib
│   ├── operator_id_model/          # Artifacts for the Operator ID Model
│   │   ├── operator_id_model_cpu_sklearn_rf_FS_True.joblib
│   │   ├── operator_id_scaler_cpu_sklearn_rf_FS_True.joblib
│   │   └── operator_id_selected_features.joblib
│   └── source_category_classifier/ # Artifacts for the Source Classifier Model
│       ├── source_classifier_pipeline.joblib
│       └── source_classifier_label_encoder.joblib
├── external_data/                  # Directory for IP enrichment data
│   ├── GeoLite2-ASN-Blocks-IPv4.csv
│   ├── amazon.json
│   ├── ServiceTags_Public_20250505.json
│   ├── gcp.json
│   ├── NordVPN-Server-IP-List.txt
│   ├── tor-exit-nodes.lst
│   └── tor-nodes.lst
└── README.md                       # This file
```

> **Note:** The model artifacts are **NOT included directly** in this repository due to size constraints. You must download them separately from the provided Google Drive link below.

## 🔧 Prerequisites

### Required Software

- **Conda (Anaconda or Miniconda)** - *Highly recommended* for environment management
  - [Download Anaconda](https://www.anaconda.com/download)
  - [Download Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- **Python 3.9** (automatically installed with Conda environment)



## 🛠️ Setup Instructions

### 1. Clone the Repository

```cmd
git clone https://github.com/FaresMallouli/AiCyber.git
cd AiCyber
```

### 2. Download Model Artifacts

> **⚠️ Critical Step: Models are required for the system to work**

The trained models are stored separately due to GitHub size limitations:

1. **Download the models ZIP file:**
   - Visit: [https://drive.google.com/file/d/1j18cP8NM9PUBkBd2K3DHoYPmxqhmPK7H/view](https://drive.google.com/file/d/1j18cP8NM9PUBkBd2K3DHoYPmxqhmPK7H/view)
   - Click "Download" to save the ZIP file to your computer

2. **Extract the models:**
   - Right-click the downloaded ZIP file
   - Select "Extract All..." or use your preferred extraction tool
   - Extract the contents to your project root directory

3. **Verify the structure:**
   After extraction, ensure your directory looks like this:
   ```
   AiCyber/
   ├── inference.py
   ├── models/                    # ← This folder should now exist
   │   ├── main_threat_detector/
   │   ├── operator_id_model/
   │   └── source_category_classifier/
   ├── external_data/             
   └── README.md
   ```

### 3. Install Npcap (Critical Step)

> **⚠️ Essential for Windows users intending live capture with NFStream**

1. Visit [https://npcap.com/#download](https://npcap.com/#download)
2. Download the **Npcap 1.82 installer** (or latest version)
3. Run installer **as Administrator**
4. Follow the installation prompts

![Npcap Download Options](assets/npcap.jpg)



### 4. Create Conda Environment

Open **Anaconda Prompt** (or **Command Prompt** if Miniconda) and run:

```cmd
# Create the environment with required packages
conda create -n sentinel_ai_env python=3.9 scikit-learn=1.2.2 numpy scipy joblib pandas lightgbm requests intervaltree -c conda-forge -y

# Activate the environment
conda activate sentinel_ai_env

# Install remaining packages via pip
pip install shap dill nfstream scapy
```

### 5. Verify Data and Model Artifacts

Ensure that the `models/` and `external_data/` directories contain all necessary files as listed in the Directory Structure section.

## 🚀 Running the Script

### Activate Environment and Navigate

```cmd
# Activate the Conda environment
conda activate sentinel_ai_env

# Navigate to project directory
cd path\to\Your_Project_Root\

# Run the script
python inference.py
```

### Providing NFStream Source

The script will prompt you to enter the NFStream source:

```
Enter NFStream source (e.g., interface name or path to PCAP file) :
```

#### For Live Network Capture:
- Enter your active network interface name (e.g., `Wi-Fi`, `Ethernet`)
- **Important:** Run **Command Prompt as Administrator** for live capture:
  ```cmd
  # Right-click Command Prompt → "Run as administrator"
  conda activate sentinel_ai_env
  python inference.py
  ```

#### For PCAP File Analysis:
- Enter the full or relative path to your capture file (e.g., `my_captures\test_traffic.pcap`)

#### Finding Your Network Interface Name:
You can find your network interface names using:
```cmd
# Method 1: Using ipconfig
ipconfig

# Method 2: Using PowerShell
Get-NetAdapter | Select-Object Name, InterfaceDescription

# Method 3: Using netsh
netsh interface show interface
```

![Exemple:](assets/interface.png)

## 📊 Example Output

```
--- Processing Flow: 10.0.0.5:12345 -> 1.2.3.4:6667 (Proto: 6, App: unknown) ---
  Main Threat Prediction: Botnet_ARES (Index: 1)
    SHAP Explanation for Main RF Detector (Predicted Class Index: 1):
      - fwd_packets_IAT_min: 0.2345
      - duration: 0.1987
      - total_payload_bytes: 0.1502
      - fwd_packets_count: -0.1200
      - syn_flag_counts: 0.0987
  Operator ID Prediction: Bot (Index: 1)
    SHAP Explanation for Operator ID RF (Predicted Class Index: 1):
      - packets_IAT_mean: 0.3110
      - duration: 0.2550
      - active_mean: -0.1800
      - fwd_init_win_bytes: 0.1520
      - bwd_packets_count: 0.1105
  Source Category Prediction (for 10.0.0.5): Cloud (ASN: 67890 'CloudProviderX', Cloud: True, Proxy: False)
--- Predictions for Flow: 10.0.0.5:12345 -> 1.2.3.4:6667 (Proto: 6, App: unknown) ---
  Threat Type: Botnet_ARES
  Operator: Bot
  Source Category: Cloud
--------------------------------------
```

Press `Ctrl+C` to stop the script.

## 🔧 Troubleshooting

### Common Windows Issues

| Issue | Solution |
|-------|----------|
| `conda: command not found` | Install Conda and restart Command Prompt |
| Environment activation issues | Use Anaconda Prompt instead of regular Command Prompt |
| `ModuleNotFoundError` | Activate `sentinel_ai_env` and reinstall missing packages |
| Permission denied for live capture | Run Command Prompt as Administrator |
| `FileNotFoundError` for models | Verify directory structure and file presence |

### NFStream Windows Errors

- **`ImportError: Npcap/Winpcap/Raw sockets not available...`**
  - Download and install Npcap from [https://npcap.com/#download](https://npcap.com/#download)
  - Choose the **Npcap 1.82 installer** for your Windows version
  - Ensure you run the installer as Administrator

- **Interface not found**
  - Use correct interface name (check with `ipconfig` or `Get-NetAdapter`)
  - Try using the full interface description name
  - Ensure Npcap is properly installed

- **Access denied during live capture**
  - Run Command Prompt as Administrator
  - Ensure Windows Defender/Antivirus isn't blocking the application

### Installation Issues

- **Conda environment creation fails**
  - Update conda: `conda update conda`
  - Try creating environment without version specifications
  - Use `conda-forge` channel: `conda create -n sentinel_ai_env -c conda-forge python=3.9`

- **pip install fails**
  - Update pip: `python -m pip install --upgrade pip`
  - Install packages individually if batch install fails
  - Use `--user` flag if permission issues occur

## 📋 Model Artifacts 

The script requires:
- **Model Artifacts:** Pre-trained `.joblib` files in `models\` subdirectories (downloaded from Google Drive)


These are outputs from the SENTINEL-AI training pipeline trained on the [Network Intrusion Detection Dataset](https://www.kaggle.com/datasets/bcccdatasets/network-intrusion-detection) and must be downloaded from the provided Google Drive link.

## ⚠️ Inference Startup Notice

When launching the app for the first time, **please wait up to 2 minutes** for the environment to initialize.  
The detection process only begins **after you open the browser interface** — initial load time may be slightly delayed but normal.


## 🏆 Competition

This project was developed for **The Fursah AI Competition - Tunisia Edition**, focusing on advanced cybersecurity threat detection and explainable AI.



**SENTINEL-AI** - *Demystifying Cyber Threats with AI*