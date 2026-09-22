# ☁️ WaferPulse — AWS Cloud Deployment & Sign-In Guide

> **Goal:** Run WaferPulse 24/7 on AWS EC2, accessible from any device (phone, laptop, iPad), with secure sign-in and automated auto-restart. Follow this step-by-step guide (~30-45 minutes).

---

## 📋 Overview of System Deployment Architecture

| Component | Technology | Purpose |
|---|---|---|
| **Cloud Host** | AWS EC2 (Ubuntu 24.04 LTS) | Always-on virtual machine in the cloud |
| **Instance Type** | `t3.large` (2 vCPUs, 8 GB RAM) | Required to run SHAP explainability and Scikit-learn models |
| **Security Group** | Port `8501` open to `0.0.0.0/0` | Enables web browser access from anywhere |
| **Daemon Manager** | Linux `systemd` (`waferpulse.service`) | Auto-starts on boot, auto-restarts on unexpected errors |
| **Sign-In Gate** | `waferpulse.core.auth` | Role-based password authentication (admin, engineer, auditor) |

---

## Step 1 — Launch AWS EC2 Instance

1. Log into your **AWS Management Console** ([aws.amazon.com](https://aws.amazon.com/)).
2. In the top search bar, type **EC2** and click **Launch instance**.
3. Fill in the instance details:
   - **Name:** `waferpulse-server`
   - **Application and OS Images:** **Ubuntu Server 24.04 LTS** (64-bit x86)
   - **Instance Type:** **`t3.large`** *(8 GB RAM is necessary for ML model inferences and SHAP calculations)*
   - **Key pair (login):** Click *Create new key pair* → Name: `waferpulse-key` → Type: **RSA**, Private key format: **`.pem`** → Click *Create key pair* (saves `waferpulse-key.pem` to your PC).
   - **Network Settings (Firewall):**
     - Check: *Allow SSH traffic from* → Select **My IP**
     - Click **Add security group rule**:
       - **Type:** Custom TCP
       - **Port range:** `8501`
       - **Source type:** Anywhere (`0.0.0.0/0`)
   - **Configure Storage:** Change `8 GiB` to **`30 GiB`** gp3 SSD.
4. Click **Launch Instance**.
5. Go to EC2 Instances view, wait until status is **Running**, and copy your **Public IPv4 address** (e.g., `54.255.120.45`).

*(Optional: Under EC2 → Elastic IPs, allocate an Elastic IP and associate it with this instance so the IP address remains fixed across server reboots).*

---

## Step 2 — Build & Upload the Code Package

Open **PowerShell** on your local Windows PC in your project directory:

```powershell
# 1. Package the slim deployment zip (automatically excludes heavy temp files, videos, and PDFs)
powershell -ExecutionPolicy Bypass -File deploy\make_deploy_zip.ps1

# 2. Upload the zip to your EC2 instance via SCP
scp -i "C:\path\to\waferpulse-key.pem" "..\waferpulse_deploy.zip" ubuntu@<YOUR-EC2-PUBLIC-IP>:~
```
*(Replace `C:\path\to\waferpulse-key.pem` with your actual key location, and `<YOUR-EC2-PUBLIC-IP>` with your EC2 IP).*

---

## Step 3 — Install & Launch on the Server

Connect to your EC2 instance via SSH:
```powershell
ssh -i "C:\path\to\waferpulse-key.pem" ubuntu@<YOUR-EC2-PUBLIC-IP>
```

Once connected to Ubuntu Linux, execute:
```bash
# 1. Unzip the deployment package
sudo apt-get install -y unzip
unzip -q waferpulse_deploy.zip -d waferpulse_app
cd waferpulse_app

# 2. Run the automated one-shot setup script
bash deploy/aws_setup.sh
```

The script will:
- Install Python 3.11 and Linux build tools
- Set up a dedicated virtual environment `.venv`
- Install all dependencies from `requirements.txt`
- Register and start the `systemd` 24/7 service `waferpulse.service`

---

## Step 4 — Sign In from Anywhere

Open your browser from **any device** (phone, tablet, or home PC):

```text
http://<YOUR-EC2-PUBLIC-IP>:8501
```

### 🔑 Default Sign-In Credentials

| Role | Username | Default Password | Permissions |
|---|---|---|---|
| **Quality Lead Administrator** | `admin` | `admin123` | Full access to pipeline runs, models, and analytics |
| **Fab Process Engineer** | `engineer` | `engineer123` | Single-wafer and batch yield prediction & verification |
| **Quality Auditor** | `auditor` | `auditor123` | Read-only inspection of reliability health certificates |

*(To change passwords or add new team members, edit `config/users.json` on the server).*

---

## Step 5 — Managing the Cloud Service

Useful commands when SSH'd into the server:

```bash
# View live application output and prediction logs:
sudo journalctl -u waferpulse -f

# Check if the service is active and healthy:
sudo systemctl status waferpulse

# Restart the service after making any code edits:
sudo systemctl restart waferpulse

# Stop the service:
sudo systemctl stop waferpulse
```

---

## Troubleshooting

| Problem | Root Cause | Solution |
|---|---|---|
| Webpage does not load | Port 8501 is closed in AWS Security Group | Go to AWS Console → EC2 → Security Groups → Inbound Rules → Ensure Port `8501` is open to `0.0.0.0/0`. |
| SSH "Permission denied" | Wrong `.pem` file permissions | Ensure you are logging in as user `ubuntu` with the exact path to `waferpulse-key.pem`. |
| Out of Memory error (`Killed`) | Instance RAM is insufficient | Ensure you chose `t3.large` (8 GB RAM). `t3.micro` or `t3.small` will run out of memory during model training/SHAP computation. |
