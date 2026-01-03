# Affine Dataset Collection

Collect real training samples from Affine environments and push to Hugging Face.

## Quick Start (GitHub Codespaces)

### 1. Open in Codespaces

Click the button below or open this repo in GitHub Codespaces:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new)

### 2. Set Environment Variables

In the Codespaces terminal:

```bash
export CHUTES_API_KEY="your-chutes-api-key"
export HF_TOKEN="your-huggingface-token"
```

### 3. Run Collection Script

```bash
python collect_affine_dataset.py
```

This will:
- Deploy LGC-V2, PRINT, and GAME Docker containers
- Collect 1000 samples total (400 LGC-V2, 300 PRINT, 300 GAME)
- Push dataset to Hugging Face: `Arielasgas/affine-training-dataset`

## Files

- `.devcontainer/devcontainer.json` - Codespaces config with Docker-in-Docker
- `collect_affine_dataset.py` - Main collection script
- `deploy_envs.sh` - Manual deployment script (alternative)

## Requirements

- GitHub Codespaces (free tier: 60 hours/month)
- Chutes API Key (from affine.io)
- Hugging Face Token

## Estimated Time

- Collection: ~30-60 minutes for 1000 samples
- Uses ~4-6 GB RAM (well within Codespaces limits)
