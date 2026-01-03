#!/usr/bin/env python3
"""
GAME Environment Dataset Collection - GitHub Codespaces

Collects 1000 samples from the GAME (Goofspiel) environment only
and pushes to Hugging Face for RL training.

Usage:
  export CHUTES_API_KEY="your-api-key"
  export HF_TOKEN="your-hf-token"
  python collect_game_dataset.py
"""

import asyncio
import os
import json
import time
from typing import Dict, List, Any

# Configuration
TOTAL_SAMPLES = 1000
HF_DATASET_NAME = "Arielasgas/affine-game-dataset"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
CHUTES_API_KEY = os.environ.get("CHUTES_API_KEY", "")

# GAME environment configuration
GAME_CONFIG = {
    "image": "affinefoundation/game:openspiel",
    "mem_limit": "8g",
    "workers": 50,
}


def check_docker():
    """Check if Docker is available."""
    import subprocess
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=30)
        return result.returncode == 0
    except Exception:
        return False


def check_environment():
    """Check required environment variables."""
    if not CHUTES_API_KEY:
        print("[ERROR] CHUTES_API_KEY required. Set with: export CHUTES_API_KEY='your-key'")
        return False
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN required. Set with: export HF_TOKEN='your-token'")
        return False
    print("[OK] Environment variables configured")
    return True


async def collect_game_samples(target: int) -> List[Dict]:
    """Collect samples from GAME environment."""
    import affinetes as af
    
    samples = []
    print(f"\nCollecting {target} samples from GAME (Goofspiel)...")
    
    try:
        # Load GAME environment
        print("Starting GAME Docker container...")
        env = af.load_env(
            image=GAME_CONFIG["image"],
            replicas=1,
            mem_limit=GAME_CONFIG["mem_limit"],
            env_vars={
                "CHUTES_API_KEY": CHUTES_API_KEY,
                "UVICORN_WORKERS": str(GAME_CONFIG["workers"]),
            }
        )
        print("[OK] GAME container running")
        
        # Collect samples
        collected = 0
        task_id = 0
        errors = 0
        
        while collected < target and errors < 100:
            try:
                task = await env.get_task(task_id=task_id)
                
                if task and "prompt" in task:
                    sample = {
                        "prompt": task["prompt"],
                        "environment": "GAME",
                        "task_id": task_id,
                        "game_type": "goofspiel",
                        "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    
                    # Include any ground truth info
                    for key in ["expected", "answer", "ground_truth", "optimal_bid"]:
                        if key in task:
                            sample[key] = task[key]
                    
                    samples.append(sample)
                    collected += 1
                    
                    if collected % 100 == 0:
                        print(f"  Progress: {collected}/{target} samples")
                
                task_id += 1
                if task_id % 20 == 0:
                    await asyncio.sleep(0.05)
                    
            except Exception as e:
                errors += 1
                task_id += 1
                continue
        
        print(f"[OK] Collected {collected} GAME samples")
        await env.cleanup()
        
    except Exception as e:
        print(f"[ERROR] Collection failed: {e}")
        import traceback
        traceback.print_exc()
    
    return samples


def push_to_huggingface(samples: List[Dict]):
    """Push dataset to Hugging Face Hub."""
    from datasets import Dataset
    from huggingface_hub import login
    
    print(f"\nPushing {len(samples)} samples to Hugging Face...")
    
    login(token=HF_TOKEN)
    dataset = Dataset.from_list(samples)
    
    dataset.push_to_hub(
        HF_DATASET_NAME,
        token=HF_TOKEN,
        private=False,
    )
    
    print(f"[SUCCESS] Dataset: https://huggingface.co/datasets/{HF_DATASET_NAME}")


def main():
    print("=" * 50)
    print("GAME Environment Dataset Collection")
    print("=" * 50)
    print(f"Target: {TOTAL_SAMPLES} GAME samples")
    print(f"Dataset: {HF_DATASET_NAME}")
    print()
    
    if not check_environment():
        return
    if not check_docker():
        print("[ERROR] Docker not available")
        return
    
    print("[OK] Docker is running")
    
    # Collect samples
    start = time.time()
    samples = asyncio.run(collect_game_samples(TOTAL_SAMPLES))
    elapsed = (time.time() - start) / 60
    
    print(f"\nCollection complete in {elapsed:.1f} minutes")
    
    if len(samples) < 10:
        print("[ERROR] Not enough samples collected")
        return
    
    # Save locally
    with open("game_dataset.json", "w") as f:
        json.dump(samples, f, indent=2)
    print(f"[OK] Saved to game_dataset.json")
    
    # Push to HuggingFace
    push_to_huggingface(samples)
    
    print("\n" + "=" * 50)
    print("DONE! GAME dataset ready for training.")
    print("=" * 50)


if __name__ == "__main__":
    main()
