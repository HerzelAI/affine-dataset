#!/usr/bin/env python3
"""
Affine Dataset Collection Script for GitHub Codespaces

This script:
1. Deploys Affine Docker environments (LGC-V2, PRINT, GAME)
2. Collects 1000 real samples (prompts + evaluations)
3. Pushes the dataset to Hugging Face

Usage in Codespaces:
1. Set environment variables:
   export CHUTES_API_KEY="your-api-key"
   export HF_TOKEN="your-hf-token"
   
2. Run the script:
   python collect_affine_dataset.py
"""

import asyncio
import os
import json
import time
from typing import Dict, List, Any
from pathlib import Path

# Configuration
TOTAL_SAMPLES = 1000
HF_DATASET_NAME = "Arielasgas/affine-training-dataset"
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # Set via: export HF_TOKEN="your-token"
CHUTES_API_KEY = os.environ.get("CHUTES_API_KEY", "")

# Environment configurations
ENVS = {
    "LGC-V2": {
        "image": "affinefoundation/lgc:pi-v2",
        "samples": 400,  # 40% of samples
        "mem_limit": "20g",
        "workers": 15,
    },
    "PRINT": {
        "image": "affinefoundation/cde:print",
        "samples": 300,  # 30% of samples
        "mem_limit": "10g",
        "workers": 15,
    },
    "GAME": {
        "image": "affinefoundation/game:openspiel",
        "samples": 300,  # 30% of samples
        "mem_limit": "8g",
        "workers": 50,
    },
}


def check_docker():
    """Check if Docker is available."""
    import subprocess
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=30)
        if result.returncode == 0:
            print("[OK] Docker is available")
            return True
        else:
            print("[ERROR] Docker not running properly")
            return False
    except Exception as e:
        print(f"[ERROR] Docker check failed: {e}")
        return False


def check_environment():
    """Check required environment variables."""
    if not CHUTES_API_KEY:
        print("[ERROR] CHUTES_API_KEY environment variable is required")
        print("Set it with: export CHUTES_API_KEY='your-api-key'")
        return False
    print("[OK] CHUTES_API_KEY is set")
    
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN environment variable is required")
        print("Set it with: export HF_TOKEN='your-token'")
        return False
    print("[OK] HF_TOKEN is set")
    
    return True


async def collect_samples_from_env(env_name: str, config: Dict, target_samples: int) -> List[Dict]:
    """Collect samples from a single Affine environment."""
    import affinetes as af
    
    samples = []
    print(f"\n{'='*50}")
    print(f"Collecting {target_samples} samples from {env_name}")
    print(f"{'='*50}")
    
    try:
        # Load environment with Docker
        print(f"Loading {env_name} Docker container...")
        env = af.load_env(
            image=config["image"],
            replicas=1,
            mem_limit=config["mem_limit"],
            env_vars={
                "CHUTES_API_KEY": CHUTES_API_KEY,
                "UVICORN_WORKERS": str(config["workers"]),
            }
        )
        
        print(f"[OK] {env_name} container started")
        
        # Collect task prompts
        collected = 0
        task_id = 0
        errors = 0
        max_errors = 50  # Stop if too many errors
        
        while collected < target_samples and errors < max_errors:
            try:
                # Get task from environment
                task = await env.get_task(task_id=task_id)
                
                if task and "prompt" in task:
                    sample = {
                        "prompt": task["prompt"],
                        "environment": env_name,
                        "task_id": task_id,
                        "metadata": {
                            "image": config["image"],
                            "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    }
                    
                    # Try to get ground truth or expected format info
                    if "expected" in task:
                        sample["expected"] = task["expected"]
                    if "answer" in task:
                        sample["answer"] = task["answer"]
                    if "ground_truth" in task:
                        sample["ground_truth"] = task["ground_truth"]
                    
                    samples.append(sample)
                    collected += 1
                    
                    if collected % 50 == 0:
                        print(f"  {env_name}: {collected}/{target_samples} samples collected")
                
                task_id += 1
                
                # Small delay to avoid overwhelming the container
                if task_id % 10 == 0:
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                errors += 1
                if errors % 10 == 0:
                    print(f"  Warning: {errors} errors so far (last: {str(e)[:50]})")
                task_id += 1
                continue
        
        print(f"[OK] Collected {collected} samples from {env_name}")
        
        # Cleanup
        await env.cleanup()
        print(f"[OK] {env_name} container stopped")
        
    except Exception as e:
        print(f"[ERROR] Failed to collect from {env_name}: {e}")
        import traceback
        traceback.print_exc()
    
    return samples


async def collect_all_samples() -> List[Dict]:
    """Collect samples from all environments."""
    all_samples = []
    
    for env_name, config in ENVS.items():
        samples = await collect_samples_from_env(env_name, config, config["samples"])
        all_samples.extend(samples)
        
        # Brief pause between environments
        await asyncio.sleep(2)
    
    return all_samples


def save_dataset_locally(samples: List[Dict], output_path: str = "affine_dataset.json"):
    """Save dataset to local JSON file."""
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"[OK] Saved {len(samples)} samples to {output_path}")
    return output_path


def push_to_huggingface(samples: List[Dict]):
    """Push dataset to Hugging Face Hub."""
    from datasets import Dataset
    from huggingface_hub import login
    
    print(f"\nPushing {len(samples)} samples to Hugging Face...")
    
    # Login
    login(token=HF_TOKEN)
    print("[OK] Logged into Hugging Face")
    
    # Create dataset
    dataset = Dataset.from_list(samples)
    print(f"[OK] Created dataset with {len(dataset)} samples")
    
    # Show sample distribution
    env_counts = {}
    for sample in samples:
        env = sample.get("environment", "unknown")
        env_counts[env] = env_counts.get(env, 0) + 1
    
    print("\nDataset distribution:")
    for env, count in env_counts.items():
        print(f"  {env}: {count} samples")
    
    # Push to Hub
    dataset.push_to_hub(
        HF_DATASET_NAME,
        token=HF_TOKEN,
        private=False,
    )
    
    print(f"\n[SUCCESS] Dataset pushed to: https://huggingface.co/datasets/{HF_DATASET_NAME}")


def main():
    print("="*60)
    print("Affine Dataset Collection")
    print("="*60)
    print(f"Target: {TOTAL_SAMPLES} samples")
    print(f"Destination: {HF_DATASET_NAME}")
    print()
    
    # Check prerequisites
    if not check_environment():
        return
    
    if not check_docker():
        return
    
    # Collect samples
    print("\nStarting sample collection...")
    start = time.time()
    
    samples = asyncio.run(collect_all_samples())
    
    elapsed = time.time() - start
    print(f"\n[OK] Collection complete in {elapsed/60:.1f} minutes")
    print(f"Total samples: {len(samples)}")
    
    if len(samples) == 0:
        print("[ERROR] No samples collected. Check your CHUTES_API_KEY and try again.")
        return
    
    # Save locally as backup
    save_dataset_locally(samples)
    
    # Push to Hugging Face
    push_to_huggingface(samples)
    
    print("\n" + "="*60)
    print("DONE! Dataset is ready for training.")
    print("="*60)


if __name__ == "__main__":
    main()
