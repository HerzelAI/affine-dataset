#!/usr/bin/env python3
"""
Balanced Affine Dataset Collection - GAME Focus with PRINT & LGC-V2 Retention

Strategy:
- 600 GAME samples (60%) - Focus on improvement
- 200 PRINT samples (20%) - Maintain performance
- 200 LGC-V2 samples (20%) - Maintain performance

This prevents catastrophic forgetting while prioritizing GAME improvement.

Usage:
  export CHUTES_API_KEY="your-api-key"
  export HF_TOKEN="your-hf-token"
  python collect_balanced_dataset.py
"""

import asyncio
import os
import json
import time
from typing import Dict, List, Any

# Configuration
HF_DATASET_NAME = "Arielasgas/affine-balanced-dataset"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
CHUTES_API_KEY = os.environ.get("CHUTES_API_KEY", "")

# Balanced distribution: GAME-heavy but includes others for retention
ENVS = {
    "GAME": {
        "image": "affinefoundation/game:openspiel",
        "samples": 600,  # 60% - Focus on improvement
        "mem_limit": "8g",
        "workers": 50,
    },
    "PRINT": {
        "image": "affinefoundation/cde:print",
        "samples": 200,  # 20% - Maintain performance
        "mem_limit": "10g",
        "workers": 15,
    },
    "LGC-V2": {
        "image": "affinefoundation/lgc:pi-v2",
        "samples": 200,  # 20% - Maintain performance
        "mem_limit": "20g",
        "workers": 15,
    },
}

TOTAL_SAMPLES = sum(cfg["samples"] for cfg in ENVS.values())


def check_docker():
    """Check if Docker is available."""
    import subprocess
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=30)
        return result.returncode == 0
    except Exception:
        return False


def pull_images():
    """Pull required Docker images."""
    import subprocess
    
    images = [
        "affinefoundation/game:openspiel",
        "affinefoundation/cde:print",
        "affinefoundation/lgc:pi-v2",
    ]
    
    print("\nPulling Docker images (this may take a few minutes)...")
    for image in images:
        print(f"  Pulling {image}...")
        try:
            result = subprocess.run(
                ["docker", "pull", image],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode == 0:
                print(f"  [OK] {image}")
            else:
                print(f"  [WARN] {image}: {result.stderr[:100]}")
        except Exception as e:
            print(f"  [ERROR] {image}: {e}")
    
    print()


def check_environment():
    """Check required environment variables."""
    if not CHUTES_API_KEY:
        print("[ERROR] CHUTES_API_KEY required")
        return False
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN required")
        return False
    print("[OK] Environment configured")
    return True


async def collect_from_env(env_name: str, config: Dict, target: int) -> List[Dict]:
    """Collect samples from a single environment."""
    import affinetes as af
    
    samples = []
    print(f"\n{'='*40}")
    print(f"Collecting {target} samples from {env_name}")
    print(f"{'='*40}")
    
    try:
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
        
        # Wait for container to initialize
        print(f"  Waiting for {env_name} to initialize...")
        await asyncio.sleep(5)
        
        # List available methods
        try:
            methods = await env.methods() if hasattr(env, 'methods') else []
            print(f"  Available methods: {methods}")
        except Exception as e:
            print(f"  Could not list methods: {e}")
        
        collected = 0
        task_id = 0
        errors = 0
        last_error = None
        
        while collected < target and errors < 100 and task_id < target + 500:
            try:
                # Try get_task first
                if hasattr(env, 'get_task'):
                    task = await env.get_task(task_id=task_id)
                elif hasattr(env, 'sample'):
                    task = await env.sample(task_id=task_id)
                elif hasattr(env, 'get_prompt'):
                    task = await env.get_prompt(task_id=task_id)
                else:
                    # Try calling directly
                    task = await env.call("get_task", task_id=task_id)
                
                if task:
                    # Handle different response formats
                    prompt = None
                    if isinstance(task, dict):
                        prompt = task.get("prompt") or task.get("question") or task.get("input")
                    elif isinstance(task, str):
                        prompt = task
                    
                    if prompt:
                        sample = {
                            "prompt": prompt,
                            "environment": env_name,
                            "task_id": task_id,
                            "weight": 1.5 if env_name == "GAME" else 1.0,
                            "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        
                        if isinstance(task, dict):
                            for key in ["expected", "answer", "ground_truth", "solution"]:
                                if key in task:
                                    sample[key] = task[key]
                        
                        samples.append(sample)
                        collected += 1
                        
                        if collected % 50 == 0:
                            print(f"  {env_name}: {collected}/{target}")
                    else:
                        errors += 1
                else:
                    errors += 1
                
                task_id += 1
                if task_id % 20 == 0:
                    await asyncio.sleep(0.05)
                    
            except Exception as e:
                last_error = str(e)
                errors += 1
                task_id += 1
        
        if collected == 0 and last_error:
            print(f"  Last error: {last_error[:200]}")
        
        print(f"[OK] Collected {collected} from {env_name} (errors: {errors})")
        await env.cleanup()
        
    except Exception as e:
        print(f"[ERROR] {env_name}: {e}")
        import traceback
        traceback.print_exc()
    
    return samples


async def collect_all() -> List[Dict]:
    """Collect from all environments."""
    all_samples = []
    
    for name, config in ENVS.items():
        samples = await collect_from_env(name, config, config["samples"])
        all_samples.extend(samples)
        await asyncio.sleep(2)
    
    return all_samples


def push_to_huggingface(samples: List[Dict]):
    """Push to HuggingFace."""
    from datasets import Dataset
    from huggingface_hub import login
    
    login(token=HF_TOKEN)
    dataset = Dataset.from_list(samples)
    
    # Show distribution
    print("\nDataset Distribution:")
    for env in ENVS.keys():
        count = len([s for s in samples if s["environment"] == env])
        print(f"  {env}: {count} samples")
    
    dataset.push_to_hub(HF_DATASET_NAME, token=HF_TOKEN, private=False)
    print(f"\n[SUCCESS] https://huggingface.co/datasets/{HF_DATASET_NAME}")


def main():
    print("=" * 50)
    print("Balanced Dataset: GAME Focus + PRINT/LGC-V2 Retention")
    print("=" * 50)
    print(f"Distribution: GAME=600, PRINT=200, LGC-V2=200")
    print(f"Total: {TOTAL_SAMPLES} samples")
    print()
    
    if not check_environment() or not check_docker():
        return
    
    # Pull Docker images first
    pull_images()
    
    start = time.time()
    samples = asyncio.run(collect_all())
    elapsed = (time.time() - start) / 60
    
    print(f"\nCollection complete in {elapsed:.1f} minutes")
    print(f"Total: {len(samples)} samples")
    
    if len(samples) < 100:
        print("[ERROR] Not enough samples")
        return
    
    # Save backup
    with open("balanced_dataset.json", "w") as f:
        json.dump(samples, f, indent=2)
    
    push_to_huggingface(samples)
    
    print("\n" + "=" * 50)
    print("DONE! Balanced dataset ready for training.")
    print("Strategy: Improve GAME, maintain PRINT & LGC-V2")
    print("=" * 50)


if __name__ == "__main__":
    main()
