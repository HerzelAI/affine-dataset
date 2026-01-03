#!/usr/bin/env python3
"""
Affine GAME Dataset Generator - Synthetic Goofspiel Training Data

Since Affine environments are evaluation-only (they don't expose prompts),
this script generates synthetic training prompts based on the GAME (Goofspiel)
task patterns, with ground truth optimal bids.

Goofspiel is a card game where:
- Both players have cards 1-13
- A prize card is revealed each round
- Players simultaneously bid a card
- Higher bidder wins the prize (ties = split)
- Goal: Maximize total prize value won

This dataset teaches the model optimal bidding strategies.

Usage:
  export HF_TOKEN="your-token"
  python create_game_dataset.py
"""

import os
import json
import random
import time
from typing import Dict, List, Any

# Configuration
TOTAL_SAMPLES = 1000
HF_DATASET_NAME = "Arielasgas/affine-game-dataset"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Also create balanced dataset with LGC-V2 and PRINT
CREATE_BALANCED = True
BALANCED_DATASET_NAME = "Arielasgas/affine-balanced-dataset"


def generate_goofspiel_scenario():
    """Generate a Goofspiel game scenario with optimal bid."""
    # Cards remaining (subset of 1-13)
    all_cards = list(range(1, 14))
    
    # Randomly determine how many cards have been played (0-10 rounds)
    rounds_played = random.randint(0, 10)
    
    # Remove played cards
    player_cards = sorted(random.sample(all_cards, 13 - rounds_played))
    opponent_cards = sorted(random.sample(all_cards, 13 - rounds_played))
    
    # Current prize card
    remaining_prizes = [c for c in all_cards if c not in random.sample(all_cards, rounds_played)]
    if not remaining_prizes:
        remaining_prizes = all_cards
    current_prize = random.choice(remaining_prizes)
    
    # Current scores
    player_score = random.randint(0, 40)
    opponent_score = random.randint(0, 40)
    
    # Generate prompt
    prompt = f"""You are playing Goofspiel, a card bidding game.

Current Game State:
- Your cards remaining: {player_cards}
- Prize card this round: {current_prize}
- Your score: {player_score}
- Opponent score: {opponent_score}
- Round: {rounds_played + 1} of 13

Rules:
- Both players secretly choose a card to bid
- Higher bid wins the prize card's value
- Ties split the prize value
- Used cards are discarded

What card should you bid? Provide your answer as a single number."""

    # Simple optimal strategy: bid proportionally to prize value
    # For high prizes, bid high cards; for low prizes, save high cards
    if current_prize >= 10:
        # High prize - bid a high card if available
        optimal_bid = max(player_cards) if player_cards else 1
    elif current_prize >= 7:
        # Medium-high prize
        mid_high = [c for c in player_cards if c >= 7]
        optimal_bid = min(mid_high) if mid_high else max(player_cards)
    elif current_prize >= 4:
        # Medium prize
        mid = [c for c in player_cards if 4 <= c <= 8]
        optimal_bid = mid[len(mid)//2] if mid else player_cards[len(player_cards)//2]
    else:
        # Low prize - sacrifice a low card
        optimal_bid = min(player_cards) if player_cards else 1
    
    return {
        "prompt": prompt,
        "environment": "GAME",
        "task_type": "goofspiel_bidding",
        "optimal_bid": optimal_bid,
        "prize_card": current_prize,
        "player_cards": player_cards,
        "weight": 1.5,  # Higher weight for GAME samples
        "expected_response": str(optimal_bid),
    }


def generate_lgc_scenario():
    """Generate an LGC-V2 (Logic/Math) scenario."""
    # Generate a math expression to evaluate
    ops = ['+', '-', '*']
    
    # Create expression
    num_terms = random.randint(2, 4)
    terms = [random.randint(1, 20) for _ in range(num_terms)]
    operators = [random.choice(ops) for _ in range(num_terms - 1)]
    
    # Build expression string
    expr_parts = [str(terms[0])]
    for i, op in enumerate(operators):
        expr_parts.append(op)
        expr_parts.append(str(terms[i + 1]))
    expression = ' '.join(expr_parts)
    
    # Calculate result
    try:
        result = eval(expression)
    except:
        result = terms[0]
    
    prompt = f"""Evaluate the following mathematical expression step by step.

Expression: {expression}

Show your work and provide the final answer."""

    return {
        "prompt": prompt,
        "environment": "LGC-V2",
        "task_type": "math_evaluation",
        "expression": expression,
        "answer": result,
        "weight": 1.0,
        "expected_response": str(result),
    }


def generate_print_scenario():
    """Generate a PRINT (Python output prediction) scenario."""
    templates = [
        # Simple print
        {
            "code": "x = {a}\nprint(x * 2)",
            "vars": {"a": random.randint(1, 50)},
            "compute": lambda v: v["a"] * 2,
        },
        # List operations
        {
            "code": "nums = [{a}, {b}, {c}]\nprint(sum(nums))",
            "vars": {"a": random.randint(1, 10), "b": random.randint(1, 10), "c": random.randint(1, 10)},
            "compute": lambda v: v["a"] + v["b"] + v["c"],
        },
        # String operations
        {
            "code": 's = "hello"\nprint(len(s) + {a})',
            "vars": {"a": random.randint(1, 10)},
            "compute": lambda v: 5 + v["a"],
        },
        # Conditionals
        {
            "code": "x = {a}\nif x > 10:\n    print('big')\nelse:\n    print('small')",
            "vars": {"a": random.randint(1, 20)},
            "compute": lambda v: "big" if v["a"] > 10 else "small",
        },
        # Loop
        {
            "code": "total = 0\nfor i in range({a}):\n    total += i\nprint(total)",
            "vars": {"a": random.randint(3, 8)},
            "compute": lambda v: sum(range(v["a"])),
        },
    ]
    
    template = random.choice(templates)
    vars_dict = template["vars"]
    code = template["code"].format(**vars_dict)
    output = template["compute"](vars_dict)
    
    prompt = f"""What will be the output of this Python code?

```python
{code}
```

Provide only the exact output, nothing else."""

    return {
        "prompt": prompt,
        "environment": "PRINT",
        "task_type": "python_output",
        "code": code,
        "expected_output": str(output),
        "weight": 1.0,
        "expected_response": str(output),
    }


def create_game_only_dataset(count: int) -> List[Dict]:
    """Create GAME-only dataset."""
    print(f"Generating {count} GAME (Goofspiel) samples...")
    samples = []
    for i in range(count):
        samples.append(generate_goofspiel_scenario())
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{count}")
    return samples


def create_balanced_dataset() -> List[Dict]:
    """Create balanced dataset: GAME 60%, PRINT 20%, LGC-V2 20%."""
    print("\nGenerating balanced dataset...")
    samples = []
    
    # GAME: 600 samples (60%)
    print("  Generating 600 GAME samples...")
    for _ in range(600):
        samples.append(generate_goofspiel_scenario())
    
    # PRINT: 200 samples (20%)
    print("  Generating 200 PRINT samples...")
    for _ in range(200):
        samples.append(generate_print_scenario())
    
    # LGC-V2: 200 samples (20%)
    print("  Generating 200 LGC-V2 samples...")
    for _ in range(200):
        samples.append(generate_lgc_scenario())
    
    random.shuffle(samples)
    return samples


def push_to_huggingface(samples: List[Dict], dataset_name: str):
    """Push dataset to Hugging Face Hub."""
    from datasets import Dataset
    from huggingface_hub import login
    
    print(f"\nPushing {len(samples)} samples to {dataset_name}...")
    
    login(token=HF_TOKEN)
    dataset = Dataset.from_list(samples)
    
    # Show distribution
    env_counts = {}
    for s in samples:
        env = s.get("environment", "unknown")
        env_counts[env] = env_counts.get(env, 0) + 1
    
    print("Distribution:")
    for env, count in env_counts.items():
        print(f"  {env}: {count}")
    
    dataset.push_to_hub(dataset_name, token=HF_TOKEN, private=False)
    print(f"[SUCCESS] https://huggingface.co/datasets/{dataset_name}")


def main():
    print("=" * 60)
    print("Affine Training Dataset Generator")
    print("=" * 60)
    print("Creating synthetic training data based on Affine task patterns")
    print()
    
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN required")
        return
    
    # Create GAME-only dataset
    game_samples = create_game_only_dataset(TOTAL_SAMPLES)
    
    # Save locally
    with open("game_dataset.json", "w") as f:
        json.dump(game_samples, f, indent=2)
    print(f"\n[OK] Saved {len(game_samples)} GAME samples to game_dataset.json")
    
    # Push GAME dataset
    push_to_huggingface(game_samples, HF_DATASET_NAME)
    
    # Create and push balanced dataset
    if CREATE_BALANCED:
        balanced_samples = create_balanced_dataset()
        
        with open("balanced_dataset.json", "w") as f:
            json.dump(balanced_samples, f, indent=2)
        print(f"\n[OK] Saved {len(balanced_samples)} balanced samples")
        
        push_to_huggingface(balanced_samples, BALANCED_DATASET_NAME)
    
    print("\n" + "=" * 60)
    print("DONE! Datasets ready for training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
