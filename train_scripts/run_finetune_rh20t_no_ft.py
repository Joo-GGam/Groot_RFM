#!/usr/bin/env python3
"""
Finetune GR00T-N1.5 with RH20T dataset WITHOUT Force/Torque data.

This script finetunes the pre-trained GR00T-N1.5 model on the RH20T dataset
WITHOUT Force/Torque sensor data. Used for ablation study to measure F/T contribution.
"""

import warnings
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.data.schema import EmbodimentTag
from gr00t.data.dataset import LeRobotSingleDataset
import torch
from gr00t.model.gr00t_n1 import GR00T_N1_5
from transformers import TrainingArguments
from gr00t.experiment.runner import TrainRunner


# ============================================================================
# Configuration
# ============================================================================

# Pre-trained model
PRE_TRAINED_MODEL_PATH = "nvidia/GR00T-N1.5-3B"

# Embodiment configuration
EMBODIMENT_TAG = EmbodimentTag.OXE_DROID  # Single arm + EEF control
EMBODIMENT_CONFIG = "rh20t_franka_no_ft"  # RH20T WITHOUT F/T!

# What to finetune
TUNE_LLM = False
TUNE_VISUAL = False
TUNE_PROJECTOR = True  # NO force encoder in this version
TUNE_DIFFUSION_MODEL = True

# Dataset configuration
DATASET_PATH = "./datasets/rh20t_cfg5_for_gr00t"
DATASET_VIDEO_BACKEND = "decord"

# Model configuration
MODEL_COMPUTE_DTYPE = "bfloat16"

# Output configuration
FINETUNED_OUTPUT_DIRECTORY = "./output/rh20t_franka_no_ft"
RUN_NAME = "rh20t_franka_no_ft"

# Training hyperparameters
BATCH_SIZE = 4
MAX_STEPS = 40000
SAVE_STEPS = 4000
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4

# ============================================================================
# Main Function
# ============================================================================

def main():
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("RH20T Franka Finetuning WITHOUT Force/Torque")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Embodiment Config: {EMBODIMENT_CONFIG}")
    print(f"Dataset Path: {DATASET_PATH}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Max Steps: {MAX_STEPS}")
    print(f"Output Dir: {FINETUNED_OUTPUT_DIRECTORY}")
    print("=" * 80)

    warnings.simplefilter("ignore", category=FutureWarning)

    # Load RH20T NO-FT configuration
    print("\n[1/5] Loading data configuration...")
    data_config = DATA_CONFIG_MAP[EMBODIMENT_CONFIG]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    # Verify NO force modality
    assert "force" not in modality_config, (
        f"Force modality found in {EMBODIMENT_CONFIG}! "
        "This should be the NO-FT version."
    )
    print(f"✓ NO Force modality (ablation study)")

    # Create dataset
    print("\n[2/5] Loading dataset...")
    train_dataset = LeRobotSingleDataset(
        dataset_path=DATASET_PATH,
        modality_configs=modality_config,
        embodiment_tag=EMBODIMENT_TAG,
        video_backend=DATASET_VIDEO_BACKEND,
        video_backend_kwargs=None,
        transforms=modality_transform,
    )
    print(f"✓ Loaded dataset with {len(train_dataset)} samples")

    # Load pre-trained model
    print("\n[3/5] Loading pre-trained model...")
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=PRE_TRAINED_MODEL_PATH,
        tune_llm=TUNE_LLM,
        tune_visual=TUNE_VISUAL,
        tune_projector=TUNE_PROJECTOR,
        tune_diffusion_model=TUNE_DIFFUSION_MODEL,
    )

    # Set compute dtype
    model.compute_dtype = MODEL_COMPUTE_DTYPE
    model.config.compute_dtype = MODEL_COMPUTE_DTYPE

    # NO force encoder initialization (key difference!)
    print("✓ NO Force encoder (using state-only input)")

    # Move to device
    model.to(device)
    print(f"✓ Model moved to {device} (compute dtype: {MODEL_COMPUTE_DTYPE})")

    # Training arguments
    print("\n[4/5] Setting up training...")
    training_args = TrainingArguments(
        output_dir=FINETUNED_OUTPUT_DIRECTORY,
        overwrite_output_dir=True,
        run_name=RUN_NAME,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=LEARNING_RATE,
        weight_decay=1e-5,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        num_train_epochs=1,
        max_steps=MAX_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        report_to="tensorboard",
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    # Create trainer
    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
    )

    print("\n[5/5] Starting training...")
    print("=" * 80)

    # Start training
    experiment.train()

    print("\n" + "=" * 80)
    print("✅ Training completed!")
    print("=" * 80)
    print(f"\nModel saved to: {FINETUNED_OUTPUT_DIRECTORY}")
    print("=" * 80)

if __name__ == "__main__":
    main()
