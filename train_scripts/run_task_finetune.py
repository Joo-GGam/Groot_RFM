#!/usr/bin/env python3
"""
Task-Specific Finetuning: Erase Task with Force/Torque data.

This script finetunes the RH20T pre-finetuned GR00T-N1.5 model
on the Erase task dataset with Force/Torque sensor data.

Pre-trained model: ./output/rh20t_franka_ft (already has force encoder)
Dataset: ./datasets/Erase_for_gr00t
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

# Pre-trained model (RH20T finetuned with force encoder)
PRE_TRAINED_MODEL_PATH = "/home/Isaac/Isaac-GR00T/output/rh20t_franka_ft_v3"

# Embodiment configuration (same as RH20T finetuning)
EMBODIMENT_TAG = EmbodimentTag.OXE_DROID  # Single arm + EEF control
EMBODIMENT_CONFIG = "rh20t_franka_with_ft"  # RH20T with F/T

# What to finetune (same as base finetuning)
TUNE_LLM = False
TUNE_VISUAL = False
TUNE_PROJECTOR = True
TUNE_DIFFUSION_MODEL = False # Freeze diffusion for task-specific finetuning

# Dataset configuration - Erase task
DATASET_PATH = "/home/Isaac/Isaac-GR00T/datasets/Erase_for_gr00t"
DATASET_VIDEO_BACKEND = "decord"

# Model configuration
MODEL_COMPUTE_DTYPE = "bfloat16"

# Output configuration
FINETUNED_OUTPUT_DIRECTORY = "/home/Isaac/Isaac-GR00T/output/erase_task_ft_v2"
RUN_NAME = "erase_task_ft"

# Training hyperparameters (lower LR for task-specific finetuning)
BATCH_SIZE = 8
MAX_STEPS = 15000
SAVE_STEPS = 2000
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-5  # Lower LR for task-specific finetuning

# ============================================================================
# Main Function
# ============================================================================

def main():
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Task-Specific Finetuning: Erase Task with Force/Torque")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Base Model: {PRE_TRAINED_MODEL_PATH}")
    print(f"Embodiment Config: {EMBODIMENT_CONFIG}")
    print(f"Dataset Path: {DATASET_PATH}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Max Steps: {MAX_STEPS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Output Dir: {FINETUNED_OUTPUT_DIRECTORY}")
    print("=" * 80)

    warnings.simplefilter("ignore", category=FutureWarning)

    # Load RH20T F/T configuration
    print("\n[1/5] Loading data configuration...")
    data_config = DATA_CONFIG_MAP[EMBODIMENT_CONFIG]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    # Verify force modality exists
    assert "force" in modality_config, (
        f"Force modality not found in {EMBODIMENT_CONFIG}. "
        "Make sure you're using 'rh20t_franka_with_ft' config."
    )
    print(f"✓ Force modality detected: {data_config.force_keys}")

    # Create dataset
    print("\n[2/5] Loading Erase task dataset...")
    train_dataset = LeRobotSingleDataset(
        dataset_path=DATASET_PATH,
        modality_configs=modality_config,
        embodiment_tag=EMBODIMENT_TAG,
        video_backend=DATASET_VIDEO_BACKEND,
        video_backend_kwargs=None,
        transforms=modality_transform,
    )
    print(f"✓ Loaded dataset with {len(train_dataset)} samples")

    # Load pre-finetuned model (already has force encoder)
    print("\n[3/5] Loading pre-finetuned model from RH20T...")
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

    # Verify force encoder exists (should already be in the pre-finetuned model)
    if hasattr(model.action_head, 'force_encoder') and model.action_head.force_encoder is not None:
        print(f"✓ Force encoder loaded from pre-finetuned model")
        print(f"  - Input dim: {model.action_head.config.force_dim}")
        print(f"  - Output dim: {model.action_head.force_encoder.layer2.W.shape[2]}")
    else:
        # Fallback: initialize force encoder if not found
        print("⚠ Force encoder not found, initializing...")
        from gr00t.model.action_head.flow_matching_action_head import CategorySpecificMLP

        force_dim = 6
        model.action_head.force_encoder = CategorySpecificMLP(
            num_categories=model.action_head.config.max_num_embodiments,
            input_dim=force_dim,
            hidden_dim=model.action_head.hidden_size,
            output_dim=model.action_head.input_embedding_dim,
        ).to(dtype=model.action_head.dtype)

        model.action_head.config.force_dim = force_dim
        model.config.action_head_cfg['force_dim'] = force_dim  # For save_pretrained

        model.action_head.set_trainable_parameters(
            tune_projector=TUNE_PROJECTOR,
            tune_diffusion_model=TUNE_DIFFUSION_MODEL
        )
        print(f"✓ Force encoder initialized")

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

    print("\n[5/5] Starting task-specific finetuning...")
    print("=" * 80)

    # Start training
    experiment.train()

    print("\n" + "=" * 80)
    print("✅ Task-specific finetuning completed!")
    print("=" * 80)
    print(f"\nModel saved to: {FINETUNED_OUTPUT_DIRECTORY}")
    print("=" * 80)

if __name__ == "__main__":
    main()
