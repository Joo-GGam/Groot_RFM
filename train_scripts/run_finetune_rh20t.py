#!/usr/bin/env python3
"""
Finetune GR00T-N1.5 with RH20T dataset including Force/Torque data.

This script finetunes the pre-trained GR00T-N1.5 model on the RH20T dataset
with Force/Torque sensor data from Franka Panda robot.
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
EMBODIMENT_CONFIG = "rh20t_franka_with_ft"  # RH20T with F/T!

# What to finetune
TUNE_LLM = False
TUNE_VISUAL = False
TUNE_PROJECTOR = True  # Includes force encoder!
TUNE_DIFFUSION_MODEL = True

# Dataset configuration
DATASET_PATH = "./datasets/rh20t_cfg5_for_gr00t"
DATASET_VIDEO_BACKEND = "decord"

# Model configuration
MODEL_COMPUTE_DTYPE = "bfloat16"

# Output configuration
FINETUNED_OUTPUT_DIRECTORY = "./output/rh20t_franka_ft_ver2"
RUN_NAME = "rh20t_franka_ft"

# Training hyperparameters
BATCH_SIZE = 4
MAX_STEPS = 180000
SAVE_STEPS = 30000
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-4

# ============================================================================
# Main Function
# ============================================================================

def main():
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("RH20T Franka Force/Torque Finetuning")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Embodiment Config: {EMBODIMENT_CONFIG}")
    print(f"Dataset Path: {DATASET_PATH}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Max Steps: {MAX_STEPS}")
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
        tune_projector=TUNE_PROJECTOR,  # Includes force encoder!
        tune_diffusion_model=TUNE_DIFFUSION_MODEL,
    )

    # Set compute dtype FIRST (before creating force encoder)
    model.compute_dtype = MODEL_COMPUTE_DTYPE
    model.config.compute_dtype = MODEL_COMPUTE_DTYPE

    # Initialize force encoder (pretrained model doesn't have it)
    if not hasattr(model.action_head, 'force_encoder') or model.action_head.force_encoder is None:
        print("Initializing force encoder...")
        from gr00t.model.action_head.flow_matching_action_head import CategorySpecificMLP

        # Create force encoder with bfloat16 dtype to match pretrained model
        force_dim = 6  # 6D F/T sensor (f_R_x, f_R_y, f_R_z, f_p_x, f_p_y, f_p_z)
        model.action_head.force_encoder = CategorySpecificMLP(
            num_categories=model.action_head.config.max_num_embodiments,
            input_dim=force_dim,
            hidden_dim=model.action_head.hidden_size,
            output_dim=model.action_head.input_embedding_dim,
        ).to(dtype=model.action_head.dtype)

        # Update config (both action_head config and model config for saving)
        model.action_head.config.force_dim = force_dim
        model.config.action_head_cfg['force_dim'] = force_dim  # For save_pretrained

        # Re-apply trainable parameter settings (to include force_encoder)
        model.action_head.set_trainable_parameters(
            tune_projector=TUNE_PROJECTOR,
            tune_diffusion_model=TUNE_DIFFUSION_MODEL
        )

        print(f"✓ Force encoder initialized")
        print(f"  - Input dim: {force_dim}")
        print(f"  - Output dim: {model.action_head.input_embedding_dim}")
    else:
        print(f"✓ Force encoder already exists")
        print(f"  - Input dim: {model.action_head.config.force_dim}")
        print(f"  - Output dim: {model.action_head.force_encoder.output_dim}")

    # Move to device (dtype conversion handled automatically via compute_dtype)
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
