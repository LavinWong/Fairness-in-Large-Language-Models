import os
import sys
from transformers import HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint

def last_checkpoint_handling(training_args, model_args):
    """HF logic for getting last checkpoint/overwriting an existing run
    """
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    checkpoint = None
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    # elif os.path.isdir(model_args.model_name_or_path):
    #     checkpoint = model_args.model_name_or_path
    return checkpoint