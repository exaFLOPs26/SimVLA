#!/bin/bash
# run_random_kitchen.sh

SEED=42
TASK_NUM=$(( (SEED + RANDOM) % 10 + 1 ))
TASK_NAME="Isaac-Kitchen-v$TASK_NUM"

echo "Running with task: $TASK_NAME"
LIVESTREAM=2 ./isaaclab.sh -p scripts/motion_planning/whole_body_curobo.py --enable_cameras --headless --task "$TASK_NAME"

