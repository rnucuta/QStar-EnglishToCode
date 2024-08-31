# Set your project name
PROJECT_NAME="Tasks_1-2"

# Set the run ID you want to keep
RUN_TO_KEEP="g2bewolj"

# List all runs, filter out the run to keep, and delete the rest
wandb runs list --project $PROJECT_NAME --format=json | \
jq -r '.[] | select(.id != "'$RUN_TO_KEEP'") | .id' | \
xargs -I {} wandb runs delete {} --project $PROJECT_NAME --yes
