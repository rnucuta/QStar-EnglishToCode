import os

import torch

import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    # Implement this if you wish to use wandb in your experiments
    try:
        wandb.login()
    except Exception as e:
        print("Error logging into wandb: {}".format(e))
        return
    wandb.init(
        project="Tasks_1-2",
        name=args.experiment_name,
        config={
            "finetune": args.finetune,
            "optimizer_type": args.optimizer_type,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "scheduler_type": args.scheduler_type,
            "num_warmup_epochs": args.num_warmup_epochs,
            "max_n_epochs": args.max_n_epochs,
            "patience_epochs": args.patience_epochs,
            "batch_size": args.batch_size,
            "test_batch_size": args.test_batch_size,
        },
    )

def initialize_model(args):
    '''
    Helper function to initialize the model. You should be either finetuning
    the pretrained model associated with the 'google-t5/t5-small' checkpoint
    or training a T5 model initialized with the 'google-t5/t5-small' config
    from scratch.
    '''
    try:
        if args.finetune:
            model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
        else:
            config = T5Config.from_pretrained('google-t5/t5-small')
            model = T5ForConditionalGeneration(config)

        if args.use_wandb:
            wandb.watch(model, log_freq=100)

        model.to(DEVICE)

    except Exception as e:
        print("Error initializing model: {}".format(e))
        model = None
    
    return model

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best):
    # Save model checkpoint to be able to load the model later
    if best:
        model.save_pretrained(os.path.join(checkpoint_dir, 'best'))
    else:
        model.save_pretrained(os.path.join(checkpoint_dir, 'latest'))

def load_model_from_checkpoint(args, best):
    # Load model from a checkpoint
    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    if best:
        model = T5ForConditionalGeneration.from_pretrained(os.path.join(checkpoint_dir, 'best'))
    else:
        model = T5ForConditionalGeneration.from_pretrained(os.path.join(checkpoint_dir, 'latest'))
    model.to(DEVICE)
    return model

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        pass

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

