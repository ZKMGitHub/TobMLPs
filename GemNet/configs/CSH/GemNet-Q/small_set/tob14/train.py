import os
import logging
import datetime
starttime = datetime.datetime.now()
#long running

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["AUTOGRAPH_VERBOSITY"] = "1"

logger = logging.getLogger()
logger.handlers = []
ch = logging.StreamHandler()
formatter = logging.Formatter(
    fmt="%(asctime)s (%(levelname)s): %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel("INFO")

import numpy as np
import yaml
import string
import ast
import random
import time
from datetime import datetime

from gemnet.model.gemnet import GemNet
from gemnet.training.trainer import Trainer
from gemnet.training.metrics import Metrics, BestMetrics
from gemnet.training.data_container import DataContainer
from gemnet.training.data_provider import DataProvider

import torch
from torch.utils.tensorboard import SummaryWriter

with open('config.yaml', 'r') as c:
    config = yaml.safe_load(c)
    
# For strings that yaml doesn't parse (e.g. None)
for key, val in config.items():
    if type(val) is str:
        try:
            config[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass

num_spherical = config["num_spherical"]
num_radial = config["num_radial"]
num_blocks = config["num_blocks"]
emb_size_atom = config["emb_size_atom"]
emb_size_edge = config["emb_size_edge"]
emb_size_trip = config["emb_size_trip"]
emb_size_quad = config["emb_size_quad"]
emb_size_rbf = config["emb_size_rbf"]
emb_size_cbf = config["emb_size_cbf"]
emb_size_sbf = config["emb_size_sbf"]
num_before_skip = config["num_before_skip"]
num_after_skip = config["num_after_skip"]
num_concat = config["num_concat"]
num_atom = config["num_atom"]
emb_size_bil_quad = config["emb_size_bil_quad"]
emb_size_bil_trip = config["emb_size_bil_trip"]
triplets_only = config["triplets_only"]
forces_coupled = config["forces_coupled"]
direct_forces = config["direct_forces"]
mve = config["mve"]
cutoff = config["cutoff"]
int_cutoff = config["int_cutoff"]
envelope_exponent = config["envelope_exponent"]
extensive = config["extensive"]
output_init = config["output_init"]
scale_file = config["scale_file"]
data_seed = config["data_seed"]
dataset = config["dataset"]
val_dataset = config["val_dataset"]
num_train = config["num_train"]
num_val = config["num_val"]
logdir = config["logdir"]
loss = config["loss"]
tfseed = config["tfseed"]
num_steps = config["num_steps"]
rho_force = config["rho_force"]
ema_decay = config["ema_decay"]
weight_decay = config["weight_decay"]
grad_clip_max = config["grad_clip_max"]
agc = config["agc"]
decay_patience = config["decay_patience"]
decay_factor = config["decay_factor"]
decay_cooldown = config["decay_cooldown"]
batch_size = config["batch_size"]
evaluation_interval = config["evaluation_interval"]
patience = config["patience"]
save_interval = config["save_interval"]
learning_rate = config["learning_rate"]
warmup_steps = config["warmup_steps"]
decay_steps = config["decay_steps"]
decay_rate = config["decay_rate"]
staircase = config["staircase"]
restart = config["restart"]
comment = config["comment"]

torch.manual_seed(tfseed)

logging.info("Start training")
num_gpus = torch.cuda.device_count()
cuda_available = torch.cuda.is_available()
logging.info(f"Available GPUs: {num_gpus}")
logging.info(f"CUDA Available: {cuda_available}")
if num_gpus == 0:
    logging.warning("No GPUs were found. Training is run on CPU!")
if not cuda_available:
    logging.warning("CUDA unavailable. Training is run on CPU!")

# Used for creating a "unique" id for a run (almost impossible to generate the same twice)
def id_generator(
    size=6, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits
):
    return "".join(random.SystemRandom().choice(chars) for _ in range(size))

# A unique directory name is created for this run based on the input
if (restart is None) or (restart == "None"):
    directory = (
        logdir
        + "/"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + id_generator()
        + "_"
        + os.path.basename(dataset)
        + "_"
        + str(comment)
    )
else:
    directory = restart

logging.info(f"Directory: {directory}")
logging.info("Create directories")

if not os.path.exists(directory):
    os.makedirs(directory, exist_ok=True)

best_dir = os.path.join(directory, "best")
if not os.path.exists(best_dir):
    os.makedirs(best_dir)
log_dir = os.path.join(directory, "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

extension = ".pth"
log_path_model = f"{log_dir}/model{extension}"
log_path_training = f"{log_dir}/training{extension}"
best_path_model = f"{best_dir}/model{extension}"

logging.info("Initialize model")
model = GemNet(
    num_spherical=num_spherical,
    num_radial=num_radial,
    num_blocks=num_blocks,
    emb_size_atom=emb_size_atom,
    emb_size_edge=emb_size_edge,
    emb_size_trip=emb_size_trip,
    emb_size_quad=emb_size_quad,
    emb_size_rbf=emb_size_rbf,
    emb_size_cbf=emb_size_cbf,
    emb_size_sbf=emb_size_sbf,
    num_before_skip=num_before_skip,
    num_after_skip=num_after_skip,
    num_concat=num_concat,
    num_atom=num_atom,
    emb_size_bil_quad=emb_size_bil_quad,
    emb_size_bil_trip=emb_size_bil_trip,
    num_targets=2 if mve else 1,
    triplets_only=triplets_only,
    direct_forces=direct_forces,
    forces_coupled=forces_coupled,
    cutoff=cutoff,
    int_cutoff=int_cutoff,
    envelope_exponent=envelope_exponent,
    activation="swish",
    extensive=extensive,
    output_init=output_init,
    scale_file=scale_file,
)
# push to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train = {}
validation = {}
test = {}

logging.info("Load dataset")
data_container = DataContainer(
    dataset, cutoff=cutoff, int_cutoff=int_cutoff, triplets_only=triplets_only
)

if val_dataset is not None:
    # Initialize DataProvider
    if num_train == 0:
        num_train = len(data_container)
    logging.info(f"Training data size: {num_train}")
    data_provider = DataProvider(
        data_container,
        num_train,
        0,
        batch_size,
        seed=data_seed,
        shuffle=True,
        random_split=True,
    )

    # Initialize validation datasets
    val_data_container = DataContainer(
        val_dataset,
        cutoff=cutoff,
        int_cutoff=int_cutoff,
        triplets_only=triplets_only,
    )
    if num_val == 0:
        num_val = len(val_data_container)
    logging.info(f"Validation data size: {num_val}")
    val_data_provider = DataProvider(
        val_data_container,
        0,
        num_val,
        batch_size,
        seed=data_seed,
        shuffle=True,
        random_split=True,
    )
else:
    # Initialize DataProvider (splits dataset into 3 sets based on data_seed and provides tf.datasets)
    num_data = len(data_container)
    logging.info(f"Training data size: {num_train}")
    logging.info(f"Validation data size: {num_val}")
    num_test = num_data - num_train - num_val
    logging.info(f"Test data size: {num_test}")
    assert num_train > 0
    assert num_val > 0
    data_provider = DataProvider(
        data_container,
        num_train,
        num_val,
        batch_size,
        seed=data_seed,
        shuffle=True,
        #random_split=True,
        split = None,
    )
    val_data_provider = data_provider
    test_data_provider = data_provider


# Initialize datasets
train["dataset_iter"] = data_provider.get_dataset("train")
validation["dataset_iter"] = val_data_provider.get_dataset("val")
test["dataset_iter"] = test_data_provider.get_dataset("test")

logging.info("Prepare training")
# Initialize trainer
trainer = Trainer(
    model,
    learning_rate=learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    warmup_steps=warmup_steps,
    weight_decay=weight_decay,
    ema_decay=ema_decay,
    decay_patience=decay_patience,
    decay_factor=decay_factor,
    decay_cooldown=decay_cooldown,
    grad_clip_max=grad_clip_max,
    rho_force=rho_force,
    mve=mve,
    loss=loss,
    staircase=staircase,
    agc=agc,
)

# Initialize metrics
train["metrics"] = Metrics("train", trainer.tracked_metrics)
validation["metrics"] = Metrics("val", trainer.tracked_metrics)
test["metrics"] = Metrics("test", trainer.tracked_metrics)

# Save/load best recorded loss (only the best model is saved)
metrics_best = BestMetrics(best_dir, validation["metrics"])

# Set up checkpointing
# Restore latest checkpoint
if os.path.exists(log_path_model):
    logging.info("Restoring model and trainer")
    model_checkpoint = torch.load(log_path_model)
    model.load_state_dict(model_checkpoint["model"])

    train_checkpoint = torch.load(log_path_training)
    trainer.load_state_dict(train_checkpoint["trainer"])
    # restore the best saved results
    metrics_best.restore()
    logging.info(f"Restored best metrics: {metrics_best.loss}")
    step_init = int(train_checkpoint["step"])
else:
    logging.info("Freshly initialize model")
    metrics_best.inititalize()
    step_init = 0



summary_writer = SummaryWriter(log_dir)
steps_per_epoch = int(np.ceil(num_train / batch_size))

for step in range(step_init + 1, num_steps + 1):

    # keep track of the learning rate
    if step % 10 == 0:
        lr = trainer.schedulers[0].get_last_lr()[0]
        summary_writer.add_scalar("lr", lr, global_step=step)

    # Perform training step
    trainer.train_on_batch(train["dataset_iter"], train["metrics"])

    # Save progress
    if step % save_interval == 0:
        torch.save({"model": model.state_dict()}, log_path_model)
        torch.save(
            {"trainer": trainer.state_dict(), "step": step}, log_path_training
        )

    # Check performance on the validation set
    if step % evaluation_interval == 0:

        # Save backup variables and load averaged variables
        trainer.save_variable_backups()
        trainer.load_averaged_variables()

        # Compute averages
        for i in range(int(np.ceil(num_val / batch_size))):
            trainer.test_on_batch(validation["dataset_iter"], validation["metrics"])

        # Update and save best result
        if validation["metrics"].loss < metrics_best.loss:
            metrics_best.update(step, validation["metrics"])
            torch.save(model.state_dict(), best_path_model)

        # write to summary writer
        metrics_best.write(summary_writer, step)

        epoch = step // steps_per_epoch
        train_metrics_res = train["metrics"].result(append_tag=False)
        val_metrics_res = validation["metrics"].result(append_tag=False)
        metrics_strings = [
            f"{key}: train={train_metrics_res[key]:.6f}, val={val_metrics_res[key]:.6f}"
            for key in validation["metrics"].keys
        ]
        logging.info(
            f"{step}/{num_steps} (epoch {epoch}): " + "; ".join(metrics_strings)
        )

        # decay learning rate on plateau
        trainer.decay_maybe(validation["metrics"].loss)

        train["metrics"].write(summary_writer, step)
        validation["metrics"].write(summary_writer, step)
        train["metrics"].reset_states()
        validation["metrics"].reset_states()

        # Restore backup variables
        trainer.restore_variable_backups()

        # early stopping
        #if step - metrics_best.step > patience * evaluation_interval:
        #    break

result = {key + "_best": val for key, val in metrics_best.items()}

for key, val in metrics_best.items():
    print(f"{key}: {val}")

# Test dataset
trainer.save_variable_backups()
trainer.load_averaged_variables()

# Compute averages
for i in range(int(np.ceil(num_test / batch_size))):
    trainer.test_on_batch(test["dataset_iter"], test["metrics"])



# write to summary writer
metrics_best.write(summary_writer, step)

epoch = step // steps_per_epoch
test_metrics_res = test["metrics"].result(append_tag=False)
test_metrics_strings = [
    f"{key}:  test={test_metrics_res[key]:.6f}"
    for key in test["metrics"].keys
]
logging.info(
    f"Test metrics: " + "; ".join(test_metrics_strings)
)

import datetime
endtime = datetime.datetime.now()
print("wall time per epoch:" + str((endtime - starttime).seconds / 1000) + "s")