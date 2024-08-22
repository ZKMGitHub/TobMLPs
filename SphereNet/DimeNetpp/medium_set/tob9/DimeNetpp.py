import torch
import yaml
import ast
#from dig.threedgraph.dataset import QM93D
from dig.threedgraph.dataset import MD17
from dig.threedgraph.dataset.PygTobermorite import Tobermorite
from dig.threedgraph.method import DimeNetPP #SchNet, DimeNetPP, ComENet
from dig.threedgraph.method import run
from dig.threedgraph.evaluation import ThreeDEvaluator

with open('config.yaml', 'r') as c:
    config = yaml.safe_load(c)
    
# For strings that yaml doesn't parse (e.g. None)
for key, val in config.items():
    if type(val) is str:
        try:
            config[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass

name = config['name']
n_train = config['n_train']
n_val = config['n_val']
seed = config['seed']
energy_and_force = config['energy_and_force']
cutoff = config['cutoff']
num_layers = config['num_layers']
hidden_channels = config['hidden_channels']
out_channels = config['out_channels']
int_emb_size = config['int_emb_size']
basis_emb_size = config['basis_emb_size']
out_emb_channels = config['out_emb_channels']
num_spherical = config['num_spherical']
num_radial = config['num_radial']
envelope_exponent = config['envelope_exponent']
num_before_skip = config['num_before_skip']
num_after_skip = config['num_after_skip']
num_output_layers = config['num_output_layers']

epochs = config['epochs']
batch_size = config['batch_size']
vt_batch_size = config['vt_batch_size']
lr = config['lr']
lr_decay_factor = config['lr_decay_factor']
lr_decay_step_size = config['lr_decay_step_size']
save_dir = config['save_dir']
log_dir= config['log_dir']

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
device

dataset = Tobermorite(root='dataset/', name=name)

split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=n_train, valid_size=n_val, seed=seed)

train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
print('train, validaion, test:', len(train_dataset), len(valid_dataset), len(test_dataset))

model = DimeNetPP(energy_and_force=energy_and_force, cutoff=cutoff, num_layers=num_layers, 
        hidden_channels=hidden_channels, out_channels=out_channels, int_emb_size=int_emb_size, 
        basis_emb_size=basis_emb_size, out_emb_channels=out_emb_channels, 
        num_spherical=num_spherical, num_radial=num_radial, envelope_exponent=envelope_exponent, 
        num_before_skip=num_before_skip, num_after_skip=num_after_skip, num_output_layers=num_output_layers 
        )
loss_func = torch.nn.L1Loss()
evaluation = ThreeDEvaluator()

run3d = run()
run3d.run(device, train_dataset, valid_dataset, test_dataset, model,
               loss_func, evaluation, epochs=epochs, batch_size=batch_size, vt_batch_size=vt_batch_size,
               lr=lr, lr_decay_factor=lr_decay_factor, lr_decay_step_size=lr_decay_step_size, 
               energy_and_force=energy_and_force, save_dir = save_dir, log_dir = log_dir)