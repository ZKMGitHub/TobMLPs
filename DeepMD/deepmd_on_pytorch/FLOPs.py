import logging
import torch

from typing import Any, Dict

from deepmd_pt import my_random
from deepmd_pt.dataset import DeepmdDataSet
from deepmd_pt.learning_rate import LearningRateExp
from deepmd_pt.loss import EnergyStdLoss
from deepmd_pt.model import EnergyModel

from deepmd_pt.embedding_net import EmbeddingNet
from deepmd_pt.fitting import EnergyFittingNet
from deepmd_pt.stat import compute_output_stats, make_stat_input, merge_sys_stat
from thop import profile


model_params =  {
	"type_map":	["Ca", "Si", "O", "H"],
	"descriptor" :{
	    "type":		"se_e2_a",
	    "sel":		[128, 128, 128, 128],
	    "rcut_smth":	0.50,
	    "rcut":		4.00,
	    "neuron":		[20, 40, 80],
	    "resnet_dt":	False,
	    "axis_neuron":	4,
	    "seed":		1,
	    "_comment":		" that's all"
	},
	"fitting_net" : {
	    "neuron":		[200, 200, 200],
	    "resnet_dt":	True,
	    "seed":		1,
	    "_comment":		" that's all"
	},
        "data_stat_nbatch": 5,
	"_comment":	" that's all"
}

training_params = {
	"training_data": {
	    "systems":     ["../benchmark_data/CSH/Tobermorite_9A/train"],
	    "batch_size":  4,
	    "_comment":	   "that's all"
	},
	"validation_data":{
	    "systems":	   ["../benchmark_data/CSH/Tobermorite_9A/val"],
	    "batch_size":  4,
	    "numb_btch":   1,
	    "_comment":	   "that's all"
	},
	"numb_steps":	237500,
	"seed":		10,
	"disp_file":	"lcurve.out",
	"disp_freq":	475,
	"save_freq":	2375,
	"_comment":	"that's all"
}

my_random.seed(training_params['seed'])
dataset_params = training_params.pop('training_data')

training_data = DeepmdDataSet(
           systems=dataset_params['systems'],
           batch_size=dataset_params['batch_size'],
           type_map=model_params['type_map']
       )

model = EnergyModel(model_params, training_data)

bdata = training_data.get_batch()
coord = torch.from_numpy(bdata['coord'])
atype = bdata['type']
natoms = bdata['natoms_vec']
box = bdata['box']
coord.requires_grad_(True)
# p_energy, p_force = model(coord, atype, natoms, box)

print("---------------------Start compute FLOPs----------------------")

from torchstat import stat

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

param = count_param(model)
print("Parameters: " + str(param))

print("--------------------------------------------------------------------")


flops,params=profile(model, inputs = (coord, atype, natoms, box, ), )
print(flops,params)