#!/usr/bin/env python3


from gears import PertData, GEARS

# get data
pert_data = PertData('./data')
# load dataset in paper: norman, adamson, dixit.
pert_data.load(data_name = 'norman')
# specify data split
pert_data.prepare_split(split = 'simulation', seed = 1)
# get dataloader with batch size
pert_data.get_dataloader(batch_size = 32, test_batch_size = 128)

# set up and train a model
gears_model = GEARS(pert_data, device = 'cuda')
gears_model.model_initialize(hidden_size = 64)
gears_model.train(epochs = 20)

# save/load model
gears_model.save_model('gears')
gears_model.load_pretrained('gears')

# predict
gears_model.predict([['CBL', 'CNN1'], ['FEV']])
gears_model.GI_predict(['CBL', 'CNN1'], GI_genes_file=None)
