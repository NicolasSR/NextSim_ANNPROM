# from nn_trainer import NN_Trainer
from nn_trainer import NN_Trainer
from sys import argv


def train(working_path, ae_config):
    training_routine=NN_Trainer(working_path, ae_config)
    training_routine.execute_training()
    del training_routine

if __name__ == "__main__":

    train_configs_list = [
   {
        "name": None,
        "architecture": {
            "q_inf_size": 20,
            "q_sup_size": 100,
            "hidden_layers": [200,200],
            "opt_strategy": {
                "learning_rate": ('sgdr', 0.001, 1e-4, 400, 10), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001), ('tri2', 0.001, 1e-6, 250)
                "batch_size": 4,
                "epochs": 800
            },
            "finetune_from": None
        },
        "dataset_path": 'rom_data_reduced/',
        "models_path_root": 'saved_models_nextsim_residual/',
        "project_parameters_file":'ProjectParametersPrimalROM.json'
   }
   ]

    working_path=argv[1]+"/"
    
    for i, train_config in enumerate(train_configs_list):
        
        print('----------  Training case ', i+1, ' of ', len(train_configs_list), '  ----------')
        train(working_path, train_config)
    
    # compss_barrier()
    print('FINISHED TRAINING')

        
