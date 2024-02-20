import os
import logging

# logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import json
import numpy as np
import scipy
import pickle

import tensorflow as tf
# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')

import logging
tf.get_logger().setLevel(logging.ERROR)

from PrePostProcessors.PODANN_prepost_processors import SVD_White_NoStand_PODANN_PrePostProcessor
from OptimizationStrategies.r_only_strategy_batches_crop import R_Only_Strategy_Cropped_KerasModel

from Kratos_Simulators.compressible_potential_flow_kratos_simulator import CompressiblePotentialFlow_KratosSimulator

from tensorflow.keras.initializers import HeNormal
# from Schedulers.lr_scheduler import get_lr_schedule_func, LR_Scheduler

tf.keras.backend.set_floatx('float64')

class NN_Trainer():
    def __init__(self,working_path,train_config):
        self.working_path=working_path
        self.train_config=train_config
        self.arch_config=self.train_config["architecture"]

    def generate_model_name_part(self):
        opt_strategy_config=self.arch_config["opt_strategy"]
        name_part='PODANN_ronly_'
        name_part+='Lay'+str(self.arch_config["hidden_layers"])+'_'
        name_part+='Emb'+str(self.arch_config["q_inf_size"])+'.'+str(self.arch_config["q_sup_size"])+'_'
        name_part+='LR'+str(opt_strategy_config["learning_rate"][0])+str(opt_strategy_config["learning_rate"][1])
        return name_part

    def setup_output_directory(self):
        model_name = self.generate_model_name_part()
        if self.train_config["name"] is None:
            self.model_path=self.working_path+self.train_config["models_path_root"]
            self.model_path+=model_name
        else:
            self.model_path=self.working_path+self.train_config["models_path_root"]+self.train_config["name"]

        while os.path.isdir(self.model_path+'/'):
            self.model_path+='_bis'
        self.model_path+='/'

        print('Created Model directory at: ', self.model_path)
        
        os.makedirs(self.model_path, exist_ok=False)
        os.makedirs(self.model_path+"best/", exist_ok=True)
        os.makedirs(self.model_path+"last/", exist_ok=True)

        return self.model_path
    
    def get_orig_fom_snapshots(self):
        dataset_path=self.train_config['dataset_path']
        S_FOM_orig=np.load(self.working_path+dataset_path+'FOM.npy')
        return S_FOM_orig
    
    def define_network(self, prepost_processor, kratos_simulation):

        input_size = self.arch_config["q_inf_size"]
        output_size = self.arch_config["q_sup_size"]-self.arch_config["q_inf_size"]

        decod_input = tf.keras.Input(shape=(input_size,), dtype=tf.float64)

        decoder_out = decod_input
        for layer_size in self.arch_config["hidden_layers"]:
            decoder_out = tf.keras.layers.Dense(layer_size, activation='elu', kernel_initializer=HeNormal(), use_bias=False, dtype=tf.float64)(decoder_out)
        decoder_out = tf.keras.layers.Dense(output_size, activation=tf.keras.activations.linear, kernel_initializer=HeNormal(), use_bias=False, dtype=tf.float64)(decoder_out)

        network = R_Only_Strategy_Cropped_KerasModel(prepost_processor, kratos_simulation, decod_input, decoder_out, name='q_sup_estimator')

        network.compile(optimizer=tf.keras.optimizers.experimental.AdamW(epsilon=1e-17), run_eagerly=network.run_eagerly)

        network.summary()

        return network
    
    def train_network(self, model, input_data, target_data, val_input, val_target):
        
        model.update_rescaling_factors(target_data[0], target_data[1])

        checkpoint_best_r_callback = tf.keras.callbacks.ModelCheckpoint(self.model_path+"best/weights_r_best.h5",save_weights_only=True,save_best_only=True,monitor="val_loss_r",mode="min")
        checkpoint_last_callback = tf.keras.callbacks.ModelCheckpoint(self.model_path+"last/weights.h5",save_weights_only=True,save_freq="epoch")
        lr_scheduler_callback = self.get_custom_LR_scheduler()
        csv_logger_callback = tf.keras.callbacks.CSVLogger(self.model_path+"train_log.csv", separator=',', append=False)

        if  not "batch_size" in self.arch_config["opt_strategy"]:
            self.arch_config["opt_strategy"]["batch_size"] = 1
        if  not "epochs" in self.arch_config["opt_strategy"]:
            self.arch_config["opt_strategy"]["epochs"] = 1

        history = model.fit(
            input_data, target_data,
            epochs=self.arch_config["opt_strategy"]["epochs"],
            shuffle=True,
            batch_size=self.arch_config["opt_strategy"]["batch_size"],
            validation_data=(val_input,val_target),
            validation_batch_size=1,
            callbacks = [
                lr_scheduler_callback,
                checkpoint_best_r_callback,
                checkpoint_last_callback,
                csv_logger_callback
            ]
        )

        return history
    
    def execute_training(self):

        self.model_path = self.setup_output_directory()

        mu_reference = np.load(self.train_config["dataset_path"]+'mu_reference.npy')
        print('mu reference: ', mu_reference)
        
        kratos_simulation = CompressiblePotentialFlow_KratosSimulator(self.working_path, self.train_config, mu_reference)

        prepost_processor=SVD_White_NoStand_PODANN_PrePostProcessor(self.working_path, self.train_config['dataset_path'])
        crop_mat_tf, crop_mat_scp = kratos_simulation.get_crop_matrix()
        sparse_avp_rearrange_matrix, sparse_vp_rearrange_matrix = kratos_simulation.get_transformation_reduced_snapshot_to_variable_vectors() 

        S_FOM_orig = self.get_orig_fom_snapshots()
        prepost_processor.configure_processor(S_FOM_orig, self.arch_config["q_inf_size"], self.arch_config["q_sup_size"], crop_mat_tf, crop_mat_scp, sparse_avp_rearrange_matrix, sparse_vp_rearrange_matrix)

        # Load the autoencoder model
        print('======= Instantiating new autoencoder =======')
        network = self.define_network(prepost_processor, kratos_simulation)
        
        if self.train_config["architecture"]["finetune_from"] is not None:
            print('======= Loading saved weights =======')
            network.load_weights(self.working_path+self.train_config["architecture"]["finetune_from"]+'model_weights.h5')

        # Get training data
        print('======= Loading training data =======')
        input_data, target_data, val_input, val_target = prepost_processor.get_training_data(self.train_config["architecture"])

        print('Shape input_data:', input_data.shape)
        for i in range(len(target_data)):
            print('Shape target_data [', i, ']: ', target_data[i].shape)
        print('Shape val_input:', val_input.shape)
        for i in range(len(val_target)):
            print('Shape target_data [', i, ']: ', val_target[i].shape)

        print('input data: ', input_data)
        print('target_data: ', target_data)

        print('======= Saving AE Config =======')
        with open(self.model_path+"train_config.npy", "wb") as ae_config_file:
            np.save(ae_config_file, self.train_config)
        with open(self.model_path+"train_config.json", "w") as ae_config_json_file:
            json.dump(self.train_config, ae_config_json_file)

        print(self.train_config)

        print('=========== Starting training routine ============')
        history = self.train_network(network, input_data, target_data, val_input, val_target)

        print('=========== Saving weights and history ============')
        network.save_weights(self.model_path+"model_weights.h5")
        with open(self.model_path+"history.json", "w") as history_file:
            json.dump(history.history, history_file)

        print(self.train_config)

        # Dettach the fake sim (To prevent problems saving the model)
        network.kratos_simulation = None

    


    
    def execute_test_dofs(self):
        import matplotlib.pyplot as plt

        self.model_path = self.setup_output_directory()

        with open('Data/mu_train.dat','rb') as mu_train_file:
            mu_train = pickle.load(mu_train_file)

        S_train = np.load('rom_data_total/SnapshotsMatrices/fom_snapshots.npy').T

        # binary_matrix=[]
        # for i, mu in enumerate(mu_train):

        #     # Create a fake Analysis stage to calculate the predicted residuals
        #     kratos_simulation = CompressiblePotentialFlow_KratosSimulator(self.working_path, self.train_config, mu)
        #     snapshot_raw = S_train[[i]]

        #     binary_vector = kratos_simulation.get_binary_vector_auxiliar_dofs()
        #     binary_matrix.append(binary_vector)

        #     del kratos_simulation
        # binary_matrix=np.array(binary_matrix)

        # np.save('aux_vel_binary_matrix.npy', binary_matrix)

        # # print(binary_matrix)
        # print(binary_matrix.shape)

        # exit()


        binary_matrix=np.load('aux_vel_binary_matrix_good.npy')
        
        # plt.plot(np.sum(binary_matrix, axis=1), 'o')
        # plt.show()

        # plt.plot(np.sum(binary_matrix, axis=0), 'o')
        # plt.show()

        not_full_list = np.argwhere(np.sum(binary_matrix, axis=0)<binary_matrix.shape[0])
        non_empty_list = np.argwhere(np.sum(binary_matrix, axis=0)>0)


        occasional_dof_nodes = np.intersect1d(not_full_list, non_empty_list)


        possible_configs=[]
        for i in range(binary_matrix.shape[0]):
            current_config=set(occasional_dof_nodes[binary_matrix[i,occasional_dof_nodes].astype(bool)])
            if current_config not in possible_configs:
                possible_configs.append(current_config)

        mu_class_1=[]
        mu_class_2=[]
        mu_class_3=[]
        for i in range(binary_matrix.shape[0]):
            current_config=set(occasional_dof_nodes[binary_matrix[i,occasional_dof_nodes].astype(bool)])
            if current_config == possible_configs[0]:
                mu_class_1.append([mu_train[i][0],mu_train[i][1]])
            elif current_config == possible_configs[1]:
                mu_class_2.append([mu_train[i][0],mu_train[i][1]])
            elif current_config == possible_configs[2]:
                mu_class_3.append([mu_train[i][0],mu_train[i][1]])
            else:
                print('UNCLASSIFIED OCCURENCE')

        mu_class_1=np.array(mu_class_1)
        mu_class_2=np.array(mu_class_2)
        mu_class_3=np.array(mu_class_3)

        plt.scatter(mu_class_1[:,0], mu_class_1[:,1])
        plt.scatter(mu_class_2[:,0], mu_class_2[:,1])
        plt.scatter(mu_class_3[:,0], mu_class_3[:,1])
        plt.grid()
        plt.xlabel('Angulo de ataque')
        plt.ylabel('Mach')
        plt.show()

        # print(len(possible_configs))
        # print(possible_configs)

        
        exit()

        print(occasional_dof_nodes)

        all_nodes_plotted=[]
        for id in [0,1,2,166,768,928]:
            # Create a fake Analysis stage to calculate the predicted residuals
            kratos_simulation = CompressiblePotentialFlow_KratosSimulator(self.working_path, self.train_config, mu_train[id])

            nodes_to_plot = occasional_dof_nodes[binary_matrix[id,occasional_dof_nodes].astype(bool)]
            all_nodes_plotted.append(nodes_to_plot)
            positions_matrix = kratos_simulation.get_node_positions(nodes_to_plot)

            plt.scatter(positions_matrix[:,0], positions_matrix[:,1])

            del kratos_simulation

        plt.show()

        print(mu_train[0])
        print(mu_train[1])
        print(mu_train[2])
        print(mu_train[166])
        print(mu_train[768])
        print(mu_train[928])

        # for nodes in all_nodes_plotted:
        #     print(np.all(all_nodes_plotted[0]==nodes))
        print(np.all(all_nodes_plotted[0]==all_nodes_plotted[4]))
        print(np.all(all_nodes_plotted[0]==all_nodes_plotted[5]))

        # print(occasional_dof_nodes[binary_matrix[0,occasional_dof_nodes]])
        # print(occasional_dof_nodes[binary_matrix[1,occasional_dof_nodes]])
        # print(occasional_dof_nodes[binary_matrix[2,occasional_dof_nodes]])

        exit()

        # for list in total_lists:
        #     print(np.all(list==total_lists[0]))
        # exit()

        snapshot = kratos_simulation.snapshots_to_system_vectors(snapshot_raw)
        snapshot_raw_recons = kratos_simulation.system_vectors_to_snapshots(snapshot)

        print(np.all(snapshot_raw==snapshot_raw_recons))

        # _, residual = kratos_simulation.get_A_b(snapshot_raw_recons)
        # print(residual)
        # plt.plot(residual)
        # plt.show()

        exit()

        # Select the type of preprocessimg (normalisation)
        prepost_processor=SVD_White_NoStand_Cropping_PODANN_PrePostProcessor(self.working_path, self.train_config['dataset_path'])

        S_FOM_orig = self.get_orig_fom_snapshots()
        prepost_processor.configure_processor(S_FOM_orig, self.arch_config["q_inf_size"], self.arch_config["q_sup_size"], crop_mat_tf, crop_mat_scp)
        # snapshot_size=S_FOM_orig.shape[1]

        # Load the autoencoder model
        print('======= Instantiating new autoencoder =======')
        network = self.define_network(prepost_processor, kratos_simulation)
        
        if self.train_config["architecture"]["finetune_from"] is not None:
            print('======= Loading saved weights =======')
            network.load_weights(self.working_path+self.train_config["architecture"]["finetune_from"]+'model_weights.h5')

        # Get training data
        print('======= Loading training data =======')
        input_data, target_data, val_input, val_target = prepost_processor.get_training_data(self.train_config["architecture"])

        print('Shape input_data:', input_data.shape)
        for i in range(len(target_data)):
            print('Shape target_data [', i, ']: ', target_data[i].shape)
        print('Shape val_input:', val_input.shape)
        for i in range(len(val_target)):
            print('Shape target_data [', i, ']: ', val_target[i].shape)

        print('input data: ', input_data)
        print('target_data: ', target_data)

        print('======= Saving AE Config =======')
        with open(self.model_path+"train_config.npy", "wb") as ae_config_file:
            np.save(ae_config_file, self.train_config)
        with open(self.model_path+"train_config.json", "w") as ae_config_json_file:
            json.dump(self.train_config, ae_config_json_file)

        print(self.train_config)

        print('=========== Starting training routine ============')
        history = self.train_network(network, input_data, target_data, val_input, val_target)

        print('=========== Saving weights and history ============')
        network.save_weights(self.model_path+"model_weights.h5")
        with open(self.model_path+"history.json", "w") as history_file:
            json.dump(history.history, history_file)

        print(self.train_config)

        # Dettach the fake sim (To prevent problems saving the model)
        network.kratos_simulation = None
        
    