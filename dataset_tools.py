import numpy as np
import pickle
from Kratos_Simulators.compressible_potential_flow_kratos_simulator import CompressiblePotentialFlow_KratosSimulator

import matplotlib.pyplot as plt

def generate_residuals():

    s_file_name='S_train.npy'
    mu_file_name='mu_train.npy'
    aux_nodes_file_name='aux_nodes.npy'
    r_file_name='R_train.npy'

    train_config = {
        "dataset_path": 'dataset/',
        "models_path_root": 'saved_models_nextsim_residual/',
        "project_parameters_file":'ProjectParametersPrimalROM.json'
    }

    s_matrix = np.load(train_config["dataset_path"]+s_file_name)
    mu_matrix = np.load(train_config["dataset_path"]+mu_file_name)
    mu_reference = np.load(train_config["dataset_path"]+'mu_reference.npy')
    aux_nodes_matrix = np.load(train_config["dataset_path"]+aux_nodes_file_name)

    print(s_matrix)
    print(mu_reference)

    kratos_simulation = CompressiblePotentialFlow_KratosSimulator('', train_config, mu_reference)
    r_matrix = kratos_simulation.get_b_batch_alt_(s_matrix, mu_matrix, aux_nodes_matrix)

    print(r_matrix)
    print(r_matrix.shape)

    np.save(train_config["dataset_path"]+r_file_name, r_matrix)

def generate_aux_nodes_matrix():

    # s_file_name='S_train.npy'
    mu_file_name='mu_train.npy'

    aux_nodes_file_name='aux_nodes.npy'

    train_config = {
        "dataset_path": 'dataset/',
        "models_path_root": 'saved_models_nextsim_residual/',
        "project_parameters_file":'ProjectParametersPrimalROM.json'
    }

    # s_matrix = np.load(train_config["dataset_path"]+s_file_name)
    mu_matrix = np.load(train_config["dataset_path"]+mu_file_name)
    mu_reference = np.load(train_config["dataset_path"]+'mu_reference.npy')

    kratos_simulaion = CompressiblePotentialFlow_KratosSimulator('', train_config, mu_reference)
    
    aux_nodes_lists_list = []
    max_len = 0
    for mu in mu_matrix:
        aux_nodes_list = kratos_simulaion.get_aux_nodes_list(mu)
        aux_nodes_lists_list.append(aux_nodes_list)
        max_len = max(max_len, len(aux_nodes_list))

    aux_nodes_matrix = np.ones((mu_matrix.shape[0],max_len), dtype=int)*(-1)
    for i, aux_nodes_list in enumerate(aux_nodes_lists_list):
        aux_nodes_matrix[i,:len(aux_nodes_list)]=aux_nodes_list
    
    print(aux_nodes_matrix)
    print(max_len)

    np.save(train_config["dataset_path"]+aux_nodes_file_name, aux_nodes_matrix)


def test_gradients(kratos_simulation, true_snapshot):
    
    true_cropped_snapshot = kratos_simulation.snapshot_to_system_vector(true_snapshot)

    true_A, true_b = kratos_simulation.get_A_b(true_snapshot)

    v=np.random.rand(true_cropped_snapshot.shape[0])
    v/=np.linalg.norm(v)
   
    print('Base noise L2 Norm: ', np.linalg.norm(v))

    eps_vec = np.logspace(1, 10, 100)/1e9

    err_vec=[]
    for eps in eps_vec:

        v_eps = v*eps
        noisy_cropped_snapshot = true_cropped_snapshot+v_eps
        noisy_snapshot = kratos_simulation.system_vector_to_snapshot(noisy_cropped_snapshot)

        noisy_A, noisy_b = kratos_simulation.get_A_b(noisy_snapshot)

        first_order_term=(-true_A@v_eps.T).T
        err_vec.append(np.linalg.norm(noisy_b-true_b-first_order_term))

    square=np.power(eps_vec,2)
    plt.plot(eps_vec, square, "--", label="square")
    plt.plot(eps_vec, eps_vec, "--", label="linear")
    plt.plot(eps_vec, err_vec, label="error")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc="upper left")
    plt.show()


if __name__=="__main__":

    # s_train = np.load('rom_data_total/SnapshotsMatrices/fom_snapshots.npy')

    # with open('Data/mu_train.dat','rb') as mu_train_file:
    #     mu_train = pickle.load(mu_train_file)

    # train_config = {
    #     "dataset_path":"",
    #     "project_parameters_file":'ProjectParametersPrimalROM.json'
    # }

    # id = 234
    # snapshot=s_train.T[id]
    # kratos_simulation = CompressiblePotentialFlow_KratosSimulator('', train_config, mu_train[id])

    # test_gradients(kratos_simulation, snapshot)

    generate_residuals()
    # generate_aux_nodes_matrix()


