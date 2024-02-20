import numpy as np
import tensorflow as tf
from KratosMultiphysics.RomApplication.randomized_singular_value_decomposition import RandomizedSingularValueDecomposition


class SVD_White_NoStand_Cropping_PODANN_PrePostProcessor():
    def __init__(self, working_path, dataset_path):
        super().__init__()
        self.crop_mat_scp=None
        self.crop_mat_tf=None
        self.phisig_inf = None
        self.phisig_inf_tf = None
        self.phisig_sup = None
        self.phisig_sup_tf = None
        self.phisig_inv_inf = None
        self.phisig_inv_inf_tf = None
        self.phisig_inv_sup = None
        self.phisig_inv_sup_tf = None
        self.sparse_avp_rearrange_matrix = None
        self.sparse_vp_rearrange_matrix = None
        self.dataset_path=working_path+dataset_path

    def configure_processor(self, S, svd_inf_size, svd_sup_size, crop_mat_tf, crop_mat_scp, sparse_avp_rearrange_matrix, sparse_vp_rearrange_matrix):
        print('Applying SVD-whitening without prior standartization for PODANN architecture')

        self.crop_mat_scp=crop_mat_scp
        self.crop_mat_tf=crop_mat_tf

        self.sparse_avp_rearrange_matrix=sparse_avp_rearrange_matrix
        self.sparse_vp_rearrange_matrix=sparse_vp_rearrange_matrix

        S_free_raw, S_bound_raw = self.subtract_bound_dofs(S)

        S_free = self.transform_full_snapshots_to_reduced_snapshots(S_free_raw)
        S_bound = self.transform_full_snapshots_to_reduced_snapshots(S_bound_raw)

        S_total = S_free + S_bound

        try:
            self.phi=np.load(self.dataset_path+'PODANN/phi_whitenostand_crop.npy')
            self.sigma=np.load(self.dataset_path+'PODANN/sigma_whitenostand_crop.npy')
        except IOError:
            print("No precomputed phi_whitenostand_crop or sigma_whitenostand_crop matrix found. Computing a new set")
            S_scaled=S_free/np.sqrt(S_free.shape[0])
            self.phi,self.sigma, _ = np.linalg.svd(S_scaled.T)
            np.save(self.dataset_path+'PODANN/phi_whitenostand_crop.npy', self.phi)
            np.save(self.dataset_path+'PODANN/sigma_whitenostand_crop.npy', self.sigma)

        phi_inf=self.phi[:,:svd_inf_size].copy()
        sigma_inf=self.sigma[:svd_inf_size].copy()
        phi_sup=np.diag(self.phi[:,svd_inf_size:svd_sup_size].copy())
        sigma_sup=np.diag(self.sigma[svd_inf_size:svd_sup_size].copy())

        self.phisig_inf=phi_inf@sigma_inf
        self.phisig_sup=phi_sup@sigma_sup
        self.phisig_inv_inf=np.linalg.inv(sigma_inf)@phi_inf.T
        self.phisig_inv_sup=np.linalg.inv(sigma_sup)@phi_sup.T

        print('Phi_inf matrix shape: ', phi_inf.shape)
        print('Sigma_inf array shape: ', sigma_inf.shape)
        print('Phi_sgs matrix shape: ', phi_sup.shape)
        print('Sigma_sgs array shape: ', sigma_sup.shape)
        print('phisig_inf matrix shape: ', self.phisig_inf.shape)
        print('phisig_sup array shape: ', self.phisig_sup.shape)
        print('phisig_inv_inf matrix shape: ', self.phisig_inv_inf.shape)
        print('phisig_inv_sup array shape: ', self.phisig_inv_sup.shape)

        self.phisig_inf_tf=tf.constant(self.phisig_inf)
        self.phisig_sup_tf=tf.constant(self.phisig_sup)
        self.phisig_inv_inf_tf=tf.constant(self.phisig_inv_inf)
        self.phisig_inv_sup_tf=tf.constant(self.phisig_inv_sup)

        ## Check reconstruction error
        S_recons_aux1=self.preprocess_nn_output_data(S_free)
        S_recons_aux2, _=self.preprocess_input_data(S_free)
        S_recons = self.postprocess_output_data(S_recons_aux1, (S_recons_aux2, S_bound))
        print('Reconstruction error SVD (Frob): ', np.linalg.norm(S_recons-S_total)/np.linalg.norm(S_total))
        err_aux=np.linalg.norm(S_total-S_recons, ord=2, axis=1)/np.linalg.norm(S_total, ord=2, axis=1)
        # print('Reconstruction error SVD (Mean L2): ', np.sum(err_aux)/S_total.shape[0])
        print('Reconstruction error SVD (Mean L2): ', np.exp(np.sum(np.log(err_aux))/S_total.shape[0]))

    def preprocess_nn_output_data(self, snapshot_free):
        # Returns q_sup from input snapshots
        output_data=snapshot_free.copy().T
        output_data=np.matmul(self.phisig_inv_sup,output_data).T
        return output_data
    
    def preprocess_nn_output_data_tf(self,snapshot_free_tensor):
        # Returns q_sup from input snapshots
        output_tensor=tf.transpose(tf.linalg.matmul(self.phisig_inv_sup_tf,snapshot_free_tensor,transpose_a=False,transpose_b=True))
        return output_tensor
    
    def preprocess_input_data(self, snapshot_free):
        # Returns q_inf from input snapshots
        output_data=snapshot_free.copy().T
        output_data=np.matmul(self.phisig_inv_inf,output_data).T
        return output_data, None  # We should remove this second output from everywhere in the code as it's no longer in use
    
    def preprocess_input_data_tf(self, snapshot_free_tensor):
        # Returns q_inf from input snapshots
        output_tensor=tf.transpose(tf.linalg.matmul(self.phisig_inv_inf_tf,snapshot_free_tensor,transpose_a=False,transpose_b=True))
        return output_tensor, None

    def postprocess_output_data(self, q_sup, aux_data):
        # Returns reconstructed u from given q_inf and q_sup
        q_inf, snapshot_bound = aux_data
        output_data_1=q_inf.copy()
        output_data_1=np.matmul(self.phisig_inf,output_data_1.T).T
        output_data_2=q_sup.copy()
        output_data_2=np.matmul(self.phisig_sup,output_data_2.T).T
        output_data = output_data_1 + output_data_2 + snapshot_bound
        return output_data
    
    def postprocess_output_data_tf(self, q_sup_tensor, aux_tensors):
        # Returns reconstructed u from given q_inf and q_sup
        q_inf_tensor, snapshot_bound_tensor = aux_tensors
        output_tensor_1=tf.transpose(tf.linalg.matmul(self.phisig_inf_tf,q_inf_tensor,transpose_b=True))
        output_tensor_2=tf.transpose(tf.linalg.matmul(self.phisig_sup_tf,q_sup_tensor,transpose_b=True))
        output_tensor = output_tensor_1 + output_tensor_2 + snapshot_bound_tensor
        return output_tensor
    
    def subtract_bound_dofs(self, total_snapshot):
        snapshot_bound=(self.crop_mat_scp@total_snapshot.T).T
        snapshot_free=total_snapshot-snapshot_bound
        return snapshot_free, snapshot_bound
    
    def get_training_data(self):

        S_train_raw_full=np.load(self.dataset_path+'S_train.npy')
        S_target_train_free_raw, S_target_train_bound_raw = self.subtract_bound_dofs(S_train_raw_full)
        S_target_train_free = self.transform_full_snapshots_to_reduced_snapshots(S_target_train_free_raw)
        S_target_train_bound = self.transform_full_snapshots_to_reduced_snapshots(S_target_train_bound_raw)
        S_target_train = S_target_train_free + S_target_train_bound
        target_aux=np.load(self.dataset_path+'R_train.npy')

        input_data, _ = self.preprocess_input_data(S_target_train_free)
        target_data=(S_target_train, target_aux, S_target_train_bound)
        
        S_val_raw_full=np.load(self.dataset_path+'S_val.npy')
        S_target_val_free_raw, S_target_val_bound_raw = self.subtract_bound_dofs(S_val_raw_full)
        S_target_val_free = self.transform_full_snapshots_to_reduced_snapshots(S_target_val_free_raw)
        S_target_val_bound = self.transform_full_snapshots_to_reduced_snapshots(S_target_val_bound_raw)
        S_target_val = S_target_val_free + S_target_val_bound
        val_target_aux=np.load(self.dataset_path+'R_val.npy')

        val_input, _ =self.preprocess_input_data(S_target_val_free)
        val_target=(S_target_val, val_target_aux, S_target_val_bound)

        return input_data, target_data, val_input, val_target
    
    def transform_full_snapshots_to_reduced_snapshots(self, full_snapshots):

        num_nodes=full_snapshots.shape[1]/2

        values = full_snapshots.reshape((full_snapshots.shape[0],num_nodes,2))
        values_avp = values[:,:,0].reshape((full_snapshots.shape[0],num_nodes)).copy()
        values_vp = values[:,:,1].reshape((full_snapshots.shape[0],num_nodes)).copy()
        
        reduced_snapshots = self.sparse_avp_rearrange_matrix.T.dot(values_avp.T).T+self.sparse_vp_rearrange_matrix.T.dot(values_vp.T).T

        return reduced_snapshots
    
    def transform_reduced_snapshots_to_full_snapshots(self, reduced_snapshot):

        values_avp = self.sparse_avp_rearrange_matrix.dot(reduced_snapshot.T).T
        values_vp = self.sparse_vp_rearrange_matrix.dot(reduced_snapshot.T).T
        values_avp = np.expand_dims(values_avp, axis=2)
        values_vp = np.expand_dims(values_vp, axis=2)

        full_snapshot=np.concatenate([values_avp,values_vp], axis=2)
        full_snapshot=full_snapshot.reshape((reduced_snapshot.shape[0],reduced_snapshot.shape[1]*2))
        
        return full_snapshot
    