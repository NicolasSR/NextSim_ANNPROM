import numpy as np
import scipy
import math

import KratosMultiphysics as KMP
import KratosMultiphysics.CompressiblePotentialFlowApplication as CPFApp
from KratosMultiphysics.CompressiblePotentialFlowApplication.potential_flow_analysis import PotentialFlowAnalysis
from KratosMultiphysics.CompressiblePotentialFlowApplication.define_wake_process_2d import DefineWakeProcess2D

import tensorflow as tf


class CompressiblePotentialFlow_KratosSimulator():

    def __init__(self, working_path, train_config, mu):
        if "project_parameters_file" in train_config:
            project_parameters_path=train_config["dataset_path"]+train_config["project_parameters_file"]
        else: 
            project_parameters_path='ProjectParameters_fom.json'
            print('LOADED DEFAULT PROJECT PARAMETERS FILE')
        with open(working_path+project_parameters_path, 'r') as parameter_file:
            parameters = KMP.Parameters(parameter_file.read())

        self.parameters = self.UpdateProjectParameters(parameters, train_config["dataset_path"], mu)

        global_model = KMP.Model()
        self.fake_simulation = PotentialFlowAnalysis(global_model, parameters)
        self.fake_simulation.Initialize()
        self.fake_simulation.InitializeSolutionStep()

        self.space = KMP.UblasSparseSpace()
        self.strategy = self.fake_simulation._GetSolver()._GetSolutionStrategy()
        self.buildsol = self.fake_simulation._GetSolver()._GetBuilderAndSolver()
        self.scheme = self.fake_simulation._GetSolver()._GetScheme()
        self.modelpart = self.fake_simulation._GetSolver().GetComputingModelPart()
        self.var_utils = KMP.VariableUtils()

        self.sub_project_parameters = self.fake_simulation.project_parameters["processes"]["boundary_conditions_process_list"]
        
        self.define_wake_process_2d_id = None
        self.apply_far_field_process_id = None
        for i in range(self.sub_project_parameters.size()):
            if self.sub_project_parameters[i]["python_module"].GetString() == "define_wake_process_2d":
                self.define_wake_process_2d_id = i
            elif self.sub_project_parameters[i]["python_module"].GetString() == "apply_far_field_process":
                self.apply_far_field_process_id = i
        
        self.free_stream_speed_of_sound = self.sub_project_parameters[self.apply_far_field_process_id]["Parameters"]["speed_of_sound"].GetDouble()


    def UpdateProjectParameters(self, parameters, dataset_path, mu):

        angle = mu[0]
        mach_infinity = mu[1]

        parameters["processes"]["boundary_conditions_process_list"][0]["Parameters"]["angle_of_attack"].SetDouble(angle)
        parameters["processes"]["boundary_conditions_process_list"][0]["Parameters"]["mach_infinity"].SetDouble(mach_infinity)

        return parameters

    def get_crop_matrix(self):
        indices=[]

        for node in self.modelpart.Nodes:
            dof_avp = node.GetDof(CPFApp.AUXILIARY_VELOCITY_POTENTIAL)
            if dof_avp.EquationId!=0 and dof_avp.IsFixed():
                indices.append([dof_avp.EquationId,dof_avp.EquationId])

            dof_vp = node.GetDof(CPFApp.VELOCITY_POTENTIAL)
            if dof_vp.IsFixed():
                indices.append([dof_vp.EquationId,dof_vp.EquationId])

        num_rows=self.modelpart.NumberOfNodes()*2
        
        values=np.ones(len(indices))
        crop_mat_tf = tf.sparse.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=[num_rows,num_rows])
        indices=np.asarray(indices)
        crop_mat_scp = scipy.sparse.coo_array((values, (indices[:,0], indices[:,1])), shape=[num_rows,num_rows]).tocsr()
        
        return crop_mat_tf, crop_mat_scp

    # def print_aux_vel_pot_indices(self):

    #     # sub_sim_parameters = self.parameters.Clone()
    #     # sub_sim_parameters = self.UpdateProjectParameters(sub_sim_parameters, mu)

    #     # sub_sim_model = KMP.Model()
    #     # sub_simulation = PotentialFlowAnalysis(sub_sim_model, sub_sim_parameters)
    #     # sub_simulation.Initialize()
    #     # sub_simulation.InitializeSolutionStep()

    #     # sub_sim_modelpart = sub_simulation._GetSolver().GetComputingModelPart()

    #     indices_list=[]
    #     # for node in sub_sim_modelpart.Nodes:
    #     for node in self.modelpart.Nodes:
    #         dof_avp = node.GetDof(CPFApp.AUXILIARY_VELOCITY_POTENTIAL)
    #         if dof_avp.EquationId!=0:
    #             indices_list.append([node.Id-1, dof_avp.EquationId])
    #     indices_list = np.array(indices_list)
    #     return indices_list
    
    # def get_binary_vector_auxiliar_dofs(self):

    #     binary_vector=np.zeros(self.modelpart.NumberOfNodes())
    #     for node in self.modelpart.Nodes:
    #         dof_avp = node.GetDof(CPFApp.AUXILIARY_VELOCITY_POTENTIAL)
    #         if dof_avp.EquationId!=0:
    #             binary_vector[node.Id-1]=1
    #     return binary_vector
    
    # def get_node_positions(self, nodes_to_plot):
    #     positions_matrix=[]
    #     for node in self.modelpart.Nodes:
    #         if node.Id-1 in nodes_to_plot:
    #             positions_matrix.append([node.X, node.Y])
    #     positions_matrix=np.array(positions_matrix)
    #     return positions_matrix
    
    # def snapshots_to_system_vectors(self, snapshots):

    #     sys_vector=self.strategy.GetSystemVector()
    #     cropped_snapshots=np.zeros((snapshots.shape[0], sys_vector.Size()))

    #     for node in self.modelpart.Nodes:
    #         dof_avp = node.GetDof(CPFApp.AUXILIARY_VELOCITY_POTENTIAL)
    #         if dof_avp.EquationId!=0:
    #             cropped_snapshots[:,dof_avp.EquationId]=snapshots[:,(node.Id-1)*2]
    #         if dof_avp.EquationId==0 and snapshots[:,(node.Id-1)*2]!=0.0:
    #             print(node.Id-1)

    #         dof_vp = node.GetDof(CPFApp.VELOCITY_POTENTIAL)
    #         cropped_snapshots[:,dof_vp.EquationId]=snapshots[:,(node.Id-1)*2+1]

    #     return cropped_snapshots
    
    # def system_vectors_to_snapshots(self, cropped_snapshots):
        
    #     snapshots=np.zeros((cropped_snapshots.shape[0], self.modelpart.NumberOfNodes()*2))

    #     for node in self.modelpart.Nodes:
    #         dof_avp = node.GetDof(CPFApp.AUXILIARY_VELOCITY_POTENTIAL)
    #         if dof_avp.EquationId!=0:
    #             snapshots[:,(node.Id-1)*2]=cropped_snapshots[:, dof_avp.EquationId]

    #         dof_vp = node.GetDof(CPFApp.VELOCITY_POTENTIAL)
    #         snapshots[:,(node.Id-1)*2+1]=cropped_snapshots[:, dof_vp.EquationId]

    #     return snapshots

    def get_aux_nodes_list(self, mu):

        self.configure_settings_to_snapshot(mu)

        aux_nodes_list = []

        dofs = self.buildsol.GetDofSet()
        for dof in dofs:
            if dof.GetVariable() == CPFApp.VELOCITY_POTENTIAL:
                aux_nodes_list.append((dof.Id()-1)*2+1)
            elif dof.GetVariable() == CPFApp.AUXILIARY_VELOCITY_POTENTIAL:
                aux_nodes_list.append((dof.Id()-1)*2)

        return aux_nodes_list
    
    def configure_settings_to_snapshot(self, mu):

        far_field_model_part = self.fake_simulation.model["MainModelPart.PotentialWallCondition2D_Far_field_Auto1"]
        fluid_model_part = far_field_model_part.GetRootModelPart()

        angle_of_attack = mu[0]
        free_stream_mach = mu[1]

        u_inf = free_stream_mach * self.free_stream_speed_of_sound

        free_stream_velocity = KMP.Vector(3)
        free_stream_velocity[0] = round(u_inf*math.cos(angle_of_attack*math.pi/180),8)
        free_stream_velocity[1] = round(u_inf*math.sin(angle_of_attack*math.pi/180),8)
        free_stream_velocity[2] = 0.0

        fluid_model_part.ProcessInfo.SetValue(CPFApp.FREE_STREAM_VELOCITY,free_stream_velocity)

        self.wake_process = DefineWakeProcess2D(self.fake_simulation.model, self.sub_project_parameters[self.define_wake_process_2d_id]["Parameters"])

        self.wake_process.ExecuteInitialize()

        ## the next two lines are needed in order to add Wake DoFs to the new Wake Elements Nodes
        ## and delete the ones that are no longer in the Wake Region.
        self.fake_simulation._GetSolver().Clear()
        self.fake_simulation._GetSolver().InitializeSolutionStep()


    def get_transformation_reduced_snapshot_to_variable_vectors(self):
        
        sys_vector = self.strategy.GetSystemVector()
        print(sys_vector.Size())

        # avp_rearrange_matrix=np.zeros((self.modelpart.NumberOfNodes(), sys_vector.Size()))
        # vp_rearrange_matrix=np.zeros((self.modelpart.NumberOfNodes(), sys_vector.Size()))
        # all_rearrange_matrix=np.zeros((self.modelpart.NumberOfNodes()*2, sys_vector.Size()))

        
        # count=0
        # for element in self.modelpart.Elements:
        #     print(element.GetDofList(self.modelpart.ProcessInfo))
        #     for dof in element.GetDofList(self.modelpart.ProcessInfo):
        #         count+=1

        # print(count)
        # exit()


        # for dof in self.fake_simulation.settings["dofs"]:
        #     if dof == CPFApp.AUXILIARY_VELOCITY_POTENTIAL:
        #         avp_rearrange_matrix[dof.Node.Id-1,dof.EquationId] = 1.0
        #         all_rearrange_matrix[(dof.Node.Id-1)*2,dof.EquationId] = 1.0
        #     elif dof == CPFApp.VELOCITY_POTENTIAL:
        #         vp_rearrange_matrix[dof.Node.Id-1,dof.EquationId] = 1.0
        #         all_rearrange_matrix[(dof.Node.Id-1)*2+1,dof.EquationId] = 1.0
        
        avp_rows_list=[]
        avp_cols_list=[]
        vp_rows_list=[]
        vp_cols_list=[]
        var_type_list=np.zeros(sys_vector.Size())
        node_id_list=np.zeros(sys_vector.Size(), dtype=int)

        dofs = self.buildsol.GetDofSet()
        for dof in dofs:
            if dof.GetVariable() == CPFApp.AUXILIARY_VELOCITY_POTENTIAL:
                avp_rows_list.append(dof.Id()-1)
                avp_cols_list.append(dof.EquationId)
                var_type_list[dof.EquationId]=1
                node_id_list[dof.EquationId]=dof.Id()
            elif dof.GetVariable() == CPFApp.VELOCITY_POTENTIAL:
                vp_rows_list.append(dof.Id()-1)
                vp_cols_list.append(dof.EquationId)
                var_type_list[dof.EquationId]=0
                node_id_list[dof.EquationId]=dof.Id()

        print(var_type_list.shape)
        print(node_id_list.shape)
        for i in range(var_type_list.shape[0]):
            print (var_type_list[i], ' ', node_id_list[i])

        ### Save only auxiliary nodes and EqnIds and generate the rest in order to fill the whole resiudal length

        exit()

        

        print(len(avp_cols_list)+len(vp_cols_list))

        self.sparse_avp_rearrange_matrix = scipy.sparse.csr_matrix((np.ones(len(avp_rows_list)), (avp_rows_list, avp_cols_list)), shape=(self.modelpart.NumberOfNodes(), sys_vector.Size()))
        self.sparse_vp_rearrange_matrix = scipy.sparse.csr_matrix((np.ones(len(vp_rows_list)), (vp_rows_list, vp_cols_list)), shape=(self.modelpart.NumberOfNodes(), sys_vector.Size()))
        # self.sparse_avp_rearrange_matrix = scipy.sparse.csr_matrix(avp_rearrange_matrix)
        # self.sparse_vp_rearrange_matrix = scipy.sparse.csr_matrix(vp_rearrange_matrix)
        # self.sparse_all_rearrange_matrix = scipy.sparse.csr_matrix(all_rearrange_matrix)

        print(self.sparse_avp_rearrange_matrix.shape)
        print(self.sparse_vp_rearrange_matrix.shape)

        return self.sparse_avp_rearrange_matrix, self.sparse_vp_rearrange_matrix#, self.sparse_all_rearrange_matrix


    def project_prediction_vectorial_optim_batch(self, y_pred):

        num_nodes=self.modelpart.NumberOfNodes()

        # values = self.sparse_avp_rearrange_matrix.T.dot(y_pred.T).T

        values = y_pred.reshape((num_nodes,2))
        values_avp = values[:,0].reshape((num_nodes)).copy()
        values_vp = values[:,1].reshape((num_nodes)).copy()

        # print(self.sparse_avp_rearrange_matrix.shape)
        # print(y_pred.shape)
        # values_avp = self.sparse_avp_rearrange_matrix.dot(y_pred.T).T[0].copy()
        # values_vp = self.sparse_vp_rearrange_matrix.dot(y_pred.T).T[0].copy()

        nodes_array=self.modelpart.Nodes
        self.var_utils.SetSolutionStepValuesVector(nodes_array, CPFApp.AUXILIARY_VELOCITY_POTENTIAL, values_avp, 0)
        self.var_utils.SetSolutionStepValuesVector(nodes_array, CPFApp.VELOCITY_POTENTIAL, values_vp, 0)
    
    # def get_A_b(self, y_pred):

    #     A = self.strategy.GetSystemMatrix()
    #     b = self.strategy.GetSystemVector()

    #     self.space.SetToZeroMatrix(A)
    #     self.space.SetToZeroVector(b)

    #     self.project_prediction_vectorial_optim_batch(self.fake_simulation, y_pred)

    #     self.buildsol.Build(self.scheme, self.modelpart, A, b)

    #     Ascipy = scipy.sparse.csr_matrix((A.value_data(), A.index2_data(), A.index1_data()), shape=(A.Size1(), A.Size2()))

    #     b = np.array(b, copy=True)

    #     return Ascipy, b
        
    def get_transformation_reduced_snapshot_to_full_snapshot(self, aux_nodes):
        sys_vector = self.strategy.GetSystemVector()
        avp_rows_list=aux_nodes[:sys_vector.Size()]
        avp_cols_list=[i for i in range(len(avp_rows_list))]

        self.sparse_avp_rearrange_matrix = scipy.sparse.csr_matrix((np.ones(len(avp_rows_list)), (avp_rows_list, avp_cols_list)), shape=(self.modelpart.NumberOfNodes()*2, sys_vector.Size()))
        
        # print(self.sparse_avp_rearrange_matrix.shape)

        
    def get_b_batch_alt_(self, y_pred_batch, mu_batch, aux_nodes_batch):

        b_list=[]

        for i, y_pred in enumerate(y_pred_batch):
            
            far_field_model_part = self.fake_simulation.model["MainModelPart.PotentialWallCondition2D_Far_field_Auto1"]
            fluid_model_part = far_field_model_part.GetRootModelPart()

            angle_of_attack = mu_batch[i,0]
            free_stream_mach = mu_batch[i,1]

            u_inf = free_stream_mach * self.free_stream_speed_of_sound

            free_stream_velocity = KMP.Vector(3)
            free_stream_velocity[0] = round(u_inf*math.cos(angle_of_attack*math.pi/180),8)
            free_stream_velocity[1] = round(u_inf*math.sin(angle_of_attack*math.pi/180),8)
            free_stream_velocity[2] = 0.0

            fluid_model_part.ProcessInfo.SetValue(CPFApp.FREE_STREAM_VELOCITY,free_stream_velocity)

            self.wake_process = DefineWakeProcess2D(self.fake_simulation.model, self.sub_project_parameters[self.define_wake_process_2d_id]["Parameters"])
            self.wake_process.ExecuteInitialize()

            ## the next two lines are needed in order to add Wake DoFs to the new Wake Elements Nodes
            ## and delete the ones that are no longer in the Wake Region.
            self.fake_simulation._GetSolver().Clear()
            self.fake_simulation._GetSolver().InitializeSolutionStep()
            
            A = self.strategy.GetSystemMatrix()
            b = self.strategy.GetSystemVector()
            self.space.SetToZeroMatrix(A)
            self.space.SetToZeroVector(b)

            self.get_transformation_reduced_snapshot_to_full_snapshot(aux_nodes_batch[i])

            self.project_prediction_vectorial_optim_batch(y_pred)

            self.buildsol.Build(self.scheme, self.modelpart, A, b)

            b_full = self.sparse_avp_rearrange_matrix.dot(np.array(b, copy=True).T).T

            b_list.append(b_full.copy())

        b_list=np.array(b_list)

        return b_list
    
    def get_b_batch_(self, y_pred_batch, mu_batch):


        b_list=[]

        for i, y_pred in enumerate(y_pred_batch):
            
            far_field_model_part = self.fake_simulation.model["MainModelPart.PotentialWallCondition2D_Far_field_Auto1"]
            fluid_model_part = far_field_model_part.GetRootModelPart()

            angle_of_attack = mu_batch[i,0]
            free_stream_mach = mu_batch[i,1]

            u_inf = free_stream_mach * self.free_stream_speed_of_sound

            free_stream_velocity = KMP.Vector(3)
            free_stream_velocity[0] = round(u_inf*math.cos(angle_of_attack*math.pi/180),8)
            free_stream_velocity[1] = round(u_inf*math.sin(angle_of_attack*math.pi/180),8)
            free_stream_velocity[2] = 0.0

            fluid_model_part.ProcessInfo.SetValue(CPFApp.FREE_STREAM_VELOCITY,free_stream_velocity)

            self.wake_process = DefineWakeProcess2D(self.fake_simulation.model, self.sub_project_parameters[self.define_wake_process_2d_id]["Parameters"])

            self.wake_process.ExecuteInitialize()

            ## the next two lines are needed in order to add Wake DoFs to the new Wake Elements Nodes
            ## and delete the ones that are no longer in the Wake Region.
            self.fake_simulation._GetSolver().Clear()
            self.fake_simulation._GetSolver().InitializeSolutionStep()
            
            A = self.strategy.GetSystemMatrix()
            b = self.strategy.GetSystemVector()
            self.space.SetToZeroMatrix(A)
            self.space.SetToZeroVector(b)

            self.get_transformation_reduced_snapshot_to_variable_vectors()

            continue

            self.project_prediction_vectorial_optim_batch(y_pred)

            self.buildsol.Build(self.scheme, self.modelpart, A, b)

            b_list.append(np.array(b, copy=True))

        b_list=np.array(b_list)

        return b_list
    
    def get_v_loss_rdiff_batch_(self, y_pred, b_true, mu_batch):

        err_r_list=[]
        v_loss_r_list=[]

        for i in range(y_pred.shape[0]):
            
            sub_project_parameters = self.project_parameters["processes"]["boundary_conditions_process_list"]
            
            for j in range(sub_project_parameters.size()):
                if sub_project_parameters[j]["python_module"].GetString() == "define_wake_process_2d":
                    sub_project_parameters[j]["Parameters"]["angle_of_attack"].SetDouble(mu_batch[i,0])
                    self.wake_process = DefineWakeProcess2D(self.model, sub_project_parameters[j]["Parameters"])
                    
                # if sub_project_parameters[i]["python_module"].GetString() == "compute_forces_on_nodes_process":
                #     self.conversion_process = ComputeForcesOnNodesProcess(self.model, sub_project_parameters[i]["Parameters"])
                # if sub_project_parameters[i]["python_module"].GetString() == "compute_lift_process":
                #     self.lift_process = ComputeLiftProcess(self.model, sub_project_parameters[i]["Parameters"])
                    
            self.wake_process.ExecuteInitialize()

            ## the next two lines are needed in order to add Wake DoFs to the new Wake Elements Nodes
            ## and delete the ones that are no longer in the Wake Region.
            self._analysis_stage._GetSolver().Clear()
            self._analysis_stage._GetSolver().InitializeSolutionStep()

            A = self.strategy.GetSystemMatrix()
            b = self.strategy.GetSystemVector()

            xD  = self.space.CreateEmptyVectorPointer()
            self.space.ResizeVector(xD, self.space.Size(b))
            foo  = self.space.CreateEmptyVectorPointer()
            self.space.ResizeVector(foo, self.space.Size(b))

            self.space.SetToZeroMatrix(A)
            self.space.SetToZeroVector(b)
            self.space.SetToZeroVector(xD)
            self.space.SetToZeroVector(foo)

            self.project_prediction_vectorial_optim_batch(y_pred[i])

            self.buildsol.Build(self.scheme, self.modelpart, A, b)

            err_r=KMP.Vector(b_true[i].copy()-b)

            v_loss_r = self.space.CreateEmptyVectorPointer()
            self.space.ResizeVector(v_loss_r, self.space.Size(b))
            self.space.SetToZeroVector(v_loss_r)

            self.space.TransposeMult(A,err_r,v_loss_r)
            
            err_r_list.append(np.expand_dims(np.array(err_r, copy=False),axis=0))
            v_loss_r_list.append(np.expand_dims(np.array(v_loss_r, copy=False),axis=0))
            # The negative sign we should apply to A is compensated by the derivative of the loss


        err_r_batch = np.concatenate(err_r_list, axis = 0)
        v_loss_r_batch = np.concatenate(v_loss_r_list, axis = 0)
        
        return err_r_batch, v_loss_r_batch