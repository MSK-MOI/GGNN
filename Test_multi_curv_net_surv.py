#from Train import trainCoxPASNet, trainEdgePathNet

import torch
import numpy as np
from scipy.stats import zscore
from math import floor
import pandas as pd
import re
import networkx as nx
import random
import time as tm
from Model import Curv_Net,CombinedFeature_net
from Survival_CostFunc_CIndex import neg_par_log_likelihood, c_index
import torch.optim as optim
import copy
from torch.nn.functional import normalize
from FindMasks import find_1_neighbors,find_edge_neighbors,find_pathway
from km import plot_KM

dtype = torch.FloatTensor
''' Net Settings'''
Out_Nodes = 10 # number of hidden nodes in the last hidden layer
Num_EPOCHS = 1800  # for training
N_keep = 10
N_Round = 1

''' load data '''
## Load HPRD network
adj=pd.read_csv("./data/adj.txt",header=None)
hprd_genes=pd.read_csv("./data/hprd.txt",header=None)
## Load Kegg pathway
Path_dir='./data/c2.cp.kegg.v5.2.symbols.xls'
hprd_genes=hprd_genes.values.tolist()
hprd_genes=[k[0] for k in hprd_genes]
adj.columns=hprd_genes
adj.index=hprd_genes

## Load multiomics data
### For TCGA
study="lgg"
omic_to_use=["RNA","CNA","Methyl"]
input_folder="./data/"+study+"_tcga/out/"

### For Commpass multiple myeloma
#study="mm"
#input_folder="./data/"+study+"_CoMMpass/out/"
#omic_to_use=["RNA","CNA"]


clinn_file=input_folder+"clinn.csv"



c_test={"RNA":[],"CNA":[],"Methyl":[]}
c_vali={"RNA":[],"CNA":[],"Methyl":[]}
c_test_f=[]
c_vali_f=[]
start = tm.time()
for r in range(N_Round):
    end = tm.time()
    print("\n\n ---------------- Start round:",r+1,"/",N_Round, "----------------\n")
    if r>0:
        print("t=",tm.strftime("%H:%M:%S", tm.gmtime(end - start)),"/",tm.strftime("%H:%M:%S", tm.gmtime((end - start)/r*N_Round)),"\n")
    tbc=pd.read_csv(clinn_file,header=0,index_col=0)
    temp=tbc[:]['OS_MONTHS'].tolist()
    Nsample=len(temp)
    time=np.zeros([Nsample,1])
    for i in range(Nsample):
        try:
            time[i,0]=float(temp[i])
        except:
            time[i,0]=0

    N_train=floor(Nsample*0.6)
    N_test=floor(Nsample*0.2)
    N_valid=floor(Nsample*0.2)
    P=np.random.permutation(Nsample)
    time=time[P,:]
    ytime_train=torch.from_numpy(time[0:N_train,:]).type(dtype).cuda()
    ytime_valid=torch.from_numpy(time[N_train:N_train+N_valid,:]).type(dtype).cuda()
    ytime_test=torch.from_numpy(time[N_train+N_valid:N_train+N_valid+N_test,:]).type(dtype).cuda()

    temp=tbc[:]['OS_STATUS'].tolist()
    E=np.zeros([Nsample,1])
    for i in range(Nsample):
        if temp[i]=='1:DECEASED' or temp[i]==1:
            E[i,0]=1
    E=E[P,:]
    yevent_train=torch.from_numpy(E[0:N_train,:]).type(dtype).cuda()
    yevent_valid=torch.from_numpy(E[N_train:N_train+N_valid,:]).type(dtype).cuda()
    yevent_test=torch.from_numpy(E[N_train+N_valid:N_train+N_valid+N_test,:]).type(dtype).cuda()

    print("Train event rate:",sum(yevent_train).item()/N_train)
    print("Test event rate:",sum(yevent_test).item()/N_test)
    print("Validation event rate:",sum(yevent_valid).item()/N_valid)

    clinn=np.zeros([Nsample,0])# No clinn input is used in this study. It is possible to add clinnical factors here. 
    clinn=clinn[P,:]
    clinn_train=torch.from_numpy(clinn[0:N_train,:]).type(dtype).cuda()
    clinn_valid=torch.from_numpy(clinn[N_train:N_train+N_valid,:]).type(dtype).cuda()
    clinn_test=torch.from_numpy(clinn[N_train+N_valid:N_train+N_valid+N_test,:]).type(dtype).cuda()

    final_feature_train=torch.zeros([N_train,0]).cuda()
    final_feature_valid=torch.zeros([N_valid,0]).cuda()
    final_feature_test=torch.zeros([N_test,0]).cuda()
    
    ''' Training '''
    # Train each omic subnetwork
    for omic in omic_to_use:
        # Build masks 
        omic_filename=input_folder+omic+".csv"
        tb=pd.read_csv(omic_filename,header=0,index_col=0)
        in_genes=tb.index.tolist()
        Adj=np.array([[adj[i][j] for j in in_genes] for i in in_genes])
        Adj=Adj-np.diag(np.diag(Adj))
        G=nx.from_numpy_matrix(Adj)
        largest_cc = max(nx.connected_components(G), key=len)
        Adj=np.array([[Adj[i][j] for j in largest_cc] for i in largest_cc])
        one_neighbors_mask=find_1_neighbors(Adj)
        one_neighbors_mask=torch.from_numpy(one_neighbors_mask).type(dtype).cuda()
        edge_mask=find_edge_neighbors(Adj)
        Edge_Nodes=edge_mask.shape[0]
        edge_mask=torch.from_numpy(edge_mask).type(dtype).cuda()

        gene_s=[in_genes[i] for i in largest_cc]
        pathway_mask=torch.from_numpy(find_pathway(Path_dir,Adj,gene_s)).type(dtype).cuda()
        Pathway_Nodes=pathway_mask.shape[0]
        
        omic_data=tb.values
        omic_data=[omic_data[i][:] for i in largest_cc]
        if omic=="Methyl":
            omic_data=1-np.array(omic_data)
        if omic=="CNA":
            omic_data=np.exp(omic_data)

        omic_data_invmea=np.multiply(omic_data,np.matmul(Adj,omic_data))
        omic_data=np.transpose(omic_data)
        #if omic=="RNA":
        omic_data=np.array(zscore(omic_data))
        omic_data=np.nan_to_num(omic_data)
        Nsample,In_Nodes=omic_data.shape
        omic_data=omic_data[P,:]
        x1_train=torch.from_numpy(omic_data[0:N_train,:]).type(dtype).cuda()
        x1_valid=torch.from_numpy(omic_data[N_train:N_train+N_valid,:]).type(dtype).cuda()
        x1_test=torch.from_numpy(omic_data[N_train+N_valid:N_train+N_valid+N_test,:]).type(dtype).cuda()

        omic_data_invmea=omic_data_invmea/np.sum(omic_data_invmea,axis=0)
        omic_data_invmea=np.transpose(omic_data_invmea)
        omic_data_invmea=np.array(zscore(omic_data_invmea))
        omic_data_invmea=np.nan_to_num(omic_data_invmea)
        omic_data_invmea=omic_data_invmea[P,:]
        x2_train=torch.from_numpy(omic_data_invmea[0:N_train,:]).type(dtype).cuda()
        x2_valid=torch.from_numpy(omic_data_invmea[N_train:N_train+N_valid,:]).type(dtype).cuda()
        x2_test=torch.from_numpy(omic_data_invmea[N_train+N_valid:N_train+N_valid+N_test,:]).type(dtype).cuda()

        omic_curv_filename=input_folder+omic+"_curv.csv"
        tbc=pd.read_csv(omic_curv_filename,header=0,index_col=0)
        edge_curv=np.array(tbc.values)
        edge_curv=edge_curv[P,:]
        x3_train=torch.from_numpy(edge_curv[0:N_train,:]).type(dtype).cuda()
        x3_valid=torch.from_numpy(edge_curv[N_train:N_train+N_valid,:]).type(dtype).cuda()
        x3_test=torch.from_numpy(edge_curv[N_train+N_valid:N_train+N_valid+N_test,:]).type(dtype).cuda()


        net = Curv_Net(In_Nodes, Edge_Nodes, Pathway_Nodes, Out_Nodes,one_neighbors_mask, edge_mask, pathway_mask,0,N_keep,0.9)
        net.update_top_gene_mask(x1_train,ytime_train,yevent_train,gene_s)
        if torch.cuda.is_available():
            net.cuda()
        opt = optim.Adam(net.parameters(), lr=0.00005, weight_decay = 0.00005)
        opt_Cindex = torch.Tensor([-float("Inf")]).cuda()
        opt_loss = torch.Tensor([float("Inf")]).cuda()
        is_new_net=False
        opt_epoch=-1	
        print("Curv_Net",omic," start training:")
        for epoch in range(Num_EPOCHS+1):
            net.train()
            opt.zero_grad() 
                
            pred = net(x1_train,x2_train, x3_train, clinn_train) 
            loss = neg_par_log_likelihood(pred, ytime_train, yevent_train) 
            loss.backward() 
            opt.step() 

            net.sc1.weight.data = net.sc1.weight.data.mul(net.Adj) #force the connections between gene layer and invmea layer
            net.sc2.weight.data = net.sc2.weight.data.mul(net.edge_mask) #force the connections between invmea layer and edge layer
            net.sc3.weight.data = net.sc3.weight.data.mul(net.pathway_mask) #force the connections between edge layer and pathway layer

            
            net.eval()
            with torch.no_grad():
                train_pred = net(x1_train,x2_train,x3_train, clinn_train)
                train_loss = neg_par_log_likelihood(train_pred, ytime_train, yevent_train).view(1,)

                eval_pred = net(x1_test,x2_test,x3_test, clinn_test)
                eval_loss = neg_par_log_likelihood(eval_pred, ytime_test, yevent_test).view(1,)

                train_cindex = c_index(train_pred, ytime_train, yevent_train)
                eval_cindex = c_index(eval_pred, ytime_test, yevent_test)
                if eval_cindex > opt_Cindex:
                    opt_loss=eval_loss
                    opt_Cindex=eval_cindex
                    copy_net = copy.deepcopy(net)
                    is_new_net=True
                    opt_epoch=epoch
            if epoch % 100 == 0:     
                print("Epoch", epoch,":\n Train Loss", train_loss.item(),", Train Cindex:",train_cindex.item(),",\n Test Loss", eval_loss.item(), ", Test Cindex:",eval_cindex.item())
                if train_cindex-eval_cindex > 0.25:
                    print("Warning: Overfitting!")

            if (epoch > 300) & (epoch % 400 == 0):
                if train_cindex-eval_cindex > 0.2:
                    is_update_layer_masks = True
                else:
                    is_update_layer_masks = False
                net.forward_update_masks(torch.cat((x1_train,x1_test)),torch.cat((x2_train,x2_test)),torch.cat((x3_train,x3_test)), torch.cat((clinn_train,clinn_test)),torch.cat((ytime_train,ytime_test)), torch.cat((yevent_train,yevent_test)),gene_s,is_update_layer_masks)
                opt_s = optim.Adam([{"params":net.sc6.parameters(), 'lr' : 0.001},{"params":net.sc7.parameters(), 'lr' : 0.001}])                
                for s in range(10):
                    opt_s.zero_grad() 
                    pred = net(x1_train,x2_train, x3_train, clinn_train) 
                    loss = neg_par_log_likelihood(pred, ytime_train, yevent_train) 
                    loss.backward()
                    opt_s.step()


                
        net=copy_net
        net.eval()
        print(omic, ' net:')
        train_pred = net(x1_train,x2_train,x3_train, clinn_train)
        train_loss = neg_par_log_likelihood(train_pred, ytime_train, yevent_train).view(1,)
        plot_KM(train_pred,ytime_train,yevent_train,study+" "+omic+" training")
        eval_pred = net(x1_test,x2_test,x3_test, clinn_test)
        eval_loss = neg_par_log_likelihood(eval_pred, ytime_test, yevent_test).view(1,)
        plot_KM(eval_pred,ytime_test,yevent_test,study+" "+omic+" validation")
        train_cindex = c_index(train_pred, ytime_train, yevent_train)
        eval_cindex = c_index(eval_pred, ytime_test, yevent_test)
        print("Data summary")
        print("Train event rate:",sum(yevent_train).item()/N_train,", average time:", torch.mean(ytime_train).item())
        print("Test event rate:",sum(yevent_test).item()/N_test,", average time:", torch.mean(ytime_test).item())
        print("Validation event rate:",sum(yevent_valid).item()/N_valid,", average time:", torch.mean(ytime_valid).item())
        print("Optimal Net at epoch",opt_epoch, ":\n Train Loss", train_loss.item(),", Train Cindex:",train_cindex.item(),",\n Test Loss", eval_loss.item(), ", Test Cindex:",eval_cindex.item())
        valid_pred = net(x1_valid,x2_valid,x3_valid, clinn_valid)
        valid_loss = neg_par_log_likelihood(valid_pred, ytime_valid, yevent_valid).view(1,)
        plot_KM(valid_pred,ytime_valid,yevent_valid,study+" "+omic+" testing")
        valid_cindex = c_index(valid_pred, ytime_valid, yevent_valid)
        print("Validation:\n Validation Loss", valid_loss.item(),", Validation Cindex:",valid_cindex.item())
        c_test[omic].append(eval_cindex.item())
        c_vali[omic].append(valid_cindex.item())
        with torch.no_grad():
            if omic=="RNA":
                weight=2
            else:
                weight=1
            final_feature_train=torch.cat((final_feature_train,weight*normalize(net.forward_feature(x1_train,x2_train,x3_train, clinn_train)).clone()),1)
            final_feature_valid=torch.cat((final_feature_valid,weight*normalize(net.forward_feature(x1_valid,x2_valid,x3_valid, clinn_valid)).clone()),1)
            final_feature_test=torch.cat((final_feature_test,weight*normalize(net.forward_feature(x1_test,x2_test,x3_test, clinn_test)).clone()),1)

    Hidden_Nodes1=Out_Nodes*len(omic_to_use)
    Hidden_Nodes2=Out_Nodes
    cfnet=CombinedFeature_net((Out_Nodes+N_keep*4)*len(omic_to_use), Hidden_Nodes1,Hidden_Nodes2)
    cfnet.cuda()
    opt_cf = optim.Adam(cfnet.parameters(), lr=0.0001)
    opt_Cindex = torch.Tensor([-float("Inf")]).cuda()
    opt_loss = torch.Tensor([float("Inf")]).cuda()
    opt_epoch=-1
    Num_EPOCHS2=500
    # Train multi-omics combined network
    for epoch in range(Num_EPOCHS2+1):
        cfnet.train()
        opt_cf.zero_grad() 
                
        pred = cfnet(final_feature_train) 
        loss = neg_par_log_likelihood(pred, ytime_train, yevent_train) 
        loss.backward() 
        opt_cf.step() 

        cfnet.eval()
        with torch.no_grad():
            train_pred = cfnet(final_feature_train)
            train_loss = neg_par_log_likelihood(train_pred, ytime_train, yevent_train).view(1,)

            eval_pred = cfnet(final_feature_test)
            eval_loss = neg_par_log_likelihood(eval_pred, ytime_test, yevent_test).view(1,)

            train_cindex = c_index(train_pred, ytime_train, yevent_train)
            eval_cindex = c_index(eval_pred, ytime_test, yevent_test)

            if eval_cindex > opt_Cindex:
                opt_loss=eval_loss
                opt_Cindex=eval_cindex
                copy_net = copy.deepcopy(cfnet)
                opt_epoch=epoch
            if epoch % 50 == 0:     
                print("Epoch", epoch,":\n Train Loss", train_loss.item(),", Train Cindex:",train_cindex.item(),",\n Test Loss", eval_loss.item(), ", Test Cindex:",eval_cindex.item())

    ''' output final results'''
    cfnet=copy_net
    cfnet.eval()
    print('Final net:')
    train_pred = cfnet(final_feature_train)
    train_loss = neg_par_log_likelihood(train_pred, ytime_train, yevent_train).view(1,)
    plot_KM(train_pred,ytime_train,yevent_train,"Training set")
    eval_pred = cfnet(final_feature_test)
    eval_loss = neg_par_log_likelihood(eval_pred, ytime_test, yevent_test).view(1,)
    plot_KM(eval_pred,ytime_test,yevent_test,"Validation set")
    train_cindex = c_index(train_pred, ytime_train, yevent_train)
    eval_cindex = c_index(eval_pred, ytime_test, yevent_test)
    print("Data summary")
    print("Train event rate:",sum(yevent_train).item()/N_train,", average time:", torch.mean(ytime_train).item())
    print("Test event rate:",sum(yevent_test).item()/N_test,", average time:", torch.mean(ytime_test).item())
    print("Validation event rate:",sum(yevent_valid).item()/N_valid,", average time:", torch.mean(ytime_valid).item())
    print("Optimal Net at epoch",opt_epoch, ":\n Train Loss", train_loss.item(),", Train Cindex:",train_cindex.item(),",\n Test Loss", eval_loss.item(), ", Test Cindex:",eval_cindex.item())
    valid_pred = cfnet(final_feature_valid)
    valid_loss = neg_par_log_likelihood(valid_pred, ytime_valid, yevent_valid).view(1,)
    plot_KM(valid_pred,ytime_valid,yevent_valid,"Test set")
    valid_cindex = c_index(valid_pred, ytime_valid, yevent_valid)
    print("Validation:\n Validation Loss", valid_loss.item(),", Validation Cindex:",valid_cindex.item())
    c_test_f.append(eval_cindex.item())
    c_vali_f.append(valid_cindex.item())
    del net
    del P
    del cfnet
    
for key in c_test:
    print("TestC_",key,"=",c_test[key])
    print("ValiC_",key,"=",c_vali[key])
    print("mean=",np.mean(c_vali[key]),"\n")

print("FinalTestC=",c_test_f)
print("FinalValiC=",c_vali_f)
print("mean=",np.mean(c_vali_f),"\n")