from audioop import bias
import torch
import torch.nn as nn
#import sparselinear as sl
import random
from Survival_CostFunc_CIndex import R_set, neg_par_log_likelihood, c_index
import numpy as np



class Curv_Net(nn.Module):
	def __init__(self, In_Nodes, Edge_Nodes, Pathway_Nodes, Out_Nodes,Adj, Edge_Mask,Pathway_Mask,clinn_Nodes,N_keep,init_mix):
		super(Curv_Net, self).__init__()
		self.activate = nn.Sigmoid()

		self.normalize=nn.functional.normalize

		self.pathway_mask = Pathway_Mask
		self.Adj = Adj
		Adj_t=np.triu(Adj.cpu().detach().numpy(),k=1)
		self.K=np.nonzero(Adj_t)
		self.edge_mask = Edge_Mask
		self.N_keep=N_keep
		if torch.cuda.is_available():
			self.Adj = self.Adj.cuda()
			self.pathway_mask = self.pathway_mask.cuda()
			self.edge_mask = self.edge_mask.cuda()
		###gene layer --> edge layer
		self.top_gene_mask=torch.zeros([In_Nodes,N_keep]).cuda()
		self.top_invmea_mask=torch.zeros([In_Nodes,N_keep]).cuda()
		self.top_curv_mask=torch.zeros([Edge_Nodes,N_keep]).cuda()
		self.top_path_mask=torch.zeros([Pathway_Nodes,N_keep]).cuda()

		
		self.sc1 = nn.Linear(In_Nodes, In_Nodes)
		self.sc1.weight.data.fill_(0.05)
		self.mp11 = nn.Parameter(init_mix*torch.ones(In_Nodes))
		self.mp12 = nn.Parameter((1-init_mix)*torch.ones(In_Nodes))
		#self.mp12 = torch.zeros(In_Nodes).cuda()################################### for test
		#self.mp11 = torch.ones(In_Nodes).cuda()################################### for test
		self.mp1=torch.ones(In_Nodes).cuda()
		self.sc2 = nn.Linear(In_Nodes, Edge_Nodes)
		self.sc2.weight.data.fill_(0.05)
		self.mp21 = nn.Parameter(init_mix*torch.ones(Edge_Nodes))
		self.mp22 = nn.Parameter((1-init_mix)*torch.ones(Edge_Nodes))
		#self.mp22 = torch.zeros(Edge_Nodes).cuda()################################### for test
		#self.mp21 = torch.ones(Edge_Nodes).cuda()################################### for test
		self.mp2=torch.ones(Edge_Nodes).cuda()
		#print(Edge_Mask.shape)
		###pathway layer --> hidden layer
		self.sc3 = nn.Linear(Edge_Nodes, Pathway_Nodes)
		self.mp3=torch.ones(Pathway_Nodes).cuda()
		self.sc3.weight.data.fill_(0.05)
		###hidden layer --> hidden layer 2
		#self.sc3 = nn.Linear(Pathway_Nodes, Out_Nodes, bias = False)
		self.sc4 = nn.Linear(Pathway_Nodes, Out_Nodes)
		self.sc5 = nn.Linear(Out_Nodes, Out_Nodes)
		###hidden layer 2 + age --> Cox layer
		self.sc6 = nn.Linear(Out_Nodes+clinn_Nodes+N_keep*4, Out_Nodes, bias = False)
		self.sc7 = nn.Linear(Out_Nodes,1, bias = False)

		#self.m=nn.ReLU()
		###

	def forward(self, x_gene, x_invmea,x_curv,clinn):
		x_cat=self.forward_feature(x_gene, x_invmea,x_curv,clinn)
		lin_pred = self.activate(self.sc6(x_cat))

		lin_pred=lin_pred-torch.mean(lin_pred, 1, True)

		lin_pred = self.sc7(lin_pred)
		return lin_pred

	def forward_feature(self, x_gene, x_invmea,x_curv,clinn):
		###force the connections between gene layer and pathway layer w.r.t. 'pathway_mask'
		kept_gene=x_gene.mm(self.top_gene_mask).clone().detach()
		self.sc1.weight.data = self.sc1.weight.data.mul(self.Adj)
		x_1 = self.activate(self.sc1(x_gene))
		#x_1=self.normalize(x_1)
		x_1=x_invmea.mul((self.mp11).mul(self.mp1))+x_1.mul((self.mp12).mul(self.mp1))
		kept_invmea=x_1.mm(self.top_invmea_mask).clone().detach()
		self.sc2.weight.data = self.sc2.weight.data.mul(self.edge_mask)
		x_1 = self.activate(self.sc2(x_1))
		#x_1=self.normalize(x_1)
		x_1=x_curv.mul((self.mp21).mul(self.mp2))+x_1.mul((self.mp22).mul(self.mp2))
		kept_curv=x_1.mm(self.top_curv_mask).clone().detach()
		self.sc3.weight.data = self.sc3.weight.data.mul(self.pathway_mask)
		x_1 = self.activate(self.sc3(x_1))
		x_1=x_1.mul(self.mp3)
		kept_path=x_1.mm(self.top_path_mask).clone().detach()


		x_1=self.activate(self.sc4(x_1))
		x_1=self.activate(self.sc5(x_1))
		#x_1 = self.ReLU(self.sc33(x_1))
		###combine age with hidden layer 2
		x_cat = torch.cat((x_1, kept_gene,kept_invmea,kept_curv,kept_path,clinn), 1)
		return x_cat

	def forward_update_masks(self, x_gene, x_invmea,x_curv,clinn,ytime, yevent,gene_s,is_update_layer_masks):
		###force the connections between gene layer and pathway layer w.r.t. 'pathway_mask'
		kept_gene=x_gene.mm(self.top_gene_mask).clone().detach()
		self.sc1.weight.data = self.sc1.weight.data.mul(self.Adj)
		x_1 = self.activate(self.sc1(x_gene))
		#x_1=self.normalize(x_1)
		x_1=x_invmea.mul((self.mp11).mul(self.mp1))+x_1.mul((self.mp12).mul(self.mp1))

		c=np.zeros(x_1.size(1))
		for i in range(x_1.size(1)):
			pred=x_1[:,i:i+1]
			c[i]=abs(c_index(pred, ytime, yevent)-0.5)
			if is_update_layer_masks & (c[i]<0.1):
				if random.random()<0.5:
					with torch.no_grad():
						self.mp1[i]=0
						#print("removing gene_invmea", i)

		toplist=c.argsort()[-self.N_keep:][::-1]
		self.top_invmea_mask.zero_()
		for i in range(self.N_keep):
			print("Gene_invmea",gene_s[toplist[i]],"c-index=",0.5+c[toplist[i]])
			self.top_invmea_mask[toplist[i],i] = 1
		
		kept_invmea=x_1.mm(self.top_invmea_mask).clone().detach()
		self.sc2.weight.data = self.sc2.weight.data.mul(self.edge_mask)
		x_1 = self.activate(self.sc2(x_1))
		#x_1=self.normalize(x_1)
		x_1=x_curv.mul((self.mp21).mul(self.mp2))+x_1.mul((self.mp22).mul(self.mp2))
    
		c=np.zeros(x_1.size(1))
		for i in range(x_1.size(1)):
			pred=x_1[:,i:i+1]
			c[i]=abs(c_index(pred, ytime, yevent)-0.5)
			if is_update_layer_masks & (c[i]<0.1):
				if random.random()<0.5:
					with torch.no_grad():
						self.mp2[i]=0
						#print("removing edge_curv", i)
		toplist=c.argsort()[-self.N_keep:][::-1]
		self.top_curv_mask.zero_()
		for i in range(self.N_keep):
			print("Edge ",gene_s[self.K[0][toplist[i]]],"-",gene_s[self.K[1][toplist[i]]]," c-index=",0.5+c[toplist[i]])
			self.top_curv_mask[toplist[i],i] = 1

		kept_curv=x_1.mm(self.top_curv_mask).clone().detach()
		self.sc3.weight.data = self.sc3.weight.data.mul(self.pathway_mask)
		x_1 = self.activate(self.sc3(x_1))
		x_1=x_1.mul(self.mp3)
		c=np.zeros(x_1.size(1)-1)
		for i in range(x_1.size(1)-1):
			pred=x_1[:,i:i+1]
			c[i]=abs(c_index(pred, ytime, yevent)-0.5)
			if is_update_layer_masks & (c[i]<0.1):
				if random.random()<0.5:
					with torch.no_grad():
						self.mp3[i]=0
						#print("removing edge_curv", i)
		toplist=c.argsort()[-self.N_keep:][::-1]
		self.top_path_mask.zero_()
		for i in range(self.N_keep):
			print("Path ",toplist[i]," c-index=",0.5+c[toplist[i]])
			self.top_path_mask[toplist[i],i] = 1

		kept_path=x_1.mm(self.top_path_mask).clone().detach()
		x_1=self.activate(self.sc4(x_1))
		x_1=self.activate(self.sc5(x_1))
		#x_1 = self.ReLU(self.sc33(x_1))
		###combine age with hidden layer 2
		x_cat = torch.cat((x_1, kept_gene,kept_invmea,kept_curv,kept_path,clinn), 1)
		lin_pred = self.activate(self.sc6(x_cat))

		lin_pred=lin_pred-torch.mean(lin_pred, 1, True)

		lin_pred = self.sc7(lin_pred)
		return lin_pred

	def update_top_gene_mask(self, x_gene,ytime, yevent,gene_s):
		c=np.zeros(x_gene.size(1))
		for i in range(x_gene.size(1)):
			pred=x_gene[:,i:i+1]
			c[i]=abs(c_index(pred, ytime, yevent)-0.5)
		toplist=c.argsort()[-self.N_keep:][::-1]
		self.top_gene_mask.zero_()
		for i in range(self.N_keep):
			print("Gene",gene_s[toplist[i]],"c-index=",0.5+c[toplist[i]])
			self.top_gene_mask[toplist[i],i] = 1

class CombinedFeature_net(nn.Module):
	def __init__(self, In_Nodes, Hidden_Nodes1,Hidden_Nodes2):
		super(CombinedFeature_net, self).__init__()
		###gene layer --> pathway layer
		self.activate = nn.Sigmoid()
		self.sc1 = nn.Linear(In_Nodes, Hidden_Nodes1)
		self.sc2 = nn.Linear(Hidden_Nodes1, Hidden_Nodes2)
		self.out = nn.Linear(Hidden_Nodes2, 1, bias = False)
		#self.out2 = nn.Linear(Hidden_Nodes1, 1, bias = False)
		###pathway layer --> hidden layer
		###

	def forward(self, x):
		###force the connections between gene layer and pathway layer w.r.t. 'pathway_mask'
		x = self.activate(self.sc1(x))
		x = self.activate(self.sc2(x))
		x=x-torch.mean(x, 1, True)
		lin_pred = self.out(x)
		#lin_pred = self.out2(x)
		return lin_pred