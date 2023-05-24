#setwd("D:/Mac_temp/Data_For_CurvNet")
setwd("/media/jiening666/Data/Mac_temp/Data_For_CurvNet")
# load kegg
library(openxlsx)
kegg<-read.xlsx("c2.cp.kegg.v5.2.symbols.xlsx",colNames = FALSE,rowNames = TRUE)
kegg<-kegg[,-1] # remove website 
kegg_genes<-c(t(kegg))
kegg_genes<-na.omit(kegg_genes)
kegg_genes<-unique(kegg_genes)

# load hprd
hprd_genes<-read.table("hprd.txt")
hprd_genes<-hprd_genes[,1]
inter_genes<-intersect(hprd_genes,kegg_genes)
adj<-read.table("adj.txt",sep = ",")
for(i in 1:length(adj)){
  adj[i,i]=0
}
colnames(adj)<-hprd_genes
row.names(adj)<-hprd_genes
adj_inter<-adj[inter_genes,inter_genes]
s<-colSums(adj_inter)
inter_genes<-inter_genes[s>0]
# data set
#setwd("D:/Mac_temp/Data_For_CurvNet/sarc_tcga")
setwd("./Data/Mac_temp/Data_For_CurvNet/lihc_tcga")

CNA <- read.table("data_cna.txt", sep="\t", stringsAsFactors=FALSE, header=TRUE)
CNA<-CNA[!duplicated(CNA[,1]),]
row.names(CNA)<-CNA[,1]
CNA<-CNA[,-c(1,2)]
CNA<-CNA[intersect(inter_genes,row.names(CNA)),-c(1,2)]

RNA<-read.table("data_mrna_seq_v2_rsem.txt", sep="\t", stringsAsFactors=FALSE, header=TRUE)
RNA<-RNA[!duplicated(RNA[,1]),]
RNA<-na.omit(RNA)
row.names(RNA)<-RNA[,1]
RNA<-RNA[,-c(1,2)]
RNA<-RNA[intersect(inter_genes,row.names(RNA)),-c(1,2)]


#methylation <- read.table("data_methylation_hm27.txt", sep="\t", stringsAsFactors=FALSE, header=TRUE)
methylation <- read.table("data_methylation_hm450.txt", sep="\t", stringsAsFactors=FALSE, header=TRUE)
methylation<-methylation[!duplicated(methylation[,1]),]
methylation<-na.omit(methylation)
row.names(methylation)<-methylation[,1]
methylation<-methylation[,-c(1,2)]
methylation<-methylation[intersect(inter_genes,row.names(methylation)),-c(1,2)]

clinn<-read.delim("data_clinical_patient.txt", sep="\t", stringsAsFactors=FALSE, header=TRUE, comment.char = "#")
na_rows<-which(clinn[,"OS_MONTHS"]=="[Not Available]")
if(length(na_rows)>0){
  clinn<-clinn[-na_rows,]
}
patient_list<-clinn[,2]
row.names(clinn)<-patient_list

p1<-colnames(CNA)
p1<-gsub(".[0-9]*$","",p1)
p1<-gsub("\\.","-",p1)
colnames(CNA)<-p1

p2<-colnames(RNA)
p2<-gsub(".[0-9]*$","",p2)
p2<-gsub("\\.","-",p2)
colnames(RNA)<-p2

p3<-colnames(methylation)
p3<-gsub(".[0-9]*$","",p3)
p3<-gsub("\\.","-",p3)
colnames(methylation)<-p3

plist<-Reduce(intersect,list(patient_list,p1,p2,p3))
CNA<-CNA[,plist]
RNA<-RNA[,plist]
methylation<-methylation[,plist]
clinn<-clinn[plist,]

if(!file.exists("out")){
dir.create("out")}

write.csv(methylation,"out/Methyl.csv")
write.csv(RNA,"out/RNA.csv")
write.csv(CNA,"out/CNA.csv")
write.csv(clinn,"out/clinn.csv")
