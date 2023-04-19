library (readxl); library(caret); library(funModeling); library(dplyr);
library(lattice); library (Hmisc); library(survival);
library(tableone); library(survminer); library (survMisc);library (ggfortify); library (ranger);  library (ggplot2); library (survMisc);
library (party); library (partykit); library(carData); library(LabRS)
library(limma); library(readr); library(calibrate); library(MASS); library(readr); 
library(LabRS); library(carData)

################################################################################

#CALCULATING OUTCOMES#

d <- read.csv("datasets/GSE117556_clinic.csv", sep = ",")

d_outcomes <- data.frame(d$Accession, d$GEO_sample_name, d$REG_DT, d$RESP_ASSESS_treclass, d$PROG_DT,  d$PFS, d$PFS_status, d$FU_DT, d$OS, d$OS_status)

#data format conversion
d_outcomes$d.REG_DT <- as.POSIXct(d_outcomes$d.REG_DT, format="%m/%d/%Y")
d_outcomes$d.PROG_DT <- as.POSIXct(d_outcomes$d.PROG_DT, format="%m/%d/%Y")
d_outcomes$d.FU_DT <- as.POSIXct(d_outcomes$d.FU_DT, format="%m/%d/%Y")

#TTP creation
d_outcomes$TTP = NA
as.numeric(d_outcomes$TTP)

#if PFS = 1, TTP == PFS, else  TTP == OS
d_outcomes$TTP <- ifelse(d_outcomes$d.PFS_status == "1",d_outcomes$d.PFS, d_outcomes$d.OS)
# 


#POD12 creation
d_outcomes$POD12 = "NA"

#se la PFS = 0, la OS = 0, il paziente non ha avuto un POD
d_outcomes$POD12 <- ifelse(d_outcomes$d.PFS_status == "0" & d_outcomes$d.OS_status == "0", "No POD", "NA")

#se il paziente ha TTP < X e non è NO POD, allora POD24= Early POD
d_outcomes$POD12[d_outcomes$TTP < 12 & d_outcomes$POD12 != "No POD" ] = "Early POD"

#se il paziente ha TTP >= X e non è NO POD, allora POD12= Late POD
d_outcomes$POD12[d_outcomes$TTP >= 12 & d_outcomes$POD12 != "No POD" ] = "Late POD"

# # Survival Early PODX vs. Late PODX vs. No POD
sopravv <-survfit(Surv(d.OS,d.OS_status)~d_outcomes$POD12,data=d_outcomes); sopravv
pp=ggsurvplot(sopravv, risk.table=TRUE, conf.int=FALSE, xlim=c(0,80), pval=TRUE,ylab="Overall Survival",
              risk.table.y.col=TRUE, risk.table.y.text=TRUE, xlab="Time (months)", break.time.by=20,
              palette= c("red","blue","gray"),legend.labs = c("Early POD12","Late POD12", "No POD"),
              legend="top", risk.table.title="No. at risk",font.x=16,font.y=16,
              font.tickslab=16,font.legend=16,pval.size=6.5,risk.table.height = 0.2); pp

# Merge Late PODX e No POD in reference group
d_outcomes$POD12[d_outcomes$POD12 == "Late POD" | d_outcomes$POD12 == "No POD" ] = "Reference"

# # Survival Early PODX vs. Reference
sopravv <-survfit(Surv(d.OS,d.OS_status)~d_outcomes$POD12,data=d_outcomes); sopravv
pp=ggsurvplot(sopravv, risk.table=TRUE, conf.int=FALSE, xlim=c(0,80), pval=TRUE,ylab="Overall Survival",
              risk.table.y.col=TRUE, risk.table.y.text=TRUE, xlab="Time (months)", break.time.by=20,
              palette= c("red","blue"),legend.labs = c("Early POD12","Reference"),
              legend="top", risk.table.title="No. at risk",font.x=16,font.y=16,
              font.tickslab=16,font.legend=16,pval.size=6.5,risk.table.height = 0.2); pp


#POD24 creation
d_outcomes$POD24 = "NA"

#se la PFS = 0, la OS = 0, il paziente non ha avuto un POD
d_outcomes$POD24 <- ifelse(d_outcomes$d.PFS_status == "0" & d_outcomes$d.OS_status == "0", "No POD", "NA")

#se il paziente ha TTP < X e non è NO POD, allora POD24= Early POD 
d_outcomes$POD24[d_outcomes$TTP < 24 & d_outcomes$POD24 != "No POD" ] = "Early POD"

#se il paziente ha TTP >= X e non è NO POD, allora POD24= Late POD
d_outcomes$POD24[d_outcomes$TTP >= 24 & d_outcomes$POD24 != "No POD" ] = "Late POD"

# # Survival Early PODX vs. Late PODX vs. No POD
sopravv <-survfit(Surv(d.OS,d.OS_status)~d_outcomes$POD24,data=d_outcomes); sopravv
pp=ggsurvplot(sopravv, risk.table=TRUE, conf.int=FALSE, xlim=c(0,80), pval=TRUE,ylab="Overall Survival",
              risk.table.y.col=TRUE, risk.table.y.text=TRUE, xlab="Time (months)", break.time.by=20,
              palette= c("red","blue","gray"),legend.labs = c("Early POD24","Late POD24", "No POD"),
              legend="top", risk.table.title="No. at risk",font.x=16,font.y=16,
              font.tickslab=16,font.legend=16,pval.size=6.5,risk.table.height = 0.2); pp

# Merge Late PODX e No POD in reference group
d_outcomes$POD24[d_outcomes$POD24 == "Late POD" | d_outcomes$POD24 == "No POD" ] = "Reference"

# # Survival Early PODX vs. Reference
sopravv <-survfit(Surv(d.OS,d.OS_status)~d_outcomes$POD24,data=d_outcomes); sopravv
pp=ggsurvplot(sopravv, risk.table=TRUE, conf.int=FALSE, xlim=c(0,50), pval=TRUE,ylab="Overall Survival",
              risk.table.y.col=TRUE, risk.table.y.text=TRUE, xlab="Time (months)", break.time.by=20,
              palette= c("red","blue"),legend.labs = c("Early POD24","Reference"),
              legend="top", risk.table.title="No. at risk",font.x=16,font.y=16,
              font.tickslab=16,font.legend=16,pval.size=6.5,risk.table.height = 0.2); pp

PFS = ifelse(d_outcomes$d.PFS == "na", NA, 0)
PFS[d_outcomes$d.PFS_status == 1] = 1

OS = ifelse(d_outcomes$d.OS == "na", NA, 0)
OS[d_outcomes$d.OS_status == 1] = 1

pod12 = as.numeric(factor(d_outcomes$POD12, levels=c("Reference", "Early POD")))-1
pod24 = as.numeric(factor(d_outcomes$POD24, levels=c("Reference", "Early POD")))-1

resp_assess = ifelse(d_outcomes$d.RESP_ASSESS_treclass == "na", NA, 0)
resp_assess[d_outcomes$d.RESP_ASSESS_treclass == "CR"] = 1

out <- data.frame(OS, PFS, pod12, pod24, resp_assess)
out_d <- data.frame(t(out))
colnames(out_d)<-d$Accession

NAME = c("OS", "PFS", "POD12", "POD24", "RESP_ASSESS")

out_d <- data.frame(NAME, out_d)

################################################################################

################################################################################

#PURIFING CIBERSORTx DATA AND DATASET PREPARING#

################################################################################

#CD3

Dato<-as.matrix(read.table("txt_CIBERSORTx/CIBERSORTxHiRes_NA_CD3_GC_Window20.txt",header=TRUE,row.names=1,sep="\t"))
str(Dato)
dim(Dato)

index1 = which(Dato==0,arr.ind = TRUE)#indici valore 1
index1bis = which(rownames(Dato)%in%rownames(index1))

indexNA = which(is.na(Dato),arr.ind = TRUE)
indexNAbis = which(rownames(Dato)%in%rownames(indexNA))

indexVar=nearZeroVar(t(Dato))

index=c(index1bis,indexNAbis,indexVar)
length(index)
Dato_clean = Dato[-index,]
dim(Dato_clean)

library(limma)
min(Dato_clean)
max(Dato_clean)
str(Dato_clean)
dev.new()
#plotDensities(log2(Dato_clean),legend=FALSE)

df = data.frame(log2(Dato_clean))
NAME = row.names(df)

gene_data <- data.frame(NAME, df)

dataset_ <- rbind(gene_data, out_d)

write.csv(dataset_, "Sha_CBSx_CD3.csv", sep=",", row.names = F)

################################################################################

#CD3_Medulla

Dato<-as.matrix(read.table("txt_CIBERSORTx/CIBERSORTxHiRes_NA_CD3_Medulla_Window20.txt",header=TRUE,row.names=1,sep="\t"))
str(Dato)
dim(Dato)

index1 = which(Dato==0,arr.ind = TRUE)#indici valore 1
index1bis = which(rownames(Dato)%in%rownames(index1))

indexNA = which(is.na(Dato),arr.ind = TRUE)
indexNAbis = which(rownames(Dato)%in%rownames(indexNA))

indexVar=nearZeroVar(t(Dato))

index=c(index1bis,indexNAbis,indexVar)
length(index)
Dato_clean = Dato[-index,]
dim(Dato_clean)

library(limma)
min(Dato_clean)
max(Dato_clean)
str(Dato_clean)
dev.new()
#plotDensities(log2(Dato_clean),legend=FALSE)

df = data.frame(log2(Dato_clean))
NAME = row.names(df)

gene_data <- data.frame(NAME, df)

dataset_ <- rbind(gene_data, out_d)

write.csv(dataset_, "Sha_CBSx_CD3_Medulla.csv", sep=",", row.names = F)

################################################################################

#CD11

Dato<-as.matrix(read.table("txt_CIBERSORTx/CIBERSORTxHiRes_NA_CD11c_GC_Window20.txt",header=TRUE,row.names=1,sep="\t"))
str(Dato)
dim(Dato)

index1 = which(Dato==0,arr.ind = TRUE)#indici valore 1
index1bis = which(rownames(Dato)%in%rownames(index1))

indexNA = which(is.na(Dato),arr.ind = TRUE)
indexNAbis = which(rownames(Dato)%in%rownames(indexNA))

indexVar=nearZeroVar(t(Dato))

index=c(index1bis,indexNAbis,indexVar)
length(index)
Dato_clean = Dato[-index,]
dim(Dato_clean)

library(limma)
min(Dato_clean)
max(Dato_clean)
str(Dato_clean)
dev.new()
#plotDensities(log2(Dato_clean),legend=FALSE)

df = data.frame(log2(Dato_clean))
NAME = row.names(df)

gene_data <- data.frame(NAME, df)

dataset_ <- rbind(gene_data, out_d)

write.csv(dataset_, "Sha_CBSx_CD11.csv", sep=",", row.names = F)

################################################################################

#CD20

Dato<-as.matrix(read.table("txt_CIBERSORTx/CIBERSORTxHiRes_NA_CD20_GC_Window20.txt",header=TRUE,row.names=1,sep="\t"))
str(Dato)
dim(Dato)

index1 = which(Dato==0,arr.ind = TRUE)#indici valore 1
index1bis = which(rownames(Dato)%in%rownames(index1))

indexNA = which(is.na(Dato),arr.ind = TRUE)
indexNAbis = which(rownames(Dato)%in%rownames(indexNA))

indexVar=nearZeroVar(t(Dato))

index=c(index1bis,indexNAbis,indexVar)
length(index)
Dato_clean = Dato[-index,]
dim(Dato_clean)

library(limma)
min(Dato_clean)
max(Dato_clean)
str(Dato_clean)
dev.new()
#plotDensities(log2(Dato_clean),legend=FALSE)

df = data.frame(log2(Dato_clean))
NAME = row.names(df)

gene_data <- data.frame(NAME, df)

dataset_ <- rbind(gene_data, out_d)

write.csv(dataset_, "Sha_CBSx_CD20.csv", sep=",", row.names = F)

################################################################################

#Stroma

Dato<-as.matrix(read.table("txt_CIBERSORTx/CIBERSORTxHiRes_NA_Stroma_Medulla_Window20.txt",header=TRUE,row.names=1,sep="\t"))
str(Dato)
dim(Dato)

index1 = which(Dato==0,arr.ind = TRUE)#indici valore 1
index1bis = which(rownames(Dato)%in%rownames(index1))

indexNA = which(is.na(Dato),arr.ind = TRUE)
indexNAbis = which(rownames(Dato)%in%rownames(indexNA))

indexVar=nearZeroVar(t(Dato))

index=c(index1bis,indexNAbis,indexVar)
length(index)
Dato_clean = Dato[-index,]
dim(Dato_clean)

library(limma)
min(Dato_clean)
max(Dato_clean)
str(Dato_clean)
dev.new()
#plotDensities(log2(Dato_clean),legend=FALSE)

df = data.frame(log2(Dato_clean))
NAME = row.names(df)

gene_data <- data.frame(NAME, df)

dataset_ <- rbind(gene_data, out_d)

write.csv(dataset_, "Sha_CBSx_Stroma.csv", sep=",", row.names = F)


