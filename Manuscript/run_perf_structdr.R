require(Matrix)
require(igraph)
require(dynwrap)
require(tidyverse)
require(data.table)
args = commandArgs(trailingOnly=TRUE)
t=10
file = args[1]
f = gsub('\\.expression.*','.rds',file)
d=readRDS(f)
g = readMM(file)

G = graph_from_adjacency_matrix(g,mode='max',weighted=T)
edges = unique(as_edgelist(G))

bscore = read.csv(gsub('mm.mtx','bc.csv',file))[,2]
allperfs = data.frame(dataset=NULL,model=NULL,threshold=NULL,keepratio=NULL,correlation=NULL,featureimp_wcor=NULL,F1_branches=NULL,him=NULL)


g.d = pmax(g[cbind(edges[,1],edges[,2])], g[cbind(edges[,2],edges[,1])])
dres.w = add_cell_graph(dataset=d,cell_graph=tibble(from=as.character(d$cell_ids[edges[,1]]),
                                   to=as.character(d$cell_ids[edges[,2]]),length=g.d,directed=F), 
                      to_keep= d$cell_ids[bscore>t])

perf.w = dyneval::calculate_metrics(d, dres.w, metrics = c("correlation","featureimp_wcor","F1_branches","him"))
allperfs=rbind(allperfs, cbind(dataset=f,model=file,threshold=t,keepratio=mean(bscore>t),perf.w[,c("correlation","featureimp_wcor","F1_branches","him")] ))

write.table(allperfs, paste0(file,'.perf.prune.txt'),sep='\t',quote=F)


