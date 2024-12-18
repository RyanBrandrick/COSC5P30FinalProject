#Brock University
#COSC 5P30 - Project
#Ryan Brandrick

#all data units are in seconds, meters, meters/second, or meters/second**2

import math
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import scipy.sparse as sp
import dgl.function as fn
from dgl.nn import GraphConv
from sklearn.metrics import roc_auc_score
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans

#The following functions are used to find the realtive mobility and the force attraction between nodes
def deltaPosition(pos, vel, acc, dt):
    #change in position given current position, velocity, acceleration and ammount of time
    return pos + vel*dt + (acc*(dt*82))/2

def relativeForce(k, q1, q2, D):
    #takes relative mobility, relative maintence, and distance between nodes
    if(D == 0):
        D = 0.001
    return k*((q1*q2)/D**2)

def nodeDistance(x_i, x_j, dx_i=0, dx_j=0):
    #takes current and future x and y positions to calculate either curren or future distance between nodes
    return x_i + dx_i - x_j - dx_j

def relativeMobility(nDist1, nDist2, dt):
    #takes node distance at two points in time, and length of time to calculate relative mobility
    return 1/(1+abs(nDist2 - nDist1)*dt)

def relativeMaintenance(range, nDist1, nDist2):
    #takes transmission range, and node distance at two points in time to determine how far nodes are beyonf communication distance
    if nDist1 <= nDist2:
        return range - nDist1
    return range + nDist2

def pairwiseRelativeForce(rf1, rf2):
    #takes two relative forces and finds hypotenuse
    #this force will be used as weight to propogate info in the GNN
    return math.sqrt(rf1**2 + rf2**2)


#input training data
def readCSV():
    with open("highD-dataset-v1.0/data/13_tracks.csv") as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)
        return rows[1:]

#selecte a random frame from the dataset
def pickRandomFrame(rows):
    randNum = random.randrange(1044991)
    randRow = rows[randNum]
    return randRow[0]

def returnFrameRows(frame):
    #return all rows at a given frame with cars moving from left to right
    frameRows = []
    for row in rows:
        if row[0] == frame and float(row[6]) >= 0:
            floatRow = [float(x) for x in row[:10]]
            frameRows.append(floatRow)
    return frameRows

rows = readCSV()

def frameGraph(frameRows, transmissionRange, deltaTime):
    #get graph from frame rows
    F_ij = []
    F_ij = np.zeros(((len(frameRows),len(frameRows))))

    #for each node pair calculate the force realtionship between them if within the transmission range
    for i in range(len(frameRows)):
        for j in range(len(frameRows)-1-i):
            j+=1+i
            Dx_ijt = nodeDistance(frameRows[i][2], frameRows[j][2])
            Dy_ijt = nodeDistance(frameRows[i][3], frameRows[j][3])
            
            if(math.sqrt(Dx_ijt**2 + Dy_ijt**2) <= transmissionRange):#if two nodes are within transmission range of each other
                Dx_ijdt = nodeDistance(deltaPosition(frameRows[i][2],frameRows[i][6],frameRows[i][8],deltaTime),deltaPosition(frameRows[j][2],frameRows[j][6],frameRows[j][8],deltaTime))
                Dy_ijdt = nodeDistance(deltaPosition(frameRows[i][3],frameRows[i][7],frameRows[i][9],deltaTime),deltaPosition(frameRows[j][3],frameRows[j][7],frameRows[j][9],deltaTime))
                
                kx_ij = relativeMobility(Dx_ijt, Dx_ijdt,deltaTime)
                ky_ij = relativeMobility(Dy_ijt, Dy_ijdt,deltaTime)
                
                q_i = q_j = relativeMaintenance(transmissionRange, Dx_ijt, Dx_ijdt)
                
                Fx_ij = relativeForce(kx_ij, q_i, q_j, Dx_ijt)
                Fy_ij = relativeForce(ky_ij, q_i, q_j, Dy_ijt)
                
                F_ij[i][j] = F_ij[j][j] = pairwiseRelativeForce(Fx_ij, Fy_ij)

            else:
                F_ij[i][j] = F_ij[j][i] = 0

    return F_ij

def dglGraph(F_ij):
    src, dst = np.nonzero(F_ij)
    g = dgl.graph((src, dst))
    return g

#load node features from csv data into graph model
def loadNodeFeatures(frameRows, g):
    node_feat = []

    for i in range(len(frameRows)):
        node_feat.append([frameRows[i][6], frameRows[i][7], frameRows[i][2], frameRows[i][3], frameRows[i][8], frameRows[i][9], frameRows[i][4], frameRows[i][5]])

    node_feat = torch.from_numpy(np.array(node_feat))
    g.ndata['feat'] = node_feat.to(torch.float32)

#link prediction
def linkPrediction(g):
    edge_ids = np.arange(g.num_edges())
    edge_ids = np.random.permutation(edge_ids)

    train_size = int(0.9*g.num_edges())
    train_mask = edge_ids[:train_size]
    test_mask = edge_ids[train_size:]
    return train_size, train_mask, test_mask

def trainingEdges(g, train_mask, train_size, test_mask):
    u,v = g.edges() #u=source nodes, v=destination nodes

    #get positive edge for training and testing of g_main
    train_pos_u, train_pos_v = u[train_mask], v[train_mask]
    test_pos_u, test_pos_v = u[test_mask], v[test_mask]

    #get negative edge for taining and testing of g_main
    #create sparse adj matrix
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(),v.numpy())))
    adj = adj.todense()+np.eye(g.num_nodes())

    u_neg, v_neg=np.where(adj==0)

    #sample negative edges
    neg_ids=np.random.choice(len(u_neg),g.num_edges())
    train_neg_u,train_neg_v=u_neg[neg_ids[:train_size]], v_neg[neg_ids[:train_size]]

    test_neg_u,test_neg_v=u_neg[neg_ids[train_size:]], v_neg[neg_ids[train_size:]]

    #positive training graph contains all positive edges
    g_train_pos=dgl.graph((train_pos_u,train_pos_v),num_nodes=g.num_nodes())
    #negative training graph contains all negative edges
    g_train_neg=dgl.graph((train_neg_u,train_neg_v),num_nodes=g.num_nodes())
    
    #graphs used in computing scores in testing
    #positive testing graph contains all positive edges
    g_test_pos=dgl.graph((test_pos_u,test_pos_v),num_nodes=g.num_nodes())
    #negative testing graph contains all negative edges
    g_test_neg=dgl.graph((test_neg_u,test_neg_v),num_nodes=g.num_nodes())
    
    return g_train_pos, g_train_neg, g_test_pos, g_test_neg

#SAGECONV
class GraphSage(nn.Module):
    #inintalize graph with 4 layers, input, 2 hidden and output
    def __init__(self,in_dim,hidden_dim, out_dim):
        super().__init__()
        self.conv1=GraphConv(in_dim,hidden_dim,weight=True)#aggregator_type="mean",
        self.conv2=GraphConv(hidden_dim,hidden_dim,weight=True)#aggregator_type="mean",
        self.conv3=GraphConv(hidden_dim,out_dim,weight=True)#aggregator_type="mean",
    
    def forward(self,g,features):
        #model forward passing with intermediate relu activation
        h=self.conv1(g,features)
        h=F.relu(h)
        h=self.conv2(g,h)
        h=F.relu(h)
        h=self.conv2(g,h)
        h=F.relu(h)
        h=self.conv3(g,h)
        return h
    
    def predict(self,g,h):
        #compute dot product of src & dst features
        #store it as a new feature for g.edata
        with g.local_scope():
            g.ndata['h']=h
            g.apply_edges(fn.u_dot_v('h','h','score'))
            return g.edata['score'][:,0]
        
    def loss(self,pos_scores,neg_scores):
        # pos_scores = scores for positive edges
        # neg_scores = scores for negative edges
        scores=torch.cat([pos_scores,neg_scores])
        labels=torch.cat([torch.ones(pos_scores.shape[0]),torch.zeros(neg_scores.shape[0])])

        return F.binary_cross_entropy_with_logits(scores,labels)
    
    def auc_score(self,pos_scores,neg_scores):
        # roc_auc_score only accepts numpy array as inputs
        scores=torch.cat([pos_scores,neg_scores]).detach().numpy()
        labels=torch.cat([torch.ones(pos_scores.shape[0]),
                          torch.zeros(neg_scores.shape[0])]).detach().numpy()
        return roc_auc_score(labels,scores)
    
def train(model,g_main,g_train_pos,g_train_neg,optimizer):
    model.train()
    
    # forward and backward
    optimizer.zero_grad()

    # prediction on g_main
    h=model(g_main,g_main.ndata['feat']) # [num_nodes,feat_dim]

    # prediction scores 
    pos_scores=model.predict(g_train_pos,h)
    neg_scores=model.predict(g_train_neg,h)
    loss=model.loss(pos_scores,neg_scores)
    loss.backward()
    optimizer.step()

    # compute auc
    auc_score=model.auc_score(pos_scores,neg_scores)

    return loss, auc_score

@torch.no_grad()
def evaluate(model,g_main,g_test_pos, g_test_neg):
    model.eval()
    h=g_main.ndata['feat'] # these features are extracted after training is finished
    
    # forward
    pos_scores=model.predict(g_test_pos,h)
    neg_scores=model.predict(g_test_neg,h)

    loss=model.loss(pos_scores,neg_scores)

    auc_score=model.auc_score(pos_scores,neg_scores)

    return loss,auc_score


##################################################


in_dim = 8
hidden_dim = 4
out_dim = 4
num_epochs = 400

#initialize gnn model
model=GraphSage(in_dim,hidden_dim, out_dim)
optimizer=torch.optim.Adam(model.parameters(), lr=0.003)#set learning rate of 0.003

#train model for set number of epochs
def trainModel(num_epochs, model, g_main, g_train_pos, g_train_neg, optimizer):
    for epoch in range(num_epochs):
        train_loss, train_auc = train(model, g_main, g_train_pos, g_train_neg, optimizer)

#read data from csv and train model
def loadAndTrainGraph(randomFrame):
    frameRows = returnFrameRows(randomFrame)
    F_ij = frameGraph(frameRows, transmissionRange, deltaTime)
    g = dglGraph(F_ij)
    loadNodeFeatures(frameRows, g)
    train_size, train_mask, test_mask = linkPrediction(g)
    g_main = dgl.remove_edges(g,test_mask)#g_main is original graph with test edges removed
    g_train_pos, g_train_neg, g_test_pos, g_test_neg = trainingEdges(g, train_mask, train_size, test_mask)
    g_main = dgl.add_self_loop(g_main)

    trainModel(num_epochs, model, g_main, g_train_pos, g_train_neg, optimizer)
    return g


##################################################

transmissionRange = 100#transmissionrange defined as 100m
deltaTime = 5#delta time is 5 seconds
random.seed(100)

randomFrame = None

#determine how many unique graphs to train the model on
for i in range(1000):#5, 50, 100, 1000 - Reduce this value to increase the speed of the program at the expense of training the model - Depending on Machine Specs, running at 1000 may take several hours
    try:
        print("Modeling for graph: ",i)
        randomFrame = pickRandomFrame(rows)
        g = loadAndTrainGraph(randomFrame)
    except dgl._ffi.base.DGLError:
        print("dgl._ffi.base.DGLError")
    
##################################################

frameRows = returnFrameRows(randomFrame)

frameID = frameRows[0][0]#get frame ID of random Frame
#return all car IDs from selected frame
cars = []
for i in range(len(frameRows)):
    if frameRows[i][0] == frameID:
        cars.append(frameRows[i][1])
        
        
F_ij = frameGraph(frameRows, transmissionRange, deltaTime)
g = loadAndTrainGraph(randomFrame)
g = dgl.add_self_loop(g)
with torch.no_grad():
    out_embeddings=model(g,g.ndata["feat"])




#K-MEANS Clustering
#cluster the nodes based on the k-means algorithm
kmeans = KMeans(n_clusters=3,random_state=111)
kmeans.fit(out_embeddings)
labels = kmeans.labels_

carLabels = zip(cars,labels)


#draw graph based on vehicle location and node cluster
w, h = 1000, 300
img = Image.new("RGB", (w,h))
draw = ImageDraw.Draw(img)

colors = []
for node in range(len(labels)):
    if labels[node] == 0:
        colors.append('white')
    elif labels[node] == 1:
        colors.append('red')
    else:
        colors.append('blue')

for i in range(len(frameRows)):
    for j in range(len(frameRows)-1-i):
        j+=1+i
        if F_ij[i][j] != 0:
            shape = [(frameRows[i][2]*2,frameRows[i][3]*15-400), (frameRows[j][2]*2,frameRows[j][3]*15-400)]
            draw.line(shape, fill ="green", width = 0)
    draw.ellipse((frameRows[i][2]*2-5,frameRows[i][3]*15-400-5, frameRows[i][2]*2+5,frameRows[i][3]*15-400+5), fill=colors[i], outline=colors[i])
            
img.show()

#Performance Evaluation, Cars Breaking Clusters - INCOMPLETE, too computationally expensive
#go back one frameID, check if any cars of cars array exist in that frame, if yes check that frames clusters and go back another frame, if none exist exit loop
#Completion of this stage was unachievable as testing of the function was too expensive of a process and could not sustainably determine if the function was behaving as expected
#this is due to there being hundreds to thousands of frame to validate each graph against