import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import operator

N = 10 ** 3
G_BA = nx.generators.barabasi_albert_graph(N, 3, seed=None)

# check edge

[e for e in G_BA.edges]
# check weight numbers
len(G_BA.edges.data())

# generate random weight(normal distribution)
mu, sigma = 0.6, 0.2  # mean and standard deviation

s=[]

# add weight list to weight
for e in G_BA.edges:
    G_BA[e[0]][e[1]]['weight'] = np.random.normal(mu, sigma)
    s.append(G_BA[e[0]][e[1]]['weight'])


# plot weight distribution
count, bins, ignored = plt.hist(s, 30, density=True)
plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
         np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
         linewidth=2, color='r')
plt.show()

# check edges' weight
[i for i in G_BA.edges.data()]


##Set attributes to each nodes
# infected
# 0 = healty ; 1 = infected ;
for i in range(len(G_BA.nodes)):
    G_BA.nodes[i]["infected"] = 0

# counter
for i in range(len(G_BA.nodes)):
    G_BA.nodes[i]["counter"] = 0

# immunity
for i in range(len(G_BA.nodes)):
    G_BA.nodes[i]["immunity"] = 0

# tem
for i in range(len(G_BA.nodes)):
    G_BA.nodes[i]["tem"] = 0

# dead
for i in range(len(G_BA.nodes)):
    G_BA.nodes[i]["dead"] = 0

# next
for i in range(len(G_BA.nodes)):
    G_BA.nodes[i]["next"] = 2

# state
for i in range(len(G_BA.nodes)):
    G_BA.nodes[i]["state"] = 'health'

# age(int)
# Level1:0~14;2:15~24;3:25~54;4:55~64;5:65+
# relate to real-world data
for i in range(len(G_BA.nodes)):
    tem = random.random()
    if (tem <= 0.1873):
        G_BA.nodes[i]["age"] = 1
    elif (tem > 0.1873 and tem <= 0.32):
        G_BA.nodes[i]["age"] = 2
    elif (tem > 0.32 and tem <= 0.7145):
        G_BA.nodes[i]["age"] = 3
    elif (tem > 0.7145 and tem <= 0.8436):
        G_BA.nodes[i]["age"] = 4
    else:
        G_BA.nodes[i]["age"] = 5

H_BA = G_BA.copy()
## main


# recover the weigth
def upprunew(G_BA, i):
    for _ in G_BA.edges(i):
        G_BA[_[0]][_[1]]['weight'] = H_BA[_[0]][_[1]]['weight']


def recover(G_BA, i):
    G_BA.nodes[i]["next"] = 2
    G_BA.nodes[i]["state"] = 'health'
    G_BA.nodes[i]["immunity"] = int(np.random.normal(60, 20))
    G_BA.nodes[i]["infected"] = 0
    G_BA.nodes[i]["counter"] = 0
    upprunew(G_BA, i)


# prune for quarantine
def prune_up(G_BA, i):
    for _ in G_BA.edges(i):
        if G_BA[_[0]][_[1]]['weight'] <= mu + sigma:
            G_BA[_[0]][_[1]]['weight'] = 0


def prune(G_BA, i, coef):
    for _ in G_BA.edges(i):
        G_BA[_[0]][_[1]]['weight'] = H_BA[_[0]][_[1]]['weight'] * coef

#prune
def policy(G_BA, coef,power):
    pow = 0.999**power
    for i in range(len(G_BA.nodes)):
        if G_BA.nodes[i]["dead"] == 0:
            # for people normal or unconscious
            if G_BA.nodes[i]["state"] == 'health':
                for _ in G_BA.edges(i):
                    G_BA[_[0]][_[1]]['weight'] = H_BA[_[0]][_[1]]['weight'] * coef *pow
    for i in range(len(G_BA.nodes)):
        if G_BA.nodes[i]["dead"] == 0:
            # for people under quarantine
            if G_BA.nodes[i]["state"] == 'q':
                # policy prune add quarantine prune
                for _ in G_BA.edges(i):
                    G_BA[_[0]][_[1]]['weight'] = H_BA[_[0]][_[1]]['weight'] * coef*pow
                    if G_BA[_[0]][_[1]]['weight'] <= mu + sigma:
                        G_BA[_[0]][_[1]]['weight'] = 0
    for i in range(len(G_BA.nodes)):
        if G_BA.nodes[i]["dead"] == 0:
            if G_BA.nodes[i]["state"] == 'h':
                # policy prune add hospital prune
                for _ in G_BA.edges(i):
                    G_BA[_[0]][_[1]]['weight'] = 0




# check infection, determine symptomatic or not
def update(G_BA, theta, coef):
    for i in range(len(G_BA.nodes)):
        if G_BA.nodes[i]["dead"] == 0:
            sum = 0
            index = 0
            lst1 = []# live neighbours list
            for _ in H_BA.edges(i):#_[1] is neighbor
                if G_BA.nodes[_[1]]["dead"] == 0:
                    lst1.append(_[1])
            for _ in lst1:
                sum = sum + G_BA[i][_]['weight'] * G_BA.nodes[_]["infected"]
            sum = sum * coef
            if sum > theta:
                G_BA.nodes[i]["tem"] = 1

    for i in range(len(G_BA.nodes)):
        if G_BA.nodes[i]["dead"] == 0:
            # update immune
            if G_BA.nodes[i]["immunity"] >= 1:
                G_BA.nodes[i]["immunity"] = G_BA.nodes[i]["immunity"] - 1
            else:
                if G_BA.nodes[i]["tem"] == 1:
                    G_BA.nodes[i]["infected"] = 1


def dead(G_BA, H_BA, i):
    G_BA.nodes[i]["dead"] = 1
    H_BA.nodes[i]["dead"] = 1

def dotest(G_BA):
    global coeff
    coeff=1
    tem = 0
    tem2 = 0
    for i in range(len(G_BA.nodes)):
        if G_BA.nodes[i]["dead"] == 0:
            tem2 += 1
        if G_BA.nodes[i]["infected"] == 1:
            tem += 1
    infectednumlst.append(tem)
    coeff =(1-tem/tem2)
    totalpopulst.append(tem2)



coeff=1


def main(G, days, p1, d1, d1_1, p2, d2, p3, d3, d3_3, p4, d4, theta_i , coef_i):

    for i in  range(days):
        dotest(G)
        policy(G, coeff,i)
        update(G,theta_i, coef_i)
        sus(G, p1, d1, d1_1, p2, d2, p3, d3, d3_3, p4, d4)




def sus(G_BA, p1, d1, d1_1, p2, d2, p3, d3, d3_3, p4, d4):
    ##Determine whether infect
    #tem = 0
    for i in range(len(G_BA.nodes)):
        if G_BA.nodes[i]["dead"] == 0:
            if (G_BA.nodes[i]["infected"] == 1):
                #tem += 1
                # Set health state to illness state
                if (G_BA.nodes[i]["state"] == 'health'):
                    if (random.random() > p1):
                        G_BA.nodes[i]["state"] = 'sym'
                    else:
                        G_BA.nodes[i]["state"] = 'asym'
                ##Determine whether Symptomatic
                if (G_BA.nodes[i]["state"] == 'sym'):
                    # incubation period d2
                    if (G_BA.nodes[i]["counter"] < d2):
                        G_BA.nodes[i]["counter"] += 1
                        # infect(G_BA, i)
                    # ask node to quarantine
                    else:
                        G_BA.nodes[i]["state"] = 'q'
                        G_BA.nodes[i]["counter"] = 0



                elif (G_BA.nodes[i]["state"] == 'asym'):
                    # first time
                    if (G_BA.nodes[i]["next"] == 2):
                        if (random.random() > p2):
                            G_BA.nodes[i]["next"] = 0
                        else:
                            G_BA.nodes[i]["next"] = 1
                    # process for recovering people
                    elif G_BA.nodes[i]["next"] == 0:
                        # processing
                        if G_BA.nodes[i]["counter"] < d1:
                            G_BA.nodes[i]["counter"] += 1
                        # determine
                        else:
                            recover(G_BA, i)
                    # process for unlucky people
                    else:
                        # processiong
                        if (G_BA.nodes[i]["counter"] < d1_1):
                            G_BA.nodes[i]["counter"] += 1
                        # determine
                        else:
                            G_BA.nodes[i]["state"] = 'sym'
                            G_BA.nodes[i]["next"] = 2
                            G_BA.nodes[i]["counter"] = 0


                elif (G_BA.nodes[i]["state"] == 'q'):
                    if (G_BA.nodes[i]["next"] == 2):
                        if (random.random() > p3):
                            G_BA.nodes[i]["next"] = 0
                        else:
                            G_BA.nodes[i]["next"] = 1
                    # lucky people
                    elif (G_BA.nodes[i]["next"] == 0):
                        # processing
                        if (G_BA.nodes[i]["counter"] < d3):
                            G_BA.nodes[i]["counter"] += 1

                        # determine
                        else:
                            recover(G_BA, i)
                    # unlucky
                    else:
                        if (G_BA.nodes[i]["counter"] < d3_3):
                            G_BA.nodes[i]["counter"] += 1
                        else:
                            G_BA.nodes[i]["state"] = 'h'
                            G_BA.nodes[i]["next"] = 2
                            G_BA.nodes[i]["counter"] = 0

                elif (G_BA.nodes[i]["state"] == 'h'):
                    if (G_BA.nodes[i]["counter"] < d4):
                        G_BA.nodes[i]["counter"] += 1
                    else:
                        if (random.random() < p4):
                            # dead
                            dead(G_BA, H_BA, i)
                        else:
                            recover(G_BA, i)
    #print(tem)






G_BA.nodes[1]["infected"] = 1
# G_BA.nodes[1]["infected"] = 1
# G_BA.nodes[0]["infected"] = 1
# G_BA.nodes[3]["infected"] = 1

# theta_i: threshold of infection
# coef_i: coefficient of infection
# coef_p: coefficient of policy


# p1:asymptomatic rate
# p2:asym to sym rate
# p3:qua to hos rate
# p4: death rate in hospital



#refresh
infectednumlst=[]
totalpopulst=[]
G_BA = H_BA.copy()
G_BA.nodes[1]["infected"] = 1

main(G = G_BA, days=365, p1=0.8, d1=7, d1_1=2, p2=0.5, d2=3, p3=0.02, d3=3, d3_3=3, p4=0.5, d4=3, theta_i = 0.696425, coef_i = 1)

infectednumlst
totalpopulst


import matplotlib.pyplot as plt
plt.plot(range(0,365),infectednumlst)
plt.plot(range(0,365),totalpopulst)