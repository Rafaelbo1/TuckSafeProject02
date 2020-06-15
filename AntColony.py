import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import pandas as pd
# Primeiro passo para importar dados
df = pd.read_csv('Capitais.csv')
X = np.array(df.drop(['km'],1))
y = np.array(df['km'])
c = pd.read_csv('city.csv')
city = np.array(c.drop(['km'],1))
n = len(y)
D = X

# Parâmetro de inicialização
m = 50
alpha = 1
beta = 5
rho = 0.1
Q = 1
Eta = np.ones((n, n))
Tau = np.ones((n, n))
Caminhos = np.zeros((m, n), dtype=np.int)
iter = 0
iter_max = 200
Route_best = np.zeros((iter_max, n))
Length_best = np.zeros((iter_max, 1))
Length_ave = np.zeros((iter_max, 1))


for i in range(n):
    for j in range(n):
        Eta[i, j] = 1/D[i, j]

while iter < iter_max:
    start = np.zeros((m, 1), dtype=np.int)
    for i in range(m):
        start[i] = random.randint(0, n-1)
    Caminhos[:, 0] = start[:, 0]
    citys_index = range(0, n)
    for i in range(m):
        tabu = []
        for j in range(1, n):
            tabu.append(Caminhos[i, j-1])
            allow = [a for a in citys_index if a not in tabu]
            P = copy.deepcopy(allow)
            for k in range(len(allow)):
                P[k] = ((Tau[int(tabu[-1]), allow[k]])**alpha)*((Eta[int(tabu[-1]), allow[k]])**beta)
            sum = np.sum(P)
            P = P/sum
            Pc = []
            s2 = 0
            for k in P:
                s2 = s2 + k
                Pc.append(s2)
            r = random.random()
            target_index = [a for a in Pc if a > r][0]
            target_index = Pc.index(target_index)
            target = allow[target_index]
            Caminhos[i, j] = int(target)
    # Calcule a distância do caminho de cada formiga
    L = [0]
    Length = L * m
    for i in range(m):
        Route = Caminhos[i, :]
        for j in range(n-1):
            Length[i] = Length[i] + D[Route[j], Route[j+1]]
        Length[i] = Length[i] + D[Route[n-1], Route[0]]

    # Calcule a distância do caminho mais curto e a distância média
    min_Length = np.min(Length)
    min_index = Length.index(min_Length)
    if iter == 0:
        Length_best[iter] = min_Length
        Route_best[iter, :] = Caminhos[min_index, :]
        Length_ave[iter] = np.mean(Length)
    elif iter >= 1:
        if min_Length-Length_best[iter-1][0] >= 0:
            Length_best[iter] = Length_best[iter-1][0]
        else:
            Length_best[iter] = min_Length
        Length_ave[iter] = np.mean(Length)
        if Length_best[iter] == min_Length:
            Route_best[iter, :] = Caminhos[min_index, :]
        else:
            Route_best[iter, :] = Route_best[iter, :]

    # Atualizar feromônio
    Delta_Tau = np.zeros((n, n))
    # Cálculo da formiga por formiga
    for i in range(m):
        # Cidade por cidade
        for j in range(n-1):
            Delta_Tau[Caminhos[i, j], Caminhos[i, j+1]] = Delta_Tau[Caminhos[i, j], Caminhos[i, j+1]] + Q/Length[i]
        Delta_Tau[Caminhos[i, n-1], Caminhos[i, 0]] = Delta_Tau[Caminhos[i, n-1], Caminhos[i, 0]] + Q / Length[i]
    Tau = (1 - rho) * Tau + Delta_Tau
    iter += 1
    Caminhos = np.zeros((m, n), dtype=np.int)
# Exibição resultado
Shortest_Length = np.min(Length_best)
index = Length_best.tolist().index(Shortest_Length)
Route_temp = Route_best[index, :]
Shortest_Route = [ int(i) for i in Route_temp ]
print("Menor distância:", Shortest_Length)
print("Caminho mais curto:", Shortest_Route)

fig, ax = plt.subplots(figsize = (70,40))
final_route = np.zeros((n, 2), dtype=np.int)
for i in range(len(Shortest_Route)):
    final_route[i, :] = (city[Shortest_Route[i], :])

ax.plot(final_route[:, 0], final_route[:, 1], "r")

for i,t in enumerate(Shortest_Route):
    txt = y[t]
    if i == 0:
        ax.annotate(("Ini.", txt), (final_route[i, 0], final_route[i, 1]))
    elif i == n-1:
        ax.annotate(("fim", txt), (final_route[i, 0], final_route[i, 1]))
    else:
        ax.annotate((txt), (final_route[i, 0], final_route[i, 1]))
plt.show()

plt.plot(Length_best, "r-")
plt.plot(Length_ave, "b-")
plt.show()