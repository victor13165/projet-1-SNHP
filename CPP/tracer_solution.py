import numpy as np
import matplotlib.pyplot as plt
#plt.switch_backend("TkAgg")

sol = np.genfromtxt('resultats.csv',delimiter=',')
#print(sol);
x = sol[0,:]
sol = sol[1:,:]
#print(sol);

fig,ax = plt.subplots();
ax.plot(x,sol[0,:],'b',label='t0')
ax.plot(x,sol[1,:],'r',label='tf')
ax.legend()
ax.grid()
plt.show(block=True)
