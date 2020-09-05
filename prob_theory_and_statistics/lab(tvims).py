
# coding: utf-8

# ## Теория вероятностей и математическая статистика
# 
# ## Задание №1
# #### Основные понятия математической статистики. Вариационный ряд. Эмпирическая  функция распределения.
# 

# In[167]:

import matplotlib.pyplot as plt
import numpy as np
import math
import random
import pandas as pd


def gen(n):
    a = -4
    b = 2
    xlist = []
    for i in range(n):
        e = random.uniform(0, 1)
        xlist.append(e * (b - a) + a)
    ylist = np.array([x ** (1/3) for x in xlist])
    ylist.sort()
    return ylist


def get_segments(n):
    M = 0
    if n <= 100:
        M = int(math.sqrt(n))
    else:
        M = int(3.05*math.log(n))
    return M


# In[168]:


# Зависимость эмпирической функции от n:
# 
# **n = 10**

# In[169]:

def F_y():
    y = np.arange(0, 1, 0.0001)
    F = np.array([(elem ** 3 + 4) / 6 for elem in y])
    plt.plot(y, F, 'r', label='RF')
    plt.ylabel('F(y)')
    plt.xlabel('y')
    plt.legend(loc='upper left')
    
def F(y, sample_y):
    func = lambda x: 1 if x < y else 0
    return sum(map(func, sample_y))/float(len(sample_y))


def plot_F(variational_ser):
    _y = np.arange(0, 1, 0.0001)
    F_y = []
    for y in _y:
        F_y.append(F(y, variational_ser))
    plt.plot(_y, F_y, label='EF')
    plt.ylabel('F*(y)')
    plt.xlabel('y')
    plt.legend(loc='upper left')


# In[170]:

n = 100
d = gen(n)
plot_F(d)
F_y()
plt.show()


# **n = 30**

# In[171]:

n = 300
d = gen(n)
plot_F(d)
F_y()

plt.show()


# **n = 100**

# In[172]:

n = 100
d = gen(n)
plot_F(d)
F_y()
plt.show()


# **n = 1000**

# In[173]:

n = 500
d = gen(n)
plot_F(d)
F_y()
plt.show()


# **Вывод:**
# 
# Все в жизни хорошо.

# ## Задание №2
# #### Статистический ряд. Построение гистограммы равноинтервальным методом.

# Построим плотность распределения, данную в условии.

# In[174]:

def g_y():
    y = np.arange(0, 1, 0.0001)
    f = np.array([elem **2 / 2 for elem in y])
    plt.plot(y, f, label='DD')
    plt.legend(loc='upper right')


# In[175]:

g_y()
plt.show()


# Построение гистограммы равноинтервальным методом.

# In[176]:

def hist_equal_interval(variety, is_line=False, is_rect=False):
    n = len(variety)
    segments = get_segments(n)
    f, _y, eeeee = plt.hist(variety, normed=True, bins=segments)
    if is_line:
        y = [(_y[i] + _y[i + 1]) / 2 for i in range(len(f))]
        if not is_rect:
            plt.clf()
        plt.plot(y, f,'r', label='HQI')
        plt.legend(loc='upper right')
    plt.ylabel('f*(y)')
    plt.xlabel('y')


# In[177]:

n = 10000

d = gen(n)
hist_equal_interval(d)
g_y()
plt.show()


# In[178]:

n = 10000

d = gen(n)
hist_equal_interval(d, True, False)
g_y()
plt.show()


# **Вывод:**
# 
# Все в жизни прекрасно.

# ## Задание №3
# #### Статистический ряд. Построение гистограммы равновероятностным методом.

# Построение гистограммы равновероятностным методом

# In[179]:


def hist_equiprobable(variety,  is_line=False, is_rect=False):
    n = len(variety)
    segments = get_segments(n)
    v = n // segments
    borders = [variety[0]] + [(variety[i] + variety[i + 1]) / 2 for i in range(v - 1, n - 1, v)] + [variety[-1]]
    f, _y, eeeee = plt.hist(variety, normed=True, bins=borders)
    if is_line:
        y = [(_y[i] + _y[i + 1]) / 2 for i in range(len(f))]
        if not is_rect:
            plt.clf()
        plt.plot(y, f, label='HQI')
        plt.legend(loc='upper right')
    plt.ylabel('f*(y)')
    plt.xlabel('y')
    


# In[180]:

n = 1000


d = gen(n)
hist_equiprobable(d, True, True)
g_y()
plt.show()


# In[ ]:




# In[ ]:



