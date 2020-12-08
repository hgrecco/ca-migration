# -*- coding: utf-8 -*-
"""
An Integrative and Modular Framework to Recapitulate Emergent Behavior in Cell Migration

Authors: Marina B. Cuenca, Lucia Canedo, Carolina Perez-Castro, Hernán E. Grecco

Frontiers on Cell and Developmental Biology DOI: 10.3389/fcell.2020.615759 //

Correspondence: cuencam@df.uba.ar, hgrecco@df.uba.ar


Created on Wed Feb 19 11:01:09 2020
@author: Marina B. Cuenca Twitter: @cuencam15
"""

import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import imageio
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm_notebook
import glob
import scipy.stats as st
from scipy.signal import convolve2d
from lmfit import Model
import pybroom as br

plt.rcParams.update({'font.size': 18})

#%%######### Initial conditions ####################


###########################################
#-------Initial condition for control-----# 
#-----------with single cells-------------#
###########################################

def icn_control(grid, u=0, dif=False, grad=False):
    
    if u == 0:
        u0 = np.zeros((grid, grid)) #initial chem concentration REPELENT (zero everywhere)
    else:
        u0 = u*np.ones((grid, grid)) #initial chem concentration ATTRACTANT (one everywhere)
    if dif == True:
        u0[int(grid/2), int(grid/2)] = 1 #for diffusion control we set an initial concentration of chemical localized
    if grad == True:
        for b in range(grid):
                    u0[:, b] = b*0.5 #gradient of chemical
        
    
    n0 = np.zeros((grid, grid)) #initial distribution of cells
    
    # Single Cell
    n0[int(grid/2), int(grid/2)] = 1
    
    
    return n0, u0

###########################################
#-------Initial condition for ------------# 
#-----------single spheres----------------#
###########################################

def icn_squared(grid, diam, cell, u=0):
    
    lower = int(grid/2 - .5*diam/cell)
    upper = int(grid/2 + .5*diam/cell)
    
    if u == 0:
        u0 = np.zeros((grid, grid)) #initial chem concentration REPELENT
    else:
        u0 = u*np.ones((grid, grid)) #initial chem concentration ATTRACTANT
    
    n0 = np.zeros((grid, grid)) #initial distribution of cells
    n0[lower:upper, lower:upper] = 1
    
    
    return n0, u0

##########################################
#----Initial condition for 2 spheres-----#
##########################################

def icn_two(grid, diam, cell, u=0):
    lower = int(grid/2 - diam/cell)
    upper = int(grid/2 + diam/cell)
    
    if u == 0:
        u0 = np.zeros((grid, grid)) #initial chem concentration REPELENT
    else:
        u0 = u*np.ones((grid, grid)) #initial chem concentration ATTRACTANT
    
    n0 = np.zeros((grid, grid)) #initial distribution of cells
    n0[lower-5:int(grid/2)-5, lower:int(grid/2)] = 1
    n0[int(grid/2)+5:upper+5, int(grid/2):upper] = 1
    
    
    return n0, u0



#%%########### Iterator ######################

######################################
#----LOOP THROUGH TIME AND SERIES----#
######################################

def iterator_m(u0, n0, var, dt, iterations, folder, name, cell, tt, save=10, progress=True, mec_stat=True, still=False, rnd = 1):
    
    # Saves initial condition to start all the iterations
    u_s = u0
    n_s = n0
    
    grid = u0.shape[0]
    ## Variables
    d, c1, c2, cf, q, alpha = var[0], var[1], var[2], var[3], var[4], var[5]
    
    ## Lambda constant
    lam = d*dt
    
    # Number of iterations in time
    t_iter = int(tt/dt)
    
    ## Difusion matrix for this particular mode and dim configuration
    ddm = dif_matrix2(lam, c2, dt, grid)
    
    matrix_u = create_matrix_u_func(grid)
    
    ## Series
   
    for itera in tqdm_notebook(range(iterations), total = iterations, unit="iterations", desc='Series', disable=progress):
        ## Matrix array for chem concentration, cells and indexes of occupied cells
        U_m, N_m, I_m = [], [], []
        M_m, P_m = [], []
        R_p, R_s = [], []
            
        u0 = u_s
        n0 = n_s
        
        ## Iterator
        for s in tqdm_notebook(range(t_iter), total = t_iter, unit='time iter', desc='Time iter', disable=progress, leave=False):
            
            ## Computes U in t + dt 
            u = matrix_u(u0, n0, ddm, c1, c2, dt)
            
            # Finds occupied cells and number of neighbors for each one
            N, index = find_n(n0)
            
            if still == True:
                n = n0
            else:
                # Probabilities and gradient
                M, per = grad2prob_m(index, cf, u, q, alpha, N, n0, mec_stat, rnd)
                
                # Computes N in t + dt   
                n, probs, f, p_m, rad_p, rad_s = matrix_n_m(n0, index, N, M, alpha, grid/2, cell)
            
            if s % save == 0:
                N_m.append(n0)
                I_m.append(index)
                U_m.append(u0)
                m = np.asarray(M)
                M_m.append(m)
                P_m.append(p_m)
                R_p.append(rad_p)
                R_s.append(rad_s)
                
 
            u0 = u
            n0 = n    
          
        np.savez(folder + '/' + name + '_it' + str(itera) + '.npz', U_m = U_m, N_m = N_m, I_m = I_m, M_m = M_m,
                 dt = dt, var = var, cell = cell, grid = grid, save = save, tt = tt, P_m = P_m,
                 iterations = iterations, name = name, rnd = rnd, R_p = R_p, R_s = R_s)
    
    
    
    filename = folder + '/' + name + '_it' + str(itera) #Ultima simulacion
    return filename, U_m, N_m, M_m



#########################################
#----THIS FUNCTION RETURNS N in t+dt----#
#########################################

def matrix_n_m(n0, index, nbr, M, alpha, centroid, cell):
    #centroid = 0
    n = np.zeros_like(n0)
    grid = n.shape[0]
    
    ## We choose cells randomly to decide if they move and where 
    ran = list(range(len(nbr)))
    random.shuffle(ran)
    pp, rad_v = [], []
    for k in ran:
        ## Coordinates of cell
        ix = index[k]
        
        r0 = np.sqrt((ix[0]-centroid)**2+(ix[1]-centroid)**2)
        
              
        ## I take a random number to decide if proliferates or moves
        r1 = random.random()
    
        #Proliferation
        if r1 < alpha:
            n = proliferation(ix, grid, n, n0)
        
        #Moves
        else:
            cells = [0, 0], [0, 1], [0, -1], [1, 0], [-1, 0], [1, -1], [1, 1], [-1, 1], [-1, -1]
            op = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            prob_m = M[k]
            probs = [prob_m[1 + i[0], 1 + i[1]] for i in cells]
            
            # Random cell with weighted prob
            r = np.random.choice(op, p=probs) 
            f = cells[r]
            
            # If it is inside the grid
            if ix[0] + f[0] > -1 and ix[0] + f[0] < grid and ix[1] + f[1] > -1 and ix[1] + f[1] < grid:
                # If the cell is empty                        
                if f!= [0, 0] and n0[ix[0] + f[0], ix[1] + f[1]] == 0 and n[ix[0] + f[0], ix[1] + f[1]] == 0:
                    n[ix[0] + f[0], ix[1] + f[1]] = 2
                    rf = np.sqrt((ix[0]+f[0]-centroid)**2+(ix[1]+f[1]-centroid)**2)
                    #fx = [ix[0] + f[0], ix[1]+f[1]]
                else:
                    n[ix[0], ix[1]] = 1
                    rf = r0
                    #fx = ix
            else:
                n[ix[0], ix[1]] = 1          
                rf = r0
                #fx = ix
        #if r0 == 0:
        #    tang = 0
        #else:
            #theta = np.arccos(np.abs((ix[1]-centroid))/r0)-np.arccos(np.abs((fx[1]-centroid))/rf)
            #tang = np.sin(theta)*rf
            rad = rf-r0
            pp.append(choice(rad, 0.1)) ##0.71 as original
            rad_v.append(rad)
    
    p_m = np.mean(pp, axis = 0)
    rad_p = len([i for i in rad_v if i>0])/len(rad_v)
    rad_s = len([i for i in rad_v if i==0])/len(rad_v)
        
    return n, probs, f, p_m, rad_p, rad_s

##########################
#----Difussion matrix----#
##########################

def dif_matrix2(lam, c2, dt, grid):

    # Tridiagonal difussion matrix
    diag0 = .5*(1-4*lam)*np.ones(grid)
    diag1 = lam*np.ones(grid-1)
    ddm = np.diagflat(diag0) + np.diagflat(diag1, 1) + np.diagflat(diag1, -1)
    
    return ddm

############################
#----MATRIX U in t + dt----#
############################

def create_matrix_u_func(grid):
    
        
    def _internal(u0, n0, ddm, c1, c2, dt):
        #return np.tensordot(ddm, u0, axes=([1,0])) + np.tensordot(u0, ddm, axes=([1,0])) + c1*dt*n0
        return np.tensordot(ddm, u0, axes=([1,0])) + np.tensordot(u0, ddm, axes=([1,0])) + c1*dt*n0-c2*dt*n0*u0
    
    return _internal
    
###########################################################
#----Ocupied cells coordinates and number of neighbors----#
###########################################################


def find_n(n0):
    m = len(n0)
    index = []
    # List with cell indexes
    for i in range(m):
        for j in range(m):
            if n0[i][j] > 0:
                index.append([i, j])
    N = []
    # Run over every neighbor and count them
    for ix in index:
        ies = [ix[0]-1, ix[0]+1]
        jes = [ix[1]-1, ix[1]+1]  
        num = 0
        for i in ies:
            if i>0 and i<m-1:
                if n0[i][ix[1]] > 0:
                            num += 1
        for j in jes:
            if j>0 and j<m-1:
                if n0[ix[0]][j] > 0:
                            num += 1
        
        # Array w number of neighbors (int)
        N.append(num)
    N = np.asarray(N)
    index = np.asarray(index)
    return N, index

####################################
#----Probabilities and gradient----#
####################################

def grad2prob_m(index, cf, u, q, alpha, N, n0, mec_stat, rnd):
    
    grid = u.shape[0]
    M, per = [], []
    # If mechanical interactions are considered
    if mec_stat == True:
        mec_grid = mech_mov(index, n0)
    else:
       mec_grid = []
       for i in range(len(index)):
           mec_grid.append(np.zeros((3,3)))
           
    # Runs over every cell
    for k, ix in enumerate(index):
        i, j = ix[0], ix[1]
        # Chemotaxis coefficient
        chi = 2*cf/(1+3*u[i, j])**2
        
    
        # If it is not at the edge of the grid
        if i>0 and i<grid-1:
            muy = chi*(u[i+1, j]-u[i-1, j])/2
        else:
            muy = 0
        
        # If it is not at the edge of the grid
        if j>0 and j<grid-1:
            mux = chi*(u[i, j+1]-u[i, j-1])/2
        else:
            mux = 0
    
        # Chemoattractant module and probability matrix for chemo
        mu = np.sqrt(mux**2+muy**2)
        C = gradient_m(mu, mux, muy)
        
#        hop = (1-q)**N[k]

        mec = mec_grid[k]
#        mec[1, 1] == 0
        sumM = sum(sum(mec))
        if sumM > 0:
#            mec = mec*hop/sumM
#            mec[1, 1] = 1-hop
            mec = q*mec/sumM
        
        #else:
        D = gkern(3,2)
        D[1, 1] == 0
        sumD = sum(sum(D))
        D = rnd*D/sumD
        ## Total probabilities
        
        
        tot_prob = D + C + mec
        ## Remove
#        tot_prob[1, 1] = 0
        
        suma = sum(sum(tot_prob))
        mat = tot_prob/suma

    
        M.append(mat) 
        
        per.append([sum(sum(D))/suma, sum(sum(C))/suma, sum(sum(mec))/suma])

    return M, per


##############################
#----Mechanical Behaviour----#
##############################

def mech_mov(ix, n0):
    
    ## Convolution Matrix
    conv = np.ones((3, 3))
    
    ## Matrix like n0 filled with ones in occupied spaces
    cells = np.zeros_like(n0)
    cells[n0 > 0] = 1   
    
    
    out = convolve2d(cells, conv)[1:-1, 1:-1]
    out = out
    
    n = len(out)
    a = np.c_[np.zeros(n), out, np.zeros(n)]
    mec_big = np.r_[[np.zeros(n+2)], a, [np.zeros(n+2)]]
    
    mec_grid = []
    for index in ix:
        x, y = index[1]+1, index[0]+1
        mec = mec_big[y-1:y+2, x-1:x+2]
        l = np.zeros((3,3))
        l[1, 1] = 1
        ll = convolve2d(l, conv)[1:-1, 1:-1]
        mec_n = mec-ll
        suma = np.sum(np.sum(mec_n))
        if suma > 0:
            mec_grid.append(mec_n/suma)
        else:
            mec_grid.append(mec_n)
                                    
    
    return mec_grid

####################################
#----Gradient movement function----#
####################################

def gradient_m(mu, ux, uy):
    C = np.zeros((3, 3))
    ## If the gradient is positive in x and bigger than in y --> moves +1 in x 
    if ux > 0 and np.abs(ux) > np.abs(uy):
        C[1, 2] = mu
    ## If the gradient is negative in x and bigger than in y --> moves -1 in x
    elif ux < 0 and np.abs(ux)>np.abs(uy) :
        C[1, 0] = mu
    ## If the gradient is positive in y and bigger than in x --> moves +1 in y
    elif uy > 0 and np.abs(uy)>np.abs(ux):
        C[2, 1] = mu
    ## If the gradient is negative in y and bigger than in x --> moves -1 in y
    elif uy < 0 and np.abs(uy)>np.abs(ux):
        C[0, 1] = mu
        
    return C

################################
#----Proliferation function----#
################################

def proliferation(ix, grid, n, n0):
    prof = 0
    ## Posible first neightbors
    vec = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    ## Random neighbor
    random.shuffle(vec)
    
    for vector in vec:
        if prof == 0:
            dy, dx = vector[0], vector[1]
            ## If the neighbor is inside the array
            if ix[0] + dy < grid and ix[1] + dx < grid and ix[0] + dy > 0 and ix[1] + dx > 0:
                ## If the neighbor is free
                if n0[ix[0] + dy, ix[1] + dx] == 0 and n[ix[0] + dy, ix[1] + dx] == 0:
                    n[ix[0] + dy, ix[1] + dx] = 1 ## I occupy the neighbor
                    n[ix[0], ix[1]] = 1 ## And also the original cell
                    prof = 1 ## Stops looking for neighbors
    
    ## If it could not proliferate it remains the same
    if prof == 0:
        n[ix[0], ix[1]] = 1
        
    return n

##############################
#----Probability Movement----#
##############################

def choice(rad, cell):
    m = np.zeros((3, 3))
    ##SI NO SE MUEVE
#    if np.abs(tang) < cell and np.abs(rad) < cell:
    if np.abs(rad) < cell:
          m[1, 1] = 1
        
    ##SI SE MUEVE SOLO RADIAL
#    elif np.abs(tang) < cell and np.abs(rad) > cell:
    else:
        if rad > 0:
            m[0, 1] = 1
            
        else:
            m[2, 1] = 1


    return m


#%%###### Other functions ##############

#########################
#----Gaussian Kernel----#
#########################
    
def gkern(kernlen=21, nsig=3):
    # Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

##########################
#----MSD single cells----#
##########################

def msd(iterations, grid, filenames, folder, plot = False):
    r_iter= []
    rx_iter, ry_iter = [], []
    r_max = []

    # For each iteration finds the cell position in each time iter and 
    #calculates de distance**2 from the center of the grid where the movement 
    #starts in module, x and y. 
    plt.rcParams["figure.figsize"] = [6,5]
    color1 = pl.cm.rainbow(np.linspace(0,1,len(filenames)))
    for m in range(iterations):
        data = np.load(filenames[m])
        I_m = data['I_m']
        r, rx, ry, rmax = [], [], [], []
        x, y = [], []
        for l, lst in enumerate(I_m):
            d, dx, dy = [], [], []
            for elements in lst:
                distance = (elements[0]-grid/2)**2+(elements[1]-grid/2)**2
                dx.append(np.abs(elements[0]-grid/2))
                dy.append(np.abs(elements[1]-grid/2))
                d.append(distance)
                x.append(elements[0]*10)
                y.append(elements[1]*10)
            if plot == True:
                plt.plot(x, y, color = color1[m])
            
            
            r.append(np.mean(d))
            rx.append(np.mean(dx))
            ry.append(np.mean(dy))
            rmax.append(np.amax(np.sqrt(d)))
        

        r_iter.append(r)
        rx_iter.append(rx)
        ry_iter.append(ry)
        r_max.append(rmax)
    
    if plot == True:    
        plt.xlabel('X position [um]')
        plt.ylabel('Y position [um]')
        #plt.xlim((0, 400))
        #plt.ylim((0, 400))
        plt.tight_layout()
        plt.savefig(folder + '/tray.png')
        #plt.close()
        plt.show()
    return r_iter, rx_iter, ry_iter, r_max

##########################################
#----Averages for multiple iterations----#
##########################################

def n_average(r_iter):
    r_average, r_error= [], []
    # Runs over the structure with length iterations and averages, extracting
    #the error
    
    for k in range(len(r_iter[0])):
        ll = []
        for j in range(len(r_iter)):
            ll.append(r_iter[j][k])
            
        r_average.append(np.mean(ll))

        r_error.append(np.std(ll)/np.sqrt(len(ll)))
    
    return r_average, r_error

##################
#----Plot MSD----#
##################

def plot_msd(I_m, r_average, r_error, rx_av, rx_error, ry_av, ry_error, folder):
    x = range(len(I_m))
    
    # Linear fit of msd data, the slot should be 1 in random walk
    m,b = pl.polyfit(x, r_average, 1) 
    
    plt.rcParams["figure.figsize"] = [15,5]
    plt.subplot(121)
    plt.plot(x, r_average, 'r') #Data
    plt.plot(x, x, '--b') #What should be
    plt.plot(x, m*x+b, '--k') #Fitting
    plt.fill_between(x, np.asarray(r_average)-np.asarray(r_error), np.asarray(r_average) + np.asarray(r_error),
        alpha=0.2, facecolor='r',
        linewidth=4, linestyle='dashdot', antialiased=True)

    plt.xlabel('Iteration')
    plt.ylabel('MSD')
    plt.legend(['Data', 'm = 1','m = %.2f' % m])
    plt.subplot(122)
    plt.plot(x, rx_av, 'b', label = 'MSD X')
    plt.plot(x, ry_av, 'r', label = 'MSD Y')
    plt.fill_between(x, np.asarray(rx_av)-np.asarray(rx_error), np.asarray(rx_av) + np.asarray(rx_error),
        alpha=0.2, facecolor='b',
        linewidth=4, linestyle='dashdot', antialiased=True)
    plt.fill_between(x, np.asarray(ry_av)-np.asarray(ry_error), np.asarray(ry_av) + np.asarray(ry_error),
        alpha=0.2, facecolor='r',
        linewidth=4, linestyle='dashdot', antialiased=True)

    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('MSD')
    plt.savefig(folder + '/MSD.png')
    plt.show()

##########################################
#----Control analysis of 2d diffusion----#
##########################################

def control_dif2D(file, folder):
       
    # Relevant parameters of the simulation
    data = np.load(file)
    U_matrix = data['U_m']
    dt = data['dt']
    save = data['save']
    d = data['var'][0]
    shape = U_matrix.shape
    mid = int(shape[1]/2)

    # Wide at half width, total concentration, maximum peak
    wide_x, wide_y, area, maximum = [], [], [], []

    time_it = []
    plt.rcParams["figure.figsize"] = [15, 4]
    plt.subplot(121)
    color1 = pl.cm.Blues(np.linspace(1,0, len(U_matrix)))
        
    for i, elements in enumerate(U_matrix):
        # The total concetration is the sum in every voxel
        area.append(np.sum(elements)) 
        # Maximum peak for each time iter
        maximum.append(np.amax(elements))
    
    # Chemical profile through a middle line
        ux = elements[mid, :]
        uy = elements[:, mid]
    
        # Calculates FWHM
        arr = np.linspace(0, shape[1]-1, shape[1]) #Array x
        
        wide_x.append(FWHM(ux, arr))
        wide_y.append(FWHM(uy, arr))
        
        plt.plot(arr, ux, alpha=.5, color = color1[i])
    
        # Time of the iteration
        time_it.append(i*save)

    plt.xlabel('Position')
    plt.ylabel('Concentration')

    # Deletes the initial conditions
    del wide_x[0], time_it[0], wide_y[0]
    # Linear fitting of parameters
    
    wide_x = np.asarray(wide_x)
    wide_y = np.asarray(wide_y)
    time_it = np.asarray(time_it)
    mx, bx = pl.polyfit(time_it, wide_x**2, 1) 
    my, by = pl.polyfit(time_it, wide_y**2, 1) 
    # Teoretical and fitted difussion coefficients
    dif_x = mx*10000*dt/2
    dif_y = my*10000*dt/2
    dif_teo = d*2/(10000*dt)# Teoretical slot

    plt.subplot(122)
    plt.plot(time_it, wide_x**2, 'r', time_it, mx*time_it+bx, '--r')
    plt.plot(time_it, wide_y**2, 'b', time_it, my*time_it+by, '--b')
    plt.plot(time_it, dif_teo*time_it, '--k')
    plt.xlabel('Iteration')
    plt.ylabel('FWHM**2')
    plt.legend(['x', 'Dx = %.2f' % dif_x, 'y', 'Dy = %.2f' % dif_y, 'Dteo = %.1f' %d])
    plt.savefig(folder + '/Difussion.png')
    plt.show()

    plt.rcParams["figure.figsize"] = [15,4]
    # Linear fittinf of maximum vs sqrt(iteration)
    intime = np.array(range(1,len(maximum)))*dt
    del maximum[0]
    maximum = np.asarray(maximum)
    ml, bl = pl.polyfit(intime, 1/maximum, 1) 
    # Toretical and fitted difussion coefficients
    dif = (ml*area[0])/(4*np.pi*save)
    d_teo = (d*(4*np.pi*save))/area[0] # Teoretical slot
    
    plt.subplot(121)
    plt.plot(range(len(area))*save, area)
    plt.xlabel('Iteration')
    plt.ylim(0, 2*area[0] )
    plt.ylabel('Total Concentration')
    plt.subplot(122)
    plt.plot(intime, 1/maximum, 'r', intime, ml*intime+bl, '--k')
    plt.plot(intime, d_teo*intime, '--b')
    plt.xlabel('Iteration^(0.5)')
    plt.ylabel('Max')
    plt.legend(['Data', 'D = %.2f' % dif, 'Dteo = %.1f' %d])
    plt.savefig(folder + '/Difussion2.png')
    plt.show()
    
##############
#----FWHM----#
##############

def FWHM(ux, arr):
    difference = max(ux) - min(ux) # Amplitud of the chemical
    HM = difference / 2 # Half maximum

    pos_extremum = ux.argmax()  # Maximum location
    # Nearest position to the maximum where half maximum is reached
    nearest_above = (np.abs(ux[pos_extremum:-1] - HM)).argmin()
    nearest_below = (np.abs(ux[0:pos_extremum] - HM)).argmin()

    f = (np.mean(arr[nearest_above + pos_extremum]) - np.mean(arr[nearest_below]))
    
    return f

#######################
#----GIF GENERATOR----#
#######################

def gif_gen(filename, folder, frames):
    data = np.load(filename)
    plt.rcParams["figure.figsize"] = [8,4]
    with imageio.get_writer(folder + '/SimulationGif.gif', mode = 'I', duration = 0.25) as writer:
        for i in range(len(data['I_m'])):
            if i%frames == 0:
                file = folder + '/It_%02d.png' % i
                plt.figure()
                plt.suptitle('Iteration ' + str(i))
                plt.subplot(121)
                
                plt.title('Cell Grid')
                plt.imshow(data['N_m'][i], 'jet', vmin=0, vmax=3)
                ax = plt.subplot(122)
                
                plt.title('Chemo Grid')
                im = ax.imshow(data['U_m'][i], vmin=0, vmax=np.amax(data['U_m']))
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
                plt.savefig(file)
                plt.close()
    
                
                image = imageio.imread(file)
                writer.append_data(image)
                
##########################
#----Plot interactive----#
##########################

# Visualizes for each time iter the cell and chemical matrix

def plot_inter(U_m, N_m, t):
    plt.rcParams["figure.figsize"] = [8,4]
    plt.rcParams.update({'font.size': 16})
    plt.figure(2)
    plt.suptitle('Iteration ' + str(t))
    plt.subplot(121)
    plt.title('Cell Grid')
    plt.imshow(N_m[t], 'jet', vmin=0, vmax=3)
    ax = plt.subplot(122)
    plt.title('Chemo Grid')
    im = ax.imshow(U_m[t], vmin=0, vmax=np.amax(U_m))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    #plt.savefig(folder + '\Control1_%02d.jpg' % i)
    plt.show()
    
##############################
#----Plot radial invasion----#
##############################

# Plots the radial relative invassion 
def plot_rmsd(I_m, r_average, r_error, rx_av, rx_error, ry_av, ry_error, folder, save):
    
    x = save*range(len(I_m))
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.plot(x, r_average, 'r')
    plt.fill_between(x, np.asarray(r_average)-np.asarray(r_error), np.asarray(r_average) + np.asarray(r_error),
        alpha=0.2, linestyle='dashdot', facecolor='r',
        linewidth=4, antialiased=True)

    plt.xlabel('Iterations')
    plt.ylabel('Relative Radial Invasion')
    plt.subplot(122)
    plt.plot(x, rx_av, 'b', label = 'X')
    plt.plot(x, ry_av, 'r', label = 'Y')
    plt.fill_between(x, np.asarray(rx_av)-np.asarray(rx_error), np.asarray(rx_av) + np.asarray(rx_error),
        alpha=0.2, facecolor='b', linewidth=4, linestyle='dashdot', antialiased=True)
    plt.fill_between(x, np.asarray(ry_av)-np.asarray(ry_error), np.asarray(ry_av) + np.asarray(ry_error),
        alpha=0.2, facecolor='r', linewidth=4, linestyle='dashdot', antialiased=True)
    plt.legend()
    plt.ylabel('MSD')
    plt.xlabel('Iteration')

    plt.savefig(folder + '/Radial_MSD.png')
    plt.show()
    
#######################
#---Difussion plot----#
#######################

# Plots the concentration profile through time
def plot_dif(file, folder, frames, save, r_max):
    
    data = np.load(file)
    U_matrix = data['U_m']
    
    shape = U_matrix.shape
    mid = int(shape[1]/2)
    arr = 10*np.linspace(0, shape[1]-1, shape[1])
    n = int(shape[0]/frames)
    
    plt.figure(figsize=(12,5))
    plt.subplot()
    #color1 = pl.cm.viridis(np.linspace(0,1,n))
    color2 = pl.cm.plasma(np.linspace(0,1,n))
    j = 0
    area = []
    plt.subplot(121)
    for i, elements in enumerate(U_matrix):
        # Total concentration
        area.append(np.sum(elements))
        if i%frames == 0 and i > 0:
            # Concentration within a line through the center of the grid
            ux = elements[mid, :]
            mux = [abs(ux[i+1]-ux[i-1])/(1+3*ux[i])**2 for i in range(1, len(ux)-1)]
            
            #plt.plot(arr, ux, color = color1[j])
            
            #plt.axvline(x=-10*r_max[i]+mid*10, color = color2[j], linestyle = '--')
            plt.plot(arr[1:-1], mux, color = color2[j])
            #plt.axvline(x=10*r_max[i]+mid*10, color = color2[j], linestyle = '--')
            j += 1
    
    #plt.ylim((0, 0.005))
    #plt.legend(['Max r', 'Gradient'])       
    plt.xlabel('Position [um]')
    #plt.xlim((500, 1500))
    plt.ylabel('Gradient')
    
    plt.subplot(122)
    plt.plot(save*range(shape[0]), area)
    plt.xlabel('Iteration')
    plt.ylabel('Total concentration')
    plt.savefig(folder + '/difussion.png')
    plt.show()
    
#######################
#-----Size plot-------#
#######################

# Plots the RRM clustering on sizes
    
def size_plot(ff, folder, fol, diam, cell, T, plot = True, hs=3):
    plt.figure(figsize=(15,5))
    # Final relative radius and error
    r_colfinal, diam_col = [], []
    final_rm, er, v,v_er = [], [], [], []
    #plt.subplot(221)
    # Each folder represent one diameter. Might contain different iterations
    i = 0

    for folders in ff:
        filenames = glob.glob(fol + '/' + folders + '/*.npz')
        file = filenames[0]
        data = np.load(file)
        iterations, grid, dt, save= np.int(data['iterations']), int(data['grid']), data['dt'], data['save']
        
        
        
        # Extracts r mean
        r_iter, rx_iter, ry_iter, r_max = msd(iterations, grid, filenames, fol)
        r_iter = np.asarray(np.sqrt(r_iter))
        
        diam_col.append([diam[i] for j in range(data['iterations'])])
        r_ave, r_error = n_average(r_iter)
        # Normalization
        rm = r_ave/r_ave[0]
        rm_e = r_error/r_ave[0]

        
        # Time un hours
        x = save*range(len(rm))*dt*T
        ind_t = closest(x, hs)
        
        # Extracts last time point radius
        final_rm.append(rm[ind_t])
        er.append(rm_e[ind_t])
        r_colfinal.append([g[ind_t]/g[0] for g in r_iter])
        
        #m, b = pl.polyfit(x[0:ind_t], rm[0:ind_t], 1)
        mod = Model(linear)
        pars = mod.make_params(a=0, b=0)
        result = mod.fit(r_ave[0:ind_t], pars, x=x[0:ind_t])
        dt = br.tidy(result)
        
        
        v.append(dt.loc[dt.name == 'a']['value'].values[0]*cell)
        v_er.append(dt.loc[dt.name == 'a']['stderr'].values[0]*cell)
     

        i+=1
        
    
    #plt.xlabel('Time [hs]')
    #plt.xlim([0, 5])
    #plt.ylabel('Relative Radial Invasion')
    #plt.legend(labels)
    plt.subplot(121)
    plt.errorbar(diam, final_rm, er)
    #plt.savefig(folder + '\\Compar.png')
    plt.xlabel('Size')
    plt.ylabel('Final Radial Invasion')
    
        
     
    
    plt.subplot(122)
    plt.errorbar(diam, v, v_er)
    plt.xlabel('Diameter [um]')
    plt.ylabel('Velocity [1/hs]')
    plt.savefig(folder + '/Velocity.png')
    plt.show()
    
    
    r_flat = [item for sublist in r_colfinal for item in sublist]
    diam_flat = [item for sublist in diam_col for item in sublist]
    
    # Returns set of parameters
    var = data['var']
    
    col6 = pd.DataFrame({'size': diam_flat})
    col7 = pd.DataFrame({'rf': r_flat})
    #col8 = pd.DataFrame({'size' : diam})
    #col9 = pd.DataFrame({'rf': final_rm})  
    #col10 = pd.DataFrame({'rfe' : er})
    # Concatenates colums
    pd2 = pd.concat([col6, col7], axis=1)

    # Saves data to csv
    pd2.to_csv(folder + '/rfinal.csv')
    
    
    col8 = pd.DataFrame({'size' : diam})
    col9 = pd.DataFrame({'vel': v})  
    col10 = pd.DataFrame({'vele' : v_er})
    # Concatenates colums
    pd1 = pd.concat([col8, col9, col10], axis=1)

    # Saves data to csv
    pd1.to_csv(folder + '/vfinal.csv')
    
    
    
    return var


###########################
#----Threshold by size----#
###########################

def thresh(ff, fol, folder, th, T, plot = True):
    # Empty arrays for small, big and mean
    r_s, r_b, r_m = [], [], []
    rmax_s,rmax_b, rmax = [], [], []
    # Goes through every folder (sizes)
    for k, folders in enumerate(ff):
        
        filenames = glob.glob(fol + '/' + folders + '/*.npz')
        file = filenames[0]
        data = np.load(file)
        iterations, grid, dt, save = np.int(data['iterations']), int(data['grid']), data['dt'], data['save']
        # Small
        if k < th:

            r_iter, rx_iter, ry_iter, r_max = msd(iterations, grid, filenames, fol)
            r_iter = np.asarray(np.sqrt(r_iter))
            r_s.append(r_iter)
            r_m.append(r_iter)
            rmax_s.append(np.mean(r_max, axis = 0))
            rmax.append(np.mean(r_max, axis = 0))

        # Big
        else:
            r_iter, rx_iter, ry_iter, r_max = msd(iterations, grid, filenames, fol)
            r_iter = np.asarray(np.sqrt(r_iter))
            #r_average, r_error = n_average(r_iter)
            r_b.append(r_iter)
            r_m.append(r_iter)
            rmax_b.append(np.mean(r_max, axis = 0))
            rmax.append(np.mean(r_max, axis = 0))
            
    #Rugosity
#    rug = [(rm-rm[0])/(rm+rm[0]) for rm in rmax]
    rug_b, rug_be = mean_arr([(rm-rm[0])/(rm+rm[0]) for rm in rmax_b])
    rug_s, rug_se = mean_arr([(rm-rm[0])/(rm+rm[0]) for rm in rmax_s])
    
    # Normalization
    rs = [r[0]/r[0][0] for r in r_s]
    rb = [r[0]/r[0][0] for r in r_b]
    rm = [r[0]/r[0][0] for r in r_m]
    # Mean and error
    r_s = np.mean(rs, axis = 0)
    r_b = np.mean(rb, axis = 0)
    r_m = np.mean(rm, axis = 0)
    rs_e = np.std(rs, axis = 0)/np.sqrt(len(rs))
    rb_e = np.std(rb, axis = 0)/np.sqrt(len(rb))
    rm_e = np.std(rm, axis = 0)/np.sqrt(len(rm))
    
    # Time in hours
    x = save*range(len(r_m))*dt*T
    
    plt.rcParams["figure.figsize"] = [15,5]
    ###############
#    color1 = pl.cm.viridis(np.linspace(0,1,len(ff)))
    
#    plt.subplot(121)
#    plt.plot(x, rug_b, 'r')
#    plt.fill_between(x, rug_b + rug_be/2, rug_b - rug_be/2, color = 'red', alpha = 0.3)
#    plt.plot(x, rug_s, 'b')
#    plt.fill_between(x, rug_s + rug_se/2, rug_s - rug_se/2, color = 'blue', alpha = 0.3)
#    plt.xlabel('Time [hs]')
#    plt.ylabel('Rugosity')
#
#    plt.subplot(122)
#    for i, rm in enumerate(rug):
#        
#        plt.plot(x, rm, color = color1[i])
#    plt.xlabel('Time [hs]')
#    plt.ylabel('Rugosity')
#    plt.savefig(folder + '/Rugosity.png')
#    plt.show()
    

    
    plt.subplot(121)
    plt.plot(x, r_m, 'g')
    plt.fill_between(x, np.asarray(r_m)-np.asarray(rm_e), np.asarray(r_m) + np.asarray(rm_e),
        alpha=0.2, linestyle='dashdot', 
        linewidth=4, antialiased=True, color = 'green')
    plt.plot(x, r_s, 'b')
    plt.fill_between(x, np.asarray(r_s)-np.asarray(rs_e), np.asarray(r_s) + np.asarray(rs_e),
        alpha=0.2, linestyle='dashdot', 
        linewidth=4, antialiased=True, color = 'blue')
    plt.plot(x, r_b, 'r')
    plt.fill_between(x, np.asarray(r_b)-np.asarray(rb_e), np.asarray(r_b) + np.asarray(rb_e),
        alpha=0.2, linestyle='dashdot', 
        linewidth=4, antialiased=True, color = 'red')

    plt.xlabel('Time [hs]')
    plt.ylabel('RRM')
    plt.legend(['Mean', 'Small', 'Big'])
    
    plt.subplot(122)
    rrm_size(ff, folder, fol, T)
    
    plt.savefig(folder + '/Subpop.png')
    if plot == True:
        plt.show()
    else:
        plt.close()
        
    col1 = pd.DataFrame({'time': x})
    col2 = pd.DataFrame({'rbig': r_b})  
    col3 = pd.DataFrame({'rbige' : rb_e})
    col4 = pd.DataFrame({'rsmall': r_s})  
    col5 = pd.DataFrame({'rsmalle' : rs_e})
    col6 = pd.DataFrame({'rmean': r_m})  
    col7 = pd.DataFrame({'rmeane' : rm_e})
    col8 = pd.DataFrame({'rug small' : rug_s})
    col9 = pd.DataFrame({'rug smalle' : rug_se})
    col10 = pd.DataFrame({'rug big' : rug_b})
    col11 = pd.DataFrame({'rug bige' : rug_be})
    # Concatenates colums
    pd1 = pd.concat([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11], axis=1)

    # Saves data to csv
    pd1.to_csv(folder + '/rtime.csv')

    
def rrm_size(ff, folder, fol, T):
    ###############
    color1 = pl.cm.viridis(np.linspace(0,1,len(ff)))
    for i, folders in enumerate(ff):
        filenames = glob.glob(fol + '/' + folders + '/*.npz')
        file = filenames[0]
        data = np.load(file)
        iterations, grid, dt, save= np.int(data['iterations']), int(data['grid']), data['dt'], data['save']

        # Extracts r mean
        r_iter, rx_iter, ry_iter, r_max = msd(iterations, grid, filenames, folder)
        r_iter = np.asarray(np.sqrt(r_iter))
        r_ave, r_error = n_average(r_iter)
        # Normalization
        rm = r_ave/r_ave[0]
        rm_e = r_error/r_ave[0]


        # Time un hours
        x = save*range(len(rm))*dt*T
        plt.plot(x, rm, color = color1[i])
        plt.fill_between(x, np.asarray(rm)-np.asarray(rm_e), np.asarray(rm) + np.asarray(rm_e),
                             alpha=0.2, antialiased=True, color = 'gray')
    #plt.xlim((0, 5))
    #plt.ylim((0, 8))
    plt.xlabel('Time [hs]')
    plt.ylabel('RRM')
    #plt.savefig(folder + '\\SingleSpheres.png')
    #plt.show()
    
    
###########################
#----Subpopulations 2D----#
###########################


def subpop(iterations, filenames, folder, quartiles = True):
    one_m, two_m = [], []
    one_rm, two_rm = [], []

    for m in range(iterations):
        data = np.load(filenames[m])
        U_m, N_m, I_m, iterations, grid = data['U_m'], data['N_m'], data['I_m'], data['iterations'], int(data['grid'])
        save = data['save']
        shape = U_m.shape
        
        # Number of cells in each state and position
        one, two, total = [], [], []
        one_r, two_r = [], []
        
        # The displacement is meassured from the center of the sphere
        for j in range(shape[0]-1):
            N = N_m[j]
            I = I_m[j]
            n1, n2 = 0, 0
            r1, r2 = [], []
            for index in I:
                x, y = index[0], index[1]
                if N[x,y] == 1:
                    n1 = n1 + 1
                    r1.append(np.sqrt((x-grid/2)**2+(y-grid/2)**2))
                elif N[x,y] == 4:
                    n1 = n1 + 1
                    r1.append(np.sqrt((x-grid/2)**2+(y-grid/2)**2))
                elif N[x,y] == 2.:
                    n2 = n2 + 1
                    r2.append(np.sqrt((x-grid/2)**2+(y-grid/2)**2))

            if n2 == 0:
                r2 = [0, 0]


            one.append(n1)
            two.append(n2)
            
            total.append(n1+n2)
            one_r.append(np.mean(r1))
            two_r.append(np.mean(r2))
            

        one_m.append(one)
        two_m.append(two)
        
        one_rm.append(one_r)
        two_rm.append(two_r)
        
        
    one, one_r, one_r1, one_r3 = quantile(one_m, one_rm)
    two, two_r, two_r1, two_r3 = quantile(two_m, two_rm)
    
        
    t = save*np.linspace(0, len(one)-1, len(one))
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.plot(t, one, 'c', t, two, 'y', t, total, 'k')
    plt.xlabel('Iteration')
    plt.ylabel('Number of cells')
    plt.legend(['Still', 'Random', 'Total'])
    plt.subplot(122)
    plt.plot(t, one_r, 'c', t, two_r, 'y')
    if quartiles == True:
        plt.plot(t, one_r1, 'c--', t, two_r1, 'y--')
        plt.plot(t, one_r3, 'c--', t, two_r3, 'y--')
    plt.xlabel('Iteration')
    plt.ylabel('Position from center')
    plt.savefig(folder +  '/Subpopulations.png')
    plt.show()
    
############################
#---Quantile extraction----#
############################

def quantile(one_m, one_rm):
    one, one_r, one_r1, one_r3 = [], [], [], []
    
    #A verages the number of cells in that state for all the iterations and
    #extracts the mean position, first and third quantiles
    
    for k in range(len(one_m[0])):
        ii = []
        ll = []
        for j in range(len(one_m)):
            ii.append(one_m[j][k])
            ll.append(one_rm[j][k])
            
        one.append(np.mean(ii))
        one_r.append(np.mean(ll))
        one_r1.append(np.percentile(ll, 25))
        one_r3.append(np.percentile(ll, 75))
    
    return one, one_r, one_r1, one_r3


############################
#------Linear LMFIT--------#
############################

def linear(x, a, b):
    return a*x+b


############################
#------Find closest--------#
############################
    
def closest(lst, K): 

    lst = np.asarray(lst) 
    idx = (np.abs(lst - K)).argmin() 
    return idx

#######################################
#------Calculates mean of array-------#
#######################################
def mean_arr(arr):
    lengths = [len(a) for a in arr]
    max_length = max(lengths)
    mean_arr = []
    std_arr = []
    
    for i in range(max_length):
        m = []
        for a in arr:
            if len(a)>i:
                m.append(a[i])
        mean_arr.append(np.mean(m))
        #std_arr.append([np.std(m), np.percentile(m, q = [25, 75])])
        std_arr.append(np.std(m))
    return mean_arr, np.asarray(std_arr)