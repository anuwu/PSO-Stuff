#!/usr/bin/env python
# coding: utf-8

# # Function Definitions

import numpy as np
import sys

def obj (x) :
    return 0.025*np.square(x) + np.sin(x)

def logistic (x, r) :
    return r * x * (1 - x) 

def getTimeSeries (length, r) :
    x0 = 0.01
    n = np.arange (1, length+1)
    xs = [x0]
    x = x0
    
    for i in range (1, length) :
        x = logistic (x, r)
        xs.append (x)
        
    xs = np.array (xs)
    return (n, xs)

def getChaosPoints (num, r) :
    if num < 1000 :
        seriesLength = 1000
    else :
        seriesLength = 2*num 
        
    return getTimeSeries (seriesLength, r)[1][-num:]

def main () :
	Nx = 100
	w = 0.7
	c1 = 1.5
	c2 = 1.5
	r1 = np.random.rand ()
	r2 = np.random.rand ()

	from statistics import mean

	xVan = 1000 * np.random.rand(Nx, 1) - 500
	vVan = 10 * np.random.rand(Nx, 1) - 5
	xChaos = 1000 * getChaosPoints (Nx, 4) - 500
	xChaos = np.reshape (xChaos, (-1,1))
	vChaos = 10 * getChaosPoints (Nx, 4) - 5
	vChaos = np.reshape (vChaos, (-1,1))
	pbestVan = xVan
	pbestChaos = xChaos

	numIter = 500
	gbestVan = min (xVan , key = lambda x : obj(x))
	gbestChaos = min (xChaos , key = lambda x : obj(x))

	################################################################################################
	# Plot 1 list -----------------> Gbest point per iteration
	gbvCache = [gbestVan]
	gbcCache = [gbestChaos]

	# Plot 2 list -----------------> Gbest point objective value per iteration
	gbvoCache = [obj(gbestVan)]
	gbcoCache = [obj(gbestChaos)]

	# Plot 3 list -----------------> Average of points 
	xavgvCache = [np.average (xVan)]
	xavgcCache = [np.average (xChaos)]

	# Plot 4 list -----------------> Average of fitness
	xoavgvCache = [np.average (obj(xVan))]
	xoavgcCache = [np.average (obj(xChaos))]

	# Plot 5 list -----------------> Average of absolute velocity
	vavgvCache = [np.average (abs(vVan))]
	vavgcCache = [np.average (abs(vChaos))]

	# Plot 6 list -----------------> Average of pbest fitness
	pbavgvCache = [np.average (obj(pbestVan))]
	pbavgcCache = [np.average (obj(pbestChaos))]
	################################################################################################

	for i in range (0, numIter) :
	    vVan = w*vVan + c1*r1*(pbestVan - xVan) + c2*r2*(gbestVan - xVan)
	    vChaos = w*vChaos + c1*r1*(pbestChaos - xChaos) + c2*r2*(gbestChaos - xChaos)
	    xVan = xVan + vVan
	    xChaos = xChaos + vChaos
	    
	    less = obj(xVan) < obj(pbestVan)
	    pbestVan = less * xVan + np.invert (less) * pbestVan
	    less = obj(xChaos) < obj(pbestChaos)
	    pbestChaos = less * xChaos + np.invert (less) * pbestChaos
	    
	    ################################################################################################
	    gbestVan = min (xVan , key = lambda x : obj(x))
	    gbestChaos = min (xChaos , key = lambda x : obj(x))
	    
	    # Appending to list for plot 1
	    gbvCache.append (gbestVan)
	    gbcCache.append (gbestChaos)
	    
	    # Appending to list for plot 2
	    gbvoCache.append (obj(gbestVan))
	    gbcoCache.append (obj(gbestChaos))
	    
	    # Appending to list for plot 3
	    xavgvCache.append (np.average (xVan))
	    xavgcCache.append (np.average (xChaos))
	    
	    # Appending to list for plot 4
	    xoavgvCache.append (np.average (obj(xVan)))
	    xoavgcCache.append (np.average (obj(xChaos)))
	    
	    # Appending to list for plot 5
	    vavgvCache.append (np.average (abs(vVan)))
	    vavgcCache.append (np.average (abs(vChaos)))
	    
	    # Appending to list for plot 6
	    pbavgvCache.append (np.average (obj(pbestVan)))
	    pbavgcCache.append (np.average (obj(pbestChaos)))
	    ################################################################################################

	cntVan = sum (np.abs(xVan - gbestVan) <= 0.01)[0]
	cntChaos = sum (np.abs(xChaos - gbestChaos) <= 0.01)[0]

	print ("\ncntVan = " + str(cntVan))
	print ("cntChaos = " + str(cntChaos))

	if abs((gbestVan-gbestChaos)/min(gbestVan,gbestChaos)) <= 0.01 :
		print ("Same gbest")
	else :
		print ("Different gbest f(" + str(gbestVan[0]) +") = " +str(obj(gbestVan)) + ", f(" + str(gbestChaos[0]) +") =" +str(obj(gbestChaos)))
	print ("Vanilla gbest = " + str(obj(gbestVan)))
	print ("Chaotic gbest = " + str(obj(gbestChaos)))
	print ("Vanilla average fitness = " + str(xoavgvCache[-1]))
	print ("Chaotic average fitness = " + str(xoavgcCache[-1]))

	if ((xoavgvCache[-1] - xoavgcCache[-1])/abs(xoavgvCache[-1]) > 0.01 ) :
		sys.exit (0)
	else :
		sys.exit (1)


if __name__ == '__main__':
	main()