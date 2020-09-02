import numpy as np
from scipy.integrate import odeint
from statistics import mean
import matplotlib.pyplot as plt
import sys

def approx (num, rel, tol, cmpr) :
	# Check if num is lower than relative
	if cmpr == -1 :
		return (rel - num)/np.abs(rel) <= tol

	# Check absolute difference to relative
	if cmpr == 0 :
		return np.abs(num - rel)/np.abs(rel) <= tol

	# Check if num is higher than relative
	if cmpr == 1 :
		return (num - rel)/np.abs(rel) <= tol

x_min = 0

def obj (x) :
    # objective 1  [x^2] ---> x_min = 0, f(x_min) = 0
    # x_min = 0
    # return np.square (x)
    
    # objective 2 [x^4 + x^3 - 10x^2 + x + 5] ---> x_min = -2.6629, f(x_min) = -37.1732
    # x_min = -2.6629,
    # return np.power(x, 4) + np.power(x, 3) - 10*np.square(x) + x + 5
    
    # objective 3 [0.025*x^2 + sin(x)] ---> x_min = 1.49593, f(x_min) = -0.94125366117
    global x_min
    x_min = -1.49593
    minimas = [-19.1433, -13.4028, -7.47114, 4.48616, 10.446, 16.324]
    maximas = [-18.4538, -11.6152, -4.96317, 1.65357, 8.28087, 14.984]
    return 0.025*np.square(x) + np.sin(x)

def objDer (x) :
	# Objective 1
	# return 2*x

	# Objective 2
	# return 4*np.pow(x,3) + 3*np.pow(x,2) - 20*np.pow(x) + 1

	# Objective 3
	return 0.05*x + np.cos(x)

def main (modbool, beta1) :
	# 10, 20, 0.2
	# 25, 20, 0.2
	# 50, 100, 1
	# 100, 1000, 10
	Nx = 25
	c1 = 0.8
	c2 = 0.9
	beta = beta1

	numIter = 50
	left = -20 
	right = 20
	intervalLength = right - left
	vmax = intervalLength/10.0

	if modbool :
		xVan = intervalLength * np.random.rand(Nx, 1) - intervalLength/2
		vVan = intervalLength/100.0 * np.random.rand(Nx, 1) - intervalLength/(2*100)
		pbestVan = xVan
		gbestVan = min (xVan , key = lambda x : obj(x))
	else :
		xMom = intervalLength * np.random.rand(Nx, 1) - intervalLength/2
		vMom = intervalLength/100.0 * np.random.rand(Nx, 1) - intervalLength/(2*100)
		momvec = np.zeros(shape = (Nx, 1))
		pbestMom = xMom
		gbestMom = min (xMom , key = lambda x : obj(x))

	################################################################################################
	
	if modbool :
		gbvCache = [gbestVan]
		gbvoCache = [obj(gbestVan)]
		xavgvCache = [np.average (xVan)]
		xoavgvCache = [np.average (obj(xVan))]
		vavgvCache = [np.average (abs(vVan))]
		pbavgvCache = [np.average (obj(pbestVan))]
	else :
		gbmCache = [gbestMom]
		gbmoCache = [obj(gbestMom)]
		xavgmCache = [np.average (xMom)]
		xoavgmCache = [np.average (obj(xMom))]
		vavgmCache = [np.average (abs(vMom))]
		pbavgmCache = [np.average (obj(pbestMom))]

	################################################################################################

	for i in range (0, numIter) :
		if modbool :
			r1 = np.random.rand (Nx, 1)
			r2 = np.random.rand (Nx, 1)

			vVan = w*vVan + c1*r1*(pbestVan - xVan) + c2*r2*(gbestVan - xVan)
			xVan = xVan + vVan

			less = obj(xVan) < obj(pbestVan)
			pbestVan[less] = xVan[less] 
			gbestVan = min (pbestVan , key = lambda x : obj(x))
		else :
			r1m = np.random.rand (Nx, 1)
			r2m = np.random.rand (Nx, 1)

			momvec = beta*momvec + (1-beta)*vMom
			vMom = momvec + c1*r1m*(pbestMom - xMom) + c2*r2m*(gbestMom - xMom)
			xMom = xMom + vMom

			less = obj(xMom) < obj(pbestMom)
			pbestMom[less] = xMom[less] 
			gbestMom = min (pbestMom , key = lambda x : obj(x))	
		
		################################################################################################
		
		if modbool :
			gbvCache.append (gbestVan)
			gbvoCache.append (obj(gbestVan))
			xavgvCache.append (np.average (xVan))
			xoavgvCache.append (np.average (obj(xVan)))
			vavgvCache.append (np.average (abs(vVan)))
			pbavgvCache.append (np.average (obj(pbestVan)))
		else :
			gbmCache.append (gbestMom)
			gbmoCache.append (obj(gbestMom))
			xavgmCache.append (np.average (xMom))
			xoavgmCache.append (np.average (obj(xMom)))
			vavgmCache.append (np.average (abs(vMom)))
			pbavgmCache.append (np.average (obj(pbestMom)))
		################################################################################################


	######################################## Shell outputs #########################################
	if modbool :
		if approx (gbestVan, x_min, 0.1, 0)[0] :
			currMinVan = True
		else :
			currMinVan = False
	else :
		if approx (gbestMom, x_min, 0.1, 0)[0] :
			currMinMom = True
		else :
			currMinMom = False

	if modbool :
		globMinVan = approx (xVan, gbestVan, 0.1, 0)
		otherVan = xVan[np.invert(globMinVan)]
		otherVanDer = np.array (list (zip (otherVan, objDer(otherVan))))
		cntVan = sum (globMinVan)[0]
	else :
		globMinMom = approx (xMom, gbestMom, 0.1, 0)
		otherMom = xMom[np.invert(globMinMom)]
		otherMomDer = np.array (list (zip (otherMom, objDer(otherMom))))
		cntMom = sum (globMinMom)[0]

	
	################################################################################################

	if modbool :
		if currMinVan :
			print ("1 " + str(cntVan))
			sys.exit (1)
		else :
			print ("0 " + str(cntVan))
			sys.exit (0)
	else :
		if currMinMom :
			print ("1 " + str(cntMom))
			sys.exit (1)
		else :
			print ("0 " + str(cntMom))
			sys.exit (0)


if __name__ == '__main__':
	model = sys.argv[1]
	if model == "pso" :
		modbool = True
	elif model == "mompso" :
		modbool = False
	else :
		print ("Incorrect model selected")
		sys.exit (1)

	beta = float(sys.argv[2])
	main(modbool, beta)
