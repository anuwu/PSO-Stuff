import numpy as np
import sys

import numpy as np

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
	global x_min
	# objective 1  [x^2] ---> x_min = 0, f(x_min) = 0
	# x_min = 0
	# return np.square (x)
	
	# objective 2 [x^4 + x^3 - 10x^2 + x + 5] ---> x_min = -2.6629, f(x_min) = -37.1732
	# x_min = -2.6629,
	# return np.power(x, 4) + np.power(x, 3) - 10*np.square(x) + x + 5
	
	# objective 3 [0.025*x^2 + sin(x)] ---> x_min = 1.49593, f(x_min) = -0.94125366117
	x_min = -1.49593
	return 0.025*np.square(x) + np.sin(x)

def objDer (x) :
	# Objective 1
	# return 2*x

	# Objective 2
	# return 4*np.pow(x,3) + 3*np.pow(x,2) - 20*np.pow(x) + 1

	# Objective 3
	return 0.05*x + np.cos(x)

def logistic (x, r) :
	return r * x * (1 - x) 

def tent (x, mu) :
	if x <= mu :
		return x/mu
	else :
		return (1-x)/(1-mu)

class chaosGenerator :
    def __init__ (self, cmap) :
        iterate = 1000 + int(np.random.rand() * 10000)
        
        self.x0 = 0.01
        self.cmap = cmap
        for i in range (1, iterate) :
            if cmap[0] == "logistic" :
                self.x0 = logistic (self.x0, cmap[1])
            elif cmap[0] == "tent" :
                self.x0 = tent (self.x0, cmap[1])
        
        
    def getTimeSeries (self, length) :
        n = np.arange (1, length+1)
        xs = [self.x0]
        x = self.x0

        for i in range (1, length) :
            if self.cmap[0] == "logistic" :
                x = logistic (x, self.cmap[1])
            elif self.cmap[0] == "tent" :
                x = tent (x, self.cmap[1])

            xs.append (x)

        xs = np.array (xs)
        self.x0 = xs[-1]
        
        return (n, xs)

    def getChaosPoints (self, num) :
        return np.reshape (self.getTimeSeries (num)[1], (-1, 1))
    
    def chaosRand (self) :
        if self.cmap[0] == "logistic" :
            self.x0 = logistic (self.x0, cmap[1])
        elif self.cmap[0] == "tent" :
            self.x0 = tent (self.x0, cmap[1])
            
        return self.x0

def main (modbool, weight) :
	# 10, 20, 0.2
	# 25, 20, 0.2
	# 50, 100, 1
	# 100, 1000, 10
	Nx = 25
	w = weight
	c1 = 2
	c2 = 2

	cmap = ["tent", 0.49999]
	#cmap = ["logistic", 4]
	chaosMan = chaosGenerator (cmap)

	numIter = 50
	left = -20 
	right = 20
	intervalLength = right - left

	if modbool :
		xVan = intervalLength * np.random.rand(Nx, 1) - intervalLength/2
		vVan = intervalLength/100.0 * np.random.rand(Nx, 1) - intervalLength/(2*100)
		pbestVan = xVan
		gbestVan = min (xVan , key = lambda x : obj(x))
	else :
		xChaos = intervalLength * chaosMan.getChaosPoints (Nx) - intervalLength/2
		vChaos = intervalLength/100.0 * chaosMan.getChaosPoints (Nx) - intervalLength/(2*100)
		pbestChaos = xChaos
		gbestChaos = min (xChaos , key = lambda x : obj(x))

	################################################################################################
	
	if modbool :
		gbvCache = [gbestVan]
		gbvoCache = [obj(gbestVan)]
		xavgvCache = [np.average (xVan)]
		xoavgvCache = [np.average (obj(xVan))]
		vavgvCache = [np.average (abs(vVan))]
		pbavgvCache = [np.average (obj(pbestVan))]
	else :
		gbcCache = [gbestChaos]
		gbcoCache = [obj(gbestChaos)]
		xavgcCache = [np.average (xChaos)]
		xoavgcCache = [np.average (obj(xChaos))]
		vavgcCache = [np.average (abs(vChaos))]
		pbavgcCache = [np.average (obj(pbestChaos))]

	################################################################################################

	for i in range (0, numIter) :
		if modbool :
			r1 = np.random.rand (Nx, 1)
			r2 = np.random.rand (Nx, 1)
			vVan = w*vVan + c1*r1*(pbestVan - xVan) + c2*r2*(gbestVan - xVan)
			xVan = xVan + vVan
			less = obj(xVan) < obj(pbestVan)
			pbestVan = less * xVan + np.invert (less) * pbestVan
			gbestVanNew = min (xVan , key = lambda x : obj(x))
			if (obj(gbestVanNew) < obj(gbestVan)) :
				gbestVan = gbestVanNew
		else :
			r1c = chaosMan.getChaosPoints (Nx)
			r2c = chaosMan.getChaosPoints (Nx)
			vChaos = w*vChaos + c1*r1c*(pbestChaos - xChaos) + c2*r2c*(gbestChaos - xChaos)
			xChaos = xChaos + vChaos
			less = obj(xChaos) < obj(pbestChaos)
			pbestChaos = less * xChaos + np.invert (less) * pbestChaos
			gbestChaosNew = min (xChaos , key = lambda x : obj(x))	
			if (obj(gbestChaosNew) < obj(gbestChaos)) :
				gbestChaos = gbestChaosNew
		
		################################################################################################
		
		if modbool :
			gbvCache.append (gbestVan)
			gbvoCache.append (obj(gbestVan))
			xavgvCache.append (np.average (xVan))
			xoavgvCache.append (np.average (obj(xVan)))
			vavgvCache.append (np.average (abs(vVan)))
			pbavgvCache.append (np.average (obj(pbestVan)))
		else :
			gbcCache.append (gbestChaos)
			gbcoCache.append (obj(gbestChaos))
			xavgcCache.append (np.average (xChaos))
			xoavgcCache.append (np.average (obj(xChaos)))
			vavgcCache.append (np.average (abs(vChaos)))
			pbavgcCache.append (np.average (obj(pbestChaos)))
		################################################################################################


	######################################## Shell outputs #########################################

	if modbool :
		if approx (gbestVan, x_min, 0.1, 0)[0] :
			currMinVan = True
		else :
			currMinVan = False
	else :
		if approx (gbestChaos, x_min, 0.1, 0)[0] :
			currMinChaos = True
		else :
			currMinChaos = False

	if modbool :
		globMinVan = approx (xVan, gbestVan, 0.1, 0)
		otherVan = xVan[np.invert(globMinVan)]
		otherVanDer = np.array (list (zip (otherVan, objDer(otherVan))))
		cntVan = sum (globMinVan)[0]
	else :
		globMinChaos = approx (xChaos, gbestChaos, 0.1, 0)
		otherChaos = xChaos[np.invert(globMinChaos)]
		otherChaosDer = np.array (list (zip (otherChaos, objDer(otherChaos))))
		cntChaos = sum (globMinChaos)[0]

	
	################################################################################################

	if modbool :
		if currMinVan :
			print ("1 " + str(cntVan))
			sys.exit (1)
		else :
			print ("0 " + str(cntVan))
			sys.exit (0)

	else :
		if currMinChaos :
			print ("1 " + str(cntChaos))
			sys.exit (1)
		else :
			print ("0 " + str(cntChaos))
			sys.exit (0)
		

if __name__ == '__main__':
	model = sys.argv[1]
	if model == "pso" :
		modbool = True
	elif model == "cpso" :
		modbool = False
	else :
		print ("Incorrect model selected")
		sys.exit (1)

	weight = float(sys.argv[2])
	main(modbool, weight)