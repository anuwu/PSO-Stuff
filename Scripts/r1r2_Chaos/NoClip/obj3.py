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
minimas = []
maximas = []

def obj (x) :
    global x_min
    global minimas
    global maximas
    # objective 1  [x^2] ---> x_min = 0, f(x_min) = 0
    # x_min = 0
    # return np.square (x)
    
    # objective 2 [x^4 + x^3 - 10x^2 + x + 5] ---> x_min = -2.6629, f(x_min) = -37.1732
    # x_min = -2.6629,
    # return np.power(x, 4) + np.power(x, 3) - 10*np.square(x) + x + 5
    
    # objective 3 [0.025*x^2 + sin(x)] ---> x_min = 1.49593, f(x_min) = -0.94125366117
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

def logisticMap (x, r) :
    return r * x * (1 - x) 

def tentMap (x, mu) :
    if x <= mu :
        return x/mu
    else :
        return (1-x)/(1-mu)

def lorenzFlow (initCond, param, t_end, length) :
    def lorenz (X, t, sigma, beta, rho) :
        x, y, z = X
        dXdt = [sigma*(y - x), x*(rho - z) - y, x*y - beta*z]
        return dXdt
    
    X0 = initCond
    t = np.linspace (0, t_end, length)
    sol = odeint (lorenz, X0, t, args = param)
    return sol

class chaosGenerator :
    def setLorenz (self, sol) :
        self.x0 = sol[-1,0]
        self.y0 = sol[-1,1]
        self.z0 = sol[-1,2]
    
    def __init__ (self, cmap) :
        
        self.cmap = cmap
        
        if cmap[0] == "logistic" or cmap[0] == "tent" :
            self.x0 = 0.01
            iterate = 1000 + int(np.random.rand() * 10000)
            funcMap = logisticMap if cmap[0] == "logistic" else tentMap
            for i in range (1, iterate) :
                self.x0 = funcMap (self.x0, cmap[1])
                
        elif cmap[0] == "lorenz" :
            self.x0 = np.random.rand ()
            self.y0 = np.random.rand ()
            self.z0 = np.random.rand ()
            sol = lorenzFlow ([self.x0, self.y0, self.z0], cmap[1], 1000, 100000)
            self.xmin = min(sol[:,0])
            self.xmax = max(sol[:,0])
            self.setLorenz (sol)
                
    def getTimeSeries (self, length) :
        if self.cmap[0] == "logistic" or self.cmap[0] == "tent" :
            n = np.arange (1, length + 1)
            xs = [self.x0]
            x = self.x0
            
            funcMap = logisticMap if self.cmap[0] == "logistic" else tentMap
            for i in range (1, length) :
                x = funcMap (x, self.cmap[1])
                xs.append (x)

            xs = np.array (xs)
            self.x0 = xs[-1]
            xs = np.reshape (xs, (-1, 1))
            
        elif self.cmap[0] == "lorenz" :
            sol = lorenzFlow ([self.x0, self.y0, self.z0], self.cmap[1], length, length * 100)
            inds = np.arange (0, length*100, 100).astype (int)
            xs = np.reshape (sol[:,0][inds], (-1, 1))
            xs = (xs - self.xmin)/(self.xmax - self.xmin)
            
            self.setLorenz (sol)
            
        return xs

    def getChaosPoints (self, num) :
        return self.getTimeSeries (num)
    
    def chaosRand (self) :
        if self.cmap[0] == "logistic" or self.cmap[0] == "tent" :
            funcMap = logisticMap if self.cmap[0] == "logistic" else tentMap
            self.x0 = funcMap (self.x0, self.cmap[1])
            
            return self.x0
        elif self.cmap[0] == "lorenz" :
            sol = lorenzFlow ([self.x0, self.y0, self.z0], self.cmap[1], 0.02, 2)
            self.setLorenz (sol)
            
            retx0 = (self.x0 - self.xmin)/(self.xmax - self.xmin)
            retx0 = 0 if retx0 < 0 else (1 if retx0 > 1 else retx0) 
            
            return retx0


def main (modbool) :
	# 10, 20, 0.2
	# 25, 20, 0.2
	# 50, 100, 1
	# 100, 1000, 10
	Nx = 25
	w = 0.1
	c1 = 2
	c2 = 2

	#cmap = ["tent", 0.49999]
	#cmap = ["logistic", 4]
	cmap = ["lorenz", (10, 8.0/3, 28)]
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
			pbestVan[less] = xVan[less] 
			gbestVan = min (pbestVan , key = lambda x : obj(x))
		else :
			r1c = chaosMan.getChaosPoints (Nx)
			r2c = chaosMan.getChaosPoints (Nx)
			vChaos = w*vChaos + c1*r1c*(pbestChaos - xChaos) + c2*r2c*(gbestChaos - xChaos)
			xChaos = xChaos + vChaos
			less = obj(xChaos) < obj(pbestChaos)
			pbestChaos[less] = xChaos[less] 
			gbestChaos = min (pbestChaos , key = lambda x : obj(x))	
		
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
			sys.exit (1)
		else :
			currMinVan = False
			sys.exit (0)
	else :
		if approx (gbestChaos, x_min, 0.1, 0)[0] :
			currMinChaos = True
			# sys.exit (1)
		else :
			currMinChaos = False
			# sys.exit (0)

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

		if currMinVan :
			print ("1 " + str(xoavgvCache[-1]))
			sys.exit (1)
		else :
			print ("0 " + str(xoavgvCache[-1]))
			sys.exit (0)
	else :
		if currMinChaos :
			print ("1 " + str(cntChaos))
			sys.exit (1)
		else :
			print ("0 " + str(cntChaos))
			sys.exit (0)
		
		if currMinChaos :
			print ("1 " + str(xoavgcCache[-1]))
			sys.exit (1)
		else :
			print ("0 " + str(xoavgcCache[-1]))
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

	main(modbool)
