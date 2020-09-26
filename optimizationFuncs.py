import numpy as np

def sphere (x) :
    """ sum(xi^2)
    f(0, ... ) = 0
    xi <- [...] """
    return np.sum (np.square(x), axis=1)

def matyas (xy) :
    """ 0.26*(x^2 + y^2) - 0.48xy
    f(0, 0) = 0
    x,y <- [-10, 10] """
    assert xy.shape[1] == 2
    x, y = xy[:,0], xy[:,1]
    return 0.26*np.sum(np.square(xy), axis=1) - 0.48*x*y

def bulkin (xy) :
    """ 100sqrt(|y - 0.01x^2|) + 0.01|x + 10|
    f(-10, 1) = 0
    x,y <- [-10, 10] """
    assert xy.shape[1] == 2
    x, y = xy[:,0], xy[:,1]
    return 100*np.sqrt(np.abs(\
                             y - 0.01*np.square(x)
                             ))
    + 0.01*np.abs(x + 10)

def rastrigin (x) :
    """ An + sum(xi^2 - Acos(2pi*xi))
    f(0, ...) = 0
    xi <- [-5.12, 5.12] """
    A = 10
    return A*x.shape[1] + np.sum(np.square(x) - A*np.cos(2*np.pi*x) , axis=1)

def alpine (x) :
    """ prod(xi*sin(xi))
    f(7.917, ...) = 2.808^n --> for maximization
    xi <- [0, 10] """
    return -np.prod (np.sqrt(x)*np.sin(x), axis=1)

def schaffer2 (xy) :
    """ 0.5 + (sin^2(x^2 - y^2) - 0.5)/(1 + 0.001(x^2 + y^2))^2
    f(0, 0) = 0
    x,y <- [-100, 100] """
    assert xy.shape[1] == 2
    x2 = np.square (xy[:,0])
    y2 = np.square (xy[:,1])
    return 0.5 + (np.square(np.sin(x2 - y2)) - 0.5)/np.square(1 + 0.001*(x2 + y2))

def schaffer4 (xy) :
    """ 0.5 + (cos^2(sin|x^2 - y^2|) - 0.5)/(1 + 0.001(x^2 + y^2))^2
    f(0, 1.25313) = 0.292579
    f(0, -1.25313) = 0.292579
    x,y <- [-100, 100] """
    assert xy.shape[1] == 2
    x2 = np.square (xy[:,0])
    y2 = np.square (xy[:,1])
    return 0.5 + (np.square(np.cos(np.sin(np.abs(x2 - y2)))) - 0.5)/np.square(1 + 0.001*(x2 + y2))

def schaffer6 (xy) :
    """ 0.5 - (sin^2(sqrt(x^2 + y^2)) - 0.5)/(1 + 0.001(x^2 + y^2))^2
    f(0, ...) = 0
    xi <- [-100, 100] """
    assert xy.shape[1] == 2
    x, y = xy[:,0], xy[:,1]
    sumxy = np.square(x) + np.square(y)
    return 0.5 - (np.square(np.sin(np.sqrt(sumxy))) - 0.5)/np.square(1 + 0.001*sumxy)

def griewank (x) :
    """ 1/4000*sum(xi^2)- prod(cos(xi/sqrt(i))) + 1
    f(0, ...) = 0
    xi <- [-600, 600] """
    return np.sum(np.square(x), axis=1)/4000 - np.prod(np.cos(x/np.sqrt(np.arange(1, x.shape[1]+1, 1))), axis=1) + 1

def rosenbrockND (x) :
    """ sum(100(x_{i+1} - x_i^2)^2 + (1-x_i)^2)
    f(0, ...) = 0
    xi <- [...] """
    assert x.shape[1] > 1
    return 100*np.sum(np.square(x[:,1:] - np.square(x[:,:-1])), axis=1) + np.sum(np.square(x[:,:-1]-1), axis=1)

def rosenbrock2D (xy) :
    assert xy.shape[1] == 2
    x, y = xy[:,0], xy[:,1]
    return np.square(1-x) + 100*np.square(y - np.square(x))

def ackley (xy) :
    """ -20exp(-0.2*sqrt(0.5*(x^2 + y^2)))
    f(0, 0) = 0
    x,y <- [-5, 5] """
    assert xy.shape[1] == 2
    x, y = xy[:,0], xy[:,1]
    x2, y2 = np.square(x), np.square(y)
    return np.e + 20 - 20*np.exp(-0.2*np.sqrt((x2+y2)/2)) - np.exp((np.cos(2*np.pi*x) + np.cos(2*np.pi*y))/2)

def beale (xy) :
    """ (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2
    f(3, 0.5) = 0
    x,y <- [-4.5, 4.5] """
    assert xy.shape[1] == 2
    x, y = xy[:,0], xy[:,1]
    return np.square(1.5 - x + x*y) + np.square(2.25 - x + x*np.square(y))\
    + np.square(2.625 - x + x*np.power(y, 3))

def goldstein (xy) :
    """ [1 + (x+y+1)^2(19 - 14x + 3x^2 - 14y + 6xy + 3y^2)]
    [30 + (2x - 3y)^2(18 - 32x + 12x^2 + 48y - 36xy + 27y^2)]

    f(0, -1) = 3
    x,y <- [-2, 2] """
    assert xy.shape[1] == 2
    x, y = xy[:,0], xy[:,1]
    return (1+np.square(x+y+1)*(19 - 14*x + 3*np.square(x) - 14*y + 6*x*y + 3*np.square(y)))*\
    (30 + np.square(2*x - 3*y)*(18 - 32*x + 12*np.square(x) + 48*y - 36*x*y + 27*np.square(y)))

def booth (xy) :
    """ (x+2y-7)^2 + (2x + y - 5)^2
    f(1, 3) = 0
    x,y <- [-10, 10] """
    assert xy.shape[1] == 2
    x, y = xy[:,0], xy[:,1]
    return np.square(x + 2*y - 7) + np.square (2*x + y - 5)

def eggholder (xy) :
    """ -(y+47)sin(sqrt|x/2 + y + 47|) - xsin(sqrt|x - y - 47|)
    f(512, 404.2319) = -959.6407
    x,y <- [-512, 512] """
    assert xy.shape[1] == 2
    x, y = xy[:,0], xy[:,1]
    return -(\
    (y+47)*np.sin(np.sqrt(np.abs(\
                                x/2 + y + 47
                                )))\
    + x*np.sin(np.sqrt(np.abs(\
                              x - y - 47
                             )))
    )

def easom (xy) :
    """ -cos(x)cos(y)exp(-((x-pi)^2 + (y-pi)^2))
    f(pi, pi) = -1
    x,y <- [-100, 100] """
    assert xy.shape[1] == 2
    x, y = xy[:,0], xy[:,1]
    return -(\
    np.cos(x) * np.cos(y) * np.exp(-(\
                                     np.square(x-np.pi) + np.square(y-np.pi)
                                    ))
    )

def mccormick (xy) :
    """ sin(x+y) + (x-y)^2 - 1.5x + 2.5y + 1
    f(-0.54719, -1.54719) = -1.9133
    x <- [-1.5, 4]
    y <- [-3, 4] """
    assert xy.shape[1] == 2
    x, y = xy[:,0], xy[:,1]
    return np.sin(x+y) + np.square(x-y) - 1.5*x + 2.5*y + 1

def styblinski (x) :
    """ sum(xi^4 - 16xi^2 + 5xi)/2
    f(-2.903534, ...) = -39.16616n
    xi <- [-5, 5] """
    return np.sum(np.power(x,4) - 16*np.square(x) + 5*x, axis=1)/2

def holdertable (xy) :
    """ -|sin(x)cos(y)exp|1 - sqrt(x^2+y^2)/pi||
    f(+/-8.05502, +/-9.64459) - -19.2085
    x,y <- [-10, 10] """
    assert xy.shape[1] == 2
    x, y = xy[:,0], xy[:,1]
    return -np.abs(\
    np.sin(x) * np.cos(y) * np.exp(np.abs(\
                                   1 - np.sqrt(np.square(x) + np.square(y))/np.pi
                                  ))
    )

def crossintray (xy) :
    """ -0.0001[|sin(x)sin(y)exp|100 - sqrt(x^2+y^2)/pi|| + 1]^0.1
    f(+/-1.34941, -1.34941) = -2.06261
    x,y <- [-10, 10] """
    assert xy.shape[1] == 2
    x, y = xy[:,0], xy[:,1]
    return -0.0001*np.power(1+np.abs(\
    np.sin(x)*np.sin(y)*np.exp(np.abs(\
                                     100 - np.sqrt(np.square(x) + np.square(y))/np.pi
                                     )+1)
    ),0.1)

def threehumpcamel (xy) :
    """ 2x^2 - 1.05x^4 + x^6/6 + xy + y^2
    f(0, 0) = 0
    x,y <- [-5, 5] """
    assert xy.shape[1] == 2
    x, y = xy[:,0], xy[:,1]
    return 2*np.square(x) - 1.05*np.power(x,4) + np.power(x,6)/6 + x*y + np.square(y)

def himmelblau (xy) :
    """ (x^2 + y - 11)^2 + (x + y^2 - 7)^2
    f(3, 2) = 0
    f(-2.805118, 3.131312) = 0
    f(-3.779310, -3.283186) = 0
    f(3.584428, -1.848126) = 0
    x,y <- [-5, 5] """
    assert xy.shape[1] == 2
    x, y = xy[:,0], xy[:,1]
    return np.square(np.square(x) + y - 11) + np.square(x + np.square(y) - 7)

def levi (xy) :
    """ sin^2(3pi*x) + (x-1)^2(1 + sin^2(3pi*y)) + (y-1)^2(1+sin^2(2pi*y))
    f(1, 1) = 0
    x,y <- [-10, 10] """
    assert xy.shape[1] == 2
    x, y = xy[:,0], xy[:,1]
    return np.square(np.sin(3*np.pi*x)) + np.square(x-1)*(1+np.square(np.sin(3*np.pi*y))) + np.square(y-1)*(1+np.square(np.sin(2*np.pi*y)))

def anuwu (x) :
    """ 0.025x^2 + sin(x)
    f(-1.49593) = -0.941254
    x <- [-20, 20] """
    assert x.shape[1] == 1
    return 0.025*np.square(x) + np.sin(x)

def ada1 (x) :
    """ -(3x^5 - x^10)
    f(1.0844717712) = -9/4
    x <- [-20, 20] """
    assert x.shape[1] == 1
    return -(3*np.power(x,5) - np.power(x,10))

def ada2 (x) :
    """ x^3 - 3x^2 + 7
    f(2) = 3
    x <- [0, 10] """
    assert x.shape[1] == 1
    return np.power(x,3) - 3*np.power(x,2) + 7

def ada3 (x) :
    """ -exp(cos(x^2)) + x^2
    f(0) = -2.71828
    x <- [-1, 1] """
    assert x.shape[1] == 1
    return (lambda x2 : -np.exp(np.cos(x2)) + x2)(np.square(x))

def ada4 (x) :
    """ x^15 - sin(x) + exp(x^6)
    f(0.651208) = 0.4747057
    x <- [0, 1] """
    assert x.shape[1] == 1
    return np.power(x,15) - np.sin(x) + np.exp(np.power(x,6))

###########################################################
# Derivatives of the 1-D functions (whenever possible) of
# the above defined optimization functions
###########################################################
funcGrads = {
sphere      : lambda x : 2*x,
rastrigin   : lambda x : 2*x + 20*np.pi*np.sin(2*np.pi*x),
griewank    : lambda x : x/2000 + np.sin(x),
alpine      : lambda x : np.sin(x)/(2*np.sqrt(x)) + np.sqrt(x)*np.cos(x),
anuwu       : lambda x : 0.05*x + np.cos(x),
ada1        : lambda x : -(15*np.power(x, 4) - 10*np.power(x, 9)),
ada2        : lambda x : 3*np.square(x) - 6*x,
ada3        : lambda x : (lambda x, x2 : 2*np.exp(np.cos(x2))*np.sin(x2)*x + 2*x)(x, np.square(x)),
ada4        : lambda x : 15*np.power(x, 14) - np.cos(x) + 6*np.exp(np.power(x, 6))*np.power(x, 5)
}
