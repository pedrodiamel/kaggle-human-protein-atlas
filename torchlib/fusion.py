# Fusion Rulers
# @autor: Pedro Diamel Marrero Fernandez
# 
# 
#       +-----+  dp_1
#     --| cnn |---------+
#       +-----+         |
#                       | 
#       +-----+  dp_2   |     +--------+
#     --| cnn |---------+ ----| Fusion |-----------+
#       +-----+         |     +--------+
#                       .        |
#                       .        SOFT, HARD, TR
#                       .        Soft: sRp, sRs, sRmx, sRmi, sRmd, sRmv
#       +-----+  dp_L   |        Hard: hWmv, hRec, hNb  
#     --| cnn |---------+        Train: tFi, tLop, tMb, tMsvm
#       +-----+



## On Combining Classifiers
# - Combining soft ruler
# - https://dspace.cvut.cz/bitstream/handle/10467/9443/1998-On-combining-classifiers.pdf?sequence=1
    
def product_ruler( dp, P=None ):
    """
    Ecuation. Josef Kittler [7]
    https://dspace.cvut.cz/bitstream/handle/10467/9443/1998-On-combining-classifiers.pdf?sequence=1
    Soft Product Rule    
    P^{(-(R-1))}( w_j )\prod_iP(w_j|x_i)  = \max_k P^{(-(R-1))}( w_k ) \prod_i
    P(w_k|x_i) (1)
    which under the assumption of equal priors, simplifies to the following:
    \prod_iP(w_j|x_i)  = \max_k \prod_i P(w_k|x_i) (2)    
    Args:
        @dp: []_nxcxl
        @P: class prior []_c
    """    
    p = dp.prod(2)
    if P is not None:
        l = dp.shape[2]
        p = P**-(l-1)*p
    return p

def sum_ruler( dp, P=None ):
    """
    Ecuation. Josef Kittler [11]
    https://dspace.cvut.cz/bitstream/handle/10467/9443/1998-On-combining-classifiers.pdf?sequence=1
    Soft Sum Ruler
    $$(1-R)P( w_j ) + \sum_iP(w_j|x_i)  = \max_k [ (1-R)P(w_k) + \sum_iP(w_k|x_i)]$$
    which under the assumption of equal priors simplifies to the following:
    $$\sum_iP(w_j|x_i)  = \max_k \sum_iP(w_k|x_i)$$    
    Args:
        @dp: []_nxcxl
    """
    p = dp.sum(2)
    if P is not None:
        l = dp.shape[2]
        p = (1-l)*P + p
    return p

def max_ruler( dp, P=None ):
    """
    Ecuation. Josef Kittler [14][15]
    https://dspace.cvut.cz/bitstream/handle/10467/9443/1998-On-combining-classifiers.pdf?sequence=1
    Soft Max Ruler
    Args:
        @dp: []_nxcxl
    """
    p = dp.max(2)
    if P is not None:
        l = dp.shape[2]
        p = (1-l)*P + l*p
    return p

def min_ruler( dp, P=None ):
    """
    Ecuation. Josef Kittler [16]
    https://dspace.cvut.cz/bitstream/handle/10467/9443/1998-On-combining-classifiers.pdf?sequence=1
    Soft Min Ruler
    Args:
        @dp: []_nxcxl
    """
    p = dp.min(2)
    if P is not None:
        l = dp.shape[2]
        p = P**-(l-1)*p
        p = (1-l)*P + l*p
    return p


def majority_ruler( dp  ):
    """
    Ecuation. Josef Kittler [20]
    https://dspace.cvut.cz/bitstream/handle/10467/9443/1998-On-combining-classifiers.pdf?sequence=1
    Soft Majority Vote Rule
    Args:
        @dp: []_nxcxl
    """
    n,c,l = dp.shape
    p = np.argmax(dp,axis=1)
    
    dki = np.zeros((n,c))
    for i in range(n):
        tup = p[i,:]
        for j in range(c):
            dki[i,j] = np.sum( tup == j )
        
    p=dki
    return p


def mean_ruler( dp ):
    """
    Ecuation. Josef Kittler [18]
    https://dspace.cvut.cz/bitstream/handle/10467/9443/1998-On-combining-classifiers.pdf?sequence=1
    Soft Median Ruler
    Args:
        @dp: []_nxcxl
    """
    p = dp.mean(2)
    return p
    

    
# # test
# #[n,c,l]
# #dp = np.random.rand(10,4,3 )
# #P = np.array([0.1,0.1,0.1,0.7])
# dp = np.array( [[[0.2,0,0,0],[0.8,1.0,1.0,1.0]],[[0.3,0.6,0.9,0.5],[0.7,0.4,0.1,0.5]]] )
# P = np.array([0.7,0.3])
# print(dp.shape)
# #print(dp[:,:,0])

# func = [product_ruler, sum_ruler, max_ruler, min_ruler, majority_ruler, mean_ruler]
# for f in func:
#     p = f(dp)
#     print( p.argmax(1) )
    
    
