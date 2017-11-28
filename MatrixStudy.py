# -*- coding: utf-8 -*-
"""
@author: Guangyu Robert Yang, 2015-2016
"""
from __future__ import division

import os

import time
import pickle
import numpy as np

import sympy
from sympy import Symbol as S
from sympy.utilities.lambdify import lambdify


def calculate_terms(weight_mode = '3dim'):
    #---------------------------------------------------------------------------------
    # Defining the matrix of the dynamical system dr/dt = -r + Mr + u
    #---------------------------------------------------------------------------------
    start = time.time()

    if weight_mode == '3dim':
        # Here have a parameter for overall PV and SST strengths
        # Specifically, have W_EE=1
        ndim = 3 # System dimensions
        # Weight matrix
        M0 = sympy.Matrix([[S('W_EE'),S('W_EP0')*S('a_P'),S('W_ES0')*S('a_S')],
                           [S('W_PE'),S('W_PP0')*S('a_P'),S('W_PS0')*S('a_S')],
                           [S('W_SE'),0*S('a_P')         ,0*S('a_S')         ]])
        pop_names = ['E','P','S'] # Excitatory, Parvalbumin, Somatostatin
        var_names = [S('W_EE'),S('W_PE'), S('W_SE'), S('W_EP0'), S('W_PP0'),
                     S('W_ES0'), S('W_PS0'), S('a_S'), S('a_P')]

    elif weight_mode == '3dim_1':
        # Here have a parameter for overall PV and SST strengths
        # Specifically, have W_EE=1
        ndim = 3 # System dimensions
        # Weight matrix
        M0 = sympy.Matrix([[1        ,S('W_EP0')*S('a_P'),S('W_ES0')*S('a_S')],
                           [S('W_PE'),S('W_PP0')*S('a_P'),S('W_PS0')*S('a_S')],
                           [S('W_SE'), 0                 , 0                 ]])
        pop_names = ['E','P','S'] # Excitatory, Parvalbumin, Somatostatin
        var_names = [S('W_PE'), S('W_SE'), S('W_EP0'), S('W_PP0'),
                     S('W_ES0'), S('W_PS0'), S('a_S'), S('a_P')]
                     
    elif weight_mode == '3dim_2':
        # Here have a parameter for overall EXC, PV, and SST strengths
        ndim = 3 # System dimensions
        # Weight matrix
        M0 = sympy.Matrix([[S('W_EE0')*S('a_E'),S('W_EP0')*S('a_P'),S('W_ES0')*S('a_S')],
                           [S('W_PE0')*S('a_E'),S('W_PP0')*S('a_P'),S('W_PS0')*S('a_S')],
                           [S('W_SE0')*S('a_E'),0*S('a_P')         ,0*S('a_S')         ]])
        pop_names = ['E','P','S'] # Excitatory, Parvalbumin, Somatostatin
        var_names = [S('W_EE0'), S('W_PE0'), S('W_SE0'), S('W_EP0'), S('W_PP0'),
                     S('W_ES0'), S('W_PS0'), S('a_E'), S('a_S'), S('a_P')]

    elif weight_mode == '3dim_3':
        # Here have a parameter for overall EXC, PV, and SST strengths
        ndim = 3 # System dimensions
        # Weight matrix
        M0 = sympy.Matrix([[S('W_EE0')*S('a_E'),S('W_EP0')*S('a_P'),S('W_ES0')*S('a_S')],
                           [S('W_PE0')*S('a_E'),S('W_PP0')*S('a_P'),S('W_PS0')*S('a_S')],
                           [S('W_SE0')*S('a_E'),S('W_SP0')*S('a_P'),S('W_SS0')*S('a_S')]])
        pop_names = ['E','P','S'] # Excitatory, Parvalbumin, Somatostatin
        var_names = [S('W_EE0'), S('W_EP0'), S('W_ES0'),
                     S('W_PE0'), S('W_PP0'), S('W_PS0'),
                     S('W_SE0'), S('W_SP0'), S('W_SS0'),
                     S('a_E')  , S('a_P')  , S('a_S')]

    elif weight_mode == '4dim':
        ndim = 4 # System dimensions
        # Weight matrix
        M0 = sympy.Matrix([[S('W_EE'),S('W_EP'),S('W_ES'), 0        ],
                           [S('W_PE'),S('W_PP'),S('W_PS'), 0        ],
                           [S('W_SE'), 0        , 0        ,S('W_SV')],
                           [S('W_VE'), 0        ,S('W_VS'), 0        ]])
        pop_names = ['E','P','S','V'] # Excitatory, Parvalbumin, Somatostatin, Vip
        var_names = [S('W_EE'), S('W_EP') ,S('W_ES'), S('W_PE'), S('W_PP'),
                     S('W_PS'), S('W_SE'), S('W_SV'), S('W_VE'), S('W_VS')]


    elif weight_mode == '4dim_1':
        ndim = 4 # System dimensions
        # Weight matrix
        M0 = sympy.Matrix([[S('W_EE'),S('W_EP'),0         , 0        ],
                           [S('W_PE'),S('W_PP'),S('W_PS'), 0        ],
                           [S('W_SE'), 0        , 0        ,S('W_SV')],
                           [S('W_VE'), 0        ,S('W_VS'), 0        ]])
        pop_names = ['E','P','S','V'] # Excitatory, Parvalbumin, Somatostatin, Vip
        var_names = [S('W_EE'), S('W_EP'), S('W_PE'), S('W_PP'),
                     S('W_PS'), S('W_SE'), S('W_SV'), S('W_VE'), S('W_VS')]

    elif weight_mode == '4dim_2':
        # Here have a parameter for overall PV and SST strengths
        ndim = 4 # System dimensions
        # Weight matrix
        # Notice here we make the approximation of no VIP, this is OK if we care about
        # the PV and SST effective inhibition
        M0 = sympy.Matrix([[S('W_EE'),S('a_P')          ,S('a_S')          , 0        ],
                           [S('W_PE'),S('W_PP')*S('a_P'),S('W_PS')*S('a_S'), 0        ],
                           [S('W_SE'), 0                 , 0                 , 0        ],
                           [S('W_VE'), 0                 ,S('W_VS')*S('a_S'), 0        ]])
        pop_names = ['E','P','S','V'] # Excitatory, Parvalbumin, Somatostatin, Vip
        var_names = [S('W_EE'), S('W_PE'), S('W_PP'), S('a_P'),
                     S('W_PS'), S('W_SE'), S('W_VE'), S('W_VS'), S('a_S')]

    elif weight_mode == '5dim':
        ndim = 5 # System dimensions
        # Weight matrix
        M0 = sympy.Matrix([[S('W_EE'), S('W_ED'),S('W_EP'),0         , 0        ],
                           [0        , 0        , 0        ,S('W_DS'), 0        ],
                           [S('W_PE'), 0        ,S('W_PP'),S('W_PS'), 0        ],
                           [S('W_SE'), 0        , 0        , 0        ,S('W_SV')],
                           [S('W_VE'), 0        , 0        ,S('W_VS'), 0        ]])
        pop_names = ['E','D','P','S','V'] # Excitatory, Dendrite, Parvalbumin, Somatostatin, Vip

        var_names = [S('W_EE'), S('W_ED'), S('W_EP') ,S('W_DS'), S('W_PE'),
                     S('W_PP'), S('W_PS'), S('W_SE'), S('W_SV'), S('W_VE'), S('W_VS')]

    elif weight_mode == '2dim':
        ndim = 2
        M0 = sympy.Matrix([[S('W_EE'),S('W_EP')],
                           [S('W_PE'),S('W_PP')]])
        pop_names = ['E','P'] # Excitatory, Parvalbumin
        var_names = [S('W_EE'), S('W_EP'), S('W_PE'), S('W_PP')]
    else:
        ValueError('Unknown weight mode')

    #---------------------------------------------------------------------------------
    # Calculate the inverse matrix and related quantities
    #---------------------------------------------------------------------------------

    M    = sympy.eye(ndim)-M0                      # identity matrix minus weight matrix
    Minv = sympy.simplify(M.inv())                 # Inversion
    Mdet = sympy.factor(sympy.simplify(M.det()))   # Determinant
    Madj = sympy.factor(sympy.simplify(Minv*Mdet)) # Adjugate matrix

    print 'Time taken {:0.4f} s'.format(time.time()-start)
    #os.system('say "your program has finished"')

    #---------------------------------------------------------------------------------
    # Storing results
    #---------------------------------------------------------------------------------

    result = {'var_names':var_names,'pop_names':pop_names,
              'Mdet':Mdet, 'Madj':Madj, 'Minv':Minv, 'M0':M0}

    with open('library/matrix_analysis_'+weight_mode,'wb') as f:
        pickle.dump(result,f)

    return result



#weight_mode = '3dim'
#weight_mode = '3dim_1'
#weight_mode = '3dim_2'
#weight_mode = '3dim_3'
weight_mode = '4dim'
load_old = False

if load_old:
    with open('library/matrix_analysis_'+weight_mode,'rb') as f:
        result = pickle.load(f)
else:
    result = calculate_terms(weight_mode)
    
def absorb_den(a):
    a = a.expand()
    if weight_mode in ['3dim','3dim_1','3dim_2','3dim_3']:
        a = a.subs(S('W_EP0')*S('a_P'),S('W_EP'))  
        a = a.subs(S('W_PP0')*S('a_P'),S('W_PP'))  
        a = a.subs(S('W_ES0')*S('a_S'),S('W_ES'))  
        a = a.subs(S('W_PS0')*S('a_S'),S('W_PS')) 
    
    if weight_mode in ['3dim_2','3dim_3']:
        a = a.subs(S('W_EE0')*S('a_E'),S('W_EE'))  
        a = a.subs(S('W_PE0')*S('a_E'),S('W_PE'))  
        a = a.subs(S('W_SE0')*S('a_E'),S('W_SE'))

    if weight_mode in ['3dim_3']:
        a = a.subs(S('W_SP0')*S('a_P'),S('W_SP'))
        a = a.subs(S('W_SS0')*S('a_S'),S('W_SS'))

    '''
    a = a.subs(S('W_PP'),S('K_PP')-1)
    a = a.subs(S('W_SS'),S('K_SS')-1)
    a = a.subs(S('W_EE'),S('K_EE')+1)
    '''
    
    return sympy.simplify(a)
    #return a

var_names = result['var_names']
pop_names = result['pop_names']
Mdet = result['Mdet']
Madj = result['Madj']
Minv = result['Minv']
M0   = result['M0']

print '\nAdjugate' 
Madj1 = absorb_den(Madj)
print Madj1
#print sympy.latex(Madj1,mode='equation')


#---------------------------------------------------------------------------------
# Obtaining functions for evaluation
#---------------------------------------------------------------------------------

#f_Mdet = lambdify(var_names,Mdet,'numpy')
#f_Madj = lambdify(var_names,Madj,'numpy')
