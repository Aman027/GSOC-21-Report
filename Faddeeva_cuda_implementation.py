import tensorflow as tf
import numpy as np
from scipy import special

# tf.config.run_functions_eagerly(True)         #Uncomment this to run functions eagerly!

@tf.function(autograph=False)
def cond1_if_helper(x,y):
    
    xLim = tf.constant(5.33, dtype=tf.float64)
    yLim = tf.constant(4.29, dtype= tf.float64)
    errf_const = tf.constant(1.12837916709551, dtype= tf.float64)

    q = (1.0 - y / yLim) * tf.math.sqrt(1.0 - (x / xLim) * (x / xLim))
    h  = 1.0 / (3.2 * q)
    nc = (7 + tf.math.floor(23.0 * q))
    xl = tf.math.pow(h, (1 - nc))
    xh = y + 0.5 / h
    yh = x
    nu = 10 + tf.math.floor(21.0 * q)
    zero =tf.zeros(tf.shape(x),dtype=tf.float64)

    @tf.function(autograph=False)
    def cond(n,nc,Rx, Ry,xl,h,Sx, Sy):
        return tf.reduce_any(tf.greater(n,0.))


    @tf.function(autograph=False)
    def body(n,nc,Rx, Ry,xl,h,Sx, Sy):
        Tx = xh + n * Rx
        Ty = yh - n * Ry
        Tn = Tx*Tx + Ty*Ty
        Rx_new = 0.5 * Tx / Tn
        Ry_new = 0.5 * Ty / Tn
    
        Saux = Sx + xl
        n_greater_zero = tf.greater(n,zero)

        Sx_new,Sy_new,xl_new = tf.unstack(
            tf.where(
                tf.logical_and(tf.greater_equal(nc,n), n_greater_zero),
                (Rx_new * Saux - Ry_new * Sy,Rx_new * Sy + Ry_new * Saux,h * xl),
                (Sx,Sy,xl)
            )
        )    

        return n -1, nc,Rx_new, Ry_new,xl_new,h,Sx_new, Sy_new

    iterators = (nu,nc,zero,zero,xl,h,zero,zero)
    
    Sx,Sy = tf.unstack(tf.while_loop(cond, body, iterators,maximum_iterations=33)[-2:])
    return_var = (errf_const * Sx,errf_const * Sy)

    return return_var

@tf.function(autograph=False)
def cond1_else_helper(x,y):
    xh = y
    yh = x
    zero =tf.constant(0.,dtype=tf.float64)
    Rx = tf.zeros_like(x, dtype=tf.float64)
    Ry =  tf.zeros_like(x, dtype=tf.float64)
    errf_const = tf.constant(1.12837916709551, dtype= tf.float64)
    
    @tf.function(autograph=False)
    def cond(n,xh,yh,Rx,Ry):
        return tf.greater(n,0.1)
    
    @tf.function(autograph=False)
    def body(n,xh,yh,Rx,Ry):
        Tx = xh + n * Rx
        Ty = yh - n * Ry
        Tn = Tx * Tx + Ty * Ty
        Rx = 0.5 * Tx / Tn
        Ry = 0.5 * Ty / Tn
        return (n-1.,xh,yh,Rx,Ry)
    
    nine = tf.constant(9., dtype=tf.float64)
    iterators = (nine,xh,yh,Rx,Ry)
    
    Rx,Ry = tf.unstack(tf.while_loop(cond, body, iterators, maximum_iterations=9)[-2:])

    return (errf_const * Rx,errf_const * Ry)

@tf.function(autograph=False)
def in_imag_negative(Wx,Wy,x,y,in_real):
    Wx =   2.0 * tf.math.exp(y * y - x * x) * tf.math.cos(2.0 * x * y) - Wx
    Wy = - 2.0 * tf.math.exp(y * y - x * x) * tf.math.sin(2.0 * x * y) - Wy;
    Wy = -1*tf.math.sign(in_real)*Wy

    return Wx,Wy

@tf.function(autograph=False)
def wofz(z):
    in_real = tf.math.real(z)
    in_imag = tf.math.imag(z)
    x = tf.math.abs(in_real)
    y = tf.math.abs(in_imag)
    
    xLim = tf.constant(5.33, dtype=tf.float64)
    yLim = tf.constant(4.29, dtype= tf.float64)
    
    cond1 = tf.math.logical_and(tf.less(y,yLim),tf.less(x,xLim))
    Wx,Wy = tf.unstack(
        tf.where(
            cond1,
            cond1_if_helper(x,y),
            cond1_else_helper(x,y)
        )
    )
    
    zero =tf.constant(0.,dtype=tf.float64)
    cond2 = tf.equal(y,zero)
    cond3 = tf.less(in_imag,zero)
    cond4 = tf.less(in_real,zero)
    
    Wx = tf.where(cond2,tf.math.exp(-x*x),Wx)
    Wy = tf.math.sign(in_real)*Wy
    
    Wx,Wy = tf.unstack(tf.where(cond3,in_imag_negative(Wx,Wy,x,y,in_real),(Wx,Wy)))
    return Wx, Wy

def unit_tests():
    
    complex_list = [(624.2,-0.26123),
      (82.22756651,-349.16044211),
      (-0.4,3.),
      (0.6,2.),
      (-1.,1.),
      (-1.,-9.),
      (-1.,9.),
      (-0.0000000234545,1.1234),
      (-3.,5.1),
      (-53,30.1),
      (0.0,0.12345),
      (11,1),
      (-22,-2),
      (9,-28),
      (21,-33),
      (1e5,1e5),
      (1e14,1e14),
      (-3001,-1000),
      (1e160,-1e159),
      (-6.01,0.01),
      (-0.7,-0.7),
      (2.611780000000000e+01, 4.540909610972489e+03),
      (0.8e7,0.3e7),
      (-20,-19.8081),
      (1e-16,-1.1e-16),
      (2.3e-8,1.3e-8),
      (6.3,-1e-13),
      (6.3,1e-20),
      (1e-20,6.3),
      (1e-20,16.3),
      (9,1e-300),
      (6.01,0.11),
      (8.01,1.01e-10),
      (28.01,1e-300),
      (10.01,1e-200),
      (10.01,-1e-200),
      (10.01,0.99e-10),
      (10.01,-0.99e-10),
      (1e-20,7.01),
      (-1,7.01),
      (5.99,7.01),
      (1,0),
      (55,0),
      (-0.1,0),
      (1e-20,0),
      (0,5e-14),
      (0,51),
      (np.inf,0),
      (-np.inf,0),
      (0,np.inf),
      (0,-np.inf),
      (np.inf,np.inf),
      (np.inf,-np.inf),
      (np.nan,np.nan),
      (np.nan,0),
      (0,np.nan),
      (np.nan,np.inf),
      (np.inf,np.nan)
      ]

    numpy_complex_list=[]
    for i in range(len(complex_list)):
        numpy_complex_list.append(np.complex128(complex_list[i][0]+1j*complex_list[i][1]))

    faddeeva_ans = []
    tf_complex_list = tf.convert_to_tensor(numpy_complex_list)
    
    result = wofz(tf_complex_list)

    for i in range(len(result)):
        print("Complex no. : ",complex_list[i],"  Scipy ans : ",special.wofz(numpy_complex_list[i])," TF ans : ",result[0][i].numpy()+1j*result[1][i].numpy())
        
unit_tests()