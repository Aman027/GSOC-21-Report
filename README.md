# Google Summer of Code 2021
### Project: Implementation of Physical Shape Function - Faddeeva
### Mentor: [Jonas Eschle](https://www.physik.uzh.ch/en/researcharea/lhcb/team/phd-students/Eschle.html)
### Organization: [CERN-HSF](https://hepsoftwarefoundation.org/)

<br/>
<br/>

Zfit is a highly scalable and customizable model manipulation and fitting library. Using Tensorflow as its backend, it has been optimised for simple and direct manipulation of probability density functions with usage target High energy physics analysis ecosystem. The main focus is on scalability, parallelisation and a user friendly experience framework (no cython, no C++ needed to extend). The basic idea is to offer a pythonic oriented alternative to the very successful RooFit library from the ROOT data analysis package.

As part of GSoC'21, I have worked on implementing a new library function named Faddeeva function or Kramp function which is a scaled complex complementary error functions which can also simplify the process of computation of various error functions of arbitrary complex arguments. I have used low-level functionality of TensorFLow (which is Numpy-like).

Here is brief summary of the all work I did in the project:

## Summary


<!-- ## Details -->
### [Approach 1 : Algorithm 916 + Continued Fraction Expansion]
For computing the Faddeeva function, we are going to use combination of two algorithms:
#### For sufficiently large |z|, we use a continued-fraction expansion for w(z) similar to those described in: 
   *    Walter Gautschi, "Efficient computation of the complex error   function," SIAM J. Numer. Anal. 7(1), pp. 187-198 (1970) 
   *    G. P. M. Poppe and C. M. J. Wijers, "More efficient computation of the complex error function," ACM Trans. Math. Soft. 16(1), pp. 38-46 (1990).

#### We switch to a completely different algorithm for smaller |z|:

*  Mofreh R. Zaghloul and Ahmed N. Ali, "Algorithm 916: Computing the Faddeeva and Voigt Functions," ACM Trans. Math. Soft. 38(2), 15(2011).

From previous experiments we know that using algorithm 916 for smaller |z|  is competitive or faster and significantly accurate than Poppe’s algorithm. 

To avoid cancellation inaccuracies in case of small |x|, and |z| we will use Taylor series approximations and continued fraction expansions for large |x|,|z| as given in [this implementation](http://ab-initio.mit.edu/Faddeeva.cc)


###   Faddeeva   **w(z) = exp(-z\*\*2) * erfc(-i*z)** , where z is a complex number.

### Benchmark Results

<table cellpadding="5" cellspacing="52">
<tr>
 <th scope="col">Input Size</th>
 <th scope="col">Scipy</th>
 <th scope="col" colspan="8">TensorFlow Implementation</th>
 <!-- <th scope="col">Time</th> -->

</tr>
<tr>
 <td>&nbsp;</td>
 <td style="padding-right: 2px">&nbsp;</td>
 <td></td>
 <td style="padding-right: 30px"> CPU</td>
 
 <td>GPU</td>
</tr>
<tr>

<tr>

 <td>1e5</td>
 <td>10.4 ms</td>
 <td ></td>
 <td "padding-right: 30px">513 ms</td>
 <td>133 ms</td>
</tr>
<tr>
 <td>1e6</td>
 <td>85 ms</td>
 <td ></td>
 <td "padding-right: 30px">6.32 s</td>
 <td>1.18 s</td>
</tr>
<tr>
 <td>2e6</td>
 <td>162 ms</td>
 <td ></td>
 <td > 22.2 s </td>
 <td>1.23 s</td>
</tr>

</table>  

For the above results, the numbers are drawn from N(0,10^6) i.e. Gaussian Distribution
The results are very highly precise and compete with Scipy Implementation.

### [Approach 2 : Optimized Approach 1 with fixed relerr]

There were a lot of table keeping operations and a lot of loops which were taking a lot of time to execute. Since TensorFlow is not very optimized with loops we optimized the previous function and used Sympy to pre process some operations to directly feed the useful values to the tensorflow functions. We fixed the number of maximum iterations in our for loops and vectorized them to have fast results. But due to this the *relerr* is fixed now to the best values 0.0. We get following results on this implementation:

### Benchmark Results

<table cellpadding="5" cellspacing="52">
<tr>
 <th scope="col">Input Size</th>
 <th scope="col">Scipy</th>
 <th scope="col" colspan="8">TensorFlow Implementation</th>
 <!-- <th scope="col">Time</th> -->

</tr>
<tr>
 <td>&nbsp;</td>
 <td style="padding-right: 2px">&nbsp;</td>
 <td></td>
 <td style="padding-right: 30px"> CPU</td>
 
 <td>GPU</td>
</tr>
<tr>

<tr>

 <td>1e5</td>
 <td>9.8 ms</td>
 <td ></td>
 <td "padding-right: 30px">513 ms</td>
 <td>78 ms</td>
</tr>
<tr>
 <td>1e6</td>
 <td>84.9 ms</td>
 <td ></td>
 <td "padding-right: 30px">6.32 s</td>
 <td>747 ms</td>
</tr>


</table>  

I have also implemented **erfcx** function or scaled complementary error function and My implementation **beats** Tensorflow Probability (TFP) implementation for the same on GPU and CPU

 ### &nbsp;   &nbsp;   &nbsp;   &nbsp;          **erfcx = exp(x\**) * erfc(x)**
 <br />


### Benchmark Results for erfcx

<table>
<tr>
 <th scope="col" >Size</th>
 <th scope="col" colspan="2.5">TFP</th>


 <th scope="col" colspan="2">My TensorFlow Implementation</th>
 <!-- <th scope="col">Time</th> -->

</tr>
<tr>
 <td>&nbsp;</td>
 <td style="padding-right: 2.5" >CPU</td>
<td>GPU</td>
 <!-- <td></td> -->
 <td > CPU</td>
 <td>GPU</td>
</tr>
<tr>

<tr>

 <td>1e5</td>
 <td>45 ms</td> 
 <!-- gpu -->
 <td >10.3 ms</td>
 <td "padding-right: 30px">25.2 ms</td>
 <td>2.52 ms</td>
</tr>
<tr>
 <td>1e6</td>
 <td>334 ms</td>
 <td >17.6 ms</td>
 <td "padding-right: 30px">168 ms</td>
 <td>5.8 ms</td>
</tr>


</table>  

<!-- ### Benchmark Results

<table cellpadding="5" cellspacing="52">
<tr>
 <th scope="col">Input Size</th>
 <!-- <th scope="col">Scipy</th> -->
 <th scope="col" colspan=2.3">TFP</th>
 <th scope="col" colspan=3">TensorFlow Implementation</th>


 <!-- <th scope="col">Time</th> -->

</tr>
<tr>
 <td>&nbsp;</td>
 <!-- <td style="padding-right: 2px">&nbsp;</td> -->
 <!-- <td></td> -->
 <td> CPU</td>
 <td>GPU</td>
  <td></td>
  
 <td> CPU</td>
 <td>GPU</td>
</tr>
<tr>

<tr>

 <td>1e5</td>
 <td>54.2 ms</td>
 <td>16.6 ms</td>

 <td ></td>
 <td>24.5 ms</td>
 <td>4.66 ms</td>
 
</tr>
<tr>

 <td>1e6</td>
 <td>396 ms</td>
 <td>18 ms</td>

 <td ></td>
 <td>241 ms</td>
 <td>7 ms</td>
 
</tr>

</table>   -->


### [Approach 3 : Lighter Implementation]


Till now, I was focusing on precision along with speed. From this implementation in [cuda](https://github.com/aoeftiger/faddeevas/blob/master/cernlib_cuda/wofz.cu), I have tried to implemented Faddeeva function in TensorFlow focussing more on the speed while trading a bitoff with precision. The results are as follows:
### Benchmark Results

<table cellpadding="5" cellspacing="52">
<tr>
 <th scope="col">Input Size</th>
 <th scope="col">Scipy</th>
 <th scope="col" colspan="8">TensorFlow Implementation</th>
 <!-- <th scope="col">Time</th> -->

</tr>
<tr>
 <td>&nbsp;</td>
 <td style="padding-right: 2px">&nbsp;</td>
 <td></td>
 <td style="padding-right: 30px"> CPU</td>
 
 <td>GPU</td>
</tr>
<tr>

<tr>

 <td>1e5</td>
 <td>9.8 ms</td>
 <td ></td>
 <td "padding-right: 30px">169 ms</td>
 <td>7.94 ms</td>
</tr>
<tr>
 <td>1e6</td>
 <td>81.4 ms</td>
 <td ></td>
 <td "padding-right: 30px">2.32 s</td>
 <td>207 ms</td>
</tr>
<tr>
 <td>2e6</td>
 <td>175 ms</td>
 <td ></td>
 <td > 4.07 s</td>
 <td>362 ms</td>
</tr>

</table>  
The precision is still well defined and can be used for scientific calculations also.

## Conclusion

Due to the the nature of algortihm of using loops and variables dependent on the past value for calculating present value, TensorFlow does not remain a good fit to implement using its lower level functionalities. Even after optimizing series expansions, lookup table, subsidiary functions, vectorizing loops, the performance on GPU reaches near Scipy. Thus, the only way to implement this even faster in TensorFlow would be using kernels as done for Scipy. This should give major gains on atleast GPU than Scipy implementation. 
