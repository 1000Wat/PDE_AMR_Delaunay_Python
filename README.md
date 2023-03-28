# PDE_AMR_Delaunay_Python

THIS IS WIP, I am only making this project public to show proof I am working on this for an application. 

Just a start of a simple algorithm to solve PDE on a non uniform multidimensionnal grid by using Delaunay tesselation and solving the equations by Newton-Broyden methods, I will describe how it works later on.  
For now there is only the uniform type of grid and it is capable to solve usual PDE such as wave equations and heat equations. 
Since it is possible to use a non uniform type of grid, I want to add the possiblity of doing AMR (Adaptative Mesh Refinement). It should be simple since the non-uniformity of the grid is already taken care of.

## How it works

The way this code works is simple. A partial differential equations can be resumed as : $F(\phi)=0$ where $F$ is an operator (a function of a function) and $\phi$ is a function of nature $E \rightarrow V$ where $E$ and $V$ are two multidimensionnal spaces (for now vector spaces, E must be at least two dimensionnal and V is the real numbers, so it correspond to a scalar field). 
We discretize the space in N points and we evaluate a proposition for phi on thoses points : $\phi_i$ for $i$ in $(1,...,N)$. We then evaluate $F(\phi_i)$ for every i (by using finite definition of the derivatives). This gives us N algebraic equations (and not differential equations) of the form $F(\phi_i)=0$ with $\phi_i$ as the unknows. We then use a Newton-Broyden method to find the closest solution next to the initial proposition for the solution : $\phi_{0_i}$.

Important notes: As you may have noticed, the form of the PDEs are not prescribed so this is THEORETICALLY a general way of solving PDEs. But this general way comes with the cost of $O(N^3)$ because the Newton-Broyden method inverts a matrix wich have a $O(N^3). Some costs also comes with the Delaunay tesselation to adopt a non-uniform grid matrix. 

## TO DO

-FIND A WAY TO MAKE THE CALCUATION OF THE JACOBIAN FASTER BY MULTIPROCESSING OR BY USING NUMPY. THIS IS A HUGE BOTTLENECK AT THE MOMENT (30sec of calculation for 800 points)
-Had ways to visualize results from more than 2D (time + x-axis).
-Had non-uniform grid such as a disk or an ring.
-Had Boundary condition more than Dirichlet, such as Neumann or periodic boundary.
-Had AMR 
-Had metrics (will need to check Differential Geometry a lot more)
