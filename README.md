# dm_sterile_production
The function $\textbf{sterile-production}$ has inputs: N; flavor (of active species); sterile mass; vacuum mixing angle; initial lepton number; and an input to produce (make_plot=True) or not produce (make_plot=False) the sterile distribution plots($\epsilon^2f_{\epsilon}$ vs $\epsilon$ plots).

$\textbf{sterile-production}$ takes in initial conditions and flavor specification and runs steps_taken, plots the produced $\nu_s$ and $\overline{\nu}_s$ distributions, calculates the produced energy density, $\Omega_s h^2$ value, and creates an .npz file with all the data. The .npz file is named as: "flavor x inital lepton number x mixing angle."

$\textbf{Bella-2}$ has two arguments, $\textbf{file-name}$ and $\textbf{k}$. The input file_name should be a string that corresponds to an .npz file. This file must be inside your local git repository to run properly. Input, $\textbf{k}$, is an integer that refers to the step size we want to take. Smaller stepsize will give us more precision, but it will take longer to run. Bigger stepsize gives less precision, but it takes a shorter time to run. 

Using these two inputs $\textbf{Bella-2}$ function returns an array of outpus from Eq.17 in the Schneider-15 paper based on differnt mass values. This function also returns the integral of Eq.17 that allows us to approximate the number of subhalos produced by the input model. The function returns two more arrays. The arrays $\textbf{lnM-vals}$ and $\textbf{sv}$ are used to plot. Return value $\textbf{lnM-vals}$ is an array containing relevant values of masses. 

$\textbf{Bella-2}$ calls several other functions in the Jupyter Notebook "PK and K". The ones not mentioned here are used to construct Eq.17. However, there are two others that complete different tasks. Functions $\textbf{make-Pk}$, $\textbf{idea-sigma8}$ have two inputs: spec-file, (a string) and omega-hh (the $\Omega_s h^2$ value extracted from the .npz file). $\textbf{ideal-sigma8}$ creates the ideal $\sigma_8$ to put into the CLASS code. $\textbf{make-Pk}$ changes CLASS settings then computes the correct value of $\sigma_8$. We then create two new .npy files with data on $k$ and $Pk$. 
