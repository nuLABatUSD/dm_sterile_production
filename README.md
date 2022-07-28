# dm_sterile_production
The function $\textbf{sterile-production}$ has inputs: N; flavor (of active species); sterile mass; vacuum mixing angle; initial lepton number; and an input to produce (make_plot=True) or not produce (make_plot=False) the sterile distribution plots($\epsilon^2f_{\epsilon}$ vs $\epsilon$ plots).

$\textbf{sterile-production}$ takes in initial conditions and flavor specification and runs steps_taken, plots the produced $\nu_s$ and $\overline{\nu}_s$ distributions, calculates the produced energy density, $\Omega_s h^2$ value, and creates an .npz file with all the data. The .npz file is named as: "flavor x inital lepton number x mixing angle."
