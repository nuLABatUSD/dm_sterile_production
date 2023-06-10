import numpy as np
import multiprocessing as mp
import time
import Num_SH_fast as NSH
import os

Run_index = 0
folder = ["New-ThreeEqual", "New-E_Mu", "New-Tau", "New-AllTau", "New-AllMu"]
file_header = ["three_equal", "mostly_emu", "mostly_tau", "all_tau", "all_mu"]

mult_factor = [[1./3, 1./3, 1./3], [10./21, 10./21, 1./21], [1./12, 1./12, 10./12], [0, 0, 1], [0, 1, 0]]

run_all = False

def parallel_equal(mix,lep):
    return NSH.solve(mix * mult_factor[Run_index][0],mix * mult_factor[Run_index][1],mix * mult_factor[Run_index][2],lep,lep,lep, folder[Run_index], file_header[Run_index], make_plot=False, run_sp_again=run_all, run_pk_again=run_all)
    


if __name__ == '__main__':
    
    if os.path.exists("{}/{}-results.npz".format(folder[Run_index], file_header[Run_index])):
        print("Summary file {}/{}-results.npz already exists. Do not overwrite. Abort.".format(folder[Run_index], file_header[Run_index]))
    elif run_all and os.path.isdir("{}".format(folder[Run_index])):
        print("Folder {} already exists. Do not overwrite data. Abort".format(folder[Run_index]))
    else:
        mixang = np.linspace( 1e-10, 3e-9, 30)
        lep0 = np.linspace(1e-3, 7e-3, 31)     
        run_list = []
        new_list = []
        for i in range(len(mixang)):
            for j in range(len(lep0)):
                run_list.append((mixang[i], lep0[j]))
                new_list.append((i,j))
            
        p = mp.Pool(4)
        new_start_time = time.time()
        
        res = p.starmap(parallel_equal, run_list)
        p.close()
        p.join()
        
        print("Parallel, elapsed time = {} seconds".format(time.time()-new_start_time))
        
    
        np.savez("{}/{}-results".format(folder[Run_index], file_header[Run_index]), results = res, mixangle = mixang, L0 = lep0, index = new_list)

