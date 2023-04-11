import numpy as np
import multiprocessing as mp
import time
import Num_SH as NSH

Run_index = 0
folder = ["ThreeEqual", "E_Mu", "Tau", "AllTau", "AllMu"]
file_header = ["three_equal", "mostly_emu", "mostly_tau", "all_tau", "all_mu"]

mult_factor = [[1./3, 1./3, 1./3], [10./21, 10./21, 1./21], [1./12, 1./12, 10./12], [0, 0, 1], [0, 1, 0]]

def parallel_equal(mix,lep):
    return NSH.solve(mix * mult_factor[Run_index][0],mix * mult_factor[Run_index][1],mix * mult_factor[Run_index][2],lep,lep,lep, folder[Run_index], file_header[Run_index], make_plot=False)
    
if __name__ == '__main__':
    mixang = np.linspace( 1e-10, 3e-9, 30)
    lep0 = np.linspace(1e-3, 7e-3, 31)     
    run_list = []
    new_list = []
    for i in range(len(mixang)):
        for j in range(len(lep0)):
            run_list.append((mixang[i], lep0[j]))
            new_list.append((i,j))
        
    p = mp.Pool(6)
    new_start_time = time.time()
    
    res = p.starmap(parallel_equal, run_list)
    
    p.close()
    p.join()
    
    print("Parallel, elapsed time = {} seconds".format(time.time()-new_start_time))
    #print(res)
    
    np.savez("{}/{}-results".format(folder[Run_index], file_header[Run_index]), results = res, mixangle = mixang, L0 = lep0, index = new_list)
    

