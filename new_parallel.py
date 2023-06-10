import numpy as np
import multiprocessing as mp
import time
import Num_SH as NSH
import os

Run_index = 4
folder = ["ThreeEqual", "E_Mu", "Tau", "AllTau", "AllMu"]
file_header = ["three_equal", "mostly_emu", "mostly_tau", "all_tau", "all_mu"]

mult_factor = [[1./3, 1./3, 1./3], [10./21, 10./21, 1./21], [1./12, 1./12, 10./12], [0, 0, 1], [0, 1, 0]]

def parallel_equal(mix,lep):
    return NSH.solve(mix * mult_factor[Run_index][0],mix * mult_factor[Run_index][1],mix * mult_factor[Run_index][2],lep,lep,lep, folder[Run_index], file_header[Run_index], make_plot=False)
    

New_index_mult = 2
New_index_L = 1

mult_L_factor = [[0.5, 0.5, 1], [1,1,0.5]]
L_folder = ["Tau-MoreL", "Tau-LessL"]
L_file_header = ["mostly_tau_tauL", "mostly_tau_emuL"]

def parallel_difflep(mix, L0):
    return NSH.solve(mix * mult_factor[New_index_mult][0],mix * mult_factor[New_index_mult][1],mix * mult_factor[New_index_mult][2],L0 * mult_L_factor[New_index_L][0],L0 * mult_L_factor[New_index_L][1],L0 * mult_L_factor[New_index_L][2], L_folder[New_index_L], L_file_header[New_index_L], make_plot=False)

if __name__ == '__main__':
    
#    if os.path.exists("{}/{}-results.npz".format(L_folder[New_index_L], L_file_header[New_index_L])):
#        print("Summary file {}/{}-results.npz already exists. Do not overwrite. Abort.".format(L_folder[New_index_L], L_file_header[New_index_L]))
#    elif os.path.isdir("{}".format(L_folder[New_index_L])):
#        print("Folder {} already exists. Do not overwrite data. Abort".format(L_folder[New_index_L]))
#    else:
        mixang = np.linspace( 1e-10, 3e-9, 30)
        lep0 = np.linspace(1e-3, 10e-3, 31)     
        run_list = []
        new_list = []
        for i in range(len(mixang)):
            for j in range(len(lep0)):
                run_list.append((mixang[i], lep0[j]))
                new_list.append((i,j))
            
        p = mp.Pool(6)
        new_start_time = time.time()
        
        res = p.starmap(parallel_equal, run_list)
    #    res = p.starmap(parallel_difflep, run_list)    
        p.close()
        p.join()
        
        print("Parallel, elapsed time = {} seconds".format(time.time()-new_start_time))
        #print(res)
        
   #     np.savez("{}/{}-results".format(L_folder[New_index_L], L_file_header[New_index_L]), results = res, mixangle = mixang, L0 = lep0, index = new_list)
    
        np.savez("{}/{}-results".format(folder[Run_index], file_header[Run_index]), results = res, mixangle = mixang, L0 = lep0, index = new_list)

