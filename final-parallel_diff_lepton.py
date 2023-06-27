import numpy as np
import multiprocessing as mp
import time
import Num_SH_fast as NSH
import os

New_index_L = 3

mult_factor = [[1./3, 1./3, 1./3], [10./21, 10./21, 1./21], [1./12, 1./12, 10./12], [1./12, 10./12, 1./12], [0.5, 0, 0.5]]

mult_L_factor = [[0.75,0.75,1.25], [1.25,1.25,0.75], [0.75,1.25,0.75], [2,0,-2]]
mix_index = [0,0,0,4]
New_index_mult = mix_index[New_index_L]

L_folder = ["New-ThreeEqual-TauL", "New-ThreeEqual-EMuL", "New-ThreeEqual-MuL", "New-OppoL"]
L_file_header = ["three_equal_tauL", "three_equal_emuL", "three_equal_muL", "oppo"]

run_all = False

def parallel_difflep(mix, L0):
    return NSH.solve(mix * mult_factor[New_index_mult][0],mix * mult_factor[New_index_mult][1],mix * mult_factor[New_index_mult][2],L0 * mult_L_factor[New_index_L][0],L0 * mult_L_factor[New_index_L][1],L0 * mult_L_factor[New_index_L][2], L_folder[New_index_L], L_file_header[New_index_L], make_plot=False, run_sp_again=run_all, run_pk_again=run_all)

if __name__ == '__main__':
    
    if os.path.exists("{}/{}-results.npz".format(L_folder[New_index_L], L_file_header[New_index_L])):
        print("Summary file {}/{}-results.npz already exists. Do not overwrite. Abort.".format(L_folder[New_index_L], L_file_header[New_index_L]))
    elif os.path.isdir("{}".format(L_folder[New_index_L])):
        print("Folder {} already exists. Do not overwrite data. Abort".format(L_folder[New_index_L]))
    else:
        print("Creating mixing model {} with lepton number model {} in folder {}".format(mult_factor[New_index_mult], mult_L_factor[New_index_L], L_folder[New_index_L]))
        
        
        mixang = np.linspace( 1e-10, 3e-9, 30)
        lep0 = np.linspace(1e-3, 10e-3, 31)     
        run_list = []
        new_list = []
        for i in range(len(mixang)):
            for j in range(len(lep0)):
                run_list.append((mixang[i], lep0[j]))
                new_list.append((i,j))
            
        p = mp.Pool(4)
        new_start_time = time.time()
        
        res = p.starmap(parallel_difflep, run_list)    
        p.close()
        p.join()
        
        print("Parallel, elapsed time = {} seconds".format(time.time()-new_start_time))
        
        np.savez("{}/{}-results".format(L_folder[New_index_L], L_file_header[New_index_L]), results = res, mixangle = mixang, L0 = lep0, index = new_list)
    

