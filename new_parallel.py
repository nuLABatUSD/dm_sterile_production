import numpy as np
import multiprocessing as mp
import time
import Num_SH as NSH


def parallel_equal(mix,lep):
    return NSH.solve(mix/3,mix/3,mix/3,lep,lep,lep, "Quick", "three_equal", make_plot=False)
    
if __name__ == '__main__':
    mixang = np.linspace( 1e-10, 3e-9, 10) #use 19
    lep0 = np.linspace(1e-3, 9e-3, 9) #use 19
    
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
    
    np.savez("DataRun/three_equal-results", results = res, mixangle = mixang, L0 = lep0, index = new_list)
    

