import time
import random
from numba import jit
from threading import Thread
import os 

class tic_toe:
    def __init__(self):
        self.t1 = 0
        self.t2 = 0

    def tic(self):
        self.t1 = time.time()
    
    def toe(self):
        self.t2 = time.time()
        print(self.t2-self.t1)


class find_pi():
    def __init__(self):
        self.i = 0
        self.n = 0

# '''Atomic Code: which you can't divide any more'''
    @staticmethod
    @jit(nopython=True, nogil=True)
    def get_points_static(m):
        # t = tic_toe()
        # t.tic()
        i, n = 0, 0
        for _ in range(m):
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            
            r = x**2 + y**2

            if r <= 1:
                i+=1
        
        n = m

        # t.toe()
        return i, n
    

    def get_points(self, m):
        (self.i, self.n) = self.get_points_static(m)
            

    def print_pi(self):
        ans = (4*self.i)/self.n
        print(ans)


import math 
def minPartitionScore(nums, K) -> int:

        n = len(nums)
        pref = [0]*(n+1)
        for i in range(1, n+1):
            pref[i] = pref[i-1] + nums[i-1]
                
        def func(idx, k):
            # if idx >= n:
                # return 0
            print(idx, k)
            if k==1:
                sm = pref[n] - pref[idx]
                return (sm*(sm+1))//2

            ans = math.inf
            
            for j in range(idx+1, n-k+2):
                sm = pref[j] - pref[idx]

                v = (sm*(sm+1))//2

                ans = min(ans, func(j, k-1)+v)

            return ans

        return func(0, K)

if __name__ == "__main__":

    arr = [1,1,1]
    k = 3
    print(minPartitionScore(arr, k))
#     t = tic_toe()
#     t.tic()
# # pi = find_pi()
# # pi.get_points(10000000)
# # pi.print_pi() 
#     find_pis = []
#     threads = []

#     n = 10000000

#     for i in range(os.cpu_count()):
#         find_pis.append(find_pi())
#         threads.append(Thread(target=find_pis[i].get_points, args = (n,)))
    
#     for thread in threads:
#         thread.start()

#     for thread in threads:
#         thread.join()

#     circle_pts = 0
#     total_pts = 0

#     for pts in find_pis:
#         circle_pts += pts.i
#         total_pts += pts.n
    
#     print(f'Circle pts: {circle_pts}, total_pts: {total_pts}, -> pi: {(4*circle_pts)/total_pts}')

#     t.toe() 


