from dask.distributed import Client, LocalCluster
from tasks import expensive
from dask import delayed
import time

def main():
    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)
    print("Dashboard:", client.dashboard_link)
    
    inputs = list(range(20))
    tasks = [delayed(expensive)(i) for i in inputs]
    total = delayed(sum)(tasks)
    
    start = time.time()
    result = total.compute()
    duration = time.time() - start

    print("Result: ", result)
    print(f"Duration: {duration:.2f}s")
    client.close()
    cluster.close()

if __name__ == "__main__":
    main()
    