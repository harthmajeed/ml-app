from dask.distributed import Client
from dask import delayed
import time

def expensive(x):
    time.sleep(5)
    return x * x

def main():
    client = Client("tcp://scheduler:8786")
    print("Connected to scheduler: ", client.scheduler_info()["address"])
    tasks = [delayed(expensive)(i) for i in range(20)]
    total = delayed(sum)(tasks)
    result = total.compute()
    print("Result: ", result)
    client.close()

if __name__ == "__main__":
    main()
    