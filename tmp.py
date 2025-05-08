import asyncio

async def fetch_data(id):
    await asyncio.sleep(1)
    return f"data {id}"

async def put_data(data_queue, nums_data):
    for i in range(nums_data):
        data = await fetch_data(i)
        await data_queue.put(data)
        print(f"Produced: {data}")
    await data_queue.put('End')

async def get_data(data_queue):
    while True:
        data = await data_queue.get()
        if data == 'End':
            break
        await asyncio.sleep(1)
        print(f"Consumed: {data}")
    print("Consumer done")

async def main():
    data_queue = asyncio.Queue()
    nums_data = 4
    await asyncio.gather(
        put_data(data_queue, nums_data),
        get_data(data_queue)
    )

await main()


# Cell output:
# Produced: data 0
# Produced: data 1
# Consumed: data 0
# Produced: data 2
# Consumed: data 1
# Produced: data 3
# Consumed: data 2
# Consumed: data 3
# Consumer done
