import asyncio
# import time

async def fetch_data(id):
  await asyncio.sleep(1)
  return f"fetched data from {id}"

async def put_data(data_queue, nums_data):
  for i in range(nums_data):
    data = await fetch_data(i)
    await data_queue.put(data)
    print(data)
  await data_queue.put(None)

async def get_data(data_queue):
  while True:
    data = await data_queue.get()
    await asyncio.sleep(1)
    print(f"got data {data}")
    if data is None:
      break

async def main():
  import queue
  data_queue = queue.Queue()
  nums_data = 4
  await asyncio.gather(
    put_data(data_queue, nums_data),
    get_data(data_queue)
  )

# asyncio.run(main())
await main()


