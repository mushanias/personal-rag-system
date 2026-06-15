import asyncio
from database import client, COLLECTION_NAME


async def main():
    if not await client.collection_exists(COLLECTION_NAME):
        print(f"Collection 不存在：{COLLECTION_NAME}")
        await client.close()
        return

    await client.delete_collection(collection_name=COLLECTION_NAME)
    await client.close()

    print(f"已删除 Collection：{COLLECTION_NAME}")


if __name__ == "__main__":
    asyncio.run(main())

# 我的代码一开始是同步写法，修改后这个脚本也顺便修改了，
# 脚本文件本身不能直接 await，
# 所以我们包一层 async def main()，最后用 asyncio.run(main()) 启动它