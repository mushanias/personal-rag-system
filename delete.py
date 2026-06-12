from database import client, COLLECTION_NAME

client.delete_collection(COLLECTION_NAME)
print("已删除")