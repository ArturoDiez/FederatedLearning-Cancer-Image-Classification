from google.cloud import storage

storage_client = storage.Client()

for bucket in storage_client.list_buckets():
    print(bucket.name)


bucket = storage_client.get_bucket("client_bucket_ot6_426127")

# blob = bucket.blob("test_path")
# blob.upload_from_filename("Test.png")



for file in storage_client.list_blobs(bucket):
    print(file.name)