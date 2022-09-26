import base64
import json
import os, os.path

from google.cloud import pubsub_v1
from google.cloud import storage

#instantiates a storage client
storage_client = storage.Client()

# Instantiates a Pub/Sub client
publisher = pubsub_v1.PublisherClient()
PROJECT_ID = 'federatedlearninginsa'

#how many clients we expect to update their weights
total_clients = 5

#function to count how many clients have updated their files
def count_files(bucketName, prefix):
    count=0
    options = {prefix: prefix,}
    blobs = storage_client.list_blobs(bucketName, prefix=prefix, delimiter='/')

    for blob in blobs:
        count+=1

    #includes directory as a 'blob' so im subtracting 1 from the total
    return count-1


def check_bucket(data, context):
    #gets the folder we are currently in
    bucket = data['bucket']
    path = data['selfLink']
    folder = path.split('%2F')[1]

    #defines the path name and calls the count function
    path_name = 'mnist_weights/' + folder + '/'
    updated_clients = count_files(bucket, path_name)


    #if more than half have been updated we send a pub/sub message
    #to alert the compute engine that it needs to process this folder
    if( updated_clients >= total_clients/2):
        #defining the publishing path
        topic_name = 'server-bucket'
        topic_path = publisher.topic_path(PROJECT_ID, topic_name)
        message_json = json.dumps({"folder": folder})
        message_bytes = message_json.encode('utf-8')

        # Publishes a message that contains the directory
        try:
            publish_future = publisher.publish(topic_path, data=message_bytes)
            publish_future.result()  # Verify the publish succeeded
            return 'Message published.'
        except Exception as e:
            print(e)
            return (e, 500)

    #otherwise it just prints to function logging
    return 'Updated Clients count: {}'.format(updated_clients)
