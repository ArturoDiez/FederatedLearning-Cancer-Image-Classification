#put these at the start of the file
import os
import json
from google.cloud import pubsub_v1
from concurrent.futures import TimeoutError

#i set the credentials from the pub-sub service account in the keys.json file
credentials_path = '/home/aoife_igoe_ai/keys.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path


timeout = 5.0                                                                       # timeout in seconds
subscriber = pubsub_v1.SubscriberClient()
subscription_path = 'projects/federatedlearninginsa/subscriptions/client-bucket-sub'
current_file = []


def callback(message):
    print(f'Received message: {message}')
    print(f'data: {message.data}')
    data = json.loads(message.data)
    current_file.append(data['folder'])

    if message.attributes:
        print("Attributes:")
        for key in message.attributes:
            value = message.attributes.get(key)
            print(f"{key}: {value}")

    message.ack()

def check_pubsub():
    global current_file
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    print(f'Listening for messages on {subscription_path}')


    with subscriber:                                                # wrap subscriber in a 'with' block to automatically call close() when done
        try:
            streaming_pull_future.result(timeout=timeout)
            #streaming_pull_future.result()                          # going without a timeout will wait & block indefinitely
        except TimeoutError:
            streaming_pull_future.cancel()                          # trigger the shutdown
            streaming_pull_future.result()                          # block until the shutdown is complete

    print('New folders: {}'.format(current_file))

#can run this to check if theres new files, it will append the folder directorys that need to be checked to the
#global 'current_file'
check_pubsub()
