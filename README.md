# Launch the programs
The client and the server can be launched in the google cloud platform using docker containers. 

# On GCP
You should make your changes locally and push it to the git repo. 
1. Do `git pull` in the FederatedLearning folder of the client/server.
2. Build the updated docker container. In the Client folder: `sudo docker build -f client.dockerfile -t client .` Server: `sudo docker build -f server.dockerfile -t server .` This may take up to around 5 minutes.
3. Run the docker containers. For the client and the server the first argument is the foldername in the google cloud platforms where the weights are stored. Only the clients have a second command line argument, which is the client id. E.g. to run the server with the directory name foo_6: `sudo docker run server foo_6`. To run the client with the directory name foo_6 and the client_id 0: `sudo docker run client foo_6 0`

# Local
You can run the client and the server locally from your command line by just executing the python functions. The server and the client still have the directory_name as the first command line argument and the clients have the client id as the second command line argument. 
1. To run the server with the directory name foo_7, execute `python3 server.py foo_7 `
2. To run the client with the directory name foo_7 and the client id 0, execute `python3 client.py foo_7 0`

# Known bugs
Sometimes the synchronization between the server and the client does not work perfectly. Therefore it is recommended to first start all clients and wait until they are running properly and are waiting for a new file by the server. Then the server should be started.

# Contributors
Aoife McGarrigle, Arturo Diez, Leonardo Vasquez, Marc Ziegler, Sebastian Kerres

# Report 
[report link](https://docs.google.com/document/d/1DlfVLCjIKz3YauWahpEX9nDpmKgEOLgqZlBAntjOKZc/edit#heading=h.62j7olksc92e)
