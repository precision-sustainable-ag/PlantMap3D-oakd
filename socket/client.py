import socket, cv2, pickle, struct
import json
import sys
import numpy as np
import json_numpy
import time
import select

host_ip = '192.168.181.1'
port = 6666

# for starting up the camera on server side
str_cmd = { "key":"Start" }
str_msg = json.dumps(str_cmd)
# for getting a preview frame from camera
img_cmd = { "key":"Preview" }
img_msg = json.dumps(img_cmd)
# for getting a set of images (1 RGB + 2 Mono + 1 Depth)
snap_cmd = { "key":"TakeSnapshot" }
snap_msg = json.dumps(snap_cmd)
# for stopping the camera on server side
img_scmd = { "key":"StopPreview" }
img_smsg = json.dumps(img_scmd)
# for closing the client connection to the server
stp_cmd = { "key":"Close" }
stp_msg = json.dumps(stp_cmd)


for i in range(1):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host_ip, port))

    # start preview
    print("\nStart")
    message = str_msg.encode()
    client_socket.sendall(message)
    time.sleep(5)

    # preview messages
    message = img_msg.encode()
    j = time.time()
    
    while(time.time()-j < 5):
        client_socket.sendall(message)
        t = time.time()
        # get length of message
        data = b""
        data = client_socket.recv(7)
        l = int( data.decode() )
        # get actual frame
        data = b""
        while len(data) < l:
            packet = client_socket.recv(1024)
            data += packet

        arr = data.decode()
        decoded_arr = json.loads(arr)
        dd = decoded_arr["data"]["pixels"]
        cv2.imwrite(f"preview/prv_{t}.jpg", np.asarray(dd))


    # ask for a snapshot
    message = snap_msg.encode()
    client_socket.sendall(message)
    t = time.time()

    data = b""
    for i in range(3):

        while len(data) < 7:
            packet = client_socket.recv(1024)
            if not packet:	break
            data += packet
        msg_size = json.loads(data[:7].decode())
        l = int(msg_size)
        print(l)
        data = data[7:]
        while len(data) < l:
            data += client_socket.recv(1024)
        decoded_arr = json.loads(data[:l].decode())
        data = data[l:]
    
        dd = decoded_arr["data"]["pixels"]
    
        if (decoded_arr["type"]=="RGB"):
            cv2.imwrite(f"preview/rgb_{t}.jpg", np.asarray(dd))
        if (decoded_arr["type"]=="MonoLeft"):
            cv2.imwrite(f"preview/left_{t}.jpg", np.asarray(dd))
        if (decoded_arr["type"]=="MonoRight"):
            cv2.imwrite(f"preview/right_{t}.jpg", np.asarray(dd))


    # stop preview
    print("\nStop Camera")
    message = img_smsg.encode()
    client_socket.sendall(message)


    # closing connection
    print("Close")
    message = stp_msg.encode()
    client_socket.sendall(message)
    client_socket.close()
    print("Client Closed")
