# ** ** ** ** ** ** ** ** ** Server ** ** ** ** ** ** ** ** ** **
import socket
import sys

# HOST = '192.168.1.27'  # this is your localhost
HOST = 'localhost'
PORT = 6666

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# socket.socket: must use to create a socket.
# socket.AF_INET: Address Format, Internet = IP Addresses.
# socket.SOCK_STREAM: two-way, connection-based byte streams.
print('socket created')

# Bind socket to Host and Port
try:
    s.bind((HOST, PORT))
except socket.error as err:
    print('Bind Failed, Error Code: ' + str(err[0]) + ', Message: ' + err[1])
    sys.exit()

print('Socket Bind Success!')

# listen(): This method sets up and start TCP listener.
s.listen(10)
print('Socket is now listening')

counter = 0
while counter < 1:
    client, addr = s.accept()
    print('Connect with ' + addr[0] + ':' + str(addr[1]))
    # receive all the bytes and write them into the file.

    file = open("ski2_copy.mp4", "wb")
    while True:
        received = client.recv(5)
        # Stop receiving.
        if received == b'':
            file.close()
            break
        # Write bytes into the file.
        file.write(received)
    print('Finish with ' + addr[0] + ':' + str(addr[1]))

    counter += 1
    # buf = client.recv(64)
    # print(buf)
    # if buf == 'exit':
    #     break
s.close()


