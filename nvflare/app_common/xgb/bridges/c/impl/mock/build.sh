gcc -fPIC -o client.o -c client.c
gcc -fPIC -o server.o -c server.c
gcc -shared -o libxgb.so client.o server.o
