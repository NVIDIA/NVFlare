gcc -fPIC -o client.o -c client.c
gcc -shared -o libxgb.so client.o
