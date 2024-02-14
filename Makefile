LDFLAGS = -pthread -lpthread
CFLAGS = -g -Wall -Werror

project: main.o layer.o neuron.o
	$(CC) $(LDFLAGS) -o project main.o layer.o neuron.o -lm

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

layer.o: layer.c
	$(CC) $(CFLAGS) -c layer.c

neuron.o: neuron.c
	$(CC) $(CFLAGS) -c neuron.c

clean:
	rm *.o project
