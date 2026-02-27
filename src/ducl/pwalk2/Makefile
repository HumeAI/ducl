CC     = gcc
CFLAGS = -O2 -Wall -std=gnu11
LDFLAGS = -lpthread

pwalk2: pwalk2.c pw2_worker.c pw2_output.c pwalk2.h pw2_uring.h
	$(CC) $(CFLAGS) $(LDFLAGS) -o pwalk2 pwalk2.c pw2_worker.c pw2_output.c

clean:
	rm -f pwalk2

.PHONY: clean
