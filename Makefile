CFLAGS=-g -std=gnu99 -O4 -Wno-unused-result -Wall -Wextra
# -pg
#CLIBS=-lm
CC=gcc
MPICC=mpicc

	 
hybrid: 3d_hybrid.c
#$^ rhs of :
	$(MPICC) $^ $(CFLAGS) -lm -o 3d_hybrid.exe -fopenmp
	 @echo "wrote executable to 3d_hybrid.exe"
	 	 
	 
serial: 3d_serial_reference.c
	$(CC) $^ $(CFLAGS) -lm -o 3d_serial.exe
	 @echo "wrote executable to 3d_serial.exe"
