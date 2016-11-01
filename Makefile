G++FLAG = -std=c++11 -O3 -m64 -mmmx -msse -msse2 -pipe
NVCCFLAG = -std=c++11 -O3 -arch sm_21 -use_fast_math

I_PATH_CPU = /usr/local/include
L_PATH_CPU = /usr/local/lib
I_PATH_GPU = /usr/local/cuda/include
L_PATH_GPU = /usr/local/cuda/lib64

LINK_OPTS_CPU = -L$(L_PATH_CPU) -lgsl -lgslcblas
LINK_OPTS_GPU = -L$(L_PATH_GPU) -lm -lGL

all: cpu gpu

cpu: cpu.o gmm.cpu.o
	g++ -o cpu cpu.o gmm.cpu.o $(LINK_OPTS_CPU)

cpu.o: src/cpu/main.cpp
	g++ $(G++FLAG) -o cpu.o -c src/cpu/main.cpp -I$(I_PATH_CPU)

gmm.cpu.o: src/cpu/gmm.cpp
	g++ $(G++FLAG) -o gmm.cpu.o -c src/cpu/gmm.cpp -I$(I_PATH_CPU)

gpu: gpu.o gmm.gpu.o
	nvcc -o gpu gpu.o gmm.gpu.o $(LINK_OPTS_GPU)

gpu.o: src/gpu/main.cpp
	g++ $(G++FLAG) -o gpu.o -c src/gpu/main.cpp -I$(I_PATH_CPU)

gmm.gpu.o: src/gpu/gmm.cu
	nvcc $(NVCCFLAG) -o gmm.gpu.o -c src/gpu/gmm.cu -I$(I_PATH_GPU)

generarDatos: generarDatos.o
	g++ -o generarDatos generarDatos.o $(LINK_OPTS_CPU)

generarDatos.o: src/generarDatos.cpp
	g++ $(G++FLAG) -c src/generarDatos.cpp -I$(I_PATH_CPU)

clean:
	rm *.o
