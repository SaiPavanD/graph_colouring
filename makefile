NVCC_COMPILER=nvcc

main: kernels.o main.o
	$(NVCC_COMPILER) -arch=compute_35 -Wno-deprecated-gpu-targets kernels.o main.o -o gc.out 

kernels.o: kernels.cu kernels.h
	$(NVCC_COMPILER) -arch=compute_35 -Wno-deprecated-gpu-targets -c kernels.cu

main.o: main.cu
	$(NVCC_COMPILER) -arch=compute_35 -Wno-deprecated-gpu-targets -c main.cu

clean: 
	rm -rf *.o img errorfile.err outputfile.log *out

run: main
	timeout 120s ./gc.out
