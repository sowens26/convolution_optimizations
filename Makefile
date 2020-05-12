all: 
	g++ sequential_basic.cpp -o sequential_basic
	g++ sequential_opt.cpp -o sequential_opt
	g++ multi_basic.cpp -o multi_basic -fopenmp
	g++ multi_opt.cpp -o multi_opt -fopenmp
	nvcc gpu_basic.cu -o gpu_basic
	nvcc gpu_opt.cu -o gpu_opt
clean:
	rm -f sequential_basic sequential_opt multi_basic multi_opt gpu_basic gpu_opt
sequential_basic:
	g++ sequential_basic.cpp -o sequential_basic
sequential_opt:
	g++ sequential_opt.cpp -o sequential_opt
multi_basic:
	g++ multi_basic.cpp -o multi_basic -fopenmp
multi_opt:
	g++ multi_opt.cpp -o multi_opt -fopenmp
gpu_basic:
	nvcc gpu_basic.cu -o gpu_basic
gpu_opt:
	nvcc gpu_opt.cu -o gpu_opt
