#include<stdint.h>
#include<stdlib.h>
#include<stdio.h>
#include<unistd.h>
#include<omp.h>
#include<chrono>

//arrs
float* arr;//host array
float* ck;//convolution kernel
float* d_in;//device arrput array
float* d_out;//device output array
float* d_ck;//device kernel array
//vars
long long m, n;//in/out dimensions
long long k;// kernel dimensions
double pixels;
//time vars
std::chrono::time_point<std::chrono::system_clock> start, stop;
std::chrono::duration<float> elapsed;
float secs;
//protos
void start_clock();
void stop_clock();
void verify_args(int, char**);
void verify_available_memory();
void init_arrays();
__global__ void convolution(float* in, float* out, float* ck, long long m, long long n, long long k);

int main(int nargs, char** args){
	verify_args(nargs, args);
	verify_available_memory();
	init_arrays();
	start_clock();
	convolution<<<m, n>>>(d_in, d_out, d_ck, m, n, k);
	cudaMemcpy(arr, d_out, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	stop_clock();
	/*
	for (long long i=0; i<m; ++i){
		for (long long j=0; j<n; ++j){
			printf("%3lf ", arr[(i*n)+j]);
		}
		printf("\n");
	}
	*/
	cudaFree(d_in); cudaFree(d_out); cudaFree(d_ck);
	cudaFree(arr); cudaFree(ck);
	return 0;
}
void start_clock(){
        start = std::chrono::system_clock::now();
};
void stop_clock(){
        stop = std::chrono::system_clock::now();
        elapsed = stop - start;
        printf("m:%llu n:%llu k:%llu\nseconds:%lf\npixels:%lf\npixels/sec:%lf\n", m, n, k,elapsed.count(), pixels, pixels/elapsed.count());
};
void verify_args(int nargs, char** args){
	if (nargs != 4){
		printf("convolution <m> <n> <k>\n");
		exit(-1);
	}
	m = atoll(args[1]);
	n = atoll(args[2]);
	k = atoll(args[3]);
	pixels = m*n;
	if (k%2==0 || k<3){
		printf("k must be at least 3 and odd\n");
		exit(-1);
	}else if(k>m || k>n){
		printf("x and n must both be larger than k\n");
		exit(-1);
	}
}
void verify_available_memory(){
	long long pages = sysconf(_SC_PHYS_PAGES);
	long long page_size = sysconf(_SC_PAGE_SIZE);
	long long bytes_free = pages * page_size;
	size_t gpu_bytes, gpu_bytes_free;
	cudaMemGetInfo( &gpu_bytes_free, &gpu_bytes );	
	//host = arr + ck
	long long bytes_needed_host = (sizeof(float)*m*n) + (sizeof(float)*k*k);
	//gpu = d_in + d_out + d_ck
	long long gpu_bytes_needed = (sizeof(float)*m*n*2) + (sizeof(float)*k*k);
	if (bytes_needed_host > bytes_free){
		printf("not enough memory available on the host\n");
		exit(-1);
	}
	else if (gpu_bytes_needed > gpu_bytes_free){
		printf("not enough memory available on the gpu\n");
		exit(-1);
	}
}
void init_arrays(){
	cudaMallocHost((void**)  &arr, sizeof(float)*m*n);
	cudaMallocHost((void**) &ck, sizeof(float)*k*k);
	cudaMalloc((void**) &d_in, sizeof(float)*m*n);
	cudaMalloc((void**) &d_out, sizeof(float)*m*n);
	cudaMalloc((void**) &d_ck, sizeof(float)*k*k);
	long long i,j;
	for(i=0; i<m; ++i) for(j=0; j<n; ++j) arr[i*n+j] = 1.;
	for(i=0; i<k; ++i) for(j=0; j<k; ++j) ck[i*k+j] = .5;
	ck[(k/2)*k+(k/2)] = k;
	cudaMemcpy(d_in, arr, m*n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ck, ck, k*k*sizeof(float), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
}
__global__ void convolution(float* in, float* out, float* ck, long long m, long long n, long long k){
	long long idx = blockIdx.x*blockDim.x + threadIdx.x;
	long long row = blockIdx.x;
	long long col = threadIdx.x;
	long long k2 = k/2;
	long long x,y;
	float sum=0.;
	for(x=0; x<k; ++x){
		for(y=0; y<k; ++y){
			if ( (row-k2+x >= 0)&&(row-k2+x < m) && (col-k2+y >= 0)&&(col-k2+y < n) )
				sum += ck[x*k+y] * in[((row-k2+x)*n) + (col-k2+y)];
			else
				sum += ck[x*k+y] * in[idx];
		}
	}
	out[idx] = sum;
}

