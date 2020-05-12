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
long long k, k2;// k = kernel dimension //k2 = k/2
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
void do_convolutions();
__global__ void convolution_top(float* in, float* out, float* ck, long long m, long long n, long long k);
__global__ void convolution_bottom(float* in, float* out, float* ck, long long m, long long n, long long k);
__global__ void convolution_left(float* in, float* out, float* ck, long long m, long long n, long long k);
__global__ void convolution_right(float* in, float* out, float* ck, long long m, long long n, long long k);
__global__ void convolution_top_left(float* in, float* out, float* ck, long long m, long long n, long long k);
__global__ void convolution_top_right(float* in, float* out, float* ck, long long m, long long n, long long k);
__global__ void convolution_bottom_left(float* in, float* out, float* ck, long long m, long long n, long long k);
__global__ void convolution_bottom_right(float* in, float* out, float* ck, long long m, long long n, long long k);
__global__ void convolution_core(float* in, float* out, float* ck, long long m, long long n, long long k);

int main(int nargs, char** args){
	verify_args(nargs, args);
	verify_available_memory();
	init_arrays();
	start_clock();
	do_convolutions();
	stop_clock();
	/*
	for (long long i=0; i<m; ++i){
		for (long long j=0; j<n; ++j){
			printf("%.2lf\t", arr[(i*n)+j]);
		}
		printf("\n");
	}
	*/
	cudaFree(d_in); cudaFree(d_out); cudaFree(d_ck);
	cudaFree(arr); cudaFree(ck);
	return 0;
}
void verify_args(int nargs, char** args){
	if (nargs != 4){
		printf("convolution <m> <n> <k>\n");
		exit(-1);
	}
	m = atoll(args[1]);
	n = atoll(args[2]);
	k = atoll(args[3]);
	k2 = k/2;
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
	ck[k2*k+k2] = k;
	cudaMemcpy(d_in, arr, m*n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ck, ck, k*k*sizeof(float), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
}
void do_convolutions(){
	convolution_top<<<k2, n-k+1>>>(d_in, d_out, d_ck, m, n, k);
	convolution_bottom<<<k2, n-k+1>>>(d_in, d_out, d_ck, m, n, k);
	convolution_left<<<m-k+1, k2>>>(d_in, d_out, d_ck, m, n, k);
	convolution_right<<<m-k+1, k2>>>(d_in, d_out, d_ck, m, n, k);
	convolution_top_left<<<k2, k2>>>(d_in, d_out, d_ck, m, n, k);
	convolution_top_right<<<k2, k2>>>(d_in, d_out, d_ck, m, n, k);
	convolution_bottom_left<<<k2, k2>>>(d_in, d_out, d_ck, m, n, k);
	convolution_bottom_right<<<k2, k2>>>(d_in, d_out, d_ck, m, n, k);
	convolution_core<<<m-k+1, n-k+1>>>(d_in, d_out, d_ck, m, n, k);
	cudaMemcpy(arr, d_out, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
}
__global__ void convolution_top(float* in, float* out, float* ck, long long m, long long n, long long k){
	long long k2 = k/2;
	long long row = blockIdx.x;
	long long col = threadIdx.x + k2;
	long long idx = row * n + col;
	long long x,y;
	float sum = 0;
	//out of bounds : top k2 rows 
	for (x=0; x<k2; ++x)
		for(y=0; y<k; ++y)
			sum += ck[x*k+y] * in[row*n + col];
	//in bounds : middle and bottom k2 rows
	for (x=k2; x<k; ++x)
		for(y=0; y<k; ++y)
			sum += ck[x*k+y] * in[((row-k2+x)*n) + (col-k2+y)];
	out[idx] = sum;
}
__global__ void convolution_bottom(float* in, float* out, float* ck, long long m, long long n, long long k){
	long long k2 = k/2;
	long long row = m - k2 + blockIdx.x;
	long long col = threadIdx.x + k2;
	long long idx = row * n + col;
	long long x,y;
	float sum = 0;
	//out of bounds : bottom k2 rows
	for (x=k2+1; x<k; ++x)
		for(y=0; y<k; ++y)
			sum += ck[x*k+y] * in[row*n + col];
	//in bounds : middle and top k2 rows
	for (x=0; x<=k2; ++x)
		for(y=0; y<k; ++y)
			sum += ck[x*k+y] * in[((row-k2+x)*n) + (col-k2+y)];
	out[idx] = sum;
}
__global__ void convolution_left(float* in, float* out, float* ck, long long m, long long n, long long k){
	long long k2 = k/2;
	long long row = k2 + blockIdx.x;
	long long col = threadIdx.x;
	long long idx = row*n + col;
	long long x,y;
	float sum = 0;
	//out of bounds : left k2 cols
	for (x=0; x<k; ++x)
		for (y=0; y<k2; ++y)
			sum += ck[x*k+y] * in[row*n + col];
	//in bounds : middle and right k2 cols
	for (x=0; x<k; ++x)
		for(y=k2; y<k; ++y)
			sum += ck[x*k+y] * in[((row-k2+x)*n) + (col-k2+y)];
	out[idx] = sum;
}
__global__ void convolution_right(float* in, float* out, float* ck, long long m, long long n, long long k){
	long long k2 = k/2;
	long long row = k2 + blockIdx.x;
	long long col = n - k2 + threadIdx.x;
	long long idx = row*n + col;
	long long x,y;
	float sum = 0;
	//out of bounds : right k2 cols
	for (x=0; x<k; ++x)
		for (y=k2+1; y<k; ++y)
			sum += ck[x*k+y] * in[row*n + col];
	//in bounds : middle and left k2 cols
	for (x=0; x<k; ++x)
		for(y=0; y<=k2; ++y)
			sum += ck[x*k+y] * in[((row-k2+x)*n) + (col-k2+y)];
	out[idx] = sum;
}
__global__ void convolution_top_left(float* in, float* out, float* ck, long long m, long long n, long long k){
	long long k2 = k/2;
	long long row = blockIdx.x;
	long long col = threadIdx.x;
	long long idx = row*n + col;
	long long x,y;
	float sum = 0;
	//out of bounds :
	// : top k2 rows
	for (x=0; x<k2; ++x)
		for (y=0; y<k; ++y)
			sum += ck[x*k+y] * in[row*n + col];
	// : left k2 cols
	for (x=k2; x<k; ++x)
		for(y=0; y<k2; ++y)
			sum += ck[x*k+y] * in[row*n + col];
	//in bounds
	for (x=k2; x<k; ++x)
		for (y=k2; y<k; ++y)
			sum += ck[x*k+y] * in[((row-k2+x)*n) + (col-k2+y)];
	out[idx] = sum;
}
__global__ void convolution_top_right(float* in, float* out, float* ck, long long m, long long n, long long k){
	long long k2 = k/2;
	long long row = blockIdx.x;
	long long col = n - k2 + threadIdx.x;
	long long idx = row*n + col;
	long long x,y;
	float sum = 0;
	//out of bounds :
	// : top k2 rows
	for (x=0; x<k2; ++x)
		for (y=0; y<k; ++y)
			sum += ck[x*k+y] * in[row*n + col];
	// : right k2 cols
	for (x=k2; x<k; ++x)
		for(y=k2+1; y<k; ++y)
			sum += ck[x*k+y] * in[row*n + col];
	//in bounds 
	for (x=k2; x<k; ++x)
		for (y=0; y<=k2; ++y)
			sum += ck[x*k+y] * in[((row-k2+x)*n) + (col-k2+y)];
	out[idx] = sum;
}
__global__ void convolution_bottom_left(float* in, float* out, float* ck, long long m, long long n, long long k){
	long long k2 = k/2;
	long long row = m - k2 + blockIdx.x;
	long long col = threadIdx.x;
	long long idx = row*n + col;
	long long x,y;
	float sum = 0;
	//out of bounds : 
	// : left k2 cols
	for (x=0; x<k; ++x)
		for (y=0; y<k2; ++y)
			sum += ck[x*k+y] * in[row*n + col];
	// : bottom k2 cols 
	for(x=k2+1; x<k; ++x)
		for (y=k2; y<k; ++y)
			sum += ck[x*k+y] * in[row*n + col];
	//in bounds
	for(x=0; x<=k2; ++x)
		for(y=k2; y<k; ++y)
			sum += ck[x*k+y] * in[((row-k2+x)*n) + (col-k2+y)];
	out[idx] = sum;
}
__global__ void convolution_bottom_right(float* in, float* out, float* ck, long long m, long long n, long long k){
	long long k2 = k/2;
	long long row = m - k2 + blockIdx.x;
	long long col = n - k2 + threadIdx.x;
	long long idx = row*n + col;
	long long x,y;
	float sum = 0;
	//out of bounds : 
	// : right k2 cols
	for (x=0; x<k; ++x)
		for (y=k2+1; y<k; ++y)
			sum += ck[x*k+y] * in[row*n + col];
	// : bottom k2 cols 
	for(x=k2+1; x<k; ++x)
		for (y=0; y<=k2; ++y)
			sum += ck[x*k+y] * in[row*n + col];
	//in bounds
	for(x=0; x<=k2; ++x)
		for(y=0; y<=k2; ++y)
			sum += ck[x*k+y] * in[((row-k2+x)*n) + (col-k2+y)];
	out[idx] = sum;
}
__global__ void convolution_core(float* in, float* out, float* ck, long long m, long long n, long long k){
	long long k2 = k/2;
	long long row = blockIdx.x + k2;
	long long col = threadIdx.x + k2;
	long long idx = row*n + col;
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
void start_clock(){
        start = std::chrono::system_clock::now();
};
void stop_clock(){
        stop = std::chrono::system_clock::now();
        elapsed = stop - start;
        printf("m:%llu n:%llu k:%llu\nseconds:%lf\npixels:%lf\npixels/sec:%lf\n", m, n, k,elapsed.count(), pixels, pixels/elapsed.count());
};
