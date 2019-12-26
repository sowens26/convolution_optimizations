/*
 	@author Samuel L Owens
	@title Convolutions: fully parallel
	2d pixel convolutions 
	K x K kernel applied to M x N pixel array
	basic multithreaded
	
		parallelize the core, the border walls, and the corners as omp tasks
		treat the core as a single task
		this will give some small speedups but not much
		the core is the bulk of the computation

		the speedup from this approach increases with the size of the kernel
		this makes sense because the only benefit would be from the borders
		the size of the border will always be the floor division of k/2

		the core is the bulk of the work
		this approach runs the core on all threads
		but relies on omp's static schedule for the chunking
		will cause the array to be carved into n_threads horizontal strips
		the strips can operate in parallel to one another

		it is possible that manually carving the array into a square grid as opposed to strips
		may provide additional speedup

	carving the core into a grid of chunks and operating on those chunks in parallel 
	should offer additional speedup
		
*/
#include<stdlib.h>
#include<stdio.h>
#include<unistd.h>
#include<omp.h>
#include<chrono>

std::chrono::time_point<std::chrono::system_clock> start, stop;
std::chrono::duration<double> elapsed;
void startClock(){
	start = std::chrono::system_clock::now();
};
void stopClock(){
	stop = std::chrono::system_clock::now();
	elapsed = stop - start;
};
void verifyArgs(int*, char***, long long*, long long*, long long*, double*);
void initArrays(double***, double***, double***, long long*, long long*, long long*);
void printArray(double***, long long*, long long*);
void freeArrays(double***, double***, double***, long long, long long, long long);
double convolutionCore( long long, long long, long long, double***, double***);
double convolutionTop( long long, long long, long long, double***, double***);
double convolutionBottom( long long, long long, long long, double***, double***);
double convolutionLeft( long long, long long, long long, double***, double***);
double convolutionRight( long long, long long, long long, double***, double***);
double convolutionTopLeft( long long, long long, long long, double***, double***);
double convolutionTopRight( long long, long long, long long, double***, double***);
double convolutionBottomLeft( long long, long long, long long, double***, double***);
double convolutionBottomRight( long long, long long, long long, double***, double***);

int main(int nargs, char** args){
	double **in, **out, **c, pixels;
	long long i, j, m, n, k;
	verifyArgs(&nargs, &args, &m, &n, &k, &pixels);
	initArrays(&in, &out, &c, &m, &n, &k);

	startClock();
  #pragma omp parallel 
  {
  #pragma omp single
  {
	//top
	#pragma omp task
	{
	for (i=0; i<(k/2); ++i)
		for (j=(k/2); j<n-(k/2); ++j)
			out[i][j] = convolutionTop( i, j, k, &in, &c );
	}
	//bottom
	#pragma omp task
	{
	for (i=m-(k/2); i<m; ++i)
		for (j=(k/2); j<n-(k/2); ++j)
			out[i][j] = convolutionBottom( i, j, k, &in, &c );
	}
	//left
	#pragma omp task
	{
	for (i=k/2; i<m-(k/2); ++i)
		for (j=0; j<(k/2); ++j)
			out[i][j] = convolutionLeft( i, j, k, &in, &c );
	}
	//right
	#pragma omp task
	{
	for (i=k/2; i<m-(k/2); ++i)
		for (j=n-(k/2); j<n; ++j)
			out[i][j] = convolutionRight( i, j, k, &in, &c );
	}
	//top-left
	#pragma omp task
	{
	for (i=0; i<(k/2); ++i)
		for (j=0; j<(k/2); ++j)
			out[i][j] = convolutionTopLeft( i, j, k, &in, &c );
	}
	//top-right
	#pragma omp task
	{
	for (i=0; i<(k/2); ++i)
		for (j=n-(k/2); j<n; ++j)
			out[i][j] = convolutionTopRight( i, j, k, &in, &c );
	}
	//bottom-left
	#pragma omp task
	{
	for (i=m-(k/2); i<m; ++i)
		for (j=0; j<(k/2); ++j)
			out[i][j] = convolutionBottomLeft( i, j, k, &in, &c );
	}
	//bottom-right
	#pragma omp task
	{
	for (i=m-(k/2); i<m; ++i)
		for (j=n-(k/2); j<n; ++j)
			out[i][j] = convolutionBottomRight( i, j, k, &in, &c );
	}
	//core
    }//end single
	#pragma omp for schedule(static)
	for (i=(k/2); i<m-(k/2); i++)
		for(j=(k/2); j<n-(k/2); j++)
			out[i][j] = convolutionCore( i, j, k, &in, &c );	
   }//end parallel
	
	stopClock();

	/*
	printArray(&in, &m, &n);
	printf("==================\n");
	printArray(&out, &m, &n);
	*/
	printf("m:%llu n:%llu k:%llu\nseconds:%lf\npixels:%lf\npixels/sec:%lf\n", m, n, k,elapsed.count(), pixels, pixels/elapsed.count());
	freeArrays(&in, &out, &c, m, n, k);
	exit(0);
}
//utils
void verifyArgs(int *nargs, char*** args, long long* m, long long* n, long long* k, double* pixels){
	if ( *nargs != 4){
		printf("convolution <m> <n> <k>\n");
		exit(0);
	}
	*m = atoll((*args)[1]);
	*n = atoll((*args)[2]);
	*k = atoll((*args)[3]);
	if(  ((*k)%2==0) || (*k<3)  ){
		printf("k must be at least 3 and odd\n");
		exit(0);
	}else if(  (*k>*m) || (*k>*n)  ){
		printf("m and n must both be larger than k\n");
		exit(0);
	}
	*pixels = (double)(*m) * (double)(*n);
	
}
void initArrays(double*** in, double*** out, double*** c, long long *m, long long *n, long long *k){
	long long i, j;
	*in = new double*[*m];
	*out = new double*[*m];
	for (i=0; i<*m; i++){
		(*in)[i] = new double[*n];
		(*out)[i] = new double[*n];
		for (j=0; j<*n; j++){
			(*in)[i][j] = 1.;
			(*out)[i][j] = 0.;
		}
	}
	*c = new double*[*k];
	for (i=0; i<*k; i++){
		(*c)[i] = new double[*k];
		for(j=0; j<*k; j++)
			(*c)[i][j] = .5 ;
	}
	(*c)[(*k)/2][(*k)/2] = *k ;
}
void printArray(double*** a, long long* m, long long* n){
	long long i, j;
	for(i=0; i<*m; ++i){
		for(j=0; j<*n; ++j){
			printf("%1.3f ", (*a)[i][j]);
		}printf("\n\n");
	}
}
void freeArrays(double*** in, double*** out, double*** c, long long m, long long n, long long k){
	long long i, j;
	for (i=0; i<m; ++i){
		delete[]((*in)[i]);
		delete[]((*out)[i]);
	}
	for (i=0; i<k; ++i){
		delete[]((*c)[i]);
	}
	delete[](*in);
	delete[](*out);
	delete[](*c);
}
//convolutions
double convolutionCore(long long i, long long j, long long k, double*** in, double*** c){
	double sum=0;
	long long k2 = k/2;
	long long x,y;
	//core contains no edge cases
	for(x=0; x<k; x++)
		for(y=0; y<k; y++)
	        	sum += (*c)[x][y] * (*in)[i-k2+x][j-k2+y];
	return sum;
}
double convolutionTop(long long i, long long j, long long k, double*** in, double*** c){
	double sum=0;
	long long k2 = k/2;
	//out of bounds
	long long x,y;
	for(x=0; x<k2; ++x){
		for(y=0; y<k; ++y){
	        	sum += (*c)[x][y] * (*in)[i][j];
		}
	}
	//in bounds
	for(x=k2; x<k; ++x)
		for(y=0; y<k; ++y)
	        	sum += (*c)[x][y] * (*in)[i-k2+x][j-k2+y];
	return sum;
}
double convolutionBottom(long long i, long long j, long long k, double*** in, double*** c){
	double sum=0;
	long long k2 = k/2;
	long long x,y;
	//out of bounds
	for(x=k2; x<k; ++x)
		for(y=0; y<k; ++y)
	        	sum += (*c)[x][y] * (*in)[i][j];
	//in bounds
	for(x=0; x<k2; ++x){
		for(y=0; y<k; ++y){
	        	sum += (*c)[x][y] * (*in)[i-k2+x][j-k2+y];
		}
	}
	return sum;
}
double convolutionLeft( long long i, long long j, long long k, double*** in, double*** c){
	double sum = 0;
	long long k2 = k/2;
	long long x,y;
	//out of bounds
	for(x=0; x<k; ++x){
		for(y=0; y<k2; ++y){
	        	sum += (*c)[x][y] * (*in)[i][j];
		}
	}
	//in bounds
	for(x=0; x<k; ++x)
		for(y=k2; y<k; ++y)
	        	sum += (*c)[x][y] * (*in)[i-k2+x][j-k2+y];
	return sum;
}
double convolutionRight( long long i, long long j, long long k, double*** in, double*** c){
	double sum = 0;
	long long k2 = k/2;
	long long x,y;
	//out of bounds
	for(x=0; x<k; ++x)
		for(y=k2; y<k; ++y)
	        	sum += (*c)[x][y] * (*in)[i][j];
	//in bounds
	for(x=0; x<k; ++x){
		for(y=0; y<k2; ++y){
	        	sum += (*c)[x][y] * (*in)[i-k2+x][j-k2+y];
		}
	}
	return sum;
}
double convolutionTopLeft( long long i, long long j, long long k, double*** in, double*** c){
	double sum = 0;
	long long k2 = k/2;
	long long x,y;
	//out of bounds
	for(x=0; x<k-k2; ++x){
		for(y=0; y<k; ++y){
	        	sum += (*c)[x][y] * (*in)[i][j];
		}
	}
	for(x=k-k2; x<k; ++x)
		for(y=0; y<k2; ++y)
	        	sum += (*c)[x][y] * (*in)[i][j];
	//in bounds
	for(x=k-k2; x<k; ++x)
		for(y=k2; y<k; ++y)
	        	sum += (*c)[x][y] * (*in)[i-k2+x][j-k2+y];
	return sum;
}
double convolutionTopRight( long long i, long long j, long long k, double*** in, double*** c){
	double sum = 0;
	long long k2 = k/2;
	long long x,y;
	//out of bounds
	for(x=0; x<k-k2; ++x){
		for(y=0; y<k; ++y){
	        	sum += (*c)[x][y] * (*in)[i][j];
		}
	}
	for(x=k-k2; x<k; ++x)
		for(y=k2; y<k; ++y)
	        	sum += (*c)[x][y] * (*in)[i][j];
	//in bounds
	for(x=k-k2; x<k; ++x)
		for(y=0; y<k2; ++y)
	        	sum += (*c)[x][y] * (*in)[i-k2+x][j-k2+y];
	return sum;
}
double convolutionBottomLeft( long long i, long long j, long long k, double*** in, double*** c){
	double sum = 0;
	long long k2 = k/2;
	long long x,y;
	//out of bounds
	for(x=k-k2; x<k; ++x){
		for(y=0; y<k; ++y){
	        	sum += (*c)[x][y] * (*in)[i][j];
		}
	}
	for(x=0; x<k-k2; ++x)
		for(y=0; y<k2; ++y)
	        	sum += (*c)[x][y] * (*in)[i][j];
	//in bounds
	for(x=0; x<k-k2; ++x)
		for(y=k2; y<k; ++y)
	        	sum += (*c)[x][y] * (*in)[i-k2+x][j-k2+y];
	return sum;
}
double convolutionBottomRight( long long i, long long j, long long k, double*** in, double*** c){
	double sum = 0;
	long long k2 = k/2;
	long long x,y;
	//out of bounds
	for(x=k-k2; x<k; ++x){
		for(y=0; y<k; ++y){
	        	sum += (*c)[x][y] * (*in)[i][j];
		}
	}
	for(x=0; x<k-k2; ++x)
		for(y=k2; y<k; ++y)
	        	sum += (*c)[x][y] * (*in)[i][j];
	//in bounds
	for(x=0; x<k-k2; ++x)
		for(y=0; y<k2; ++y)
	        	sum += (*c)[x][y] * (*in)[i-k2+x][j-k2+y];
	return sum;
}
