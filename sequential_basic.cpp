/*
   	@author Samuel L Owens
	@title Convolutions: basic sequential
	sequential 2d pixel convolutions 
	K x K kernel applied to M x N pixel array
	loop over every cell in array
	iterate over every column in the row then move to the next row
	check for validity on every cell within the every convolution
		that is k*k*m*m iterations of the 4 part if statement within the convolution
	pass all variables by value not reference
*/
#include<stdlib.h>
#include<stdio.h>
#include<omp.h>
#include<chrono>

float** in;
float** out;
float** c;
std::chrono::time_point<std::chrono::system_clock> start, stop;
std::chrono::duration<double> elapsed;

float convolution( int, int, int, long long, long long);
int main(int nargs, char** args){
	//verify nargs
	if (nargs != 4){
		printf("convolution <m> <n> <k>\n");
		exit(0);
	}
	long long m = atoll(args[1]);
	long long n = atoll(args[2]);
	long long k = atoll(args[3]);
	double pixels = (double)m*(double)n;
	//verify args
	if (k%2==0 || k<3){
		printf("k must be at least 3 and odd\n");
		exit(0);
	}else if(k>m || k>n){
		printf("m and n must both be larger than k\n");
		exit(0);
	}

	//init arrays ===
	in = new float*[m];
	out = new float*[m];
	for (int i=0; i<m; i++){
		in[i] = new float[n];
		out[i] = new float[n];
		for (int j=0; j<n; j++){
			in[i][j] = 1.;
			out[i][j] = 0.;
		}
	}
	c = new float*[k];
	for (int i=0; i<k; i++){
		c[i] = new float[k];
		for(int j=0; j<k; j++)
			c[i][j] = .5 ;
	}
	c[k/2][k/2] = k ;
	//===

	start = std::chrono::system_clock::now();
	for (int i=0; i<m; ++i){
		for (int j=0; j<n; ++j){
			out[i][j] = convolution(i,j,k,m,n);
		}
	}
	stop = std::chrono::system_clock::now();
	elapsed = stop - start;
	/*
	//PRINT ARRAY
	for(long long i=0; i<m; ++i){
		for(long long j=0; j<n; ++j){
			printf("%1.2f ", in[i][j]);
		}
		printf("\n\n");
	}
	printf("===============\n");
	for(long long i=0; i<m; ++i){
		for(long long j=0; j<n; ++j){
			printf("%1.2f ", out[i][j]);
		}
		printf("\n\n");
	}
	//////
	*/
	delete[](in);
	delete[](out);
	delete[](c);
	printf("m:%llu n:%llu k:%llu\nseconds:%lf\npixels:%lf\npixels/sec:%lf\n", m, n, k,elapsed.count(), pixels, pixels/elapsed.count());
	return 0;
}
float convolution(int i, int j, int k, long long m, long long n){
	float sum=0;
	int k2 = k/2;
	for(int x=0; x<k; x++)
		for(int y=0; y<k; y++)
			sum += ( i-k2+x >= 0 && i-k2+x<m && j-k2+y >= 0 && j-k2+y < n) ?  c[x][y] * in[i-k2+x][j-k2+y] : in[i][j] * c[x][y] ;
	return sum;
}
