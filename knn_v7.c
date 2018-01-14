#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>


typedef struct DataSetStruct
{
	int N;
	int D;
	double* data; // Continous array all the data. Perfect for MPI
	double** dataPoints; //Array of pointers pointing on segments of data.
	int* label;
	int* index;
} DataSet;



typedef struct neighbourPointStruct
{
	//DataPoint* dpoint; //A dataPoint and its distance from another datapoint
	int index;
	double dist;
} nnPoint;

double calc_dist(double* A, double* B, int D);
void distance_matrix_SEQ(DataSet *dataSet, nnPoint*** distMatrix);
int cmpfunc (const void * a, const void * b);
void readData(const char* filename1,const char* filename2,DataSet *dataSet, int N, int D);
void knn(DataSet *dataSet, int K, nnPoint*** KNN );
void test_distance_matrix();
void print_dataset(DataSet* dataSet);
void allocate_empty_dataset(DataSet* dataSet, int N, int D);



struct timeval startwtime, endwtime;

int N,D;
DataSet dataSet;

int main(int argc, char** argv){
	int i;
        double  seq_time;

	readData("./data/mnist_trainX.txt","./data/mnist_trainL.txt", &dataSet,100,300);
	

	int K = 3;
	nnPoint** KNN;
        gettimeofday (&startwtime, NULL);
	{
	        knn(&dataSet, K, &KNN);
        }
        gettimeofday (&endwtime, NULL);
        seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
		      + endwtime.tv_sec - startwtime.tv_sec);
	printf("\n Serial Time: %f\n", seq_time);

	printf("\n*** Distance Matrix SERIAL ***\n");
	for (i = 0; i<dataSet.N; i++){
		printf("\n");
		int k;

		for(k=0; k<K; k++)
		{
			printf("[%d %lf]\t", KNN[i][k].index, KNN[i][k].dist);
		}

		printf("\n");
	}
  

	return 0;
}




void knn(DataSet *dataSet, int K, nnPoint*** KNN )
{
	int N = dataSet->N;
	int D = dataSet->D;

	// N-Array of knns' for each element. size(KNN)=[N,K]
	*KNN = (nnPoint**) malloc(N*sizeof(nnPoint*));

	// Calculate the Distance Matrix.
	nnPoint ** distMatrix;
	distance_matrix_SEQ( dataSet, &distMatrix);

	int i;
	for(i=0; i<N; i++)
		(*KNN)[i] = (nnPoint*) malloc(K*sizeof(nnPoint));

		for(i=0; i<N; i++)
		{
			 qsort(distMatrix[i], N, sizeof(nnPoint), cmpfunc);
			 int j;
			 for(j=0; j<K; j++){ // Disregard first element
			 	(*KNN)[i][j].dist = distMatrix[i][j+1].dist;
			 	(*KNN)[i][j].index = distMatrix[i][j+1].index;
			 }

		}

	// Free distance matrix, AFTER we are done with the
	for(i=0; i<N; i++)
		free(distMatrix[i]);
	free(distMatrix);
}

void distance_matrix_SEQ(DataSet *dataSet, nnPoint*** distMatrix)
{
	int N = dataSet->N;
	int D = dataSet->D;

	*distMatrix = (nnPoint**) malloc(N*sizeof(nnPoint*));
	int i;
	for(i=0; i<N; i++)
		(*distMatrix)[i] = (nnPoint*) malloc(N*sizeof(nnPoint));

	for(i=0; i<N; i++)
	{
		(*distMatrix)[i][i].dist = 0;
		(*distMatrix)[i][i].index = dataSet->index[i];

		int j;
		for(j=i+1; j<N; j++)
		{
			double d = calc_dist( dataSet->dataPoints[i], dataSet->dataPoints[j], D);
			(*distMatrix)[i][j].dist = d;
			(*distMatrix)[i][j].index =  dataSet->index[j];

			(*distMatrix)[j][i].dist = d;
			(*distMatrix)[j][i].index =  dataSet->index[i];
		}
	}
}

double calc_dist(double* A, double* B, int D){

	int i;
	double temp, dist = 0;
	for(i=0; i<D; i++){
		temp = A[i]-B[i];
		dist += temp*temp;
	}
	//printf("Dist = %f\n", dist);
	return dist;
}

/**
	Used for qsort. Compares two nnPoint, by distance
*/
int cmpfunc (const void * a, const void * b) {
   nnPoint* p1 = (nnPoint*) a;
   nnPoint* p2 = (nnPoint*) b;

   return p1->dist - p2->dist;
}


void readData(const char* filename1,const char* filename2,DataSet *dataSet, int N, int D)
{
	
        // Open File
	FILE *fp1,*fp2;
	fp1 = fopen( filename1, "r" );
        fp2 = fopen( filename2, "r" );
	if(fp1 == NULL){
		printf("PROBLEM Opening file\n");
		return;
	}

	

	allocate_empty_dataset(dataSet,N,D);
	int i,j;	
	for(i=0; i<(N); i++)
	{
		dataSet->dataPoints[i] = (double*) malloc((D)*sizeof(double));
		
		for(j=0; j<(D); j++)
		{
			double  temp = 3;;
			if (EOF == fscanf(fp1, "%lf\t", &temp))
				printf("ERROR Reading datapoint in %d row, %d column",i,j);
			dataSet->dataPoints[i][j] = temp;
		}

		int temp = -10;
		if (EOF == fscanf(fp2, "%d\n", &temp))
				printf("ERROR Reading label in %d row",i);
		
		dataSet->label[i] = temp;
                dataSet->index[i] = i;
	}

	fclose( fp1 );
        fclose( fp2 );
}

void allocate_empty_dataset(DataSet* dataSet, int N, int D){
	int i,j;

	dataSet->data = (double*) malloc(N*D*sizeof(double));
	dataSet->N = N;
	dataSet->D = D;
	dataSet->dataPoints = (double**) malloc(N*sizeof(double*));

	dataSet->index = (int*) malloc(N*sizeof(int));
	dataSet->label = (int*) malloc(N*sizeof(int));

	for(i=0; i<N; i++)
	{
		dataSet->index[i] = -1;
		dataSet->label[i] = -1;

		// Each dataPoint points to a row of the whole data matrix
		dataSet->dataPoints[i] = &(dataSet->data[D*i]);
		for(j=0; j<D; j++)
			dataSet->dataPoints[i][j] = 11112; // For debbuging purposes.
	}

}


void deallocate_dataset(DataSet* dataSet)
{
	int i;

	// All Datapoint structs is now freed
	free(dataSet->dataPoints);

	// All data are now freed. Data has size [1,N*D], so one 'free()' will suffice.
	free(dataSet->data);
}

/**
	Resizes the dataSet's allocated memory.
	The address and size of the actual struct remains unchanged.
	int start = i-th datapoint that will be the first entry to the new data set.
*/
void reallocate_dataset(DataSet* dataSet, int newN)
{
	int D = dataSet->D;
	dataSet->N = newN;

	dataSet->data = (double*) realloc(dataSet->data, newN*D*sizeof(double));

	dataSet->dataPoints = (double**) realloc( dataSet->dataPoints, newN*sizeof(double*) );
	dataSet->index = (int*) realloc(dataSet->index, newN*sizeof(int));
	dataSet->label = (int*) realloc(dataSet->label, newN*sizeof(int));

	// re init the pointer of each datapoint, in case realloc()
	// moved the content to another memory location
	int i;
	for(i=0; i<newN; i++)
		dataSet->dataPoints[i] = &(dataSet->data[D*i]);
}


/**
	Prints array of NxD elements
*/
void print_dataset(DataSet* dataSet)
{
	int i,j;
	for (i = 0; i<dataSet->N; i++)
	{
		printf("\t%d: [", dataSet->index[i]);
		for (j = 0; j < dataSet->D; j++)
		{
			printf("%lf ",dataSet->dataPoints[i][j]);
		}
		printf("]\t%d\n", dataSet->label[i]);
	}
}


void print_array(double** A, int N, int M)
{
	int i,j;
	for (i = 0; i<N; i++){
		for (j = 0; j<M; j++)
			printf("%f ",(A)[i][j]);
		printf("\n");
	}
}
