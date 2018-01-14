#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <mpi/mpi.h>
	
#include "data_types.h"
struct timeval startwtime, endwtime;
double  seq_time;

#define DATA_TAG 3
#define LABEL_TAG 4	
#define INDEX_TAG 5
#define POINTS 100
#define  MASTER	0

//double calc_dist(double* A, double* B, int i , int j , int D);
double calc_dist(double* A, double* B, int D);
//void distance_matrix_SEQ(DataSet *dataSet1,DataSet *dataSet2, nnPoint*** distMatrix,int chunksize,int offset,int offset1);
void distance_matrix_SEQ(DataSet *dataSet, DataSet *recvdataset, nnPoint*** distMatrix);
int cmpfunc (const void * a, const void * b);
//void readData(const char* filename1,DataSet *dataSet);
//void knn(DataSet *dataSet, int K, nnPoint*** KNN ,int offset,int chunksize);
void knn(DataSet *dataSet, DataSet *recvdataset,int K, nnPoint*** KNN,nnPoint ** distMatrix );

void read_data(const char* filename,DataSet* dataSet);

void test_distance_matrix();
void print_dataset(DataSet* dataSet);
void allocate_empty_dataset(DataSet* dataSet, int N, int D);

int N=100,D=30;
nnPoint*** KNN;



int main(int argc, char** argv){

	/* Init MPI */
		int rank, size, P;
		
		MPI_Init(&argc, &argv);
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &size);

		
		DataSet dataSet; // the personal dataset of each task

		allocate_empty_dataset(&dataSet, N,D); //allocating fr our dataset

		int K=3; //how many neighbours  I am looking for

		MPI_Status status;
				

        int   numtasks, taskid, dest,offset,chunksize,rc,tag1,source;
        offset=0;
       
		chunksize = (POINTS / numtasks); // how many elements for each task

		dataSet.N = chunksize;

		dataSet.D = D;

		//  the 

		// the personal Distance Matrix
		nnPoint ** distMatrix;

		for(int i=0; i<N; i++){

			(distMatrix)[i] = (nnPoint *)malloc(chunksize*sizeof(nnPoint));
		}

		
        if (taskid == MASTER){

        		offset = 0;
        		
        		//MASTER reads our data 

        		read_data("myfile.txt", &dataSet);



                //now the MASTER is distributing the data array to the others

                for (dest=1; dest<numtasks; dest++) {

                        MPI_Send(&offset,1, MPI_INT, dest, tag1, MPI_COMM_WORLD);

                        MPI_Send(&(dataSet.dataPoints[offset]), chunksize*D, MPI_DOUBLE,dest,DATA_TAG,MPI_COMM_WORLD);

                        //2MPI_Send(&(dataSet.label[offset]),chunksize, MPI_INT,dest,LABEL_TAG,MPI_COMM_WORLD);

                        MPI_Send(&(dataSet.index[offset]), chunksize, MPI_INT,dest,INDEX_TAG,MPI_COMM_WORLD);

                        printf("Sent %d elements to task %d offset= %d\n",chunksize,dest,offset);
                        offset = offset + chunksize;
                }
        }
        // Master does its part of the work
        //offset = 0; 
        //knn(&dataSet, K, &KNN ,offset, chunksize);
        //printf("%p\n",k);
        source =0;
        //each non-MASTER task receive his portion of array
         if (taskid > MASTER) {
         		printf("i have reached here and i am rank :%d\n",rank );
                source = MASTER;
                MPI_Recv(&offset,1, MPI_INT, source, tag1, MPI_COMM_WORLD, &status);
                MPI_Recv(&(dataSet.dataPoints[offset]), chunksize*D, MPI_DOUBLE, source, DATA_TAG, MPI_COMM_WORLD, &status);
                //MPI_Recv(&(dataSet.label[offset]), chunksize, MPI_INT, source, LABEL_TAG,MPI_COMM_WORLD, &status);
                MPI_Recv(&(dataSet.index[offset]), chunksize, MPI_INT, source, INDEX_TAG,MPI_COMM_WORLD, &status);
         		
         		// in my block of data allocate memory
         		//(*distMatrix [offset] )  = (nnPoint*) malloc(POINTS*sizeof(nnPoint)); 
         }
         
         source =1;
         DataSet recvDataset;
		 allocate_empty_dataset(&recvDataset, N,D);
		 recvDataset = dataSet;

         for (dest=0; dest<numtasks; dest++) {

         		//each task (including the master) send their array to the others 
		        

                MPI_Send(&offset,1, MPI_INT, dest, tag1, MPI_COMM_WORLD);

                MPI_Send(&(dataSet.dataPoints[offset*D]), chunksize*D, MPI_DOUBLE,dest,DATA_TAG,MPI_COMM_WORLD);

                //MPI_Send(&(dataSet.label[offset]),chunksize, MPI_INT,dest,LABEL_TAG,MPI_COMM_WORLD);

                MPI_Send(&(dataSet.index[offset]), chunksize, MPI_INT,dest,INDEX_TAG,MPI_COMM_WORLD);
				

                MPI_Recv(&offset,1, MPI_INT, source, tag1, MPI_COMM_WORLD, &status);

                MPI_Recv(&dataSet.dataPoints[offset*D], chunksize*D, MPI_DOUBLE, source, DATA_TAG, MPI_COMM_WORLD, &status);
                
                //MPI_Recv(&dataSet.label[offset], chunksize, MPI_INT, source, LABEL_TAG,MPI_COMM_WORLD, &status);
                
                MPI_Recv(&dataSet.index[offset], chunksize, MPI_INT, source, INDEX_TAG,MPI_COMM_WORLD, &status);
						        				
        		//distance_matrix_SEQ(&dataSet,&recvDataset,distMatrix,chunksize,offset,offest1);
                knn(&recvDataset,&dataSet,K,KNN,distMatrix);

        		source= source +1 ;

        }
        
  		if (taskid > MASTER) {

  			MPI_Send(&distMatrix, chunksize*N, MPI_DOUBLE, 0,0, MPI_COMM_WORLD);

		}
		if (taskid == MASTER){ //Qsort of the distance matrix + knn +printing
			offset = 0;
			for (int source1=1; source1<numtasks; dest++) {
				
				MPI_Recv(&distMatrix[offset], chunksize*N, MPI_DOUBLE,source1,0,MPI_COMM_WORLD, &status);

				offset = offset +chunksize;
			}
			for(int m=0; m<N; m++){

				 	qsort(distMatrix[m], N, sizeof(nnPoint), cmpfunc);	 
				 
			}	
			for(int i=0; i<N; i++){

				int j;

				for(j=0; j<K; j++){ // Disregard first element

				 	(*KNN)[i][j].dist = distMatrix[i][j+1].dist;

				    (*KNN)[i][j].index = distMatrix[i][j+1].index;

				}
			}
		 	for (int i=0; i<N;i++){
				for(int k=0; k<K; k++){
								
					printf("[%d %lf]\t", (*KNN)[i][k].index, (*KNN)[i][k].dist);
				}

				printf("\n");

			}
		}
         //printf("%p\n", k);

		MPI_Finalize();
		return 	0;
}	






/**

	Calculates distance between datapoint A and B. 

	Each of these are double arrays of length D.

*/

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


int cmpfunc (const void * a, const void * b) {
   nnPoint* p1 = (nnPoint*) a;
   nnPoint* p2 = (nnPoint*) b;

   return p1->dist - p2->dist;
}

void knn(DataSet *dataSet, DataSet *recvdataset,int K, nnPoint*** KNN,nnPoint ** distMatrix )//nnPoint ** distMatrix )


{
	int N = dataSet->N;

	int D = dataSet->D;

	distance_matrix_SEQ(dataSet,recvdataset, &distMatrix);

	//end - Calculate the Distance Matrix

	//int i,m;



} 



void distance_matrix_SEQ(DataSet *dataSet, DataSet *recvdataset, nnPoint*** distMatrix)

{		

	int N = recvdataset->N;

	int D = recvdataset->D;

	

	
	

	int i;



	for(i=0; i<N; i++)

	{

		(*distMatrix)[i][i].dist = 0;

		(*distMatrix)[i][i].index = dataSet->index[i];



		int j;

		for(j=0; j<N; j++)

		{

			double d = calc_dist( dataSet->dataPoints[i], recvdataset->dataPoints[j], D);

			(*distMatrix)[i][j].dist = d;

			(*distMatrix)[i][j].index =  dataSet->index[j];

			

			(*distMatrix)[j][i].dist = d;

			(*distMatrix)[j][i].index =  dataSet->index[i];

		}

	}

}

void read_data(const char* filename,DataSet* dataSet)
	{
		// Open File
		printf ( "%s \n" , filename);
		FILE *fp1,*fp2;
		fp1 = fopen( "/home/allstarabey/Desktop/serial/myFile.txt", "r" );
		fp2 = fopen( "/home/allstarabey/Desktop/serial/labels.txt", "r" );
	
		allocate_empty_dataset(dataSet,N,D);
		if(fp1== 0){
			printf("PROBLEM Opening file\n");
			return;
		}
	
		// Read File Header and parse dataSet size and dimensionality
		
		char buff[15];	
		int i,j;	
		for(i=0; i<N; i++)
		{
			
			for(j=0; j<D; j++)
			{
				
				double  temp;
				
				if (EOF == fscanf(fp1,"%lf,",&temp))
					printf("ERROR Reading datapoint in %d row, %d column",i,j);
				//printf("%lf \n",temp);
				dataSet->dataPoints[i][j] = temp;
			}
	
			int temp;
			if (EOF == fscanf(fp2, "%d\n",&temp))
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
	for(i=0; i<newN; i++){
		dataSet->dataPoints[i] = &(dataSet->data[D*i]);
	}
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


