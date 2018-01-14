



#ifndef DATA_TYPES_ODY_H
#define DATA_TYPES_ODY_H


#define DATA_TAG 3
#define LABEL_TAG 4	
#define INDEX_TAG 5




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

#endif 
