#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include <time.h>

#define ROOT 0

// procNum = 4
// n = 20

int Size = 4;
double *A;
double *B;
double *C;

void PrintMatrix(double *matr) {
    for (int i = 0; i < Size; ++i) {
        for (int j = 0; j < Size; ++j) {
            printf("%7.4f ", matr[i * Size + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void SeqMupliplication(double *A, double *B, int Size) {
    for (int i = 0; i < Size; i++) {
        for (int j = 0; j < Size; j++) {
            double temp = 0.0;
            for (int k = 0; k < Size; k++) {
                temp += A[i * Size + k] * B[k * Size + j];
            }
            printf("%7.4f ", temp);
        }
        printf("\n");
    }
    printf("\n");
}

void RandInit(double *matr, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matr[i * n + j] = (rand() % 100000) / 10000.0;
        }
    }
}

void InitMatrices(double *A, double *B, double *C) {
    srand((unsigned) time(NULL));
    RandInit(A, Size);
    RandInit(B, Size);
    for (int i = 0; i < Size * Size; i++) {
        C[i] = 0.0;
    }
}

void Transpose(double *B) {
    double temp;
    for (int i = 0; i < Size; i++) {
        for (int j = i + 1; j < Size; j++) {
            temp = B[i * Size + j];
            B[i * Size + j] = B[j * Size + i];
            B[j * Size + i] = temp;
        }
    }
}

void LocalMultiplication(double *a, double *b, double *c, int ProcNum, int ProcRank, int IterNum, int RowsInTape) {
    int BlockNumInC = (ProcRank + IterNum) % ProcNum;
    int Offset = BlockNumInC * RowsInTape;
    for (int i = 0; i < RowsInTape; ++i) {
        for (int j = 0; j < RowsInTape; ++j) {
            c[i * Size + j + Offset] = 0.0;
            for (int k = 0; k < Size; ++k) {
                c[i * Size + j + Offset] += a[i * Size + k] * b[j * Size + k];
            }
        }
    }
}


void main(int argc, char *argv[]) {

    int ProcNum, ProcRank;
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    int MaxProcRank = ProcNum - 1;

    int RowsInTape = Size / ProcNum;
    int ElementsInTape = Size * RowsInTape;
//
    double *bufA = (double *) malloc(Size * RowsInTape * sizeof(double));
    double *bufB = (double *) malloc(Size * RowsInTape * sizeof(double));
    double *bufC = (double *) malloc(Size * RowsInTape * sizeof(double));

    if (ProcRank == ROOT) {
        A = (double *) malloc(Size * Size * sizeof(double));
        B = (double *) malloc(Size * Size * sizeof(double));
        C = (double *) malloc(Size * Size * sizeof(double));

        InitMatrices(A, B, C);
        printf("Matrix A\n");
        PrintMatrix(A);
        printf("Matrix B\n");
        PrintMatrix(B);
        Transpose(B);
    }

    MPI_Scatter(A, ElementsInTape, MPI_DOUBLE, bufA, ElementsInTape, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    MPI_Scatter(B, ElementsInTape, MPI_DOUBLE, bufB, ElementsInTape, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    int CurrIterNum = 0;
    LocalMultiplication(bufA, bufB, bufC, ProcNum, ProcRank, CurrIterNum, RowsInTape);

    int NextProcRank = ProcRank == MaxProcRank ? ROOT : ProcRank + 1;
    int PrevProcRank = ProcRank == ROOT ? MaxProcRank : ProcRank - 1;

    MPI_Status Status;

    int TotalIterCount = ProcNum;
    for (CurrIterNum = 1; CurrIterNum < TotalIterCount; CurrIterNum++) {
        MPI_Sendrecv_replace(bufB, ElementsInTape, MPI_DOUBLE, NextProcRank, 0, PrevProcRank, 0, MPI_COMM_WORLD,
                             &Status);
        LocalMultiplication(bufA, bufB, bufC, ProcNum, ProcRank, CurrIterNum, RowsInTape);
    }

    MPI_Gather(bufC, ElementsInTape, MPI_DOUBLE, C, ElementsInTape, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    free(bufA);
    free(bufB);
    free(bufC);

    if (ProcRank == ROOT) {
        printf("Matrix C\n");
        PrintMatrix(C);

        printf("TEST\n");
        Transpose(B);
        SeqMupliplication(A, B, Size);
        free(A);
        free(B);
        free(C);
    }

    MPI_Finalize();
}
