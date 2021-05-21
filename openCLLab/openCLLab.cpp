#define _CRT_SECURE_NO_WARNINGS

#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <random>

#include <CL/cl.h>


void printError(const std::string& message, const cl_uint& errorCode) {
    if (errorCode != 0) {
        std::cerr << "Error of " << message << ". Code is " << errorCode << std::endl;
        exit(errorCode);
    }
}

void printMatrix(const cl_uint const* A, cl_uint numOfRows, cl_uint numOfCols) {
    for (size_t i = 0; i < numOfRows; i++) {
        for (size_t j = 0; j < numOfCols; j++) {
            std::cout << A[i* numOfCols +j] << "\t ";
        }
        std::cout << std::endl;
    }
}

cl_uint* matrixTranspose(const cl_uint* Amatrix, size_t numOfRows, size_t numOfCols) {
    cl_uint* Cmatrix = new cl_uint[numOfRows * numOfCols];

    for (size_t y = 0; y < numOfRows; y++) {
        for (size_t x = 0; x < numOfCols; x++) {
            Cmatrix[x * numOfRows + y] = Amatrix[y * numOfCols + x];
        }
    }
    return Cmatrix;
}


void randomMatrixFilling(cl_uint* data, const cl_uint size) {
    cl_uint max = 1000;
    cl_uint min = 200;
    for (size_t i = 0; i < size; ++i)
        data[i] = cl_uint(rand()%(max-min) + min);
}

int main(int argc, char** argv) {
    /*
    if (argc < 5)
        printError("not enough aruments.\n1:M-dimetion, 2:K-dimetion, 3:N-dimetion, 4:Path to kernal, 5(optional):CPU verification", 101);

    cl_uint M = std::atoi(argv[1]); 
    cl_uint N = std::atoi(argv[2]); 
    std::string pathToFile = argv[3];
    bool checkResult = false;
    if (argc == 5) {
        if(std::string(argv[4]) == "1")
            checkResult = true;
    }*/

    cl_uint M = 5;
    cl_uint N = 5;
    std::string pathToFile = "matrixTranspose.cl";
    bool checkResult = true;

    cl_int err = 0;

    if (M < 1 || N < 1)
        printError("invalid dimetions of matrix", 101);

    cl_device_id device_id;
    cl_context context;
    cl_command_queue commands;
    cl_program program;
    cl_kernel kernel1, kernel2, kernel3;

    // A,B and C matrix creation and initialization
    cl_uint size_A = M * N;
    cl_uint size_C = M * N;

    cl_uint* Amatrix = (cl_uint*)malloc(sizeof(cl_uint) * size_A);
    cl_uint* Cmatrix = (cl_uint*)malloc(sizeof(cl_uint) * size_C);
    cl_uint* Dmatrix = (cl_uint*)malloc(sizeof(cl_uint) * size_C);
    cl_uint* Ematrix = (cl_uint*)malloc(sizeof(cl_uint) * size_C);
    randomMatrixFilling(Amatrix, size_A);

    // Get platforms
    cl_uint dev_cnt = 2;
    cl_platform_id *platform_ids = new cl_platform_id[dev_cnt];
    clGetPlatformIDs(dev_cnt, platform_ids, NULL);

    // Connect to GPU
    err = clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    printError("creating device", err);  

    {
        // print some information    
        char* sValue[1024];
        clGetDeviceInfo(device_id, CL_DEVICE_NAME, 1024, (void*)sValue, NULL);
        printf("Device name: \t\t\t%s\n", sValue);

        clGetDeviceInfo(device_id, CL_DEVICE_VERSION, 1024, (void*)sValue, NULL);
        printf("Hardware version: \t\t%s\n", sValue);

        cl_uint uintValue;
        clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(uintValue), &uintValue, NULL);
        printf("Parallel compute units: \t%d\n", uintValue);

        clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(uintValue), &uintValue, NULL);
        printf("Max clock frequency: \t\t%dMHz\n", uintValue);

        clGetDeviceInfo(device_id, CL_DEVICE_ADDRESS_BITS, sizeof(uintValue), &uintValue, NULL);
        printf("Max wavefront dimetion: \t%d\n", uintValue);

        size_t maxWorkGroupSize;
        clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
        printf("Max work group size: \t\t%d\n", maxWorkGroupSize);

        size_t dimetion[3];
        clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(dimetion), &dimetion, NULL);
        printf("Max work group dimetions: \t{%d, %d, %d} \n\n", dimetion[0], dimetion[1], dimetion[2]);
    }

    // Context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    printError("failed creating a context", err);

    // Command queue
    commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    printError("failed creating a comand queue", err);

    // Load kernel, program initialization
    std::ifstream sourceFile(pathToFile);
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    if(sourceCode.size() == 0)
        printError("loading a sourse file", 102);
    char* KernelSource = new char[sourceCode.length() + 1];
    strcpy(KernelSource, sourceCode.c_str());

    // Program initialization
    program = clCreateProgramWithSource(context, 1, (const char**)&KernelSource, NULL, &err);
    printError("creating programm", err);

    // Build program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    printError("creating programm", err);

    // Kernel
    kernel1 = clCreateKernel(program, "matTransposeGlobal", &err);
    printError("creating kernel", err);
    kernel2 = clCreateKernel(program, "matTransposeLocal", &err);
    printError("creating kernel", err);
    kernel3 = clCreateKernel(program, "matTransposeLocalWithoutBankConflict", &err);
    printError("creating kernel", err);

    // Buffers creating
    cl_mem bufferA;
    cl_mem bufferC;
    cl_mem bufferD;
    cl_mem bufferE;

    bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_uint) * size_A, Amatrix, &err);
    bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_uint) * size_C, NULL, &err);
    bufferD = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_uint) * size_C, NULL, &err);
    bufferE = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_uint) * size_C, NULL, &err);

    if (!bufferA || !bufferC || bufferD || bufferE ||err) {
        printError("creating buffers", err);
    }

    // Kernel args
    err = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void*)&bufferA);
    printError("setting kernel argument", err);
    err = clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void*)&bufferC);
    printError("setting kernel argument", err);
    err = clSetKernelArg(kernel1, 2, sizeof(cl_int), (void*)&M);
    printError("setting kernel argument", err);
    err = clSetKernelArg(kernel1, 3, sizeof(cl_int), (void*)&N);
    printError("setting kernel argument", err);

    // Kernel args
    err = clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void*)&bufferA);
    printError("setting kernel argument", err);
    err = clSetKernelArg(kernel2, 1, sizeof(cl_mem), (void*)&bufferD);
    printError("setting kernel argument", err);
    err = clSetKernelArg(kernel2, 2, sizeof(cl_int), (void*)&M);
    printError("setting kernel argument", err);
    err = clSetKernelArg(kernel2, 3, sizeof(cl_int), (void*)&N);
    printError("setting kernel argument", err);

    // Kernel args
    err = clSetKernelArg(kernel3, 0, sizeof(cl_mem), (void*)&bufferA);
    printError("setting kernel argument", err);
    err = clSetKernelArg(kernel3, 1, sizeof(cl_mem), (void*)&bufferE);
    printError("setting kernel argument", err);
    err = clSetKernelArg(kernel3, 2, sizeof(cl_int), (void*)&M);
    printError("setting kernel argument", err);
    err = clSetKernelArg(kernel3, 3, sizeof(cl_int), (void*)&N);
    printError("setting kernel argument", err);

    // local and global sizes
    size_t localWorkSize[2] = {32,32};

    size_t globalWorkSize[2] = { 
        ceil(double(N) / 32.0) * 32, 
        ceil(double(M) / 32.0) * 32
    };

    std::cout << "Globals sizes: " << globalWorkSize[0] << " " << globalWorkSize[1] << std::endl;

    //Perform first matrix transposing
    cl_event waitEvent;
    err = clEnqueueNDRangeKernel(commands, kernel1, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &waitEvent);
    clWaitForEvents(1, &waitEvent);
    clFinish(commands);
    cl_ulong time_start;
    cl_ulong time_end;
    clGetEventProfilingInfo(waitEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(waitEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double nanoSeconds = time_end - time_start;
    std::cout << "GPU calculation time: \t" << nanoSeconds / 1000000.0 << "ms \n";
    printError("executing kernel!", err);
    // Where is my results
    err = clEnqueueReadBuffer(commands, bufferC, CL_TRUE, 0, sizeof(cl_uint) * size_C, Cmatrix, 0, NULL, NULL);
    printError("read output array!", err);

    //Perform second matrix transposing
    cl_event waitEvent1;
    err = clEnqueueNDRangeKernel(commands, kernel2, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &waitEvent1);
    clWaitForEvents(1, &waitEvent1);
    clFinish(commands);
    cl_ulong time_start1;
    cl_ulong time_end1;
    clGetEventProfilingInfo(waitEvent1, CL_PROFILING_COMMAND_START, sizeof(time_start1), &time_start1, NULL);
    clGetEventProfilingInfo(waitEvent1, CL_PROFILING_COMMAND_END, sizeof(time_end1), &time_end1, NULL);
    nanoSeconds = time_end1 - time_start1;
    std::cout << "GPU calculation time: \t" << nanoSeconds / 1000000.0 << "ms \n";
    printError("executing kernel!", err);
    // Where is my results
    err = clEnqueueReadBuffer(commands, bufferD, CL_TRUE, 0, sizeof(cl_uint) * size_C, Dmatrix, 0, NULL, NULL);
    printError("read output array!", err);

    //Perform third matrix transposing
    cl_event waitEvent2;
    err = clEnqueueNDRangeKernel(commands, kernel3, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &waitEvent2);
    clWaitForEvents(1, &waitEvent2);
    clFinish(commands);
    cl_ulong time_start2;
    cl_ulong time_end2;
    clGetEventProfilingInfo(waitEvent2, CL_PROFILING_COMMAND_START, sizeof(time_start2), &time_start2, NULL);
    clGetEventProfilingInfo(waitEvent2, CL_PROFILING_COMMAND_END, sizeof(time_end2), &time_end2, NULL);
    nanoSeconds = time_end2 - time_start2;
    std::cout << "GPU calculation time: \t" << nanoSeconds / 1000000.0 << "ms \n";
    printError("executing kernel!", err);
    // Where is my results
    err = clEnqueueReadBuffer(commands, bufferE, CL_TRUE, 0, sizeof(cl_uint) * size_C, Ematrix, 0, NULL, NULL);
    printError("read output array!", err);

    if (checkResult) {

        auto start = std::chrono::high_resolution_clock::now();
        cl_uint* CmatrixOnCPU = matrixTranspose(Amatrix, M, N);

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
        std::cout << "CPU calculation time: \t" << duration.count() / 1000000.0 << "ms \n";

        bool isEqualC = true, isEqualD = true, isEqualE = true;

        for (size_t  i = 0; i < size_C; ++i) {
            if (CmatrixOnCPU[i] != Cmatrix[i]) {
                isEqualC = false;
            }
            if (CmatrixOnCPU[i] != Dmatrix[i]) {
                isEqualD = false;
            }
            if (CmatrixOnCPU[i] != Ematrix[i]) {
                isEqualE = false;
            }
        }

        if (isEqualC) {
            std::cout << "CPU and GPU calculations with using global memory IS equal!\n"; 
        }
        else {
            std::cout << "CPU and GPU calculations with using global memory IS NOT equal!\n";
        }

        if (isEqualD) {
            std::cout << "CPU and GPU calculations with using local memory IS equal!\n";
        }
        else {
            std::cout << "CPU and GPU calculations with using local memory IS NOT equal!\n";
        }

        if (isEqualE) {
            std::cout << "CPU and GPU calculations with using local memory without bank conflict IS equal!\n";
        }
        else {
            std::cout << "CPU and GPU calculations with using local memory without bank conflict IS NOT equal!\n";
        }

        free(CmatrixOnCPU);
    }
    /*
    printMatrix(Amatrix, M, N);
    std::cout << "--------------------------------------------------\n";
    printMatrix(Cmatrix, N, M);
    std::cout << "--------------------------------------------------\n";
    printMatrix(Dmatrix, N, M);
    std::cout << "--------------------------------------------------\n";
    printMatrix(Ematrix, N, M);*/

    // It's cleaning time
    free(Amatrix);
    free(Cmatrix);
    free(Dmatrix);
    free(Ematrix);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferC);
    clReleaseMemObject(bufferD);
    clReleaseMemObject(bufferE);
    clReleaseProgram(program);
    clReleaseKernel(kernel1);
    clReleaseKernel(kernel2);
    clReleaseKernel(kernel3);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return 0;
}
