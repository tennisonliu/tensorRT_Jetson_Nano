#ifndef _TENSORNET_H_
#define _TENSORNET_H_

#include "NvInfer.h"
#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>

// Logger for GIE info/warning/errors
class Logger : public nvinfer1::ILogger
{
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) override
    {
        switch (severity)
        {
            case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
            case Severity::kERROR: std::cerr << "ERROR: "; break;
            case Severity::kWARNING: std::cerr << "WARNING: "; break;
            case Severity::kINFO: std::cerr << "INFO: "; break;
            default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }
};

const int maxBatchSize = 1;  //only batch=1 is available
const int OUTPUT_LEN = 1001;

nvinfer1::ICudaEngine* createTrtFromUFF(char* modelpath);
void prepareBuffer(nvinfer1::ICudaEngine* engine);

void inference(nvinfer1::ICudaEngine* engine,
               int dim_in, float* data_in,
               int dim_out, int* data_out);

#endif // _TENSORNET_H_