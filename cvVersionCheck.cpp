#include <iostream>
#include "core/version.hpp"

int main(int argc, char ** argv) {
    std::cout << "OpenCV version:" << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << std::endl;
    return 0;
}
