#include <string>
#include <iostream>
#include <source_dir.h>

#ifndef PYTHON_PATH
#define PYTHON_PATH ""
#endif

#if defined(_WIN32)
#include <direct.h>
#define cross_getcwd _getcwd
#else
#include <unistd.h>
#define cross_getcwd getcwd
#endif


void python(std::string& args)
{
    std::string filename;
    {
        char curdir[1024];
        (void)cross_getcwd(curdir, 1024);
        filename = curdir;
    }
#if defined(_WIN32)
    std::string slash = "\\";
#else
    std::string slash = "/";
#endif
    filename = MAIN_PATH + slash + "plots" + slash + "python" + slash + "plot.py";

    (void)std::system((PYTHON_PATH + slash + "python3 \"" + filename + "\"" + args).c_str());

}


int main(int argc, char* argv[])
{
    std::string args = "";
    for (int i = 1; i < argc; i++){
        std::string tmp = argv[i];
        args = args + " " + tmp;
    }
        
    python(args); 
}