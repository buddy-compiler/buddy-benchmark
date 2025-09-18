//===- Main.cpp -----------------------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This is the main file of the audio-plot executable.
//
//===----------------------------------------------------------------------===//

#include <source_dir.h>
#include <string>

#if defined(_WIN32)
#include <direct.h>
#define cross_getcwd _getcwd
#else
#include <unistd.h>
#define cross_getcwd getcwd
#endif

void python(std::string &args) {
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
  filename = MAIN_PATH + slash + "utils" + slash + "plots" + slash + "python" + slash + "plot.py";

#ifndef PYTHON_PATH
  (void)std::system(("python3 \"" + filename + "\"" + args).c_str());
#else
  (void)std::system(
      (PYTHON_PATH + slash + "python3 \"" + filename + "\"" + args).c_str());
#endif
}

int main(int argc, char *argv[]) {
  std::string args = "";
  for (int i = 1; i < argc; i++) {
    std::string tmp = argv[i];
    args = args + " " + tmp;
  }

  python(args);
}
