#include <benchmark/benchmark.h>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>

void generateResultMLIRPolybench2mm(size_t);
void generateResultMLIRPolybench3mm(size_t);
void generateResultMLIRPolybenchAdi(size_t);
void generateResultMLIRPolybenchAtax(size_t);
void generateResultMLIRPolybenchBicg(size_t);
void generateResultMLIRPolybenchCholesky(size_t);
void generateResultMLIRPolybenchCorrelation(size_t);
void generateResultMLIRPolybenchCovariance(size_t);
void generateResultMLIRPolybenchDeriche(size_t);
void generateResultMLIRPolybenchDoitgen(size_t);
void generateResultMLIRPolybenchDurbin(size_t);
void generateResultMLIRPolybenchFdtd2D(size_t);
void generateResultMLIRPolybenchFloydWarshall(size_t);
void generateResultMLIRPolybenchGemm(size_t);
void generateResultMLIRPolybenchGemver(size_t);
void generateResultMLIRPolybenchGesummv(size_t);
void generateResultMLIRPolybenchGramschmidt(size_t);
void generateResultMLIRPolybenchHeat3D(size_t);
void generateResultMLIRPolybenchJacobi1D(size_t);
void generateResultMLIRPolybenchJacobi2D(size_t);
void generateResultMLIRPolybenchLu(size_t);
void generateResultMLIRPolybenchLudcmp(size_t);
void generateResultMLIRPolybenchMvt(size_t);
void generateResultMLIRPolybenchNussinov(size_t);
void generateResultMLIRPolybenchSeidel2D(size_t);
void generateResultMLIRPolybenchSymm(size_t);
void generateResultMLIRPolybenchSyr2k(size_t);
void generateResultMLIRPolybenchSyrk(size_t);
void generateResultMLIRPolybenchTrisolv(size_t);
void generateResultMLIRPolybenchTrmm(size_t);

void registerMLIRPolybench2mm(const std::set<std::string> &disabledSizes);
void registerMLIRPolybench3mm(const std::set<std::string> &disabledSizes);
void registerMLIRPolybenchAdi(const std::set<std::string> &disabledSizes);
void registerMLIRPolybenchAtax(const std::set<std::string> &disabledSizes);
void registerMLIRPolybenchBicg(const std::set<std::string> &disabledSizes);
void registerMLIRPolybenchCholesky(const std::set<std::string> &disabledSizes);
void registerMLIRPolybenchCorrelation(
    const std::set<std::string> &disabledSizes);
void registerMLIRPolybenchCovariance(
    const std::set<std::string> &disabledSizes);
void registerMLIRPolybenchDeriche(const std::set<std::string> &disabledSizes);
void registerMLIRPolybenchDoitgen(const std::set<std::string> &disabledSizes);
void registerMLIRPolybenchDurbin(const std::set<std::string> &disabledSizes);
void registerMLIRPolybenchFdtd2D(const std::set<std::string> &disabledSizes);
void registerMLIRPolybenchFloydWarshall(
    const std::set<std::string> &disabledSizes);
void registerMLIRPolybenchGemm(const std::set<std::string> &disabledSizes);
void registerMLIRPolybenchGemver(const std::set<std::string> &disabledSizes);
void registerMLIRPolybenchGesummv(const std::set<std::string> &disabledSizes);
void registerMLIRPolybenchGramschmidt(
    const std::set<std::string> &disabledSizes);
void registerMLIRPolybenchHeat3D(const std::set<std::string> &disabledSizes);
void registerMLIRPolybenchJacobi1D(const std::set<std::string> &disabledSizes);
void registerMLIRPolybenchJacobi2D(const std::set<std::string> &disabledSizes);
void registerMLIRPolybenchLu(const std::set<std::string> &disabledSizes);
void registerMLIRPolybenchLudcmp(const std::set<std::string> &disabledSizes);
void registerMLIRPolybenchMvt(const std::set<std::string> &disabledSizes);
void registerMLIRPolybenchNussinov(const std::set<std::string> &disabledSizes);
void registerMLIRPolybenchSeidel2D(const std::set<std::string> &disabledSizes);
void registerMLIRPolybenchSymm(const std::set<std::string> &disabledSizes);
void registerMLIRPolybenchSyr2k(const std::set<std::string> &disabledSizes);
void registerMLIRPolybenchSyrk(const std::set<std::string> &disabledSizes);
void registerMLIRPolybenchTrisolv(const std::set<std::string> &disabledSizes);
void registerMLIRPolybenchTrmm(const std::set<std::string> &disabledSizes);

const std::map<std::string, size_t> SIZE_IDS = {
    {"mini", 0}, {"small", 1}, {"medium", 2}, {"large", 3}, {"extralarge", 4}};

std::set<std::string> parseDisabledSizes(const std::string &arg) {
  std::set<std::string> disabledSizes;
  std::stringstream ss(arg);
  std::string size;
  while (std::getline(ss, size, ',')) {
    disabledSizes.insert(size);
  }
  return disabledSizes;
}

int main(int argc, char **argv) {
  std::string arg = (argc > 1) ? argv[1] : "run-benchmark";
  std::set<std::string> disabledSizes;

  for (int i = 2; i < argc; ++i) {
    std::string argStr = argv[i];
    if (argStr.find("--disable-sizes=") == 0) {
      disabledSizes = parseDisabledSizes(argStr.substr(16));
    }
  }

  if (arg == "run-benchmark") {
    std::cout << "Disabled dataset sizes: [ ";
    for (const auto &size : disabledSizes) {
      std::cout << size << " ";
    }
    std::cout << "]" << std::endl;

    registerMLIRPolybench2mm(disabledSizes);
    registerMLIRPolybench3mm(disabledSizes);
    registerMLIRPolybenchAdi(disabledSizes);
    registerMLIRPolybenchAtax(disabledSizes);
    registerMLIRPolybenchBicg(disabledSizes);
    registerMLIRPolybenchCholesky(disabledSizes);
    registerMLIRPolybenchCorrelation(disabledSizes);
    registerMLIRPolybenchCovariance(disabledSizes);
    registerMLIRPolybenchDeriche(disabledSizes);
    registerMLIRPolybenchDoitgen(disabledSizes);
    registerMLIRPolybenchDurbin(disabledSizes);
    registerMLIRPolybenchFdtd2D(disabledSizes);
    registerMLIRPolybenchFloydWarshall(disabledSizes);
    registerMLIRPolybenchGemm(disabledSizes);
    registerMLIRPolybenchGemver(disabledSizes);
    registerMLIRPolybenchGesummv(disabledSizes);
    registerMLIRPolybenchGramschmidt(disabledSizes);
    registerMLIRPolybenchHeat3D(disabledSizes);
    registerMLIRPolybenchJacobi1D(disabledSizes);
    registerMLIRPolybenchJacobi2D(disabledSizes);
    registerMLIRPolybenchLu(disabledSizes);
    registerMLIRPolybenchLudcmp(disabledSizes);
    registerMLIRPolybenchMvt(disabledSizes);
    registerMLIRPolybenchNussinov(disabledSizes);
    registerMLIRPolybenchSeidel2D(disabledSizes);
    registerMLIRPolybenchSymm(disabledSizes);
    registerMLIRPolybenchSyr2k(disabledSizes);
    registerMLIRPolybenchSyrk(disabledSizes);
    registerMLIRPolybenchTrisolv(disabledSizes);
    registerMLIRPolybenchTrmm(disabledSizes);

    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();

  } else if (arg.rfind("run-validation", 0) == 0) {
    size_t size_id = SIZE_IDS.at(argv[2]);

    generateResultMLIRPolybench2mm(size_id);
    generateResultMLIRPolybench3mm(size_id);
    generateResultMLIRPolybenchAdi(size_id);
    generateResultMLIRPolybenchAtax(size_id);
    generateResultMLIRPolybenchBicg(size_id);
    generateResultMLIRPolybenchCholesky(size_id);
    generateResultMLIRPolybenchCorrelation(size_id);
    generateResultMLIRPolybenchCovariance(size_id);
    generateResultMLIRPolybenchDeriche(size_id);
    generateResultMLIRPolybenchDoitgen(size_id);
    generateResultMLIRPolybenchDurbin(size_id);
    generateResultMLIRPolybenchFdtd2D(size_id);
    generateResultMLIRPolybenchFloydWarshall(size_id);
    generateResultMLIRPolybenchGemm(size_id);
    generateResultMLIRPolybenchGemver(size_id);
    generateResultMLIRPolybenchGesummv(size_id);
    generateResultMLIRPolybenchGramschmidt(size_id);
    generateResultMLIRPolybenchHeat3D(size_id);
    generateResultMLIRPolybenchJacobi1D(size_id);
    generateResultMLIRPolybenchJacobi2D(size_id);
    generateResultMLIRPolybenchLu(size_id);
    generateResultMLIRPolybenchLudcmp(size_id);
    generateResultMLIRPolybenchMvt(size_id);
    generateResultMLIRPolybenchNussinov(size_id);
    generateResultMLIRPolybenchSeidel2D(size_id);
    generateResultMLIRPolybenchSymm(size_id);
    generateResultMLIRPolybenchSyr2k(size_id);
    generateResultMLIRPolybenchSyrk(size_id);
    generateResultMLIRPolybenchTrisolv(size_id);
    generateResultMLIRPolybenchTrmm(size_id);

  } else {
    std::cout
        << "Usage: " << argv[0] << std::endl
        << "  run-benchmark [--disable-sizes=<sizes-to-disable>]" << std::endl
        << "  run-validation <size-id>" << std::endl
        << "By default, run-benchmark will be executed with all dataset sizes "
           "enabled."
        << std::endl;
    return 1;
  }

  return 0;
}
