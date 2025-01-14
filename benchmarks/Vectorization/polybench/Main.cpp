#include <benchmark/benchmark.h>
#include <cstring>
#include <iostream>
#include <string>

void registerMLIRPolybench2mm();
void registerMLIRPolybench3mm();
void registerMLIRPolybenchAdi();
void registerMLIRPolybenchAtax();
void registerMLIRPolybenchBicg();
void registerMLIRPolybenchCholesky();
void registerMLIRPolybenchCorrelation();
void registerMLIRPolybenchCovariance();
void registerMLIRPolybenchDeriche();
void registerMLIRPolybenchDoitgen();
void registerMLIRPolybenchDurbin();
void registerMLIRPolybenchFdtd2D();
void registerMLIRPolybenchFloydWarshall();
void registerMLIRPolybenchGemm();
void registerMLIRPolybenchGemver();
void registerMLIRPolybenchGesummv();
void registerMLIRPolybenchGramschmidt();
void registerMLIRPolybenchHeat3D();
void registerMLIRPolybenchJacobi1D();
void registerMLIRPolybenchJacobi2D();
void registerMLIRPolybenchLu();
void registerMLIRPolybenchLudcmp();
void registerMLIRPolybenchMvt();
void registerMLIRPolybenchNussinov();
void registerMLIRPolybenchSeidel2D();
void registerMLIRPolybenchSymm();
void registerMLIRPolybenchSyr2k();
void registerMLIRPolybenchSyrk();
void registerMLIRPolybenchTrisolv();
void registerMLIRPolybenchTrmm();

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

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " run-benchmark | run-validation=<size_id>" << std::endl;
    return 1;
  }

  std::string arg = argv[1];

  if (arg == "run-benchmark") {
    registerMLIRPolybench2mm();
    registerMLIRPolybench3mm();
    registerMLIRPolybenchAdi();
    registerMLIRPolybenchAtax();
    registerMLIRPolybenchBicg();
    registerMLIRPolybenchCholesky();
    registerMLIRPolybenchCorrelation();
    registerMLIRPolybenchCovariance();
    registerMLIRPolybenchDeriche();
    registerMLIRPolybenchDoitgen();
    registerMLIRPolybenchDurbin();
    registerMLIRPolybenchFdtd2D();
    registerMLIRPolybenchFloydWarshall();
    registerMLIRPolybenchGemm();
    registerMLIRPolybenchGemver();
    registerMLIRPolybenchGesummv();
    registerMLIRPolybenchGramschmidt();
    registerMLIRPolybenchHeat3D();
    registerMLIRPolybenchJacobi1D();
    registerMLIRPolybenchJacobi2D();
    registerMLIRPolybenchLu();
    registerMLIRPolybenchLudcmp();
    registerMLIRPolybenchMvt();
    registerMLIRPolybenchNussinov();
    registerMLIRPolybenchSeidel2D();
    registerMLIRPolybenchSymm();
    registerMLIRPolybenchSyr2k();
    registerMLIRPolybenchSyrk();
    registerMLIRPolybenchTrisolv();
    registerMLIRPolybenchTrmm();

    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();

  } else if (arg.rfind("run-validation=", 0) == 0) {
    size_t size_id = std::stoul(arg.substr(std::strlen("run-validation=")));
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
    std::cerr << "Invalid argument: " << arg << std::endl;
    return 1;
  }

  return 0;
}
