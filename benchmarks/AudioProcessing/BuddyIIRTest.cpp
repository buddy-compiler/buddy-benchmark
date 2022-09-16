
#include <fstream>

#include "Utils/Container.h"
#include <kfr/base.hpp>
#include <kfr/dft.hpp>
#include <kfr/dsp.hpp>
#include <kfr/io.hpp>

using namespace kfr;

extern "C" {
void _mlir_ciface_MLIR_iir(MemRef<float, 1> *inputBuddyConv1D,
                               MemRef<float, 2> *kernelBuddyConv1D,
                               MemRef<float, 1> *outputBuddyConv1D);

void _mlir_ciface_buddy_iir(MemRef<float, 1> *inputBuddyConv1D,
                               MemRef<float, 2> *kernelBuddyConv1D,
                               MemRef<float, 1> *outputBuddyConv1D);
}

bool testAudios(univector<float> aud1, univector<float> aud2){
   
    if(aud1.size() != aud2.size()){
        std:: cout<< "Dimensions are not equal\n";
        return 0;
    }

    for(int i=0; i<aud1.size(); i++){
        if(aud1[i] != aud2[i]){
            std:: cout << "Values are not equal at " << i << "\n";
            std:: cout << "audio_1 value " << aud1[i] << "\n";
            std:: cout << "audio_2 value " << aud2[i] << "\n";
            if (i >= 10){
                break;
            }
        }
    }

    return 1;
}

bool testImplementation(int argc, char *argv[]){

    // univector<float, 6> kernel;
    // biquad_params<float> bq = {biquad_lowpass(0.3, -1.0)};
    univector<float> kernel;
    // zpk<fbase> filt                       = iir_lowpass(bessel<fbase>(12), 1000, 48000);
    zpk<fbase> filt                       = iir_lowpass(butterworth<fbase>(10), 1000, 48000);
    std::vector<biquad_params<fbase>> bqs = to_sos(filt);
    
    for(int i=0; i<bqs.size(); ++i){
        auto bq = bqs[i];
        kernel.push_back(bq.b0);
        kernel.push_back(bq.b1);
        kernel.push_back(bq.b2);
        kernel.push_back(bq.a0);
        kernel.push_back(bq.a1);
        kernel.push_back(bq.a2);

    }
    std::cout<<"Size: "<<bqs.size()<<"\n";
    for(int i=0; i<kernel.size(); i++){
        std::cout<<kernel[i]<<", ";
    }

    std::cout<<"\n";
    univector<float, 200000> aud;

    intptr_t sizeofKernel[2];
    sizeofKernel[0] = bqs.size();
    sizeofKernel[1] = 6;
    intptr_t sizeofAud{aud.size()};
    
    // MemRef copys all data, so data here are actually not accessed.
    MemRef<float, 2> kernelRef(sizeofKernel);
    MemRef<float, 1> audRef(&sizeofAud);
        
    audio_reader_wav<float> reader(open_file_for_reading(
      "../../benchmarks/AudioProcessing/Audios/NASA_Mars.wav"));
    reader.read(aud.data(), aud.size()); 

    std:: cout<<"Input: "<<" \n";
    for(int i=0; i<20; i++){
        // aud[i] = 1;
        std:: cout<<aud[i]<<",";
    }
    std:: cout<<"\n";
    kernelRef = std::move(MemRef<float, 2>(kernel.data(), sizeofKernel));
    audRef = std::move(MemRef<float, 1>(aud.data(), &sizeofAud));
    
    auto am = kernelRef.getData();
    for(int i=0; i<kernel.size(); i++){
        std::cout<<"Debug: "<<am[i]<<"\n";
    }
    MemRef<float, 1> generateResult(&sizeofAud);
    _mlir_ciface_buddy_iir(&audRef, &kernelRef, &generateResult);

    audio_writer_wav<float> writer_buddy(
        open_file_for_writing("./ResultBuddyBiquad.wav"),
        audio_format{1 /* channel */, audio_sample_type::i24,
                    100000 /* sample rate */});
    writer_buddy.write(generateResult.getData(), generateResult.getSize());
    writer_buddy.close();

    univector<float> result_biquad = biquad<32>(bqs, aud);
    audio_writer_wav<float> writer_kfr(open_file_for_writing("./ResultKFRBiquad.wav"),
                                    audio_format{1 /* channel */,
                                                audio_sample_type::i24,
                                                100000 /* sample rate */});
    writer_kfr.write(result_biquad.data(), result_biquad.size());
    writer_kfr.close();

    
    auto x = generateResult.getData();
    std:: ofstream myfile;
    myfile.open("output_results.txt");
    for (int i=0; i<200000; i++){
        myfile << "BBiquad: "<<x[i]<<"\t KfrBiquad: "<<result_biquad[i]<<"\n";
    }
    myfile.close();

    univector<float, 200000> buddy_aud;
    univector<float, 200000> kfr_aud;
   
    audio_reader_wav<float> reader_buddy(open_file_for_reading(
      "./ResultBuddyBiquad.wav"));
    reader_buddy.read(buddy_aud.data(), buddy_aud.size());
    
    audio_reader_wav<float> reader_kfr(open_file_for_reading(
      "./ResultKFRBiquad.wav"));
    reader_kfr.read(kfr_aud.data(), kfr_aud.size());
   
    std:: ofstream file;
    file.open("read_results.txt");
    for (int i=0; i<200000; i++){
        file << "BBiquad: "<<buddy_aud[i]<<"\t KfrBiquad: "<<kfr_aud[i]<<"\n";
    }
    file.close();

    bool rs = testAudios(kfr_aud, buddy_aud);

    return 1;
}

int main(int argc, char* argv[]){
    testImplementation(argc, argv);
    return 0;
}