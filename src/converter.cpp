#include "converter.h"

int Converter::from_bin_to_sigmf(const fd::path& filepath) {
    std::cout << "Processing file: " << filepath << std::endl;
    
    if (filepath.extension() != ".iq" && filepath.extension() != ".bin"){
        std::cerr << "Error: File extension not valid " << filepath << std::endl;
        return -1;
    }

    //py::scoped_interpreter guard{}; // start the interpreter and keep it alive

    py::module my_module = py::module::import("RFDataFactory.SigMF.sigmf_converter");
    py::function convert_bin_to_mat = my_module.attr("convert_bin_to_mat");
    convert_bin_to_mat(std::string(filepath), std::string("/root/libiq-101/iq_samples_mat"));

    std::cout << "Arriva2" << std::endl;

    py::function convert_mat_to_sigmf = my_module.attr("convert_mat_to_sigmf");
    convert_mat_to_sigmf(std::string("/root/libiq-101/iq_samples_sigmf/"), std::string("/root/libiq-101/iq_samples_mat/"));

    std::cout << "Arriva3" << std::endl;

    return 0;
}


class IQFile {
public:
    uint32_t linkSpeed;
    double fc;
    double bw;
    double fs;
    double gain;
    double numSamples;
    uint32_t bitWidth;
    std::string fpgaVersion;
    std::string fwVersion;
    double sampleStartTime;
    std::vector<int8_t> iq;

    void read(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);

        uint32_t endianness;
        file.read(reinterpret_cast<char*>(&endianness), sizeof(endianness));
        if (endianness == 0x00000000) {
            std::cout << "Reading in big endian file\n";
        } else if (endianness == 0x01010101) {
            std::cout << "Reading in little endian file\n";
        } else {
            std::cout << "Reading in file with unknown endianness\n";
        }

        file.read(reinterpret_cast<char*>(&linkSpeed), sizeof(linkSpeed));
        file.read(reinterpret_cast<char*>(&fc), sizeof(fc));
        file.read(reinterpret_cast<char*>(&bw), sizeof(bw));
        file.read(reinterpret_cast<char*>(&fs), sizeof(fs));
        file.read(reinterpret_cast<char*>(&gain), sizeof(gain));
        file.read(reinterpret_cast<char*>(&numSamples), sizeof(numSamples));
        file.read(reinterpret_cast<char*>(&bitWidth), sizeof(bitWidth));

        char fpgaVersionBuffer[32];
        file.read(fpgaVersionBuffer, sizeof(fpgaVersionBuffer));
        fpgaVersion = std::string(fpgaVersionBuffer, sizeof(fpgaVersionBuffer));

        char fwVersionBuffer[32];
        file.read(fwVersionBuffer, sizeof(fwVersionBuffer));
        fwVersion = std::string(fwVersionBuffer, sizeof(fwVersionBuffer));

        file.read(reinterpret_cast<char*>(&sampleStartTime), sizeof(sampleStartTime));

        // Leggi i dati I/Q
        while (!file.eof()) {
            int8_t i, q;
            file.read(reinterpret_cast<char*>(&i), sizeof(i));
            file.read(reinterpret_cast<char*>(&q), sizeof(q));
            if (!file.eof()) {
                iq.push_back(i);
                iq.push_back(q);
            }
        }

        std::cout << "endianess: " << endianness << std::endl;
        std::cout << "Link Speed: " << linkSpeed << std::endl;
        std::cout << "fc: " << fc << std::endl;
        std::cout << "bw: " << bw << std::endl;
        std::cout << "fs: " << fs << std::endl;
        std::cout << "gain: " << gain << std::endl;
        std::cout << "Number of Samples: " << numSamples << std::endl;
        std::cout << "Bit Width: " << bitWidth << std::endl;
        std::cout << "FPGA Version: " << fpgaVersion << std::endl;
        std::cout << "Firmware Version: " << fwVersion << std::endl;
        std::cout << "Sample Start Time: " << sampleStartTime << std::endl;
    }
};

void saveData(const std::string& filename, const IQFile& iqFile) {
    // Apri il file in modalità binaria
    std::ofstream file(filename, std::ios::binary);

    // Scrivi i dati nel file
    file.write(reinterpret_cast<const char*>(iqFile.iq.data()), iqFile.iq.size());
}

int Converter::from_bin_to_mat(const fd::path& filepath) {
    std::cout << "Processing file: " << filepath << std::endl;
    
    if (filepath.extension() != ".iq" && filepath.extension() != ".bin"){
        std::cerr << "Error: File extension not valid " << filepath << std::endl;
        return -1;
    }

    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::cout << std::ctime(&now_c) << " - Clearing everything out\n";

    std::string outdir = "/root/libiq-101/iq_samples_mat/";

    // Crea la directory se non esiste
    std::filesystem::create_directories(outdir);

    // Apri il file in lettura
    IQFile iqFile;
    iqFile.read(filepath.string());

    // Calcola la durata
    double dur = iqFile.iq.size() / iqFile.fs;

    // Salva i dati
    std::cout << std::ctime(&now_c) << " - Saving I/Q\n";
    std::string filename = outdir + filepath.stem().string() + ".mat"; // Modifica con il tuo formato di output

    // Utilizza la tua libreria o funzione personalizzata per salvare i dati
    saveData(filename, iqFile);

    std::cout << std::ctime(&now_c) << " - Done\n";

    return 0;
}
