#include "from_bin_to_sigmf.h"

int Converter::from_bin_to_sigmf(const fd::path& filepath) {
    std::cout << "Processing file: " << filepath << std::endl;
    
    if (filepath.extension() == ".iq" || filepath.extension() == ".bin"){
        std::cerr << "Error: File extension not valid " << filepath << std::endl;
        return -1;
    }
    /*
    ifstream fin(filepath, ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        return -1;
    } 
    
    fs::path outdir = filepath;

    uint32_t endianness;
    double fc;
    double bw;
    double fs;
    double gain;
    double numSamples;
    double sampleStartTime;
    uint32_t bitWidth;
    char fpgaVersion[32];
    char fwVersion[32];
    uint32_t linkSpeed;

    fin.read(reinterpret_cast<char*>(&endianness), sizeof(uint32_t));

    if (endianness == 0x00000000)
        std::cout << "Reading in big endian file" << std::endl;
    else if (endianness == 0x01010101)
        std::cout << "Reading in little endian file" << std::endl;
    else
        cerr << "Warning: Reading in file with unknown endianness" << std::endl;

    fin.read(reinterpret_cast<char*>(&linkSpeed), sizeof(uint32_t));
    fin.read(reinterpret_cast<char*>(&fc), sizeof(double));
    fin.read(reinterpret_cast<char*>(&bw), sizeof(double));
    fin.read(reinterpret_cast<char*>(&fs), sizeof(double));
    fin.read(reinterpret_cast<char*>(&gain), sizeof(double));
    fin.read(reinterpret_cast<char*>(&numSamples), sizeof(double));
    fin.read(reinterpret_cast<char*>(&bitWidth), sizeof(uint32_t));
    fin.read(fpgaVersion, 32);
    fin.read(fwVersion, 32);
    fin.read(reinterpret_cast<char*>(&sampleStartTime), sizeof(double));

    if (0 < bitWidth && bitWidth <= 8) {
        // Read int8_t data
        vector<int8_t> iq(2 * numSamples);
        fin.read(reinterpret_cast<char*>(iq.data()), 2 * numSamples * sizeof(int8_t));
    } else if (8 < bitWidth && bitWidth <= 16) {
        // Read int16_t data
        vector<int16_t> iq(2 * numSamples);
        fin.read(reinterpret_cast<char*>(iq.data()), 2 * numSamples * sizeof(int16_t));
    } else {
        cerr << "Unsupported bit width" << std::endl;
        return -1;
    }

    fin.close();

    double dur = numSamples / fs;
    std::cout << "Saving I/Q" << std::endl;
    string name = filepath.stem().string();
    string filename = name + ".mat";
    ofstream fout((outdir / filename).string(), ios::binary);

    if (!fout.is_open()) {
        cerr << "Error: Could not create output file " << filename << std::endl;
        return -1;
    }

    vector<int16_t> iq(2 * numSamples);

    fout.write(reinterpret_cast<const char*>(iq.data()), iq.size() * sizeof(int16_t));
    fout.write(reinterpret_cast<const char*>(&fs), sizeof(double));
    fout.write(reinterpret_cast<const char*>(&fc), sizeof(double));
    fout.write(reinterpret_cast<const char*>(&dur), sizeof(double));
    fout.write(reinterpret_cast<const char*>(&bw), sizeof(double));
    fout.write(reinterpret_cast<const char*>(&gain), sizeof(double));
    fout.write(reinterpret_cast<const char*>(&bitWidth), sizeof(uint32_t));
    fout.write(fpgaVersion, 32);
    fout.write(fwVersion, 32);
    fout.write(reinterpret_cast<const char*>(&sampleStartTime), sizeof(double));
    fout.write(reinterpret_cast<const char*>(&linkSpeed), sizeof(uint32_t));

    fout.close();

    std::cout << "Done" << std::endl;
    */
    return 0;
}
