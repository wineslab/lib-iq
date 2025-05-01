#include "converter.h"

int Converter::from_csv_or_bin_to_mat(const std::string& input_file_path, const std::string& output_file_path) {
    std::filesystem::path input_filepath(input_file_path);

    std::cout << "Processing file: " << input_filepath << std::endl;

    if (!std::filesystem::exists(input_filepath)) {
        std::cerr << "Error: File does not exist: " << input_filepath << std::endl;
        return EXIT_FAILURE;
    }

    std::string ext = input_filepath.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext != ".iq" && ext != ".bin" && ext != ".csv") {
        std::cerr << "Error: Unsupported file extension. Only .iq, .bin, and .csv are supported." << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<double> real, imag;

    if (ext == ".csv") {
        std::ifstream csv_file(input_file_path);
        if (!csv_file.is_open()) {
            std::cerr << "Error: Cannot open CSV file." << std::endl;
            return EXIT_FAILURE;
        }

        std::string line;
        std::getline(csv_file, line);  // header
        std::istringstream header_stream(line);
        std::vector<std::string> headers;
        std::string col;
        while (std::getline(header_stream, col, ',')) {
            col.erase(0, col.find_first_not_of(" \t\r\n"));
            col.erase(col.find_last_not_of(" \t\r\n") + 1);
            headers.push_back(col);
        }

        int real_index = -1, imag_index = -1;
        for (size_t i = 0; i < headers.size(); ++i) {
            if (headers[i] == "Real") real_index = static_cast<int>(i);
            if (headers[i] == "Imaginary") imag_index = static_cast<int>(i);
        }

        if (real_index == -1 || imag_index == -1) {
            std::cerr << "Error: Columns 'Real' and 'Imaginary' not found." << std::endl;
            return EXIT_FAILURE;
        }

        while (std::getline(csv_file, line)) {
            if (line.empty()) continue;
            std::istringstream line_stream(line);
            std::vector<std::string> tokens;
            std::string token;
            while (std::getline(line_stream, token, ',')) {
                token.erase(0, token.find_first_not_of(" \t\r\n"));
                token.erase(token.find_last_not_of(" \t\r\n") + 1);
                tokens.push_back(token);
            }

            size_t max_index = static_cast<size_t>(std::max(real_index, imag_index));
            if (tokens.size() <= max_index) continue;

            try {
                double r = std::stod(tokens[real_index]);
                double i = std::stod(tokens[imag_index]);
                real.push_back(r);
                imag.push_back(i);
            } catch (...) {
                continue; // skip line if parsing fails
            }
        }
    } else {
        // Binary reader for .bin/.iq files assuming int16_t interleaved
        std::ifstream file(input_file_path, std::ios::binary);
        if (!file) {
            std::cerr << "Error: File cannot be opened." << std::endl;
            return EXIT_FAILURE;
        }

        std::vector<int16_t> buffer;
        int16_t value;
        while (file.read(reinterpret_cast<char*>(&value), sizeof(value))) {
            buffer.push_back(value);
        }

        if (buffer.size() % 2 != 0) {
            std::cerr << "Error: Odd number of samples in binary file." << std::endl;
            return EXIT_FAILURE;
        }

        for (size_t i = 0; i < buffer.size(); i += 2) {
            real.push_back(static_cast<double>(buffer[i]));
            imag.push_back(static_cast<double>(buffer[i + 1]));
        }
    }

    if (real.empty()) {
        std::cerr << "Error: No IQ samples loaded." << std::endl;
        return EXIT_FAILURE;
    }

    std::filesystem::path output_dir = std::filesystem::path(output_file_path).parent_path();
    if (!std::filesystem::exists(output_dir)) {
        std::filesystem::create_directories(output_dir);
    }

    std::shared_ptr<mat_t> matfp(Mat_CreateVer(output_file_path.c_str(), NULL, MAT_FT_MAT73), Mat_Close);
    if (!matfp) {
        std::cerr << "Error: Unable to create .mat file." << std::endl;
        return EXIT_FAILURE;
    }

    size_t dims[2] = {1, real.size()};

    matvar_t* mat_real = Mat_VarCreate("real", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, real.data(), 0);
    matvar_t* mat_imag = Mat_VarCreate("imag", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, imag.data(), 0);

    if (!mat_real || !mat_imag) {
        std::cerr << "Error: Failed to create MATLAB variables." << std::endl;
        return EXIT_FAILURE;
    }

    size_t scalar_dims[2] = {1, 1};

    Mat_VarWrite(matfp.get(), Mat_VarCreate("freq_lower_edge", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, scalar_dims, &this->freq_lower_edge, 0), MAT_COMPRESSION_NONE);
    Mat_VarWrite(matfp.get(), Mat_VarCreate("freq_upper_edge", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, scalar_dims, &this->freq_upper_edge, 0), MAT_COMPRESSION_NONE);
    Mat_VarWrite(matfp.get(), Mat_VarCreate("sample_rate", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, scalar_dims, &this->sample_rate, 0), MAT_COMPRESSION_NONE);
    Mat_VarWrite(matfp.get(), Mat_VarCreate("frequency", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, scalar_dims, &this->frequency, 0), MAT_COMPRESSION_NONE);

    Mat_VarWrite(matfp.get(), Mat_VarCreate("global_index", MAT_C_UINT64, MAT_T_UINT64, 2, scalar_dims, &this->global_index, 0), MAT_COMPRESSION_NONE);
    Mat_VarWrite(matfp.get(), Mat_VarCreate("sample_start", MAT_C_UINT64, MAT_T_UINT64, 2, scalar_dims, &this->sample_start, 0), MAT_COMPRESSION_NONE);

    size_t hw_dims[2] = {1, this->hw.length() + 1};
    size_t version_dims[2] = {1, this->version.length() + 1};

    Mat_VarWrite(matfp.get(), Mat_VarCreate("hw", MAT_C_CHAR, MAT_T_UTF8, 2, hw_dims, this->hw.c_str(), 0), MAT_COMPRESSION_NONE);
    Mat_VarWrite(matfp.get(), Mat_VarCreate("version", MAT_C_CHAR, MAT_T_UTF8, 2, version_dims, this->version.c_str(), 0), MAT_COMPRESSION_NONE);

    std::cout << "Successfully wrote IQ data to: " << output_file_path << std::endl;
    return EXIT_SUCCESS;
}

void create_sigmf_meta(std::shared_ptr<mat_t> matfp, const std::string& output_file_path){
    sigmf::SigMF<sigmf::VariadicDataClass<sigmf::core::GlobalT>, sigmf::VariadicDataClass<sigmf::core::CaptureT>, sigmf::VariadicDataClass<sigmf::core::AnnotationT> > sigmf_file;
    auto new_capture = sigmf::VariadicDataClass<sigmf::core::CaptureT>();
    auto new_annotation = sigmf::VariadicDataClass<sigmf::core::AnnotationT>();
    std::shared_ptr<matvar_t> matvar = nullptr;
    while ((matvar = std::shared_ptr<matvar_t>(Mat_VarReadNext(matfp.get()), Mat_VarFree)) != nullptr) {
        switch (matvar->class_type) {
            case MAT_C_DOUBLE: {
                double *data = static_cast<double*>(matvar->data);
                if(strcmp(matvar->name, "freq_lower_edge")==0){
                    new_annotation.access<sigmf::core::AnnotationT>().freq_lower_edge = *data;
                }
                if(strcmp(matvar->name, "freq_upper_edge")==0){
                    new_annotation.access<sigmf::core::AnnotationT>().freq_upper_edge = *data;
                }
                if(strcmp(matvar->name, "frequency")==0){
                    new_capture.access<sigmf::core::CaptureT>().frequency = *data;
                }
                if(strcmp(matvar->name, "sample_rate")==0){
                    sigmf_file.global.access<sigmf::core::GlobalT>().sample_rate = *data;
                }
                break;
            }
            case MAT_C_UINT64: {
                uint64_t *data = static_cast<uint64_t*>(matvar->data);
                if(strcmp(matvar->name, "global_index")==0){
                    new_capture.access<sigmf::core::CaptureT>().global_index = *data;
                }
                if(strcmp(matvar->name, "sample_start")==0){
                    new_capture.access<sigmf::core::CaptureT>().sample_start = *data;
                }
                break;
            }
            case MAT_C_CHAR: {
                std::string data(static_cast<char*>(matvar->data), matvar->nbytes);
                data.erase(std::remove(data.begin(), data.end(), '\0'), data.end());
                if(strcmp(matvar->name, "hw")==0){
                    sigmf_file.global.access<sigmf::core::GlobalT>().hw = data;
                }
                if(strcmp(matvar->name, "version")==0){
                    sigmf_file.global.access<sigmf::core::GlobalT>().version = data;
                }
                break;
            }
            default:
                std::cerr << "Unhandled data type" << std::endl;
                break;
        }
        matvar = nullptr;
    }

    sigmf_file.captures.emplace_back(new_capture);
    sigmf_file.annotations.emplace_back(new_annotation);

    std::stringstream json_output;
    json_output << json(sigmf_file).dump(4) << std::flush;

    std::ofstream file_out(output_file_path);
    if (file_out.is_open())
    {
        file_out << json_output.str();
        file_out.close();
        std::cout << "File saved in: " << output_file_path << std::endl;
    }
    else 
    {
        std::cerr << "Error: File cannot be opened" << std::endl;
    }
}


int Converter::from_mat_to_sigmf(const std::string& input_file_path, const std::string& output_file_path) {
    std::filesystem::path input_filepath = input_file_path;

    std::cout << "processing file at: " << input_filepath << std::endl;

    if (!std::filesystem::exists(input_filepath)) {
        std::cerr << "Error: File does not exist in: " << input_filepath << std::endl;
        return EXIT_FAILURE;
    }

    if (input_filepath.extension() != ".mat"){
        std::cerr << "Error: File extension not valid, .mat is required " << input_filepath << std::endl;
        return EXIT_FAILURE;
    }

    std::filesystem::path output_dir = std::filesystem::path(output_file_path).parent_path();
    if (!std::filesystem::exists(output_dir)) {
        std::filesystem::create_directories(output_dir);
    }

    std::shared_ptr<mat_t> matfp(Mat_Open(input_file_path.c_str(), MAT_ACC_RDONLY), Mat_Close);
    if (matfp == nullptr) {
        std::cerr << "Error opening MAT file " << input_file_path << std::endl;
        return EXIT_FAILURE;
    }

    create_sigmf_meta(matfp, output_file_path);

    return EXIT_SUCCESS;
}