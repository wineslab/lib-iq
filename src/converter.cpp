#include "converter.h"

int Converter::from_bin_to_mat(const std::string& input_file_path, const std::string& output_file_path) {
    std::filesystem::path input_filepath = input_file_path;

    std::cout << "Processing file: " << input_filepath << std::endl;
    std::cout << "from_bin_to_mat" << std::endl;
    
    if (!std::filesystem::exists(input_filepath)) {
        std::cerr << "Error: File does not exist in: " << input_filepath << std::endl;
        return -1;
    }

    if (input_filepath.extension() != ".iq" && input_filepath.extension() != ".bin"){
        std::cerr << "Error: File extension not valid, .iq or .bin is required " << input_filepath << std::endl;
        return -1;
    }
    
    // Apri il file binario in modalità lettura
    std::ifstream file(input_filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Errore nell'apertura del file." << std::endl;
        return -1;
    }

    // Leggi i dati dal file
    std::vector<int16_t> iq_sample; // Utilizza int16_t per interpretare i dati come interi a 16 bit
    int16_t value;
    while (file.read(reinterpret_cast<char*>(&value), sizeof(value))) {
        iq_sample.push_back(value);
    }

    std::filesystem::path output_dir = std::filesystem::path(output_file_path).parent_path();
    if (!std::filesystem::exists(output_dir)) {
        std::filesystem::create_directories(output_dir);
    }

    mat_t *matfp = Mat_CreateVer(output_file_path.c_str(), NULL, MAT_FT_MAT73);
    if (matfp == NULL) {
        std::cout << "Errore nell'apertura del file .mat" << std::endl;
        return EXIT_FAILURE;
    }

    size_t dims[2] = {1, 1};
    size_t dimshw[2] = {1, hw.length() + 1};
    size_t dimsversion[2] = {1, version.length() + 1};


    Mat_VarWrite(matfp, Mat_VarCreate("freq_lower_edge", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, &freq_lower_edge, 0), MAT_COMPRESSION_NONE);
    Mat_VarWrite(matfp, Mat_VarCreate("freq_upper_edge", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, &freq_upper_edge, 0), MAT_COMPRESSION_NONE);
    Mat_VarWrite(matfp, Mat_VarCreate("frequency", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, &frequency, 0), MAT_COMPRESSION_NONE);
    Mat_VarWrite(matfp, Mat_VarCreate("sample_rate", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, &sample_rate, 0), MAT_COMPRESSION_NONE);
    
    Mat_VarWrite(matfp, Mat_VarCreate("global_index", MAT_C_UINT64, MAT_T_UINT64, 2, dims, &global_index, 0), MAT_COMPRESSION_NONE);
    Mat_VarWrite(matfp, Mat_VarCreate("sample_start", MAT_C_UINT64, MAT_T_UINT64, 2, dims, &sample_start, 0), MAT_COMPRESSION_NONE);


    Mat_VarWrite(matfp, Mat_VarCreate("hw", MAT_C_CHAR, MAT_T_UTF8, 2, dimshw, hw.c_str(), 0), MAT_COMPRESSION_NONE);
    Mat_VarWrite(matfp, Mat_VarCreate("version", MAT_C_CHAR, MAT_T_UTF8, 2, dimsversion, version.c_str(), 0), MAT_COMPRESSION_NONE);

    Mat_Close(matfp);
    return 0;
}

void create_sigmf_meta(mat_t *matfp, const std::string& output_file_path){

    sigmf::SigMF<sigmf::VariadicDataClass<sigmf::core::GlobalT>, sigmf::VariadicDataClass<sigmf::core::CaptureT>, sigmf::VariadicDataClass<sigmf::core::AnnotationT> > sigmf_file;
    auto new_capture = sigmf::VariadicDataClass<sigmf::core::CaptureT>();
    auto new_annotation = sigmf::VariadicDataClass<sigmf::core::AnnotationT>();
    matvar_t* matvar = NULL;
    while ((matvar = Mat_VarReadNext(matfp)) != NULL) {
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
                fprintf(stderr, "Tipo di dati non gestito\n");
                break;
        }
        Mat_VarFree(matvar);
        matvar = NULL;
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
        std::cout << "Impossibile aprire il file.";
    }
}

int Converter::from_mat_to_sigmf(const std::string& input_file_path, const std::string& output_file_path) {
    std::filesystem::path input_filepath = input_file_path;

    std::cout << "processing file at: " << input_filepath << std::endl;

    if (!std::filesystem::exists(input_filepath)) {
        std::cerr << "Error: File does not exist in: " << input_filepath << std::endl;
        return -1;
    }

    if (input_filepath.extension() != ".mat"){
        std::cerr << "Error: File extension not valid, .mat is required " << input_filepath << std::endl;
        return -1;
    }

    // Crea la directory di output se non esiste
    std::filesystem::path output_dir = std::filesystem::path(output_file_path).parent_path();
    if (!std::filesystem::exists(output_dir)) {
        std::filesystem::create_directories(output_dir);
    }

    // Apri il file .mat
    mat_t *matfp = Mat_Open(input_file_path.c_str(), MAT_ACC_RDONLY);
    if (matfp == NULL) {
        std::cout << "Errore nell'apertura del file .mat" << std::endl;
        return EXIT_FAILURE;
    }

    create_sigmf_meta(matfp, output_file_path);
    std::cout << "Dati SigMF salvati in miofile.sigmf-meta" << std::endl;

    Mat_Close(matfp);

    return 0;
}