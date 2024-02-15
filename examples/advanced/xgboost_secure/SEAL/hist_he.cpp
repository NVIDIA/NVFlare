#include "hist_he.h"

using namespace std;
using namespace seal;


void example_secure_hist()
{
#if (!defined(SEAL_USE_ZSTD) && !defined(SEAL_USE_ZLIB))
    cout << "Neither ZLIB nor Zstandard support is enabled; this example is not available." << endl;
    cout << endl;
    return;
#else
    // Part 1: HE initialization
    // file for storing the serialized parameters
    ofstream params_file_out;
    params_file_out.open("./temp/params.bin", ios::binary);

    // parameter generation
    EncryptionParameters params_out(scheme_type::ckks);
    size_t poly_modulus_degree = 8192;
    std::cout << "Max bit count: " << CoeffModulus::MaxBitCount(poly_modulus_degree) << endl;
    params_out.set_poly_modulus_degree(poly_modulus_degree);
    params_out.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 30, 30, 30, 60}));

    // parameter serialization and save via EncryptionParameters::save function
    auto size = params_out.save(params_file_out, compr_mode_type::zstd);
    params_file_out.close();
    // print the size
    std::cout << "EncryptionParameters total size in bytes: " << size << endl;

    // generate context with parameters
    SEALContext context(params_out);
    print_parameters(context);

    // generate keys
    // addition + mask multiplication, so need relinearization key
    // need public key for mask encryption
    // need Galois keys for rotation in order to perform dot product
    KeyGenerator keygen(context);
    auto secret_key = keygen.secret_key();
    PublicKey public_key;
    keygen.create_public_key(public_key);
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    GaloisKeys gal_keys;
    keygen.create_galois_keys(gal_keys);
    // save secret key
    ofstream key_file_out;
    key_file_out.open("./temp/secret_key.bin", ios::binary);
    secret_key.save(key_file_out, compr_mode_type::zstd);
    key_file_out.close();
    // save public key
    key_file_out.open("./temp/public_key.bin", ios::binary);
    public_key.save(key_file_out, compr_mode_type::zstd);
    key_file_out.close();
    // save relin keys
    key_file_out.open("./temp/relin_keys.bin", ios::binary);
    relin_keys.save(key_file_out, compr_mode_type::zstd);
    key_file_out.close();
    // save Galois keys
    key_file_out.open("./temp/gal_keys.bin", ios::binary);
    gal_keys.save(key_file_out, compr_mode_type::zstd);
    key_file_out.close();


    // Part 2: label-owner side simulation
    // parameter deserialization and load via EncryptionParameters::load function
    EncryptionParameters params_in;
    ifstream params_file_in;
    params_file_in.open("./temp/params.bin", ios::binary);
    params_in.load(params_file_in);
    params_file_in.close();
    // set up context
    SEALContext context_label_owner(params_in);
    // load keys from file
    SecretKey secret_key_in;
    PublicKey public_key_in;
    ifstream key_file_in;
    key_file_in.open("./temp/secret_key.bin", ios::binary);
    secret_key_in.load(context_label_owner, key_file_in);
    key_file_in.close();
    key_file_in.open("./temp/public_key.bin", ios::binary);
    public_key_in.load(context_label_owner, key_file_in);
    key_file_in.close();

    // set up encryptor and decryptor
    Encryptor encryptor(context_label_owner, public_key);
    Decryptor decryptor(context_label_owner, secret_key);

    // CKKS encoder
    CKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;

    // encrypt a vector with length of slot_count
    // value being 0, 1, 2, 3, ..., slot_count - 1
    vector<double> input;
    input.reserve(slot_count);
    double curr_point = 0;
    double step_size = 1.0;
    for (size_t i = 0; i < slot_count; i++)
    {
        input.push_back(curr_point/1000);
        curr_point += step_size;
    }
    cout << "Input vector: " << endl;
    print_vector(input, 3, 7);
    // encode and encrypt
    Plaintext x_plain;
    double scale = pow(2.0, 40);
    cout << "Encode input vectors." << endl;
    encoder.encode(input, scale, x_plain);
    Ciphertext x_encrypted;
    encryptor.encrypt(x_plain, x_encrypted);

    // save encrypted vector
    ofstream data_file_out;
    data_file_out.open("./temp/data.bin", ios::binary);
    auto size_sym_encrypted = x_encrypted.save(data_file_out);
    data_file_out.close();
    std::cout << "Size of encrypted vector: " << size_sym_encrypted << std::endl;


    // Part 3: non-label-owner side simulation
    // parameter deserialization and load via EncryptionParameters::load function
    params_file_in.open("./temp/params.bin", ios::binary);
    params_in.load(params_file_in);
    params_file_in.close();
    // set up context
    SEALContext context_non_label_owner(params_in);
    // load keys from file
    key_file_in.open("./temp/public_key.bin", ios::binary);
    public_key_in.load(context_non_label_owner, key_file_in);
    key_file_in.close();
    RelinKeys relin_keys_in;
    key_file_in.open("./temp/relin_keys.bin", ios::binary);
    relin_keys_in.load(context_non_label_owner, key_file_in);
    key_file_in.close();
    GaloisKeys gal_keys_in;
    key_file_in.open("./temp/gal_keys.bin", ios::binary);
    gal_keys_in.load(context_non_label_owner, key_file_in);
    key_file_in.close();

    // set up evaluator
    Evaluator evaluator(context_non_label_owner);
    // set up encryptor
    Encryptor encryptor_non_label_owner(context_non_label_owner, public_key_in);
    // set up encoder
    CKKSEncoder encoder_non_label_owner(context);

    // simulate an index vector of length slot_count
    // range 0-255
    int max_index = 255;
    vector<double> index_vector;
    index_vector.reserve(slot_count);
    double random_index;
    for (size_t i = 0; i < slot_count; i++)
    {
        // generate index from 0 to 255
        random_index = round(i / 16);
        index_vector.push_back(random_index);
    }
    cout << "Index vector: " << endl;
    print_vector(index_vector, 3, 7);

    // load the encrypted vector
    ifstream data_file_in;
    data_file_in.open("./temp/data.bin", ios::binary);
    Ciphertext x_encrypted_in;
    x_encrypted_in.load(context_non_label_owner, data_file_in);
    data_file_in.close();

    // initialize result vector as all 0 with length max_index
    vector<double> result_vector;
    result_vector.reserve(max_index + 1);
    for (size_t i = 0; i < max_index + 1; i++)
    {
        result_vector.push_back(0);
    }
    // encode and encrypt the result vector
    Plaintext result_plain;
    encoder_non_label_owner.encode(result_vector, scale, result_plain);
    Ciphertext result_encrypted;
    encryptor_non_label_owner.encrypt(result_plain, result_encrypted);
    // the following computation involves two multiplications, hence the result also needs
    // to be rescaled twice
    Plaintext plain_one;
    encoder_non_label_owner.encode(1.0, scale, plain_one);
    // twice rescale
    evaluator.multiply_plain_inplace(result_encrypted, plain_one);
    evaluator.rescale_to_next_inplace(result_encrypted);
    evaluator.multiply_inplace(result_encrypted, result_encrypted);
    evaluator.relinearize_inplace(result_encrypted, relin_keys_in);
    evaluator.rescale_to_next_inplace(result_encrypted);

    // compute for each index
    for (size_t i = 0; i <= max_index; i++)
    {
        std::cout << "Computing for index " << i << std::endl;
        // generate a mask vector with length slot_count
        vector<double> mask;
        mask.reserve(slot_count);
        for (size_t j = 0; j < slot_count; j++)
        {
            if (index_vector[j] == i)
            {
                mask.push_back(1);
            }
            else
            {
                mask.push_back(0);
            }
        }
        // encode and encrypt the mask vector
        Plaintext mask_plain;
        encoder_non_label_owner.encode(mask, scale, mask_plain);
        Ciphertext mask_encrypted;
        encryptor_non_label_owner.encrypt(mask_plain, mask_encrypted);

        // multiply the mask vector with the encrypted vector
        Ciphertext masked_single_bin;
        evaluator.multiply(x_encrypted_in, mask_encrypted, masked_single_bin);
        // relinearize and rescale the result
        evaluator.relinearize_inplace(masked_single_bin, relin_keys_in);
        evaluator.rescale_to_next_inplace(masked_single_bin);
        // accumulate the result vector using rotation, each time offset reduce by half
        int rotation_offset = slot_count;
        while (rotation_offset >= 1)
        {
            rotation_offset /= 2;
            Ciphertext rotated_masked_single_bin;
            evaluator.rotate_vector(masked_single_bin, rotation_offset, gal_keys, rotated_masked_single_bin);
            evaluator.add_inplace(masked_single_bin, rotated_masked_single_bin);
            // after the final rotation with offset 1, stop the process
            if (rotation_offset == 1)
            {
                rotation_offset = 0;
            }
        }

        // add the result to the result vector according to current index
        // generate a mask vector with length slot_count
        vector<double> mask_hist;
        mask_hist.reserve(slot_count);
        for (size_t j = 0; j < slot_count; j++)
        {
            if (j == i)
            {
                mask_hist.push_back(1);
            }
            else
            {
                mask_hist.push_back(0);
            }
        }
        // encode and encrypt the mask vector
        Plaintext mask_hist_plain;
        encoder_non_label_owner.encode(mask_hist, masked_single_bin.scale(), mask_hist_plain);
        Ciphertext mask_hist_encrypted;
        encryptor_non_label_owner.encrypt(mask_hist_plain, mask_hist_encrypted);
        // For applying the mask_hist, we need to make sure its parms_id is the same as the masked_single_bin
        // in order to do that, it needs multiply with a plaintext 1
        Plaintext plain_one;
        encoder_non_label_owner.encode(1.0, scale, plain_one);

        // perform the same relinearize and rescale operations
        evaluator.multiply_plain_inplace(mask_hist_encrypted, plain_one);
        // rescale the result
        evaluator.rescale_to_next_inplace(mask_hist_encrypted);

        // multiply the mask vector with the encrypted vector
        evaluator.multiply_inplace(masked_single_bin, mask_hist_encrypted);
        // relinearize and rescale the result
        evaluator.relinearize_inplace(masked_single_bin, relin_keys_in);
        evaluator.rescale_to_next_inplace(masked_single_bin);

        // lets decrypt the masked_single_bin and print it
        Plaintext masked_single_bin_plain;
        decryptor.decrypt(masked_single_bin, masked_single_bin_plain);
        vector<double> masked_single_bin_decoded;
        encoder_non_label_owner.decode(masked_single_bin_plain, masked_single_bin_decoded);

        // add the masked_single_bin to the result_encrypted
        result_encrypted.scale() = masked_single_bin.scale();
        evaluator.add_inplace(result_encrypted, masked_single_bin);
    }


    // Part 4: label owner decrypts the result
    Plaintext result_encrypted_plain;
    decryptor.decrypt(result_encrypted, result_encrypted_plain);
    vector<double> result_encrypted_decoded;
    encoder.decode(result_encrypted_plain, result_encrypted_decoded);
    std::cout << "result: " << std::endl;
    print_vector(result_encrypted_decoded, 3, 7);

#endif
}

int main()
{
    example_secure_hist();
    return 0;
}
