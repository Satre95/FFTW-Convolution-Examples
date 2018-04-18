#include <cassert>
#include <cstring>
#include <algorithm>
#include "fftw_wrappers.hpp"

using namespace std;

FFTW_R2C_1D_Executor::FFTW_R2C_1D_Executor(int n_real_samples) :
input_size(n_real_samples),
input_buffer(fftw_alloc_real(n_real_samples)),
output_size(n_real_samples/2 + 1),
output_buffer(fftw_alloc_complex(n_real_samples/2 + 1))
{
    plan = fftw_plan_dft_r2c_1d(n_real_samples, input_buffer, output_buffer, FFTW_ESTIMATE);
}

FFTW_R2C_1D_Executor::~FFTW_R2C_1D_Executor()
{
    fftw_destroy_plan(plan);
    fftw_free(input_buffer);
    fftw_free(output_buffer);
}

void FFTW_R2C_1D_Executor::set_input_zeropadded(const double* buffer, int size)
{
    assert(size <= input_size);
    memcpy(input_buffer, buffer, sizeof(double)*size);
    memset(&input_buffer[size], 0, sizeof(double)*(input_size - size));
}

void FFTW_R2C_1D_Executor::set_input_zeropadded(const vector<double>& vec)
{
    set_input_zeropadded(&vec[0], vec.size());
}

void FFTW_R2C_1D_Executor::execute()
{
    fftw_execute(plan);
}

vector<std::complex<double>> FFTW_R2C_1D_Executor::get_output()
{
    std::vector<std::complex<double>> out;
    out.reserve(output_size);
    for (int i = 0; i < output_size; ++i)
        out.emplace_back(output_buffer[i][0], output_buffer[i][1]);
    return out;
}

//-----------------------------------------------------------------

FFTW_C2R_1D_Executor::FFTW_C2R_1D_Executor(int n_real_samples) : 
input_size(n_real_samples/2 + 1),
input_buffer(fftw_alloc_complex(n_real_samples/2 + 1)),
output_size(n_real_samples),
output_buffer(fftw_alloc_real(n_real_samples))
{
    plan = fftw_plan_dft_c2r_1d(n_real_samples, input_buffer, output_buffer, FFTW_ESTIMATE);
}

FFTW_C2R_1D_Executor::~FFTW_C2R_1D_Executor()
{
    fftw_destroy_plan(plan);
    fftw_free(input_buffer);
    fftw_free(output_buffer);
}

void FFTW_C2R_1D_Executor::set_input(const std::complex<double>* buffer, int size)
{
    assert(size == input_size);
    memcpy(input_buffer, buffer, sizeof(fftw_complex)*size);
    memset(&input_buffer[size], 0, sizeof(fftw_complex)*(input_size - size));
}

void FFTW_C2R_1D_Executor::set_input(const vector<std::complex<double>>& vec)
{
    set_input(&vec[0], vec.size());
}

void FFTW_C2R_1D_Executor::execute()
{
    fftw_execute(plan);
}

vector<double> FFTW_C2R_1D_Executor::get_output()
{
    return vector<double>(output_buffer, output_buffer + output_size);
}

//-----------------------------------------------------------------

FFTW_R2C_2D_Executor::FFTW_R2C_2D_Executor(int n0_real_samples, int n1_real_samples) :
input_size_0(n0_real_samples), input_size_1(n1_real_samples),
input_buffer(fftw_alloc_real(n0_real_samples * n1_real_samples)),
output_size_0(n0_real_samples), output_size_1(n1_real_samples),
output_buffer(fftw_alloc_complex(n0_real_samples * n1_real_samples))
{
    plan = fftw_plan_dft_r2c_2d(input_size_0, input_size_1, input_buffer, output_buffer, FFTW_MEASURE);
    std::fill(input_buffer, input_buffer + (input_size_0 * input_size_1), 0.0);
    fftw_complex temp = {0.0, 0.0};
    std::memset(output_buffer, 0, sizeof(fftw_complex) * output_size_0 * output_size_1);
}

FFTW_R2C_2D_Executor::~FFTW_R2C_2D_Executor() {
    fftw_free(input_buffer);
    fftw_free(output_buffer);
    fftw_destroy_plan(plan);
}

void FFTW_R2C_2D_Executor::set_input_zeropadded(const double* buffer, int size0, int size1) {
    assert(size0 <= input_size_0);
    assert(size1 <= input_size_1);
    std::copy(buffer, buffer + (size0 * size1), input_buffer);
}

void FFTW_R2C_2D_Executor::set_input_zeropadded(const std::vector<double *> & vec, int rowSize) {
    assert(rowSize <= input_size_0);
    assert(vec.size() <= input_size_1);
    for (size_t i = 0; i < vec.size(); ++i)
    {
        for (size_t j = 0; j < rowSize; ++j)
        {
            double * row = vec.at(i);
            input_buffer[i * input_size_0 + j] = row[j];
        }
    }
}

void FFTW_R2C_2D_Executor::execute() {
    fftw_execute(plan);
}

vector<std::complex<double>> FFTW_R2C_2D_Executor::get_output(){
    std::vector<std::complex<double>> out;
    out.reserve(output_size_0 * output_size_1);
    for (int i = 0; i < output_size_1; ++i) {
        for (int j = 0; j < output_size_0; ++j) {
            out.emplace_back(output_buffer[i * output_size_0 + j][0], output_buffer[i * output_size_0 + j][1]);
        }
    }
    return out;
}

