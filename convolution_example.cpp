#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>
#include <random>
#include "fftw_wrappers.hpp"

#define ROUND(x) int(x + 0.5)

using namespace std;

void print_double_array(const double* arr, int length)
{
    for (int i = 0; i < length; ++i) {
        cout << arr[i] << ", ";
    }
    cout << endl;
}
void print_complex_array(const std::complex<double>* arr, int length)
{
    for (int i = 0; i < length; ++i) {
        cout << arr[i].real() << "+" << arr[i].imag() << "i, ";
    }
    cout << endl;
}

void print_complex_array(const fftw_complex * arr, int length)
{
    for (int i = 0; i < length; ++i) {
        cout << arr[i][0] << "+" << arr[i][1] << "i, ";
    }
    cout << endl;
}

template<class T>
void print_vector(const vector<T>& vec)
{
    for (unsigned int i = 0; i < vec.size(); ++i) {
        cout << vec[i] << ", ";
    }
    cout << endl;
}

/// Prints out a 2D array. Assumes array is stored in ROW major
template <class T>
void print_2d_array(const T * const arr, int n0, int n1) {
    for (int i = 0; i < n1; ++i)
    {
        // cout << "|\t";
        for (int j = 0; j < n0; ++j)
        {
            cout << arr[i * n0 + j] << "\t|\t";
        }
        cout << endl;
    }
}

void print_2d_complex_array(const fftw_complex * const arr, int n0, int n1) {
    for (int i = 0; i < n1; ++i) {
        // cout << "|\t";
        for (int j = 0; j < n0; ++j) {
            cout << arr[i * n0 + j][0] << "+" << arr[i * n0 + j][1] << "i |\t";
        }
        cout << endl;

    }

}


// This function computes the discrete convolution of two arrays:
// result[i] = a[i]*b[0] + a[i-1]*b[1] + ... + a[0]*b[i]
// a and b can be vectors of different lengths, this function is careful to never
// exceed the bounds of the vectors.
vector<double> convolve(const vector<double>& a, const vector<double>& b)
{
    int n_a = a.size();
    int n_b = b.size();
    vector<double> result(n_a + n_b - 1);

    for (int i = 0; i < n_a + n_b - 1; ++i) {
        double sum = 0.0;
        for (int j = 0; j <= i; ++j) {
            sum += ((j < n_a) && (i-j < n_b)) ? a[j]*b[i-j] : 0.0;
        }
        result[i] = sum;
    }
    return result;
}

template <class T>
vector<T> vector_elementwise_multiply(const vector<T> a, const vector<T> b)
{
    assert(a.size() == b.size());
    vector<T> result(a.size());
    for (int i = 0; i < result.size(); ++i) {
        result[i] = a[i]*b[i];
    }
    return result;
}

// Convolution of real vectors using the Fast Fourier Transform and the convolution theorem.
// See https://en.wikipedia.org/wiki/Convolution
vector<double> fftw_convolve(vector<double>& a, vector<double>& b)
{
    // Recall that element-wise
    int padded_length = a.size() + b.size() - 1;
    
    // Compute Fourier transform of vector a
    
    FFTW_R2C_1D_Executor fft_a(padded_length);
    fft_a.set_input_zeropadded(a);

    cout << "a: ";
    print_double_array(fft_a.input_buffer, fft_a.input_size);

    fft_a.execute();

    cout << "FFT(a): ";
    print_complex_array(fft_a.output_buffer, fft_a.output_size);
    cout << endl;

    // Compute Fourier transform of vector b
    
    FFTW_R2C_1D_Executor fft_b(padded_length);
    fft_b.set_input_zeropadded(b);

    cout << "b: ";
    print_double_array(fft_b.input_buffer, fft_b.input_size);

    fft_b.execute();

    cout << "FFT(b): ";
    print_complex_array(fft_b.output_buffer, fft_b.output_size);
    cout << endl;

    // Perform element-wise product of FFT(a) and FFT(b)
    // then compute inverse fourier transform.
    FFTW_C2R_1D_Executor ifft(padded_length);
    assert (ifft.input_size == fft_a.output_size);
    ifft.set_input(vector_elementwise_multiply(fft_a.get_output(), fft_b.get_output()));

    ifft.execute();

    // FFTW returns unnormalized output. To normalize it one must divide each element
    // of the result by the number of elements.
    assert(ifft.output_size == padded_length);
    vector<double> result = ifft.get_output();
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] /= padded_length;
    }

    return result;
}

vector<double> fftw_convolve_2d(double * a_2d, double * b_2d, int n0, int n1) {
    FFTW_R2C_2D_Executor fft_a(n0, n1);
    fft_a.set_input_zeropadded(a_2d, n0, n1);

    cout << "a:" << endl;
    print_2d_array(fft_a.input_buffer, fft_a.input_size_0, fft_a.input_size_1);

    fft_a.execute();

    cout << "FFT(a):" << endl;
    print_2d_complex_array(fft_a.output_buffer, fft_a.output_size_0, fft_a.output_size_1);

    cout << endl;

    FFTW_R2C_2D_Executor fft_b(n0, n1);
    fft_b.set_input_zeropadded(b_2d, n0, n1);

    cout << "b:" << endl;
    print_2d_array(fft_b.input_buffer, fft_b.input_size_0, fft_b.input_size_1);

    fft_b.execute();

    cout << "FFT(b):" << endl;
    print_2d_complex_array(fft_b.output_buffer, fft_b.output_size_0, fft_b.output_size_1);


    FFTW_C2R_2D_Executor i_fft_a(n0, n1);
    i_fft_a.set_input(fft_a.get_output().data(), fft_a.output_size_0, fft_a.output_size_1);

    i_fft_a.execute();

    cout << "Inverset FFT(A):" << endl;
    std::vector<double> i_fft_a_output(i_fft_a.get_output());
    for (size_t i = 0; i < i_fft_a_output.size(); ++i)
        i_fft_a_output.at(i) /= (i_fft_a.output_size_0 * i_fft_a.output_size_1);
    print_2d_array(i_fft_a_output.data(), i_fft_a.output_size_0, i_fft_a.output_size_1);

    for (int i = 0; i < n1; ++i)
    {
        for (int j= 0; j < n0; ++j)
        {
            assert(ROUND(i_fft_a_output.at(i * n0 + j)) == ROUND(a_2d[i * n0 + j]));
        }
    }

    return vector<double>();
}

int main()
{
    cout << "**** 1D Operations ***********************************************" << endl << endl;;
    vector<double> a;
    a.push_back(2);
    a.push_back(1);
    a.push_back(1);
    a.push_back(1);
    a.push_back(1);
    a.push_back(1);
    a.push_back(1);
    a.push_back(1);
    a.push_back(1);
    cout << "First vector (a): ";
    print_vector(a);

    vector<double> b;
    b.push_back(1);
    b.push_back(0);
    b.push_back(7);
    cout << "Second vector (b): ";
    print_vector(b);

    cout << "==== Naive convolution ===========================================\n";

    vector<double> result_naive = convolve(a, b);
    cout << "Naive convolution result:\n";
    print_vector(result_naive);

    cout << "==== FFT convolution =============================================\n";

    vector<double> result_fft = fftw_convolve(a, b);
    cout << "FFT convolution result:\n";
    print_vector(result_fft);

    cout << endl << "**** 2D Operations ***********************************************" << endl << endl;;

    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<> distro(1, 100);

    int n0 = 5, n1 = 5;
    double * a_2D = new double[n0 * n1];
    for (int i = 0; i < n0 * n1; ++i)
    {
        a_2D[i] = distro(generator);
    }
    cout << "First Grid (a, " << n0 << "x" << n1 << ") is: " << endl;
    print_2d_array(a_2D, n0, n1);

    double * b_2D = new double[n0 * n1];
    for (int i = 0; i < n0 * n1; ++i)
    {
        b_2D[i] = distro(generator);
    }
    cout << "Second Grid (b, " << n0 << "x" << n1 << ") is: " << endl;
    print_2d_array(b_2D, n0, n1);

    cout << "==== FFT convolution =============================================" << endl;
    vector<double> result_fft_2d = fftw_convolve_2d(a_2D, b_2D, n0, n1);




    delete[] a_2D;
    delete[] b_2D;
}

