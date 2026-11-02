#include "algoat/succinct/wavelet_matrix.hpp"

namespace algoat::succinct {

template class WaveletMatrix<uint8_t>;
template class WaveletMatrix<uint16_t>;
template class WaveletMatrix<uint32_t>;

} // namespace algoat::succinct
