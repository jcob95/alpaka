/* Copyright 2019 Benjamin Worpitz, Matthias Werner
 *
 * This file exemplifies usage of Alpaka.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED “AS IS” AND ISC DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL ISC BE LIABLE FOR ANY
 * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
 * IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <alpaka/alpaka.hpp>

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <time.h>
#include <cstdlib>
#include <exception>
#include <map>
#include <random>
#include <string>
#include <iomanip>
#include <sstream>

#include "args.hxx"

double delta = 0.5;
double divisor = 2 / (256.0 * 2 * delta * delta);


//#############################################################################
//! A vector addition kernel.
class VectorAddKernel
{
public:
    //-----------------------------------------------------------------------------
    //! The kernel entry point.
    //!
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam TElem The matrix element type.
    //! \param acc The accelerator to be executed on.
    //! \param A The first source vector.
    //! \param B The second source vector.
    //! \param C The destination vector.
    //! \param numElements The number of elements.
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc,
        typename TElem,
        typename TIdx>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        TElem const * const A,
        TElem const * const B,
        TElem * const C,
        TIdx const & numElements) const
    -> void
    {
        static_assert(
            alpaka::dim::Dim<TAcc>::value == 1,
            "The VectorAddKernel expects 1-dimensional indices!");

        TIdx const gridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        TIdx const threadElemExtent(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
        TIdx const threadFirstElemIdx(gridThreadIdx * threadElemExtent);

        if(threadFirstElemIdx < numElements)
        {
            // Calculate the number of elements to compute in this thread.
            // The result is uniform for all but the last thread.
            TIdx const threadLastElemIdx(threadFirstElemIdx+threadElemExtent);
            TIdx const threadLastElemIdxClipped((numElements > threadLastElemIdx) ? threadLastElemIdx : numElements);

            for(TIdx i(threadFirstElemIdx); i<threadLastElemIdxClipped; ++i)
            {
                C[i] = A[i] + B[i];
            }
        }
    }
};

//A and B are the samples we are calculating between
class compute_distance{
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc,
        typename TElem,
        typename TIdx>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        TElem const * const A,
        TElem const * const B,
        TElem * const C,
        TIdx const & numElementsA,
        TIdx const & numElementsB) const
        -> void {
            static_assert(
            alpaka::dim::Dim<TAcc>::value == 5,
            " Expect 5-dimensional indices phase space variables\n Use dummy values of zeros if needed!");

        TIdx const gridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        TIdx const threadElemExtent(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
        TIdx const threadFirstElemIdx(gridThreadIdx * threadElemExtent);

        if(threadFirstElemIdx < numElementsA)
        {
            // Calculate the number of elements to compute in this thread.
            // The result is uniform for all but the last thread.
            TIdx const threadLastElemIdx(threadFirstElemIdx+threadElemExtent);
            TIdx const threadLastElemIdxClipped((numElementsA > threadLastElemIdx) ? threadLastElemIdx : numElementsA);
            //compute the T value contributon
            for(TIdx i(threadFirstElemIdx); i<threadLastElemIdxClipped; i++)
            { 
                double mySum = 0;
                auto event_A = A[i];
                for(TIdx j(i); j < numElementsB;j++){
                    double dist = 0;
                    double diff = (event_A[0] - B[j][0]);
                    dist += diff*diff;
                    diff = (event_A[1] - B[j][1]);
                    dist += diff*diff;
                    diff = (event_A[2] - B[j][2]);
                    dist += diff*diff;
                    diff = (event_A[3] - B[j][3]);
                    dist += diff*diff;
                    diff = (event_A[4] - B[j][4]);
                    dist += diff*diff;
                    mySum += exp(-0.5*dist/delta/delta);
                }
                C[i] += mySum;
            }
        }
    }
};
struct Event {
  double s12;
  double s13;
  double s24;
  double s34;
  double s134;
  double half_mag_squared;
};

const std::vector<Event> read_file(const std::string filename, const size_t n_events) {
  std::fstream file(filename, std::ios_base::in);
  if (!file)
    throw std::runtime_error("Error opening file " + filename);

  std::vector<Event> events;
  events.reserve(std::min((size_t)5000000, n_events));

  std::string line;
  while (std::getline(file, line) && events.size() < n_events) {
    std::istringstream iss(line);
    Event e;
    iss >> e.s12 >> e.s13 >> e.s24 >> e.s34 >> e.s134;
    if (iss.fail())
      throw std::runtime_error("Error reading line " + std::to_string(events.size()+1) + " in " + filename);
    e.half_mag_squared = 0.5 * (e.s12*e.s12 + e.s13*e.s13 + e.s24*e.s24 + e.s34*e.s34 + e.s134*e.s134);
    events.push_back(e);
  }
  return events;
}
auto main(int argc, char *argv[])
-> int
{

args::ArgumentParser parser("CPU based energy test");
args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
args::Flag permutations_only(parser, "permutations only", "Only calculate permutations", {"permutations-only"});
args::Flag output_write(parser, "output write", "write output Tvalues", {"output-write"});
args::ValueFlag<size_t> n_permutations(parser, "n_permutations", "Number of permutations to run", {"n-permutations"});
args::ValueFlag<size_t> max_events_1(parser, "max events 1", "Maximum number of events to use from dataset 1", {"max-events-1"});
args::ValueFlag<size_t> max_events_2(parser, "max events 2", "Maximum number of events to use from dataset 2", {"max-events-2"});
args::ValueFlag<size_t> max_events(parser, "max events", "Max number of events in each dataset", {"max-events"});
args::ValueFlag<size_t> seed(parser, "seed", "seed for permutations", {"seed"});
args::ValueFlag<size_t> max_permutation_events_1(parser, "max permutation events 1", "Max number of events in dataset 1 for permutations",
                                                {"max-permutation-events-1"});
args::ValueFlag<double> delta_value(parser, "delta value", "delta_value", {"delta-value"});
args::Positional<std::string> filename_1(parser, "dataset 1", "Filename for the first dataset");
args::Positional<std::string> filename_2(parser, "dataset 2", "Filename for the second dataset");
args::Positional<std::string> output_fn(parser, "output filename", "Output filename for the permutation test statistics", {"output-fn"});

try {
parser.ParseCLI(argc, argv);
if (!filename_1 || !filename_2)
    throw args::ParseError("Two dataset filenames must be given");
if ((max_events_1 || max_events_2) && max_events)
    throw args::ParseError("--max-events cannot be used with --max-events-1 or --max-events-2");
} catch (args::Help) {
std::cout << parser;
return 0;
} catch (args::ParseError e) {
std::cerr << e.what() << std::endl;
std::cerr << parser;
return 1;
}

// set delta
if (delta_value) {
delta = args::get(delta_value);
}
std::cout << "Distance parameter set to " << delta << std::endl;
divisor = 2 / (256.0 * 2 * delta * delta);

// Parse the maximum number of events to use
size_t data_1_limit = std::numeric_limits<size_t>::max();
size_t data_2_limit = std::numeric_limits<size_t>::max();

if (max_events) {
data_1_limit = args::get(max_events);
data_2_limit = args::get(max_events);
} else {
if (max_events_1)
    data_1_limit = args::get(max_events_1);
if (max_events_2)
    data_2_limit = args::get(max_events_2);
}
// Fallback for the CI with disabled sequential backend
#if defined(ALPAKA_CI) && !defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    return EXIT_SUCCESS;
#else
    // Define the index domain
    using Dim = alpaka::dim::DimInt<1u>;
    using Idx = std::size_t;

    // Define the accelerator
    //
    // It is possible to choose from a set of accelerators
    // that are defined in the alpaka::acc namespace e.g.:
    // - AccGpuCudaRt
    // - AccCpuThreads
    // - AccCpuFibers
    // - AccCpuOmp2Threads
    // - AccCpuOmp2Blocks
    // - AccCpuOmp4
    // - AccCpuTbbBlocks
    // - AccCpuSerial
    using Acc = alpaka::acc::AccGpuCudaRt<Dim, Idx>;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;

    // Defines the synchronization behavior of a queue
    //
    // choose between Blocking and NonBlocking
    using QueueProperty = alpaka::queue::Blocking;
    using QueueAcc = alpaka::queue::Queue<Acc, QueueProperty>;

    // Select a device
    DevAcc const devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));

    // Create a queue on the device
    QueueAcc queue(devAcc);

    // Define the work division
    Idx const numElements(123456);
    Idx const elementsPerThread(3u);
    alpaka::vec::Vec<Dim, Idx> const extent(numElements);

    // Let alpaka calculate good block and grid sizes given our full problem extent
    alpaka::workdiv::WorkDivMembers<Dim, Idx> const workDiv(
        alpaka::workdiv::getValidWorkDiv<Acc>(
            devAcc,
            extent,
            elementsPerThread,
            false,
            alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted));

    // Define the buffer element type
    using Data = std::uint32_t;

    // Get the host device for allocating memory on the host.
    using DevHost = alpaka::dev::DevCpu;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    DevHost const devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));

    // Allocate 3 host memory buffers
    using BufHost = alpaka::mem::buf::Buf<DevHost, Data, Dim, Idx>;
    BufHost bufHostA(alpaka::mem::buf::alloc<Data, Idx>(devHost, extent));
    BufHost bufHostB(alpaka::mem::buf::alloc<Data, Idx>(devHost, extent));
    BufHost bufHostC(alpaka::mem::buf::alloc<Data, Idx>(devHost, extent));

    // Initialize the host input vectors A and B
    Data * const pBufHostA(alpaka::mem::view::getPtrNative(bufHostA));
    Data * const pBufHostB(alpaka::mem::view::getPtrNative(bufHostB));
    Data * const pBufHostC(alpaka::mem::view::getPtrNative(bufHostC));

    // C++11 random generator for uniformly distributed numbers in {1,..,42}
    std::random_device rd{};
    std::default_random_engine eng{ rd() };
    std::uniform_int_distribution<Data> dist(1, 42);

    for (Idx i(0); i < numElements; ++i)
    {
        pBufHostA[i] = dist(eng);
        pBufHostB[i] = dist(eng);
        pBufHostC[i] = 0;
    }

    // Allocate 3 buffers on the accelerator
    using BufAcc = alpaka::mem::buf::Buf<DevAcc, Data, Dim, Idx>;
    BufAcc bufAccA(alpaka::mem::buf::alloc<Data, Idx>(devAcc, extent));
    BufAcc bufAccB(alpaka::mem::buf::alloc<Data, Idx>(devAcc, extent));
    BufAcc bufAccC(alpaka::mem::buf::alloc<Data, Idx>(devAcc, extent));

    // Copy Host -> Acc
    alpaka::mem::view::copy(queue, bufAccA, bufHostA, extent);
    alpaka::mem::view::copy(queue, bufAccB, bufHostB, extent);
    alpaka::mem::view::copy(queue, bufAccC, bufHostC, extent);

    // Instantiate the kernel function object
    VectorAddKernel kernel;

    // Create the kernel execution task.
    auto const taskKernel(alpaka::kernel::createTaskKernel<Acc>(
        workDiv,
        kernel,
        alpaka::mem::view::getPtrNative(bufAccA),
        alpaka::mem::view::getPtrNative(bufAccB),
        alpaka::mem::view::getPtrNative(bufAccC),
        numElements));

    // Enqueue the kernel execution task
    alpaka::queue::enqueue(queue, taskKernel);

    // Copy back the result
    alpaka::mem::view::copy(queue, bufHostC, bufAccC, extent);
    alpaka::wait::wait(queue);

    bool resultCorrect(true);
    for(Idx i(0u);
        i < numElements;
        ++i)
    {
        Data const & val(pBufHostC[i]);
        Data const correctResult(pBufHostA[i] + pBufHostB[i]);
        if(val != correctResult)
        {
            std::cerr << "C[" << i << "] == " << val << " != " << correctResult << std::endl;
            resultCorrect = false;
        }
    }

    if(resultCorrect)
    {
        std::cout << "Execution results correct!" << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "Execution results incorrect!" << std::endl;
        return EXIT_FAILURE;
    }
#endif
}
