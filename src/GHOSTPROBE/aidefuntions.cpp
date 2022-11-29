#include "aidefunctions.h"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>

using namespace std;

int aidefunctions::findIndex(const vector<PLMD::AtomNumber> arr, PLMD::AtomNumber item) {
    auto ret = find(arr.begin(),arr.end(),item);
    if (ret != arr.end())
        return ret - arr.begin();
    return -1;
}

unsigned aidefunctions::get_random_integer(unsigned start, unsigned end)
{
    random_device rd;                                   // only used once to initialise (seed) engine
    mt19937 rng(rd());                                  // random-number engine used (Mersenne-Twister in this case)
    uniform_int_distribution<unsigned> uni(start,end); // guaranteed unbiased
    auto random_integer = uni(rng);
    return random_integer;
}