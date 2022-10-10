#include <vector>
#include "colvar/Colvar.h"
#include "colvar/ActionRegister.h"

using namespace std;
class aidefunctions
{
public:
  static int findIndex(const vector<PLMD::AtomNumber> arr, PLMD::AtomNumber item);
  static unsigned get_random_integer(unsigned start, unsigned end);
};