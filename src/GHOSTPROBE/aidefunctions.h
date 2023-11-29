#include <vector>
#include "colvar/Colvar.h"
#include "core/ActionRegister.h"

using namespace std;
class aidefunctions
{
public:
  static int findIndex(const vector<PLMD::AtomNumber> arr, PLMD::AtomNumber item);
  static unsigned get_random_integer(unsigned start, unsigned end);
  static vector<vector<double>> read_xyz(string filename, int frame_id);
};