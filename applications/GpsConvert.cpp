/**
 * @author  Steven Lovegrove
 * Copyright (C) 2010  Steven Lovegrove
 *                     Imperial College London
 **/

#include <iostream>
#include <iomanip>
#include <fstream>
#include <GeographicLib/UTMUPS.hpp>

using namespace std;

void Usage(char* name)
{
  cout << name << " filename" << endl;
  cout << "  filename: filename of newline delimited lat long tuples." << endl;
}

int main( int argc, char* argv[] )
{
  // Generate gps data from file with:
  // cat gps.txt | sed -e'/POSITI/ !d' -e's/.*TI,\(.*\),\(.*\),.*$/\1 \2/' > latlon.txt

  if(argc == 2 )
  {
    int numCoords = 0;
    int _z = GeographicLib::UTMUPS::STANDARD;
    bool _np;

    std::ifstream f(argv[1]);
    if( f.is_open() )
    {
      while(!f.eof() && !f.fail())
      {
        double lat;
        double lon;
        f >> lat;
        f >> lon;
        if( !f.fail() )
        {
            double x_meters;
            double y_meters;
            GeographicLib::UTMUPS::Forward(lat, lon, _z,_np,x_meters,y_meters,_z);

            if(numCoords==0) {
                cerr << "Zone:   " << _z << endl;
                cerr << "Northp: " << _np << endl;
            }

            numCoords++;
            cout << setprecision(12) << x_meters << " " << y_meters << endl;
        }
      }
      f.close();
    }
  }else{
    Usage(argv[0]);
  }

  return 0;
}
