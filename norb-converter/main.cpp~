#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

struct Header {
    int magic;   // 4 bytes
    int ndim;    // 4 bytes, little endian
    int dim[3];
};

struct MyHeader
{
    int count;
    int height;
    int width;
    
    MyHeader(): count(0), height(0), width(0) {}
};

// The magic number encodes the element type of the matrix:
enum MagicNums {
    SingleF    = 0x1E3D4C51 // for a single precision matrix
  , Packed     = 0x1E3D4C52 // for a packed matrix
  , DoubleF    = 0x1E3D4C53 // for a double precision matrix
  , IntegerF   = 0x1E3D4C54 // for an integer matrix
  , ByteF      = 0x1E3D4C55 // for a byte matrix
  , ShortF     = 0x1E3D4C56 // for a short matrix
};

size_t typeOfData(ifstream &stream)
{    
  int magic = 0;
  stream.read((char*)&magic, sizeof(magic));

  if (magic == ByteF)  
      return 1;
      
  if (magic == IntegerF)
      return 4;
      
  return 0;
}

MyHeader sizeOfFile(ifstream &stream)
{
    int ndim = 0;
    stream.read((char*)&ndim, sizeof(ndim));
    
    int dim[3];
    stream.read((char*)&dim, sizeof(dim));

    MyHeader header;

    if (ndim == 1)
    {
         header.count = dim[0];
         header.width = 1;
         header.height = 1;
    }
    
    if (ndim == 4)
    {
        header.count  = dim[0] * dim[1];
        header.width  = dim[2];
        
        int wdim = 0;
        stream.read((char*)&wdim, sizeof(wdim));
        header.height = wdim;
    }
    
    return header;
}

template <typename T>
void convertData(MyHeader header, ifstream &inFile, ofstream &outFile)
{
    outFile.write((char*)&header, sizeof(header));
    
    size_t size = header.count * header.height * header.width;
    for (size_t i = 0; i < size; ++i)
    {
        T val;
        inFile.read((char*)&val, sizeof(val));
        
        double converted = static_cast<double>(val);
        outFile.write((char*)&converted, sizeof(converted));
    }        
}

int main(int argc, char **argv) 
{
    if (argc != 3)
    {
        cout << "Specify 2 (in, out) file path." << endl;
        return 0;
    }
    
    string inPath   = argv[1];
    string outPath  = argv[2];
    
    cout << "try to read " << inPath << endl;

    ifstream inFile(inPath.c_str(), ios::binary);
    size_t type = typeOfData(inFile);
    MyHeader header = sizeOfFile(inFile);
    cout << "count  " << header.count  << endl
         << "width  " << header.width  << endl
         << "height " << header.height << endl;

    cout << "try to convert " << outPath << endl;
    ofstream outFile(outPath.c_str(), ios::binary);
    
    if (type == 1)
    {
        convertData<char>(header, inFile, outFile);
    }
    else if (type == 4)
    {
        convertData<int> (header, inFile, outFile);
    }   

    cout << "Success!" << endl;
    return 0;
}



