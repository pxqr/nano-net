using System;
using System.IO;

using Nanon.FileFormat;

namespace Nanon.Converter
{
	public class Converter
	{    
		static readonly int labelMagicNumber = 0x00000801;
		static readonly int imageMagicNumber = 0x00000803;
		static string incorrectMagicNumberMessage = 
		      "Incorrect type value. (magic number)";
    
		static Header ReadLabelHeader(BinaryReader reader)
		{
			var itemsCount = reader.ReadUInt32();
			if (BitConverter.IsLittleEndian) 
			  itemsCount = ReverseBytes(itemsCount);
      		return new Header((int)itemsCount, 1, 1);
		}
		
		static Header ReadImageHeader(BinaryReader reader) {
			var icnt = reader.ReadUInt32();
			var rcnt = reader.ReadUInt32();
			var ccnt = reader.ReadUInt32();
			
			if (BitConverter.IsLittleEndian) 
			{
				icnt = ReverseBytes(icnt);
				rcnt = ReverseBytes(rcnt);
				ccnt = ReverseBytes(ccnt);
			}
			
			return new Header((int)icnt, (int)rcnt, (int)ccnt);
		}
			
		public static UInt32 ReverseBytes(UInt32 value)
		{
		  return (value & 0x000000FFU) << 24 | (value & 0x0000FF00U) << 8 |
		         (value & 0x00FF0000U) >> 8 | (value & 0xFF000000U) >> 24;
		}
		
	    static Header ReadHeader (BinaryReader reader) {
			var type = reader.ReadUInt32();
			
			if (BitConverter.IsLittleEndian)
				type = ReverseBytes(type);
			
			if (type == labelMagicNumber) 
           		return ReadLabelHeader(reader);
			else 
				if (type == imageMagicNumber) 
          			return ReadImageHeader(reader);
					
		    throw new Exception (incorrectMagicNumberMessage);
		}
    
		public static void Convert (BinaryReader reader, BinaryWriter writer)
		{
			Console.WriteLine("Read header.");
			var header = ReadHeader(reader);
			
			Console.WriteLine("Read body.");
			byte[] bytes = new byte[header.Size];
			reader.Read(bytes, 0, header.Size);
			
			Console.WriteLine("Convert body.");
			var doubles = new double[header.Size];
			for (var i = 0; i < header.Size; ++i)
				doubles[i] = bytes[i];
			
			Console.WriteLine("Form and check data file structure.");
			
			var body   = new Body(doubles);
			var file   = new DataFile(header, body);
			
			Console.WriteLine("Write to output.");
			file.ToBytes(writer);
		}
	}
}

