using System;
using System.IO;

namespace Nanon.FileFormat
{	
	public class Body
	{
		readonly double[] data;
		
		public Body(double[] rawData)
		{
			data = rawData;
		}
		
		public Body(BinaryReader reader, Header header) 
		{
	    	var size  = header.Size;
			data = new double[size];
			
			var raw = new byte[size * 8];
			reader.BaseStream.Read(raw, 0, size * 8);
			Buffer.BlockCopy(raw, 0, data, 0, size*8);
		}
		
		public void ToBytes(BinaryWriter writer) 
		{
			foreach(var val in data) 
				writer.Write(val);
		}

		public double[] Data {
			get {
				return this.data;
			}
		}
	}
}

