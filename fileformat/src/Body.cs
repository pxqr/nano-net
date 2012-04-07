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
			for (var i = 0; i < size; ++i)
				data[i] = reader.ReadDouble();
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

