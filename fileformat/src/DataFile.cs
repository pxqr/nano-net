using System;
using System.IO;

namespace Nanon.FileFormat
{
	public class DataFile
	{
		readonly Header header;
		readonly Body   body;
		
		public DataFile(Header fileHeader, Body fileBody) {
			header = fileHeader;
			body   = fileBody;
		}
		
		public DataFile(BinaryReader reader)
		{
			header = new Header(reader);
			body   = new Body(reader, header);
		}
		
		public void ToBytes(BinaryWriter writer) {
			header.ToBytes(writer);
			body.ToBytes(writer);
		}

		public Body Body {
			get {
				return this.body;
			}
		}

		public Header Header {
			get {
				return this.header;
			}
		}
	}
}

