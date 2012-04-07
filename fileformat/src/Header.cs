using System;
using System.IO;

namespace Nanon.FileFormat
{
	public class Header
	{
		readonly int itemCount;
		readonly int rowCount;
		readonly int columnCount;
		
		public Header(int itemCnt, int rowCnt, int columnCnt)
		{
			itemCount   = itemCnt;
			rowCount    = rowCnt;
			columnCount = columnCnt;
		}
		
		public Header(BinaryReader reader) {
			itemCount   = reader.ReadInt32();
			rowCount    = reader.ReadInt32();
			columnCount = reader.ReadInt32();
		}
		
		public void ToBytes(BinaryWriter writer) {
			writer.Write(itemCount);
			writer.Write(rowCount);
			writer.Write(columnCount);
		}
		
		public int ItemSize {
		    get {
				return rowCount * columnCount;
			}	
		}
		
		public int Size {
		    get {
				return itemCount * ItemSize;
			}
		}

		public int ColumnCount {
			get {
				return this.columnCount;
			}
		}

		public int ItemCount {
			get {
				return this.itemCount;
			}
		}

		public int RowCount {
			get {
				return this.rowCount;
			}
		}
	}
}

