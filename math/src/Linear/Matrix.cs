using System;
using System.IO;
using System.Linq;

using Nanon.FileFormat;

namespace Nanon.Math.Linear
{	
	public class Matrix : IVector
	{		
		double[] cells;
		int    width;
		int    height;
		
		public Matrix(int w, int h, double[] c) {
			if (c.Length != w * h) 
				throw new Exception();
					
			cells  = c;
			width  = w;
			height = h;
		}
		
		public static Matrix RandomUnform(int w, int h, double range)
		{
			var res  = new Matrix(w, h);
			var rand = new System.Random(w * h);
			
			res.Transform(dummy => 2 * range * rand.NextDouble());
			
			return res;
		}
		
		public Matrix(int w, int h) {
			cells  = new double[w * h];
			width  = w;
			height = h;
		}
		
		public Matrix(int length) {
			cells  = new double[length];
			width  = length;
			height = 1;
		}
		
		public static Matrix[] FromDataFile(DataFile file) {
			var h     = file.Header;
			var size  = h.ItemSize;
			var count = h.ItemCount;
			
			var width  = h.ColumnCount;
			var height = h.RowCount;
			var data   = file.Body.Data;
			
		    var a = new Matrix[count];
			
			for (var i = 0; i < count; ++i) {
				var table = new double[size];
				Array.Copy(data, i * size, table, 0, size);
				a[i] = new Matrix(width, height, table);
			}				                  
			return a;
		}
		
		public static Matrix[] FromFile(string filename) 
		{
			using (var reader = new BinaryReader(File.Open(filename, FileMode.Open)))
			{
				var file = new DataFile(reader);
				return Matrix.FromDataFile(file);
			}
		}
		
		public static Matrix Concat(Matrix[] ms) 
		{
			if (ms.Length == 0)
				return new Matrix(0, 0, new double[]{});
			
			var size    = ms[0].Size;
			var length  = ms.Length;
			var concatedSize = size * length;
			var doubles = new double[concatedSize];
			
			var start = 0;  // mutable
			foreach(var m in ms)
			{
				Array.Copy(m.Cells, 0, doubles, start, size);
				start += size;
			}
			
			return new Matrix(concatedSize, 1, doubles);
		}
		
		public Vector ToVector
		{
			get
			{
		   	  return new Vector(cells);	
			}
		}
		
		public Matrix ZeroCopy()
		{
			return new Matrix(width, height, new double[Size]);
		}
			        
	    public int Size 
		{
		    get 
			{		
				return width * height;
			}
        }
		
		public int Height {
			get {
				return this.height;
			}
		}

		public int Width {
			get {
				return this.width;
			}
		}

		public double[] Cells {
			get {
				return this.cells;
			}
		}
		
		public double this[int i]  
		{
			get
			{
				return cells[i];
			}
		}
		
		// for normalization
		public double Sum
		{
			get
			{
				return cells.Sum();
			}
		}
		
		public double Mean
		{
			get
			{
				return cells.Average();
			}
		}
		
		public double Std
		{
			get
			{
				var mean = Mean;	
				return System.Math.Sqrt(Array
					     .ConvertAll(cells, x => {
							var a = x - mean;
							return a * a;
						 })
					     .Sum() / cells.Length);
			}
		}
		
		public Matrix Map(Func<double, double> f) 
		{
			var mapped = Array.ConvertAll(cells, x => f(x));
			return new Matrix(width, height, mapped);
		}
		
		public void Transform(Func<double, double> f) 
		{
			var size = Size;
			
			for (var i = 0; i < size; ++i)
			  cells[i] = f(cells[i]);
		}
		
		public double Dot(Matrix rhs)
		{
			if (Size != rhs.Size)
				throw new Exception("Cant dot inequal vectors.");
			
			var size = Size;
			double dot  = 0;
			
			for (var i = 0; i < size; ++i)
				dot += cells[i] * rhs.cells[i];
				
			return dot;
		}
		
		public static Matrix operator -(Matrix lhs, Matrix rhs)
		{
			if (lhs.Size != rhs.Size)
				throw new Exception("Cant substract inequal vectors or matrices.");
			
			var size = lhs.Size;
			var sub  = lhs.ZeroCopy();
			
			for (var i = 0; i < size; ++i)
				sub.cells[i] = lhs.cells[i] - rhs.cells[i];
			
			return sub;
		}
		
		public static Matrix operator +(Matrix lhs, Matrix rhs)
		{
			if (lhs.Size != rhs.Size)
				throw new Exception("Cant add inequal vectors or matrices.");
			
			var size = lhs.Size;
			var adds = lhs.ZeroCopy();
			
			for (var i = 0; i < size; ++i)
				adds.cells[i] = lhs.cells[i] + rhs.cells[i];
			
			return adds;
		}
		
		public void AddInplace(Matrix rhs)
		{
			if (Size != rhs.Size)
				throw new Exception("Cant add inequal vectors or matrices.");
			
			var size = Size;
			
			for (var i = 0; i < size; ++i)
				cells[i] += rhs.cells[i];
		}
		
		public void AddInplace(Vector rhs)
		{
			if (Size != rhs.Size)
				throw new Exception("Cant add inequal vectors or matrices.");
			
			var size = Size;
			
			var rcells = rhs.Cells;
			
			for (var i = 0; i < size; ++i)
				cells[i] += rcells[i];
		}
		
		public void SubInplace(Vector rhs)
		{
			if (Size != rhs.Size)
				throw new Exception("Cant add inequal vectors or matrices.");
			
			var size = Size;
			
			var rcells = rhs.Cells;
			
			for (var i = 0; i < size; ++i)
				cells[i] -= rcells[i];
		}
		
		public static Matrix operator -(double lhs, Matrix rhs)
		{
			var size = rhs.Size;
			var sub  = rhs.ZeroCopy();
			
			for (var i = 0; i < size; ++i)
				sub.cells[i] = lhs - rhs.cells[i];
			
			return sub;
		}
		
		public static Matrix operator -(Matrix lhs, double rhs)
		{
			var size = lhs.Size;
			var sub  = lhs.ZeroCopy();
			
			for (var i = 0; i < size; ++i)
				sub.cells[i] = lhs.cells[i] - rhs;
			
			return sub;
		}
		
		public static Matrix operator +(double lhs, Matrix rhs)
		{
			var size = rhs.Size;
			var adds = rhs.ZeroCopy();
			
			for (var i = 0; i < size; ++i)
				adds.cells[i] = lhs + rhs.cells[i];
			
			return adds;
		}
		
		public static Matrix operator *(double lhs, Matrix rhs)
		{
			var size = rhs.Size;
			var adds = rhs.ZeroCopy();
			
			for (var i = 0; i < size; ++i)
				adds.cells[i] = lhs * rhs.cells[i];
			
			return adds;
		}
		
		// Matrix(m x n) -> Vector(n x 1) -> Vector(m x 1)
		public static void MultiplyVertical(Matrix lhs, Vector rhs, Vector res)
		{
			var vector = rhs.Cells;
			var matrix = lhs.Cells;
			var width  = lhs.Width;
			var height = lhs.Height;
			
			for(var rowIndex = 0; rowIndex < height; ++rowIndex)
			{
				var rowOffset = rowIndex * width;
				double acc = 0.0d;
				
				for(var i = 0; i < width; ++i)
				{
					acc += matrix[rowOffset + i] * vector[i];
				}
				
				res.Cells[rowIndex] = acc;
			}
		}
		
		//   a = 1    b = 3 4   res = 1 * 3  1 * 4 
		//       2                    2 * 3  2 * 4
		public static void MultiplyWithTrasposed(Vector lhs, Vector rhs, Matrix res)
		{
			var sizeL = lhs.Size;
			var sizeR = rhs.Size;
			
			if (res.height != sizeL || res.width != sizeR)
				throw new ArgumentException("Incorrect sizes.");
			    
			var lhsCells = lhs.Cells;
			var rhsCells = rhs.Cells;
				
			var resCells = res.cells;
			
			for (var j = 0; j < sizeL; ++j)
			{
				var offset = j * sizeR;
				var left   = lhsCells[j];
				for (var i = 0; i < sizeR; ++i)
			      resCells[i + offset] = left * rhsCells[i];
			}
		}
		
		public static void MultiplyWithTrasposedWithBias(Vector lhs, Vector rhs, Matrix res)
		{
			var sizeL = lhs.Size;
			var sizeR = rhs.Size;
			
			if (res.height != sizeL || res.width != (sizeR + 1))
				throw new ArgumentException("Incorrect sizes.");
			    
			var lhsCells = lhs.Cells;
			var rhsCells = rhs.Cells;
				
			var resCells = res.cells;
			
			for (var j = 0; j < sizeL; ++j)
			{
				var offset = j * sizeR;
				var left   = lhsCells[j];
				resCells[offset] = left;
				for (var i = 0; i < sizeR - 1; ++i)
			      resCells[i + 1 + offset] = left * rhsCells[i];
			}
		}
		
		// Matrix(m + 1 x n) -> Vector(n x 1) -> Vector(m x 1)
		public static void MultiplyVerticalWithBias(Matrix lhs, Vector rhs, Vector res)
		{
			if ((lhs.width - 1) != rhs.Size || lhs.height != res.Size)
				throw new ArgumentException("Incorrect size!");
			
			var vector = rhs.Cells;
			var matrix = lhs.Cells;
			var mwidth = lhs.Width;
			var vwidth = rhs.Size;
			var height = lhs.Height;
			
			for(var rowIndex = 0; rowIndex < height; ++rowIndex)
			{
				var rowOffset = rowIndex * mwidth;
				// bias term
				double acc = matrix[rowOffset];
				
				for(var i = 0; i < vwidth; ++i)
				{
					acc += matrix[rowOffset + i + 1] * vector[i];
				}
				
				res.Cells[rowIndex] = acc;
			}
		}
		
		// Vector(1 x m) -> Matrix(m x n) -> Vector(1 x n)
		public static void MultiplyHorizontal(Vector lhs, Matrix rhs, Vector res)
		{
			if (lhs.Size != rhs.Height || rhs.Width != res.Size)
				throw new ArgumentException("Incorrect size!");
			
			var vector = lhs.Cells;
			var matrix = rhs.Cells;
			var width  = rhs.Width;
			var height = rhs.Height;
			
			for(var colIndex = 0; colIndex < width; ++colIndex)
			{
				double acc = 0.0d;
				
				for(var rowIndex = 0; rowIndex < height; ++rowIndex)
				{
					acc += matrix[colIndex + rowIndex * height] * vector[rowIndex];
				}
				
				res.Cells[colIndex] = acc;
			}
		}
		
		// Vector(1 x m) -> Matrix(m x n) -> Vector(1 x n - 1)
		public static void MultiplyHorizontalWithoutBias(Vector lhs, Matrix rhs, Vector res)
		{
			if (lhs.Size != rhs.Height || (rhs.Width - 1) != res.Size)
				throw new ArgumentException("Incorrect size!");
			
			var vector = lhs.Cells;
			var matrix = rhs.Cells;
			var width  = rhs.Width;
			var height = rhs.Height;
			
			for(var colIndex = 1; colIndex < width; ++colIndex)
			{
				double acc = 0.0d;
				
				for(var rowIndex = 0; rowIndex < height; ++rowIndex)
				{
					acc += matrix[colIndex + rowIndex * height] * vector[rowIndex];
				}
				
				res.Cells[colIndex - 1] = acc;
			}
		}
		
		public static void Convolve(Matrix matrix, Matrix kernel, Matrix res)
		{
			Func<int, bool> even = x => (x & 1) == 0;
			
			if (even(kernel.width) || even(kernel.height))
			    throw new ArgumentException("Kernel width and height should be odd!");
			
			var widthCutOff  = kernel.width / 2;
			var heightCutOff = kernel.height / 2;
			
			var inputWidth   = matrix.width;
			var inputHeight  = matrix.height;
			
			var outputWidth  = res.width;
			var outputHeight = res.height;
			
			var kernelWidth  = kernel.width;
			var kernelHeight = kernel.height;
			
			// check sizes
			/////////  here ///////		
			
			var resCells     = res.cells;
			var inputCells   = matrix.cells;
			var kernelCells  = kernel.cells;
				
			for (var row = heightCutOff; row < inputHeight - heightCutOff; ++row)
			    for (var col = widthCutOff; col < inputWidth - widthCutOff; ++col)
				{
					var acc = 0.0d;
				
			    	for (var j = -heightCutOff; j <= heightCutOff; ++j)
					{
						//inputOffset  = col + (row + j) * inputWidth;
						//kernelOffset = 0; 
					                
			     		//for (var i = -widthCutOff; i <= widthCutOff; ++i)
						///	acc += inputCells[inputOffset++] * kernelCells[kernelOffset++];
					}
							
				    var resIndex = (col - widthCutOff) + (row - heightCutOff) * outputWidth;
					resCells[resIndex] = acc;
				}
		}
	}
}