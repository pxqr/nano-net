using System;
using System.IO;
using System.Linq;

using Nanon.FileFormat;

namespace Nanon.Math.Linear
{	
	public class Matrix : IVector, IMatrix<Matrix>
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
		
		static int randomSeed = 0;
		
		public static Matrix RandomUnform(int w, int h, double range)
		{
			var res  = new Matrix(w, h);
			var rand = new System.Random(randomSeed += (w * h));
			
			res.Transform(dummy => 2 * range * rand.NextDouble() - range);
			
			return res;
		}
		
		public static Matrix RandomNormal(int w, int h, double range)
		{
			var res  = new Matrix(w, h);
			var rand = new System.Random(randomSeed += (w * h));
			
			var halfWidth  = w / 2;
			var halfHeight = h / 2;
			
			var coeff  = System.Math.Sqrt(2 * System.Math.PI) * range;
			var coeff2 = - 1 / (2 * range * range);
			
			for (var j = 0; j < h; ++j)
				for (var i = 0; i < w; ++i)
				{	
					var deltaX = (i - halfWidth);
					var deltaY = (j - halfHeight);
					var deltaX2 = deltaX * deltaX;
					var deltaY2 = deltaY * deltaY;
					var val = coeff * System.Math.Exp(coeff2 * deltaX2 * deltaY2);
					res.cells[i + j * w] = val * (rand.NextDouble() - 0.5d);
				}
			
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
			using (var stream = new BufferedStream(File.Open(filename, FileMode.Open), 20000))
			{
				using (var reader = new BinaryReader(stream))
				{
					var file = new DataFile(reader);
					return Matrix.FromDataFile(file);
				}
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
		
		public static void Transform(Func<double, double> f, Matrix src, Matrix res) 
		{
			var size = src.Size;
			
			for (var i = 0; i < size; ++i)
			  res.cells[i] = f(src.cells[i]);
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
		
		public static void MultiplyWithTrasposedWithBiasAdd(Vector lhs, Vector rhs, Matrix res)
		{
			var sizeL = lhs.Size;
			var sizeR = rhs.Size;
			
			if (res.height != sizeL || res.width != sizeR + 1)
				throw new ArgumentException("Incorrect sizes.");
			    
			var lhsCells = lhs.Cells;
			var rhsCells = rhs.Cells;
				
			var resCells = res.cells;
			
			for (var j = 0; j < sizeL; ++j)
			{
				var offset = j * (sizeR + 1);
				var left   = lhsCells[j];
				
				resCells[offset++] += left;
				
				for (var i = 0; i < sizeR; ++i)
			      resCells[i + offset] += left * rhsCells[i];
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
				double acc = matrix[rowOffset++];
				
				for(var i = 0; i < vwidth; ++i)
				{
					acc += matrix[rowOffset + i] * vector[i];
				}
				
				res.Cells[rowIndex] = acc;
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
			
			for(var i = 1; i < width; ++i)
			{
				double acc = 0.0d;
					
				for(var j = 0; j < height; ++j)
				{
					acc += matrix[i + j * width] * vector[j];
				}
				
				res.Cells[i - 1] = acc;
			}
		}
		
		#region IMatrix[Matrix] implementation
		
		public void SetToZero()
		{
			var size  = Size;
			
			for (var i = 0; i < size; ++i)
				cells[i] = 0.0d;
		}
		
		public Vector Unwind 
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
		
		public Matrix Copy()
		{
			return new Matrix(width, height, ToVector.Copy().Cells);
		}
		
		public double Sum
		{
			get
			{
				return cells.Sum();
			}
		}
		
		unsafe public void Convolve(Matrix kernel, Matrix res)
		{
			Func<int, bool> even = x => (x & 1) == 0;
			
			if (even(kernel.width) || even(kernel.height))
			    throw new ArgumentException("Kernel width and height should be odd!");
			
			var inputWidth   = width;
			var inputHeight  = height;
			
			var outputWidth  = res.width;
			var outputHeight = res.height;
			
			var kernelWidth  = kernel.width;
			var kernelHeight = kernel.height;
			
			if (inputWidth  != (outputWidth  + kernelWidth  - 1) || 
			    inputHeight != (outputHeight + kernelHeight - 1))
			    throw new ArgumentException("Incorrect sizes!");
			
			var resCells     = res.cells;
			var inputCells   = cells;
			var kernelCells  = kernel.cells;
				
			for (var row = 0; row <= inputHeight - kernelHeight; ++row)
			    for (var col = 0; col <= inputWidth - kernelWidth; ++col)
				{
					var acc = 0.0d;
				
					var inputOffset  = col + row * inputWidth;
					var kernelOffset = 0;
			    	
					for (var j = 0; j < kernelHeight; ++j)
					{
						for (var i = 0; i < kernelWidth; ++i)
							acc += inputCells[inputOffset + i] * kernelCells[kernelOffset + i];
					                
					                
					    inputOffset += inputWidth;
						kernelOffset += kernelWidth;
					}
							
				    var resIndex = col + row * outputWidth;
					resCells[resIndex] = acc;
				}
		}
		
		unsafe public void Deconvolve1(Matrix kernel, Matrix res)
		{
			res.SetToZero();
			
			var inputWidth   = width;
			var inputHeight  = height;
			
			var outputWidth  = res.width;
			var outputHeight = res.height;
			
			var kernelWidth  = kernel.width;
			var kernelHeight = kernel.height;
			
			for (var row = 0; row < inputHeight; ++row)
			    for (var col = 0; col < inputWidth; ++col)
				{
					var inputOffset = col + row * inputWidth;
					var inputVal = cells[inputOffset];
					var resOffset = col + row * outputWidth;
					
					for (var j = 0; j < kernelHeight; ++j)
					{
						for (var i = 0; i < kernelWidth; ++i)
							res.Cells[i + resOffset] += inputVal;
						
						resOffset += outputWidth;
					}
				}
			
			var ksum = kernel.Sum;
			res.Mul(ksum, res);
		}
		
		unsafe public void Deconvolve(Matrix kernel, Matrix res)
		{
			res.SetToZero();
			
			var inputWidth   = width;
			var inputHeight  = height;
			
			var outputWidth  = res.width;
			var outputHeight = res.height;
			
			var kernelWidth  = kernel.width;
			var kernelHeight = kernel.height;
			
			for (var row = 0; row < inputHeight; ++row)
			    for (var col = 0; col < inputWidth; ++col)
				{
					var inputOffset = col + row * inputWidth;
					var inputVal = cells[inputOffset];
					var resOffset = col + row * outputWidth;
					
					for (var j = 0; j < kernelHeight; ++j)
					{
						for (var i = 0; i < kernelWidth; ++i)
							res.Cells[i + resOffset] += inputVal * kernel.cells[i + j * kernelWidth];
						
						resOffset += outputWidth;
					}
				}
		}

		unsafe public void InvolveAdd(Matrix err, Matrix kernel)
		{
			var inputWidth   = width;
			var inputHeight  = height;
			
			var kernelWidth  = kernel.width;
			var kernelHeight = kernel.height;
			
			var errCells     = err.cells;
			var inputCells   = cells;
			var kernelCells  = kernel.cells;
				
			for (var row = 0; row <= inputHeight - kernelHeight; ++row)
			    for (var col = 0; col <= inputWidth - kernelWidth; ++col)
				{
			    	var errorIndex   = col + row  * err.width; 
					var errorVal     = errCells[errorIndex];
					var inputOffset  = col + row * inputWidth;
					var kernelOffset = 0;
				
					for (var j = 0; j < kernelHeight; ++j)
					{
						for (var i = 0; i < kernelWidth; ++i)
							kernelCells[kernelOffset + i] += inputCells[inputOffset + i] * errorVal;
						
						kernelOffset += kernelWidth;
						inputOffset  += inputWidth;
					}
				}
		}
		
		public void DownsampleBy2(Matrix res)
		{
			for (int j = 0; j < res.height; ++j)
			{
				var inputOffset = 2 * j * width;
				
				for (int i = 0; i < res.width; ++i)
				{
					var targetIndex = i + j * res.width;
					var topIndex = 2 * i + inputOffset;
					var bottomIndex = topIndex + width;
					
					res.cells[targetIndex] = 0.25d * (
							cells[topIndex]    + cells[1 + topIndex] +
							cells[bottomIndex] + cells[1 + bottomIndex]
							);
				}
			}
		}
		
		public void UpsampleBy2(Matrix res)
		{
			for (int j = 0; j < height; ++j)
			{
				var inputOffset  = j * width;
				var outputOffset = 2 * j * res.width;
					
				for (int i = 0; i < width; ++i)
				{
					var val = cells[i + inputOffset];
					var topOffset     = 2 * i + outputOffset;
					var bottomOffset  = topOffset + res.width;
					
					res.cells[topOffset]        = val;
					res.cells[topOffset + 1]    = val;
					res.cells[bottomOffset]     = val;
					res.cells[bottomOffset + 1] = val;
				}
			}
		}

		public void Add (double val, Matrix res)
		{
			var size = Size;
			
			for (var i = 0; i < size; ++i)
				res.cells[i] = cells[i] + val;
		}

		public void Sub (double val, Matrix res)
		{
			var size = Size;
			
			for (var i = 0; i < size; ++i)
				res.cells[i] = cells[i] - val;
		}

		public void Add(Matrix rhs, Matrix res)
		{
			if (Size != rhs.Size)
				throw new Exception("Cant add inequal vectors or matrices.");
			
			var size = Size;
			
			for (var i = 0; i < size; ++i)
				res.cells[i] = cells[i] + rhs.cells[i];
		}

		public void Sub(Matrix rhs, Matrix res)
		{
			if (Size != rhs.Size)
				throw new Exception("Cant add inequal vectors or matrices.");
			
			var size = Size;
			
			for (var i = 0; i < size; ++i)
				res.cells[i] = cells[i] - rhs.cells[i];
		}

		public void Mul(double val, Matrix res)
		{
			Transform(x => x * val, res);
		}

		public void Mul(Matrix rhs, Matrix res)
		{
			if (Size != rhs.Size)
				throw new Exception("Cant mul inequal vectors or matrices.");
			
			var size = Size;
			
			for (var i = 0; i < size; ++i)
				res.cells[i] = cells[i] * rhs.cells[i];
		}
		
		public void Transform (Func<double, double> f, Matrix res)
		{
			var size = Size;
			
			for (var i = 0; i < size; ++i)
			  res.cells[i] = f(cells[i]);
		}
		
		public void Transform(Func<double, double> f) 
		{
			var size = Size;
			
			for (var i = 0; i < size; ++i)
			  cells[i] = f(cells[i]);
		}
		
		#endregion
	}
}