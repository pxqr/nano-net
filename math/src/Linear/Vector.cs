using System;
using System.Linq;

namespace Nanon.Math.Linear
{
	public interface IVector
	{
		double[] Cells { get; }
		Vector ToVector { get; }
	}
	
	public class Vector : IVector
	{
		double[] cells;
		
		//
		// Basic constructors.
		//
		public Vector(int size)
		{
			cells = new double[size];
		}
		
		public Vector(double[] comp)
		{
			cells = comp;
		}
		
		public Vector ToVector
		{
			get
			{
				return this;	
			}
		}
		
		public int IndexOfMax
		{
			get
			{
				var size  = Size;
				var index = 0;
				var max   = -1000000000000000000000000000.0d;
				
				for (var i = 0; i < size; ++i)
					if (cells[i] > max)
				    {
						index = i;
					    max   = cells[i];
				    }
				
				return index;
			}
		}
		
		// Raw 
		public double[] Cells {
			get {
				return this.cells;
			}
		}		
		
		// 
		public int Size
		{
			get
			{
				return cells.Length;	
			}
		}
		
		
		// 
		public Vector ZeroCopy()
		{
			return new Vector(Size);	
		}
		
		public void SetToZero()
		{
			Transform(x => 0);
		}
		
		public static Vector FromIndex(int index, int size)
		{
			if (size <= index)
				throw new ArgumentException("size less than index");
			
			var v = new Vector(size);
			v.cells[index] = 1.0;
			return v;
		}
		
		public static Vector Prepend(Vector vector)
		{
			var oldSize = vector.Size;
			var newSize = oldSize + 1;
			var res = new Vector(newSize);
			
			Array.Copy(vector.cells, 0, res.cells, 1, oldSize);
			res.cells[0] = 1;
			
			return res;
		}
		
		//  Print vector to console output.
		public void Show(int width)
		{
			for (var i = 0; i < Size; ++i)
			{
				if (i % width == 0) 
					Console.WriteLine();
				
				Console.Write("{0,6} ", cells[i]);
			}
		}
		
		public void Show()
		{
			Show(Size);	
		}
		
		//////////////////////////////////////////////////////////////////////////////
		//                             Unsafe BLAS section
		//  Lack error cheking and impure => use only in well-tested performance critical
		//  code after rigorous profiling. Otherwise use "pretty BLAS" methods.
		//////////////////////////////////////////////////////////////////////////////
		
		// Dot product of two vectors.
		public double DotProduct(Vector lhs)
		{
			double product = 0;
			
			var size = Size;
			var rhsc = cells;
			var lhsc = lhs.cells;
			
			for (var i = 0; i < size; ++i)
				product += rhsc[i] * lhsc[i];
			
			return product;
		}
		
		public double DotProductWithBias(Vector lhs)
		{
			double product = cells[0];
			
			var size = lhs.Size;
			var rhsc = cells;
			var lhsc = lhs.cells;
			
			for (var i = 0; i < size; ++i)
				product += rhsc[i + 1] * lhsc[i];
			
			return product;
		}
		
		public void Transform(Func<double, double> f)
		{
			var size = Size;
			
			for (var i = 0; i < size; ++i)
			{
				cells[i] = f(cells[i]);
			}
		}
		
		public static  void  Transform(Func<double, double> f, Vector src, Vector res)
		{
			var size = src.Size;
			
			for (var i = 0; i < size; ++i)
			{
				res.cells[i] = f(src.cells[i]);
			}
		}		
		
		//
		public void Multiply(double scalar, Vector res)
		{
			var size = Size;
			var resCells = res.cells;
			
			for (var i = 0; i < size; ++i)
				resCells[i] = scalar * cells[i];
		}
		
		
		public void Multiply(Vector lhs, Vector res)
		{
			var size = lhs.Size;
			var resCells = res.cells;
			var lhsCells = this.cells;
			var rhsCells = lhs.cells;
			
			for (var i = 0; i < size; ++i)
				resCells[i] = rhsCells[i] * lhsCells[i];
		}
		
		public void Add(Vector lhs, Vector res)
		{
			var size = lhs.Size;
			var resCells = res.cells;
			var lhsCells = this.cells;
			var rhsCells = lhs.cells;
			
			for (var i = 0; i < size; ++i)
				resCells[i] = rhsCells[i] + lhsCells[i];
		}
		
		public void Sub(Vector lhs, Vector res)
		{
			var size = lhs.Size;
			var resCells = res.cells;
			var lhsCells = this.cells;
			var rhsCells = lhs.cells;
			
			for (var i = 0; i < size; ++i)
				resCells[i] = rhsCells[i] - lhsCells[i];
		}
		
		//////////////////////////////////////////////////////////////////////////////
		//                        "Pretty" BLAS
		//  It's can create new objects during procedure lifetime
		//  thus not so much efficient. In inner loops, it would be better to 
		//  replace each this procedure to more efficient not "pretty" analog.
		//////////////////////////////////////////////////////////////////////////////
		
		// Dot product of two vectors. Commutative.
		public static double operator * (Vector rhs, Vector lhs)
		{
			if (rhs.Size != lhs.Size)
				throw new Exception("Dot product of different " +
									"length vectors doesnt exist.");
		   	return rhs.DotProduct(lhs);
		}
		
		// Component-wise scalar x vector multiplication.
		// Commutative and associative.
		public static Vector operator * (double rhs, Vector lhs)
		{
			var res = lhs.ZeroCopy();
			lhs.Multiply(rhs, res);
			return res;
		}
		
		// Component-wise vector x scalar multiplication.
		// Commutative and associative.
		public static Vector operator * (Vector rhs, double lhs)
		{
			var res = rhs.ZeroCopy();
			rhs.Multiply(lhs, res);
			return res;
		}
		
		// Component-wise scalar and scalar addition. 
		// Associative and commutative.
		public static Vector operator + (Vector rhs, Vector lhs)
		{
			if (rhs.Size != lhs.Size)
				throw new Exception("Sum of different " +
									"length vectors doesnt exist.");
			
			var res = rhs.ZeroCopy();
			rhs.Add(lhs, res);
			return res;
		}
		
		// Component-wise scalar and scalar substraction. 
		// Associative and commutative.
		public static Vector operator - (Vector rhs, Vector lhs)
		{
			if (rhs.Size != lhs.Size)
				throw new Exception("Difference between of different " +
									"length vectors doesnt exist.");
			var res = rhs.ZeroCopy();
			rhs.Sub(lhs, res);
			return res;
		}
		
		//////////////////////////////////////////////////////////////////////////////
		//             Utilities
		//////////////////////////////////////////////////////////////////////////////
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
		
		public double EuclideanNorm
		{
			get
			{
			  return System.Math.Sqrt(this * this);	
			}
		}
		
		public Vector Normalized
		{
			get
			{
				var mean = Mean;
				var std  = Std;
				
				double epsilon = 0.0001;
				if (System.Math.Abs(std) < epsilon)
						std = 1.0;
				
				return new Vector(
					Array.ConvertAll(cells, x => (x - mean) / std));
			}
		}
		
		
		//              Functional like
		public Vector Map(Func<double, double> f)
		{
			return new Vector(Array.ConvertAll(cells, x => f(x)));
		}
		

		
		public Vector ZipWith(Vector lhs, Func<double, double, double> f)
		{
			var size = Size;
			var res  = ZeroCopy();
			
			var lftCells = lhs.cells;
		    var rhsCells = this.cells;
			var resCells = res.cells;
			
			for (int i = 0; i < size; ++i)
				resCells[i] = f(lftCells[i], rhsCells[i]);
				
		    return res;
		}
		
		public double this [int i]
		{
			get 
			{
			    return cells[i];	
			}
			set 
			{
				cells[i] = value;	
			}
		}
	}
}

