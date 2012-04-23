using System;
using Nanon.Math.Linear;

namespace Nanon.Test
{
	public class MatrixTest
	{
		static void TestConvolution()
		{
			var a = new Matrix(4, 4, new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
			var b = new Matrix(3, 3, new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9});
			var c = new Matrix(2, 2);
			a.Convolve(b, c);
			c.ToVector.Show(2);
		}
		
		static void TestDeconv()
		{
			var a = new Matrix(4, 4);
			var b = new Matrix(3, 3, new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
			var c = new Matrix(2, 2, new double[] { 1, 2, 3, 4 });
			c.Deconvolve(b, a);
			a.ToVector.Show(4);
		}
		
		static void SubsamplingTest()
		{
			var a = new Matrix(4, 4, new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });
			var c = new Matrix(2, 2);
			a.DownsampleBy2(c);
			c.ToVector.Show(2);
			
			var b = new Matrix(4, 4);
			c.UpsampleBy2(b);
			b.ToVector.Show (4);
		}
		
		static void ShiftTest()
		{
			var a = new Vector(new double[] { 1, 2, 3, 4, 5, 6 });
			var b = a.Copy();
			

		}
		
		
		public static void Test ()
		{
			ShiftTest();
			TestDeconv();
			TestConvolution();
			
			SubsamplingTest();			
		}
	}
}

