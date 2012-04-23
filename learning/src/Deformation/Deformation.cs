using System;
using System.Collections.Generic;
using Nanon.Math.Linear;
using Nanon.Data;

namespace Nanon.Learning.Deformation
{
	public class Deformation
	{
		static Random rand = new Random();
		
		public static Matrix Shift(Matrix x, int maxShiftX, int maxShiftY)
		{
			var xShift = rand.Next(-maxShiftX, maxShiftX);
			var yShift = rand.Next(-maxShiftY, maxShiftY);
			var res = x.Copy();
			res.Shift(xShift, yShift);
			return res;
		}
		
		public static Matrix Distortion(Matrix x, int kw, int kh, double std, double strength)
		{
			var dispWidth  = x.Width  + kw - 1;
			var dispHeight = x.Height + kh - 1;
			
			var hdispRand = Matrix.RandomUnform(dispWidth, dispHeight, strength);
			var vdispRand = Matrix.RandomUnform(dispWidth, dispHeight, strength);
			
			var kernel = new Matrix(kw, kh);
			kernel.MakeGaussian(std);
			
			var hdisp = new Matrix(x.Width, x.Height);
			var vdisp = new Matrix(x.Width, x.Height);
			
			hdispRand.Convolve(kernel, hdisp);
			vdispRand.Convolve(kernel, vdisp);
			
			var res = x.ZeroCopy();
			x.Distort(hdisp, vdisp, res);
			return res;
		}
	}
}

