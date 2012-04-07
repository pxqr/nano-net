using System;

namespace Nanon.Math.Sigmoid
{
	public class Sigmoids
	{
		public static double Logistic(double x)
		{
			return 1 / (1 + System.Math.Exp(-x));
		}
		
		public static double DLogisticX(double logisticX)
		{
			return logisticX * (1 - logisticX);
		}
		
		public static double DLogistic(double x)
		{
			var logisticX = Logistic(x);
			return DLogistic(logisticX);
		}
		

		
		public static double Tanh(double x)
		{
			var exp2X = System.Math.Exp(2 * x);
			return (exp2X - 1) / (exp2X + 1);
		}		
		
		public static double DTanhX(double tanhx)
		{
			return 1 - tanhx * tanhx;
		}		
		
		public static double DTanh(double x)
		{
			var tanhx = Tanh(x);
			return DTanhX(tanhx);
		}
	}
}

