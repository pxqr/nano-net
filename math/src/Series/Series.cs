using System;

namespace Nanon.Math.Series
{
	public class Series
	{
		public static double HarmonicSeries(int i) 
		{
			return 1.0d / (double)i;
		}
		
		public static double CostantSeries(int i)
		{
			return (double)i;
		}
	}
}

