using System;

using Nanon.Data;
using Nanon.Math.Linear;
using Nanon.Model.Regression;

namespace Nanon.Learning.Tools
{
	public class RegressionTester<InputT> : ITester<InputT, Vector>
	{
		IRegression<InputT, Vector> regression;
			
		public RegressionTester(IRegression<InputT, Vector> regressionA)
		{
			regression = regressionA;
		}
		
		// cost
		public double Test(IDataSet<InputT, Vector> dataSet)
		{
			int setSize = dataSet.Size;
			
			if (setSize == 0)
				return 0.0d;
			
			var costAcc = 0.0d;
			
			foreach(var x in dataSet.Set)
			{
				costAcc += Cost(regression.Predict(x.Item1), x.Item2);
			}
				
			return costAcc / (double)setSize;
		}
		
		public static double Cost(Vector prediction, Vector output)
		{
			return (prediction - output).EuclideanNorm;
		}
	}
}

