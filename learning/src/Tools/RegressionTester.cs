using System;

using Nanon.Data;
using Nanon.Math.Linear;
using Nanon.Model.Regression;

namespace Nanon.Learning.Tools
{
	public class RegressionTester<InputT, OutputT> : ITester<InputT, OutputT>
	{
		IRegression<InputT, OutputT> regression;
		Func<OutputT, OutputT, double> cost;
			
		public RegressionTester(IRegression<InputT, OutputT> regressionA, Func<OutputT, OutputT, double> costA)
		{
			regression = regressionA;
			cost = costA;
		}
		
		// cost
		public double Test(IDataSet<InputT, OutputT> dataSet)
		{
			int setSize = dataSet.Size;
			
			if (setSize == 0)
				return 0.0d;
			
			var costAcc = 0.0d;
			
			foreach(var x in dataSet.Set)
			{
				costAcc += cost(regression.Predict(x.Item1), x.Item2);
			}
				
			return costAcc / (double)setSize;
		}
		
	}
}

