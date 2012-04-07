using System;

using Nanon.Data;
using Nanon.Model;

namespace Nanon.Learning.Tools
{
	public class HypothesisTester<InputT, OutputT> : ITester<InputT, OutputT>
	{
		IHypothesis<InputT, OutputT> hypothesis;
		
		public HypothesisTester(IHypothesis<InputT, OutputT> hypothesisA)
		{
			hypothesis = hypothesisA;
		}
		
		public double Test(IDataSet<InputT, OutputT> dataSet)
		{
			int setSize = dataSet.Size;
			
			if (setSize == 0)
				return 0.0d;
			
			var costAcc = 0.0d;
			
			foreach(var x in dataSet.Set)
			{
				costAcc += hypothesis.Cost(x.Item1, x.Item2);
			}
				
			return costAcc / (double)setSize;
		}
	}
}

