using System;
using System.Linq;

using Nanon.Math.Linear;
using Nanon.Data;
using Nanon.Model;
using Nanon.Learning.Optimization;

namespace Nanon.Learning.Tools
{
	public class Trainer<InputT, OutputT>
	{
		IOptimizer<InputT, OutputT> optimizer;
		
		public Trainer(IOptimizer<InputT, OutputT> optimizerA)
		{
			optimizer = optimizerA;
		}
		
		public void Train(IHypothesis<InputT, OutputT> hypothesis, IDataSet<InputT, OutputT> dataSet)
		{
			optimizer.Optimize(hypothesis, dataSet.Set);
		}
	}
}

