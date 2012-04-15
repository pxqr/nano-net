using System;
using System.Linq;

using Nanon.Math.Linear;
using Nanon.Data;
using Nanon.Model;
using Nanon.Learning.Optimization;
using System.Diagnostics;

namespace Nanon.Learning.Tools
{
	public class Trainer<InputT, OutputT>
	{
		IOptimizer<InputT, OutputT> optimizer;
		bool showInfo = true;
		
		public Trainer(IOptimizer<InputT, OutputT> optimizerA)
		{
			optimizer = optimizerA;
		}
		
		public void Train(IHypothesis<InputT, OutputT> hypothesis, IDataSet<InputT, OutputT> dataSet)
		{
			var timer = new Stopwatch();
			timer.Start();
			
			optimizer.Optimize(hypothesis, dataSet.Set);
			
			timer.Stop();
			if (showInfo)
				Console.WriteLine("Training time {0} ms.", timer.ElapsedMilliseconds);
		}

		public bool ShowInfo {
			get {
				return this.showInfo;
			}
			set {
				showInfo = value;
			}
		}
	}
}

