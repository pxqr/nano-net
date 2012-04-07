using System;

using Nanon.Math;
using Nanon.ML;
using Nanon.ML.Regression;

namespace Nanon.ML.Optimization
{
	public class OnlineGradientDescent
	{
		int iterationCount  = 100;
		int refreshRate     = 1;
		double learningRate = 1;
		
		public OnlineGradientDescent(int iterationCountP, double learningRateP, int refreshRateP = 1)
		{
			iterationCount = iterationCountP;
			learningRate   = learningRateP;
			refreshRate    = refreshRateP;
		}
		
		public void Minimize(IHypothesis hypothesis, Vector[] inputs, double[] outputs)
		{
			if (inputs.Length == 0) 
				return;
			
			var inputCount   = inputs.Length;
			var inputSize    = inputs[0].Size;
				
			//  calculate invert avg multiplier
			var contribution  = 1.0d / (double)inputSize;
			//  prevent to passing small inputs away without weigths have been changed
			var currentRefreshRate = System.Math.Min(refreshRate, inputCount);
				
			for (var dummy = 0; dummy < iterationCount; ++dummy)
			{
				var gradAcc = new Vector(inputSize);
				
				for (var i = 0; i < inputCount; ++i)
				{
					// find gradient of cost(loss) function
					gradAcc += hypothesis.Gradient(inputs[i], outputs[i]);
					
					if ((i % currentRefreshRate) == 0)
					{
						// do gradient step
						hypothesis.Weights = hypothesis.Weights - learningRate * contribution * gradAcc;
						// reset accumulator
						gradAcc.Transform(x => 0.0d);
					}
				}
				hypothesis.Weights = hypothesis.Weights - learningRate * contribution * gradAcc;
			}
		}
	}
}

