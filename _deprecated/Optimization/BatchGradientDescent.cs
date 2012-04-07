using System;

using Nanon.Math;
using Nanon.ML;
using Nanon.ML.Regression;

namespace Nanon.ML.Optimization
{
	//  
	//  http://www.willamette.edu/~gorr/classes/cs449/momrate.html
	//
	public class BatchGradientDescent
	{
		int iterationCount  = 100;
		double learningRate = 1;
		
		public enum Rate { Const };
		
		public BatchGradientDescent(int iterationCnt, double learningRateP)
		{
			iterationCount = iterationCnt;
			learningRate   = learningRateP;
		}
		
		public void Minimize(IHypothesis hypothesis, Vector[] inputs, double[] outputs)
		{
			if (iterationCount < 1) return;
			
			// first iteration
			var prevGrad  = hypothesis.BatchGradient(inputs, outputs);
			hypothesis.Weights = hypothesis.Weights - learningRate * prevGrad;
			
			// other iterations
			for (var dummy = 1; dummy < iterationCount; ++dummy)
			{
				// find gradient of cost(loss) function
				var grad = hypothesis.BatchGradient(inputs, outputs);
				// find Euclidean norm of delta of last two gradients
				var dGrad = (grad - prevGrad);
				var dGradNorm = dGrad.EuclideanNorm;
				// adopt adaptive learning rate
				var rate = learningRate * dGradNorm;
				// do gradient step
				hypothesis.Weights = hypothesis.Weights - (learningRate * dGradNorm) * grad;
				
				// store gradient thus we can easily find delta of gradient
				prevGrad = grad;
			}
		}
	}
}