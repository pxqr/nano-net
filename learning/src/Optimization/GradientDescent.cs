using System;
using System.Collections.Generic;
using System.Linq;

using Nanon.Math.Linear;
using Nanon.Math.Series;
using Nanon.Model;

namespace Nanon.Learning.Optimization
{
	//
	//  Straightforward stohastic gradient descent. 
	//  It can be used in either online or batch learning.
	//  
	public class GradientDescent<InputT, OutputT> : IOptimizer<InputT, OutputT>
	{
		const double DoNotCheckCost = -1.0d;
			
		int iterationCount  = 100;
		int initialStepSize = 1;
		double learningRate = 1;
		bool showInfo = true;
		
		//  learningProgression series should ever divergent!
		Func<int, double> learningProgression =  Series.HarmonicSeries;
		
		public GradientDescent(int iterationCountP, double learningRateP, Func<int, double> learningProgressionA, int initialStepSizeP = 1)
		{
			iterationCount  = iterationCountP;
			learningRate    = learningRateP;
			initialStepSize = initialStepSizeP;
			learningProgression = learningProgressionA;
		}
		
		public bool ShowInfo 
		{
			get
			{
				return showInfo;
			}
			set
			{
				showInfo = value;
			}
		}
		
		//  check whether average cost if less than costThreshold or not
		//  costIsGoodEnough hyp = (((< threshold) . average . map (cost hyp) .) . zip)
		double Cost(IHypothesis<InputT, OutputT> hypothesis, IEnumerable<Tuple<InputT, OutputT> > exsamples)
		{
			var inputCount   = exsamples.Count();
			double costs     = 0.0d;  // acc
			
			foreach(var ex in exsamples)
			{
				costs += hypothesis.Cost(ex.Item1, ex.Item2);	
			}
			
			var avgCost      = (costs / (double) inputCount);
			
			return avgCost;
		}
				
		double GradSumm(Vector[] gradients)
		{
			var acc = 0.0d;
			foreach(var gradient in gradients)
				acc += gradient.ToVector.Map(System.Math.Abs).Mean;
			
			return acc;
		}
		
		void Update(IHypothesis<InputT, OutputT> hypothesis, Vector[] grads, double coeff)
		{
			var layerNum = 1;
			// correct hypothesis
			foreach(var gradient in grads)
			{
				var finalFactor = coeff * (1.0d / (double)layerNum);
				gradient.Multiply(finalFactor, gradient);
				//++layerNum;
			}
			
			hypothesis.Correct(grads);
		}
		
		double DoGradientStep(IHypothesis<InputT, OutputT> hypothesis, IEnumerable<Tuple<InputT, OutputT> > exsamples, double coeff, int stepSize)
		{
			var inputCount   = exsamples.Count();
			
			//  prevent to passing a small inputs away without weigths have been changed
			var boundedStepSize = System.Math.Min(stepSize, inputCount);
				
			var gradAcc   = hypothesis.ZeroGradient;
			var gradCount = gradAcc.Length;
			
			var i = 0;
			var batchSize = 0;
			
			foreach(var ex in exsamples)
			{
				var currGrad = hypothesis.Gradient(ex.Item1, ex.Item2);
				
				// gather gradient to the accumulator
				for (var gradientIndex = 0; gradientIndex < gradCount; ++gradientIndex)
				  gradAcc[gradientIndex].Add(currGrad[gradientIndex], gradAcc[gradientIndex]);
				++batchSize;
				
				// gradient step - correct hypothesis 
				var isLastSample   = i == (inputCount - 1);
				var batchIsFull    = batchSize == boundedStepSize;
				var isTimeToUpdate = batchIsFull || isLastSample;
				
				if (isTimeToUpdate)
				{
					//  calculate invert avg multiplier
					var factor  = (coeff / (double)batchSize) * learningRate;
					
					Update(hypothesis, gradAcc, factor);
					
					if (!isLastSample)
					{
						// reset accumulators
						foreach(var grad in gradAcc)
							grad.SetToZero();
						
						batchSize = 0;
					}
				}
				++i;
			}
			
			//  diagnostic info
			return GradSumm(gradAcc);
		}
		
		public void Optimize(IHypothesis<InputT, OutputT> hypothesis, IEnumerable<Tuple<InputT, OutputT> > exsamples)
		{
			if (exsamples.Count() == 0)
				return;
			
			var stepSize = initialStepSize;
			var cost = Cost(hypothesis, exsamples);
				
			if (showInfo)
			{
				Console.WriteLine("initial cost   : {0}", cost);
			}
			
			for (var iteration = 1; iteration <= iterationCount; ++iteration)
			{
				var coeff =  learningProgression(iteration);
				var lastGradSum = DoGradientStep(hypothesis, exsamples, coeff, stepSize);
				var newCost = Cost(hypothesis, exsamples);
				
				if (showInfo)
				{
					Console.WriteLine("obtain new cost : {0} last grad sum : {1}", newCost, lastGradSum);
					if (newCost > cost)
						Console.WriteLine("Warning: probably weights will divergent!");
				}
				cost = newCost;
				stepSize *= 2;
			}
		}
	}
}