using System;

using Nanon.Math;

namespace Nanon.ML.Regression
{
	public abstract class IHypothesis
	{
		public abstract double Predict(Vector input);
		public abstract double Cost(Vector input, double output);
		public abstract Vector Gradient(Vector input, double output);
		public abstract Vector Weights { get; set; }
		
		public double[] BatchPredict(Vector[] input)
		{
			var predictions = new double[input.Length];
			
			for (var i = 0; i < input.Length; ++i)
				predictions[i] = Predict(input[i]);
				
			return predictions;
		}
		
		public double BatchCost(Vector[] inputs, double[] outputs)
		{
			var length = inputs.Length;
			double cost = 0;
			
			for (var i = 0; i < length; ++i)
				cost += Cost(inputs[i], outputs[i]);
				
			return cost / (double)length;
		}
		
		public Vector BatchGradient(Vector[] inputs, double[] outputs)
		{
			var length = inputs.Length;
			var size   = inputs[0].Size;
			var grad   = new Vector(size);
			
			for (var i = 0; i < length; ++i)
			{
				grad += Gradient(inputs[i], outputs[i]);
			}
			
			return (1 / (double)length) * grad;
		}
		
	}
}

