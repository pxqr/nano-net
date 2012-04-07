using System;

using Nanon.Math;

namespace Nanon.ML.Regression
{
	public class LinearHypothesis : IHypothesis
	{
		Vector weights;
		
		public LinearHypothesis(int size)
		{
			weights = new Vector(size);
		}
		
		public override double Predict(Vector input)
		{
			return weights * input;
		}		
		
		public override double Cost(Vector input, double output)
		{
			var prediction = Predict(input);
			var sqrtCost = prediction - output;
			return 0.5 * sqrtCost * sqrtCost;
		}
		
		public override Vector Gradient(Vector input, double output)
		{
			var cost = Cost(input, output);
			return cost * input;
		}
		
		public override Vector Weights
		{
			get
			{
				return weights;	
			}
			
			set 
			{
				weights = value;	
			}
		}
	}
}