using System;

using Nanon.Math;

namespace Nanon.ML.Regression
{
	public class LogisticHypothesis : IHypothesis
	{
		Vector weights;
		
		public LogisticHypothesis(int size)
		{
			weights = new Vector(size);
		}
		
		public override double Predict(Vector input)
		{
			return Utils.Sigmoid(weights * input);
		}		
		
		public override double Cost(Vector input, double output)
		{
			var h = Predict(input);
			var ifmatch = output * System.Math.Log(h);
			var ifnotmh = (1 - output) * System.Math.Log(1 - h);
			return - (ifmatch + ifnotmh);
		}
		
		public override Vector Gradient(Vector input, double output)
		{
			var cost = Cost(input, output);
			return cost * input;
		}
		
		public override Vector Weights
		{
			get	{ return weights; }
			set { weights = value;	}
		}		
	}
}

